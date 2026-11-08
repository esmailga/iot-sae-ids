"""Efficiency measurements: inference latency (CPU and GPU), model size, TFLite size, memory.

All timings use ``time.perf_counter``. Inference protocol (spec §10):
batch size 64, 10 warm-up batches, then mean ± std over 100 timed batches;
per-flow latency = batch time / 64.

Notes on fairness:
* sklearn models are timed with ``n_jobs=1``: thread-pool start-up for a 64-row batch would
  otherwise dominate, and single-core is closer to an edge device. This is recorded.
* Keras models are called eagerly (``model(x, training=False)``), the usual low-latency path.
* CPU timing of a Keras model uses a CPU-placed clone so weights are not copied from the GPU
  on every call.
"""

from __future__ import annotations

import gc
import logging
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np

from ids.models.base import DetectorModel

logger = logging.getLogger(__name__)


def _batches(X: np.ndarray, batch_size: int, n: int, seed: int) -> list[np.ndarray]:
    """``n`` random contiguous float32 batches of ``batch_size`` rows (with wrap-around)."""
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(X, dtype="float32")
    if len(X) < batch_size:
        reps = int(np.ceil(batch_size / max(len(X), 1)))
        X = np.tile(X, (reps, 1))
    starts = rng.integers(0, len(X) - batch_size + 1, size=n)
    return [np.ascontiguousarray(X[s:s + batch_size]) for s in starts]


def time_inference(predict_fn: Callable[[np.ndarray], Any], X: np.ndarray, batch_size: int = 64,
                   warmup: int = 10, n_batches: int = 100, seed: int = 0) -> dict[str, float]:
    """Time ``predict_fn`` on random batches. Returns seconds and per-flow milliseconds."""
    batches = _batches(X, batch_size, warmup + n_batches, seed)
    for b in batches[:warmup]:
        predict_fn(b)
    times = np.empty(n_batches)
    for i, b in enumerate(batches[warmup:]):
        t0 = time.perf_counter()
        predict_fn(b)
        times[i] = time.perf_counter() - t0
    return {
        "batch_size": batch_size, "warmup_batches": warmup, "timed_batches": n_batches,
        "batch_mean_s": float(times.mean()), "batch_std_s": float(times.std(ddof=1)),
        "batch_median_s": float(np.median(times)),
        "per_flow_mean_ms": float(times.mean() / batch_size * 1e3),
        "per_flow_std_ms": float(times.std(ddof=1) / batch_size * 1e3),
        "throughput_flows_per_s": float(batch_size / times.mean()),
    }


def peak_memory(predict_fn: Callable[[np.ndarray], Any], X: np.ndarray, batch_size: int = 64,
                n_batches: int = 20, seed: int = 0) -> dict[str, Optional[float]]:
    """Peak memory during inference.

    * ``tracemalloc_peak_mb``: peak Python/NumPy allocations (does not see TF's allocator).
    * ``rss_peak_delta_mb``: max increase of process RSS over the baseline (psutil).
    """
    batches = _batches(X, batch_size, n_batches, seed + 1)
    gc.collect()
    out: dict[str, Optional[float]] = {"tracemalloc_peak_mb": None, "rss_peak_delta_mb": None}
    try:
        import psutil

        proc = psutil.Process()
        base = proc.memory_info().rss
        peak = base
    except Exception:  # pragma: no cover
        proc = None
    tracemalloc.start()
    try:
        for b in batches:
            predict_fn(b)
            if proc is not None:
                peak = max(peak, proc.memory_info().rss)
        _, tm_peak = tracemalloc.get_traced_memory()
        out["tracemalloc_peak_mb"] = tm_peak / 1e6
    finally:
        tracemalloc.stop()
    if proc is not None:
        out["rss_peak_delta_mb"] = (peak - base) / 1e6
    return out


@contextmanager
def _single_thread_sklearn(det: DetectorModel) -> Iterator[dict]:
    """Temporarily set ``n_jobs=1`` on sklearn estimators that support it."""
    changed: list[tuple[Any, Any]] = []
    model = getattr(det, "model", None)
    est_list = [model] + (list(getattr(model, "named_steps", {}).values()) if model else [])
    for est in est_list:
        if est is not None and hasattr(est, "n_jobs"):
            changed.append((est, est.n_jobs))
            est.n_jobs = 1
    try:
        yield {"sklearn_n_jobs": 1 if changed else None}
    finally:
        for est, nj in changed:
            est.n_jobs = nj


@contextmanager
def _keras_on_cpu(det: DetectorModel) -> Iterator[None]:
    """Swap ``det.model`` for a CPU-placed clone inside a ``/CPU:0`` device scope."""
    import tensorflow as tf
    from tensorflow import keras

    with tf.device("/CPU:0"):
        clone = keras.models.clone_model(det.model)
        clone.set_weights(det.model.get_weights())
        orig = det.model
        det.model = clone
        try:
            yield
        finally:
            det.model = orig


def gpu_available() -> bool:
    """True if TensorFlow sees a GPU."""
    try:
        import tensorflow as tf

        return len(tf.config.list_logical_devices("GPU")) > 0
    except Exception:
        return False


def measure_inference(det: DetectorModel, X: np.ndarray, eff_cfg: dict, seed: int) -> dict:
    """Latency on CPU (always, if enabled) and GPU (TF models, if available) + peak memory."""
    kw = dict(batch_size=int(eff_cfg["batch_size"]), warmup=int(eff_cfg["warmup_batches"]),
              n_batches=int(eff_cfg["timed_batches"]), seed=seed)
    mem_batches = int(eff_cfg.get("memory_batches", 20))
    out: dict[str, Any] = {}
    if det.uses_tensorflow:
        import tensorflow as tf

        if eff_cfg.get("measure_gpu", True) and gpu_available():
            with tf.device("/GPU:0"):
                out["gpu"] = time_inference(det.predict_batch, X, **kw)
        if eff_cfg.get("measure_cpu", True):
            with _keras_on_cpu(det):
                out["cpu"] = time_inference(det.predict_batch, X, **kw)
                out["memory"] = peak_memory(det.predict_batch, X, kw["batch_size"], mem_batches, seed)
    else:
        with _single_thread_sklearn(det) as info:
            out["cpu"] = time_inference(det.predict_batch, X, **kw)
            out["memory"] = peak_memory(det.predict_batch, X, kw["batch_size"], mem_batches, seed)
            out.update(info)
    return out


def file_size_mb(path: Path) -> float:
    """Size of a file (or total of a directory) in MB (10^6 bytes)."""
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return p.stat().st_size / 1e6


def tflite_export(keras_model: Any, out_dir: Path, X_sample: np.ndarray, eff_cfg: dict,
                  seed: int) -> dict:
    """Convert a Keras model to TFLite float32 and dynamic-range int8; report size and latency.

    Latency uses the TFLite interpreter on CPU with the same batch protocol.
    """
    import tensorflow as tf

    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {}
    for variant in ("float32", "dynamic_int8"):
        try:
            conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            if variant == "dynamic_int8":
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
            blob = conv.convert()
            path = out_dir / f"sae_{variant}.tflite"
            path.write_bytes(blob)
            entry: dict[str, Any] = {"size_mb": len(blob) / 1e6}
            interp = tf.lite.Interpreter(model_content=blob, num_threads=1)
            inp = interp.get_input_details()[0]
            bs = int(eff_cfg["batch_size"])
            interp.resize_tensor_input(inp["index"], [bs, X_sample.shape[1]])
            interp.allocate_tensors()
            outd = interp.get_output_details()[0]

            def _run(b: np.ndarray) -> np.ndarray:
                interp.set_tensor(inp["index"], b)
                interp.invoke()
                return interp.get_tensor(outd["index"])

            entry["cpu"] = time_inference(_run, X_sample, bs, int(eff_cfg["warmup_batches"]),
                                          int(eff_cfg["timed_batches"]), seed)
            res[variant] = entry
        except Exception as exc:  # conversion support varies by TF version
            logger.warning("TFLite %s export failed: %s", variant, exc)
            res[variant] = {"error": str(exc)}
    return res
