"""One experiment run: (dataset, model, seed[, sample]) -> results/raw/<run>.json.

Steps: prepare data (cached, shared by all models) -> fit (timed) -> score val/test ->
choose threshold (SAE) -> detection metrics + per-category recall -> save model ->
efficiency (latency CPU/GPU, params, size, memory, TFLite for SAE) -> write JSON + scores.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from ids.data.prepare import prepare
from ids.detection.threshold import evaluate_thresholds, headline_key, predict
from ids.evaluation.efficiency import file_size_mb, measure_inference, tflite_export
from ids.evaluation.metrics import binary_metrics, per_category_recall
from ids.models import MODEL_NAMES, make_model
from ids.utils.config import load_config
from ids.utils.env_info import hardware_info, library_versions
from ids.utils.seed import set_global_seed
from ids.utils.timing import Timer

logger = logging.getLogger(__name__)


def run_name(dataset: str, model: str, seed: int, sample: Optional[float]) -> str:
    """File stem for a run, e.g. ``ton_iot__sae__s0`` or ``ton_iot__sae__s0__frac0.02``."""
    return f"{dataset}__{model}__s{seed}" + (f"__frac{sample:g}" if sample else "")


def result_path(cfg: dict, dataset: str, model: str, seed: int, sample: Optional[float]) -> Path:
    """Where a run's JSON is written."""
    return Path(cfg["paths"]["results_raw"]) / f"{run_name(dataset, model, seed, sample)}.json"


def tuned_sae_params(cfg: dict, dataset: str) -> Optional[dict]:
    """Load ``results/best_sae_<dataset>.yaml`` if present."""
    path = Path(cfg["paths"]["results_root"]) / f"best_sae_{dataset}.yaml"
    if not path.is_file():
        return None
    with open(path) as fh:
        best = yaml.safe_load(fh) or {}
    params = best.get("params", best)
    logger.info("Using tuned SAE params from %s: %s", path, params)
    return params


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, dt.datetime)):
        return str(o)
    return repr(o)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written result


def run_experiment(
    dataset: str,
    model: str,
    seed: int,
    sample: Optional[float] = None,
    overrides: Optional[list[str]] = None,
    config_path: Optional[str] = None,
    force: bool = False,
    use_tuned: bool = True,
    verbose: int = 0,
) -> dict:
    """Run one experiment and write its JSON. Returns the result dict (or the existing one)."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_NAMES}")
    cfg = load_config(dataset, overrides, config_path)
    out_path = result_path(cfg, dataset, model, seed, sample)
    if out_path.is_file() and not force:
        logger.info("Skipping %s (result exists)", out_path.name)
        return json.loads(out_path.read_text())

    name = run_name(dataset, model, seed, sample)
    started = dt.datetime.now(dt.timezone.utc)
    set_global_seed(seed)
    with Timer() as t_prep:
        prep = prepare(cfg, seed, sample, plots=sample is None)
    Xtr, ytr, Xva, yva, Xte, yte = (prep.X["train"], prep.y["train"], prep.X["val"],
                                    prep.y["val"], prep.X["test"], prep.y["test"])

    sae_params = tuned_sae_params(cfg, dataset) if (model == "sae" and use_tuned) else None
    set_global_seed(seed)
    det = make_model(model, cfg, seed, sae_params=sae_params, verbose=verbose)
    logger.info("[%s] fitting on d=%d features", name, prep.d)
    with Timer() as t_fit:
        det.fit(Xtr, ytr, Xva, yva)
    logger.info("[%s] training time %.2fs", name, t_fit.seconds)

    with Timer() as t_score:
        val_scores = det.score(Xva)
        test_scores = det.score(Xte)

    result_metrics: dict[str, Any] = {}
    if model == "sae":
        thr = evaluate_thresholds(val_scores, yva, test_scores, yte, cfg["threshold"]["percentiles"])
        section, key = headline_key(cfg["threshold"]["headline"])
        chosen = thr[section] if key is None else thr[section][key]
        tau = chosen["tau"]
        det.set_threshold(tau)
        y_pred = predict(test_scores, tau)
        result_metrics["thresholds"] = thr
        result_metrics["headline_threshold"] = {"name": cfg["threshold"]["headline"], "tau": tau}
        val_pred = predict(val_scores, tau)
    else:
        y_pred = (test_scores > det.default_threshold).astype("int8")
        val_pred = (val_scores > det.default_threshold).astype("int8")
    result_metrics["test"] = binary_metrics(yte, y_pred, test_scores)
    result_metrics["val"] = binary_metrics(yva, val_pred, val_scores)
    result_metrics["per_category_test"] = per_category_recall(prep.m["test"], yte, y_pred)

    model_dir = Path(cfg["paths"]["models_dir"]) / name
    model_path = det.save(model_dir)
    eff: dict[str, Any] = {
        "training_time_s": t_fit.seconds,
        "scoring_time_val_test_s": t_score.seconds,
        "n_params": det.n_params(),
        "model_file": model_path.name,
        "model_size_mb": file_size_mb(model_path),
    }
    eff.update(measure_inference(det, Xte, cfg["efficiency"], seed))
    if model == "sae" and cfg["efficiency"].get("tflite", True):
        eff["tflite"] = tflite_export(det.model, model_dir, Xte, cfg["efficiency"], seed)

    scores_path = Path(cfg["paths"]["results_raw"]) / "scores" / f"{name}.npz"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_path, y_test=yte, test_scores=test_scores, y_pred=y_pred,
                        m_test=prep.m["test"], y_val=yva, val_scores=val_scores)

    result = {
        "run": {"name": name, "dataset": dataset, "model": model, "seed": seed,
                "sample": sample, "started_utc": started.isoformat(),
                "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "prepare_time_s": t_prep.seconds, "used_tuned_sae_params": sae_params is not None},
        "data": {"d": prep.d, "feature_names": prep.feature_names, "counts": prep.counts,
                 "split_sizes": prep.stats.get("split_sizes"),
                 "normal_rows": prep.stats.get("normal_rows"),
                 "cleaning": {k: v for k, v in prep.stats.items()
                              if k.startswith(("train_", "dataset_", "official_"))}},
        "model": det.info,
        "history": det.history,
        "metrics": result_metrics,
        "efficiency": eff,
        "env": {"versions": library_versions(), "hardware": hardware_info()},
        "config": cfg,
    }
    _write_json_atomic(out_path, result)
    h = result_metrics["test"]
    logger.info("[%s] test F1=%.4f recall=%.4f FPR=%.4f ROC-AUC=%s -> %s", name, h["f1"],
                h["recall"], h["fpr"], h.get("roc_auc"), out_path)
    return result
