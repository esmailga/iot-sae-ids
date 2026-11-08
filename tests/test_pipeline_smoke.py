"""End-to-end smoke test on synthetic data: every dataset x model through run_experiment.

Must finish in < 60 s. TF models are included only when TensorFlow is installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from ids.experiment import result_path, run_experiment
from ids.utils.config import DATASETS, load_config
from ids.utils.seed import tf_available
from tests.conftest import RT_CONFIRMED

# Keep the 12-run synthetic pipeline under 60 s on Colab (T4, 2 vCPU) as well as locally.
# Every code path still runs once; only repetition counts are reduced. The GPU timing pass
# is skipped here (it is the same code under a device scope) and covered by smoke_test.py.
FAST = [
    "features.rf_n_estimators=10",
    "baselines.rf.n_estimators=10",
    "sae.epochs=2",
    "baselines.lstm.epochs=1",
    "baselines.lstm.units=[4,2]",
    "efficiency.timed_batches=3",
    "efficiency.warmup_batches=1",
    "efficiency.memory_batches=2",
    "efficiency.measure_gpu=false",
    "efficiency.tflite=false",
]


def test_full_pipeline_smoke(raw_root: Path) -> None:
    models = ["rf", "svm"] + (["sae", "lstm"] if tf_available() else [])
    t0 = time.perf_counter()
    per_run: dict[str, float] = {}
    for ds in DATASETS:
        ov = FAST + (RT_CONFIRMED if ds == "rt_iot2022" else [])
        for model in models:
            t_run = time.perf_counter()
            res = run_experiment(ds, model, seed=0, overrides=ov)
            per_run[f"{ds}/{model}"] = round(time.perf_counter() - t_run, 1)
            path = result_path(load_config(ds, ov), ds, model, 0, None)
            assert path.is_file()
            saved = json.loads(path.read_text())
            m = saved["metrics"]["test"]
            for k in ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"):
                assert k in m
            assert 0 <= m["f1"] <= 1
            e = saved["efficiency"]
            assert e["training_time_s"] > 0 and e["n_params"] > 0 and e["model_size_mb"] > 0
            assert e["cpu"]["per_flow_mean_ms"] > 0
            assert saved["env"]["versions"]["sklearn"]
            assert saved["metrics"]["per_category_test"]
            if model == "sae":
                assert set(saved["metrics"]["thresholds"]["percentile"]) >= {"p95", "p99"}
                assert saved["metrics"]["headline_threshold"]["name"] == "val_youden"
                assert {"val_youden", "val_f1"} <= set(saved["metrics"]["thresholds"])
            scores = np.load(Path(saved["config"]["paths"]["results_raw"]) / "scores"
                             / f"{saved['run']['name']}.npz")
            assert len(scores["test_scores"]) == saved["data"]["split_sizes"]["test"]
            # Resumability: second call skips and returns the stored result.
            again = run_experiment(ds, model, seed=0, overrides=ov)
            assert again["run"]["finished_utc"] == res["run"]["finished_utc"]
    total = time.perf_counter() - t0
    assert total < 60, f"synthetic pipeline took {total:.1f}s; per run (s): {per_run}"


def test_all_models_share_features(raw_root: Path) -> None:
    ov = FAST
    a = run_experiment("ton_iot", "rf", 1, overrides=ov)
    b = run_experiment("ton_iot", "svm", 1, overrides=ov)
    assert a["data"]["feature_names"] == b["data"]["feature_names"]
    assert a["data"]["split_sizes"] == b["data"]["split_sizes"]


@pytest.mark.parametrize("bad", ["nope"])
def test_unknown_model(raw_root: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        run_experiment("ton_iot", bad, 0, overrides=FAST)
