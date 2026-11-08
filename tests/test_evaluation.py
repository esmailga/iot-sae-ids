"""Tests for metrics (vs sklearn), statistics and efficiency helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ids.evaluation.efficiency import peak_memory, time_inference
from ids.evaluation.metrics import binary_metrics, per_category_recall
from ids.evaluation.stats import (
    cohens_dz,
    compare_to_reference,
    mean_std,
    paired_tests,
    rank_biserial,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_binary_metrics_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.3 + rng.random(500) * 0.8, 0, 1)
    p = (s > 0.5).astype(int)
    m = binary_metrics(y, p, s)
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert m["precision"] == pytest.approx(precision_score(y, p))
    assert m["recall"] == pytest.approx(recall_score(y, p))
    assert m["f1"] == pytest.approx(f1_score(y, p))
    assert m["fpr"] == pytest.approx(fp / (fp + tn))
    assert m["confusion"] == {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s))


def test_metrics_degenerate() -> None:
    m = binary_metrics(np.zeros(5), np.zeros(5), np.zeros(5))
    assert m["precision"] == 0 and m["recall"] == 0 and m["roc_auc"] is None


def test_per_category_recall() -> None:
    m = np.array(["normal", "normal", "dos", "dos", "dos", "scan"])
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0, 1, 1, 1, 0, 0])
    out = per_category_recall(m, y, p)
    assert out["dos"]["recall"] == pytest.approx(2 / 3) and out["dos"]["n"] == 3
    assert out["scan"]["recall"] == 0.0
    assert out["normal"]["normal"] is True and out["normal"]["recall"] == 0.5


def test_paired_tests_match_scipy() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.9, 0.02, 15)
    b = a - rng.normal(0.01, 0.01, 15)
    r = paired_tests(a, b)
    assert r["t_p"] == pytest.approx(sps.ttest_rel(a, b).pvalue)
    assert r["w_p"] == pytest.approx(sps.wilcoxon(a, b).pvalue)
    assert r["cohens_dz"] == pytest.approx((a - b).mean() / (a - b).std(ddof=1))
    assert -1 <= r["rank_biserial"] <= 1


def test_effect_sizes_edge_cases() -> None:
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert rank_biserial(np.zeros(3)) == 0.0
    assert cohens_dz(np.zeros(4)) == 0.0
    r = paired_tests(np.ones(5), np.ones(5))
    assert r["w_p"] == 1.0 and not r["w_significant"]


def test_compare_to_reference_and_mean_std() -> None:
    rows = []
    rng = np.random.default_rng(1)
    for ds in ("a", "b", "c"):
        for seed in range(5):
            for model, mu in (("sae", 0.95), ("rf", 0.9), ("svm", 0.95)):
                rows.append({"dataset": ds, "seed": seed, "model": model,
                             "f1": mu + rng.normal(0, 0.005)})
    df = pd.DataFrame(rows)
    res = compare_to_reference(df, ["f1"])
    allscope = res[(res.scope == "all") & (res.baseline == "rf")].iloc[0]
    assert allscope.n_pairs == 15 and allscope.mean_diff > 0 and allscope.w_significant
    assert set(res.scope) == {"all", "a", "b", "c"}
    ms = mean_std(df, ["dataset", "model"], ["f1"])
    assert {"f1_mean", "f1_std", "n"} <= set(ms.columns) and (ms.n == 5).all()


def test_time_inference_and_memory() -> None:
    X = np.random.default_rng(0).random((100, 8)).astype("float32")
    calls = []
    out = time_inference(lambda b: calls.append(b.shape) or b.sum(), X, 64, 10, 100)
    assert len(calls) == 110 and calls[0] == (64, 8)
    assert out["per_flow_mean_ms"] == pytest.approx(out["batch_mean_s"] / 64 * 1e3)
    mem = peak_memory(lambda b: np.zeros((1000, 1000)), X, 64, n_batches=3)
    assert mem["tracemalloc_peak_mb"] >= 7.9  # one 8 MB float64 array


def test_time_inference_small_input() -> None:
    X = np.ones((10, 3), dtype="float32")
    out = time_inference(lambda b: b, X, 64, 1, 5)
    assert out["timed_batches"] == 5
