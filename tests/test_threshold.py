"""Tests for threshold selection and threshold-based evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from ids.detection.threshold import (
    best_f1_threshold,
    best_youden_threshold,
    evaluate_thresholds,
    headline_key,
    percentile_threshold,
    predict,
)


def _errors(seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    err = np.where(y == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)) ** 2
    return err, y


def test_predict_is_strictly_greater() -> None:
    np.testing.assert_array_equal(predict(np.array([0.1, 0.5, 0.9]), 0.5), [0, 0, 1])


def test_percentile_threshold_fpr() -> None:
    rng = np.random.default_rng(0)
    normal = rng.exponential(size=10_000)
    for p in (90, 95, 99):
        tau = percentile_threshold(normal, p)
        assert tau == pytest.approx(np.percentile(normal, p))
        assert predict(normal, tau).mean() == pytest.approx(1 - p / 100, abs=2e-3)


def test_best_f1_matches_brute_force() -> None:
    err, y = _errors()
    tau, f1 = best_f1_threshold(err, y)
    assert f1 == pytest.approx(f1_score(y, predict(err, tau)))
    # Brute force over every distinct cut
    u = np.unique(err)
    cands = np.r_[u[0] - 1, (u[:-1] + u[1:]) / 2]
    brute = max(f1_score(y, predict(err, t), zero_division=0) for t in cands)
    assert f1 == pytest.approx(brute)


def test_best_f1_with_ties_and_perfect_separation() -> None:
    err = np.array([0.1, 0.1, 0.2, 0.2, 0.9, 0.9, 1.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    tau, f1 = best_f1_threshold(err, y)
    assert f1 == pytest.approx(1.0)
    assert 0.2 < tau < 0.9


def test_best_f1_all_attack_cut() -> None:
    err = np.array([0.5, 0.4, 0.3])
    y = np.array([1, 1, 1])
    tau, f1 = best_f1_threshold(err, y)
    assert f1 == pytest.approx(1.0)
    assert predict(err, tau).all()


def test_best_youden_matches_brute_force() -> None:
    err, y = _errors(3)
    tau, j = best_youden_threshold(err, y)
    p = predict(err, tau)
    tpr = (p[y == 1] == 1).mean()
    fpr = (p[y == 0] == 1).mean()
    assert j == pytest.approx(tpr - fpr)
    u = np.unique(err)
    cands = np.r_[u[0] - 1, (u[:-1] + u[1:]) / 2]
    brute = max((predict(err, t)[y == 1] == 1).mean() - (predict(err, t)[y == 0] == 1).mean()
                for t in cands)
    assert j == pytest.approx(brute)


def test_youden_ignores_prevalence_where_f1_does_not() -> None:
    # Attack-majority validation: F1 favours flagging nearly everything; J does not.
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(200, int), np.ones(1800, int)]
    err = np.r_[rng.normal(0, 1, 200), rng.normal(1.5, 1, 1800)]
    tau_f1, _ = best_f1_threshold(err, y)
    tau_j, _ = best_youden_threshold(err, y)
    fpr = lambda t: predict(err, t)[y == 0].mean()  # noqa: E731
    assert tau_j > tau_f1 and fpr(tau_j) < fpr(tau_f1)


def test_evaluate_thresholds_structure() -> None:
    ev, yv = _errors(1)
    et, yt = _errors(2)
    out = evaluate_thresholds(ev, yv, et, yt, [90, 95, 97.5, 99, 99.5, 99.9])
    assert set(out["percentile"]) == {"p90", "p95", "p97.5", "p99", "p99.5", "p99.9"}
    taus = [out["percentile"][k]["tau"] for k in ("p90", "p95", "p99", "p99.9")]
    assert taus == sorted(taus)
    # Higher percentile -> lower (or equal) validation FPR
    fprs = [out["percentile"][k]["val"]["fpr"] for k in ("p90", "p95", "p99")]
    assert fprs == sorted(fprs, reverse=True)
    assert out["val_f1"]["uses_attack_labels"] is True
    assert out["val_youden"]["uses_attack_labels"] is True
    assert -1 <= out["val_youden"]["val_youden_j"] <= 1
    assert out["threshold_free"]["test"]["roc_auc"] == pytest.approx(roc_auc_score(yt, et))


def test_headline_key() -> None:
    assert headline_key("val_f1") == ("val_f1", None)
    assert headline_key("val_youden") == ("val_youden", None)
    assert headline_key("p95") == ("percentile", "p95")
    with pytest.raises(ValueError):
        headline_key("bogus")
