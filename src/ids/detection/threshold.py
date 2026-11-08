"""Reconstruction-error thresholds for the SAE.

Decision rule everywhere: a flow is an attack iff ``error > tau``.

1. Unsupervised: tau = p-th percentile of the *normal* validation errors.
2. Validation-optimised (uses attack labels for threshold selection only):
   * ``val_youden`` — tau maximising Youden's J = TPR - FPR (HEADLINE). J does not depend
     on the attack/normal ratio, so it is not pulled towards "flag everything" when attacks
     are the majority of the validation split (they are in all three datasets).
   * ``val_f1`` — tau maximising attack-class F1 (the original proposal; reported for
     transparency — on attack-majority data it tends to accept a high FPR).
3. Threshold-free: ROC-AUC / PR-AUC of the raw errors (see :mod:`ids.evaluation.metrics`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ids.evaluation.metrics import binary_metrics, threshold_free

HEADLINE_CHOICES = ("val_youden", "val_f1")  # or "p<percentile>", e.g. "p95"


def predict(errors: np.ndarray, tau: float) -> np.ndarray:
    """Binary predictions (1 = attack) for ``errors > tau``."""
    return (np.asarray(errors) > tau).astype("int8")


def percentile_threshold(normal_val_errors: np.ndarray, p: float) -> float:
    """tau = p-th percentile (0-100) of normal validation errors."""
    if len(normal_val_errors) == 0:
        raise ValueError("No normal validation errors to take a percentile of")
    return float(np.percentile(np.asarray(normal_val_errors, dtype="float64"), p))


@dataclass
class _Cuts:
    """Confusion counts at every distinct cut of the descending-sorted scores.

    Cut ``k`` flags the top ``idx[k] + 1`` rows as attacks.
    """

    s_sorted: np.ndarray
    idx: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    pos: int
    neg: int

    def tau(self, k: int) -> float:
        """Threshold realising cut ``k``: midpoint to the next distinct value, or just
        below the minimum for the "everything is an attack" cut."""
        i = self.idx[k]
        if i + 1 < len(self.s_sorted):
            return float(0.5 * (self.s_sorted[i] + self.s_sorted[i + 1]))
        return float(np.nextafter(self.s_sorted[i], -np.inf))


def _cuts(errors: np.ndarray, y_true: np.ndarray) -> _Cuts:
    s = np.asarray(errors, dtype="float64")
    y = np.asarray(y_true).astype(int)
    if s.size == 0:
        raise ValueError("empty errors")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # A cut after position i is valid only where the next value is strictly smaller.
    idx = np.flatnonzero(np.r_[s_sorted[:-1] > s_sorted[1:], True])
    return _Cuts(s_sorted, idx, np.cumsum(y_sorted)[idx], np.cumsum(1 - y_sorted)[idx],
                 int(y.sum()), int((y == 0).sum()))


def best_f1_threshold(errors: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Return ``(tau, f1)`` maximising F1 of ``errors > tau`` on labelled data.

    Exact and vectorised over every distinct cut. Ties pick the larger tau (fewer alarms).
    """
    c = _cuts(errors, y_true)
    denom = 2 * c.tp + c.fp + (c.pos - c.tp)
    f1 = np.divide(2 * c.tp, denom, out=np.zeros(len(denom)), where=denom > 0)
    k = int(np.argmax(f1))  # first max = largest tau (scores sorted descending)
    return c.tau(k), float(f1[k])


def best_youden_threshold(errors: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Return ``(tau, J)`` maximising Youden's J = TPR - FPR of ``errors > tau``.

    Exact and vectorised over every distinct cut. Ties pick the larger tau (fewer alarms).
    """
    c = _cuts(errors, y_true)
    tpr = c.tp / c.pos if c.pos else np.zeros(len(c.tp))
    fpr = c.fp / c.neg if c.neg else np.zeros(len(c.fp))
    j = tpr - fpr
    k = int(np.argmax(j))
    return c.tau(k), float(j[k])


def _scored(tau: float, val_errors: np.ndarray, val_y: np.ndarray, test_errors: np.ndarray,
            test_y: np.ndarray) -> dict:
    return {"tau": tau,
            "val": binary_metrics(val_y, predict(val_errors, tau)),
            "test": binary_metrics(test_y, predict(test_errors, tau), test_errors)}


def evaluate_thresholds(
    val_errors: np.ndarray,
    val_y: np.ndarray,
    test_errors: np.ndarray,
    test_y: np.ndarray,
    percentiles: Iterable[float],
) -> dict:
    """Compute every threshold on validation data and score it on test.

    Returns:
        ``{"percentile": {"p95": {"p", "tau", "val", "test"}, ...},
           "val_youden": {"tau", "val_youden_j", "uses_attack_labels", "val", "test"},
           "val_f1": {"tau", "val_f1", "uses_attack_labels", "val", "test"},
           "threshold_free": {"val": {...}, "test": {...}}}``
    """
    val_errors = np.asarray(val_errors)
    val_y = np.asarray(val_y).astype(int)
    normal_val = val_errors[val_y == 0]
    args = (val_errors, val_y, test_errors, test_y)
    out: dict = {"percentile": {}}
    for p in percentiles:
        tau = percentile_threshold(normal_val, p)
        out["percentile"][f"p{p:g}"] = {"p": float(p), **_scored(tau, *args)}
    tau, j = best_youden_threshold(val_errors, val_y)
    out["val_youden"] = {"val_youden_j": j, "uses_attack_labels": True, **_scored(tau, *args)}
    tau, f1 = best_f1_threshold(val_errors, val_y)
    out["val_f1"] = {"val_f1": f1, "uses_attack_labels": True, **_scored(tau, *args)}
    out["threshold_free"] = {"val": threshold_free(val_y, val_errors),
                             "test": threshold_free(test_y, test_errors)}
    return out


def headline_key(name: str) -> tuple[str, str | None]:
    """Map config ``threshold.headline`` (``val_youden``, ``val_f1`` or e.g. ``p95``)."""
    if name in HEADLINE_CHOICES:
        return name, None
    if name.startswith("p"):
        return "percentile", name
    raise ValueError(f"Unknown headline threshold {name!r}")
