"""Reconstruction-error thresholds for the SAE.

Decision rule everywhere: a flow is an attack iff ``error > tau``.

1. Unsupervised: tau = p-th percentile of the *normal* validation errors.
2. Validation-optimised: tau maximising F1 on the labelled validation split
   (uses attack labels for threshold selection only).
3. Threshold-free: ROC-AUC / PR-AUC of the raw errors (see :mod:`ids.evaluation.metrics`).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ids.evaluation.metrics import binary_metrics, threshold_free


def predict(errors: np.ndarray, tau: float) -> np.ndarray:
    """Binary predictions (1 = attack) for ``errors > tau``."""
    return (np.asarray(errors) > tau).astype("int8")


def percentile_threshold(normal_val_errors: np.ndarray, p: float) -> float:
    """tau = p-th percentile (0-100) of normal validation errors."""
    if len(normal_val_errors) == 0:
        raise ValueError("No normal validation errors to take a percentile of")
    return float(np.percentile(np.asarray(normal_val_errors, dtype="float64"), p))


def best_f1_threshold(errors: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Return ``(tau, f1)`` maximising F1 of ``errors > tau`` on labelled data.

    Exact and vectorised: every distinct cut between sorted error values is evaluated.
    tau is placed at the midpoint between adjacent distinct errors (and just below the
    minimum for the "everything is an attack" cut). Ties in F1 pick the larger tau
    (fewer alarms).
    """
    s = np.asarray(errors, dtype="float64")
    y = np.asarray(y_true).astype(int)
    if s.size == 0:
        raise ValueError("empty errors")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    pos = int(y.sum())
    # Valid cut after index i only where the next value is strictly smaller (or at the end).
    last = np.r_[s_sorted[:-1] > s_sorted[1:], True]
    idx = np.flatnonzero(last)
    tp_c, fp_c = tp[idx], fp[idx]
    fn_c = pos - tp_c
    denom = 2 * tp_c + fp_c + fn_c
    f1 = np.divide(2 * tp_c, denom, out=np.zeros_like(denom, dtype="float64"), where=denom > 0)
    # Candidate "nothing flagged" (tau >= max): F1 = 0, only chosen if all else is 0 too.
    best = int(np.argmax(f1))  # first max = largest tau among ties (sorted descending)
    i = idx[best]
    if i + 1 < len(s_sorted):
        tau = 0.5 * (s_sorted[i] + s_sorted[i + 1])
    else:
        tau = float(np.nextafter(s_sorted[i], -np.inf))
    return float(tau), float(f1[best])


def evaluate_thresholds(
    val_errors: np.ndarray,
    val_y: np.ndarray,
    test_errors: np.ndarray,
    test_y: np.ndarray,
    percentiles: Iterable[float],
) -> dict:
    """Compute every threshold on validation data and score it on test.

    Returns:
        ``{"percentile": {"p95": {"tau":..., "test": metrics, "val": metrics}, ...},
           "val_f1": {"tau":..., "val_f1":..., "test": metrics, "val": metrics},
           "threshold_free": {"val": {...}, "test": {...}}}``
    """
    val_errors = np.asarray(val_errors)
    val_y = np.asarray(val_y).astype(int)
    normal_val = val_errors[val_y == 0]
    out: dict = {"percentile": {}}
    for p in percentiles:
        tau = percentile_threshold(normal_val, p)
        out["percentile"][f"p{p:g}"] = {
            "p": float(p), "tau": tau,
            "val": binary_metrics(val_y, predict(val_errors, tau)),
            "test": binary_metrics(test_y, predict(test_errors, tau), test_errors),
        }
    tau, f1 = best_f1_threshold(val_errors, val_y)
    out["val_f1"] = {
        "tau": tau, "val_f1": f1,
        "uses_attack_labels": True,
        "val": binary_metrics(val_y, predict(val_errors, tau)),
        "test": binary_metrics(test_y, predict(test_errors, tau), test_errors),
    }
    out["threshold_free"] = {"val": threshold_free(val_y, val_errors),
                             "test": threshold_free(test_y, test_errors)}
    return out


def headline_key(name: str) -> tuple[str, str | None]:
    """Map config ``threshold.headline`` (``val_f1`` or ``p95``) to a lookup path."""
    if name == "val_f1":
        return "val_f1", None
    if name.startswith("p"):
        return "percentile", name
    raise ValueError(f"Unknown headline threshold {name!r}")
