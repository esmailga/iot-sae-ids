"""Detection metrics: Accuracy, Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, confusion matrix,
and per-attack-category recall. Label convention: 1 = attack (positive), 0 = normal."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return ``{"tn","fp","fn","tp"}`` counts."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def threshold_free(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Optional[float]]:
    """ROC-AUC and PR-AUC (average precision) from continuous scores (higher = attack)."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    s = np.asarray(scores, dtype="float64")
    return {"roc_auc": float(roc_auc_score(y_true, s)),
            "pr_auc": float(average_precision_score(y_true, s))}


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray] = None
) -> dict:
    """All binary detection metrics. ``scores`` (optional) enables ROC-AUC / PR-AUC."""
    c = confusion(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    out = {
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "fpr": _div(fp, fp + tn),
        "confusion": c,
    }
    if scores is not None:
        out.update(threshold_free(y_true, scores))
    return out


def per_category_recall(
    y_multi: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict[str, float | int]]:
    """Recall (detection rate) for each attack category, over attack rows only.

    For normal categories the 'recall' reported is the fraction correctly kept as normal
    (i.e. 1 - FPR within that category), flagged with ``"normal": True``.
    """
    y_multi = np.asarray(y_multi).astype(str)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out: dict[str, dict[str, float | int]] = {}
    for cat in sorted(np.unique(y_multi)):
        mask = y_multi == cat
        is_normal = bool((y_true[mask] == 0).all())
        correct = (y_pred[mask] == (0 if is_normal else 1)).sum()
        out[cat] = {"n": int(mask.sum()), "recall": _div(correct, mask.sum()),
                    "normal": is_normal}
    return out
