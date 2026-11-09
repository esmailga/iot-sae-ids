"""Result figures (300-dpi PNG, light background).

Colour: fixed categorical order per model, validated for colour-vision deficiency
(SAE blue, RF orange, SVM aqua, LSTM yellow; adjacent CVD ΔE ≥ 9.1). Aqua and yellow are
below 3:1 contrast on white, so every chart also carries a legend or direct labels, lines get
distinct dash styles, and every figure has a matching table in results/tables/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve  # noqa: E402

from ids.reporting import DATASET_LABEL, DATASET_ORDER, MODEL_LABEL, MODEL_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

COLOR = {"sae": "#2a78d6", "rf": "#eb6834", "svm": "#1baf7a", "lstm": "#eda100"}
DASH = {"sae": "-", "rf": "--", "svm": "-.", "lstm": ":"}
NORMAL_C, ATTACK_C = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
DPI = 300


def apply_style() -> None:
    """Recessive axes and grid, thin marks, readable ink."""
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2.0,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def _datasets(df: pd.DataFrame) -> list[str]:
    return [d for d in DATASET_ORDER if (df.dataset == d).any()]


# ----------------------------------------------------------------------------- bar charts

def metric_bars(df: pd.DataFrame, col: str, label: str, path: Path, log: bool = False,
                lower_better: bool = False, extra: Optional[dict] = None,
                ylim: Optional[tuple[float, float]] = None) -> Path:
    """Small multiples (one panel per dataset): mean ± std of ``col`` per model.

    ``extra``: {dataset: [(label, value, colour)]} additional hatched bars (e.g. SAE TFLite).
    """
    dss = _datasets(df)
    fig, axes = plt.subplots(1, len(dss), figsize=(3.1 * len(dss), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        sub = df[df.dataset == ds]
        names, means, stds, cols, hatches = [], [], [], [], []
        for m in MODEL_ORDER:
            v = sub[sub.model == m][col].dropna()
            names.append(MODEL_LABEL[m])
            means.append(v.mean() if len(v) else np.nan)
            stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
            cols.append(COLOR[m])
            hatches.append(None)
        for lab, val, c in (extra or {}).get(ds, []):
            names.append(lab), means.append(val), stds.append(0.0), cols.append(c)
            hatches.append("////")
        x = np.arange(len(names))
        for i in range(len(names)):
            if np.isnan(means[i]):
                ax.text(x[i], 0.02, "not run", rotation=90, ha="center", va="bottom",
                        transform=ax.get_xaxis_transform(), color=INK2, fontsize=7)
                continue
            ax.bar(x[i], means[i], width=0.68, color="white" if hatches[i] else cols[i],
                   edgecolor=cols[i], hatch=hatches[i], linewidth=1.2 if hatches[i] else 0,
                   yerr=stds[i] or None, capsize=3, error_kw={"elinewidth": 1, "ecolor": INK2})
        ax.set_xticks(x, names, rotation=30 if extra else 0, ha="right" if extra else "center")
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_title(DATASET_LABEL[ds])
        ax.grid(axis="x", visible=False)
        if log:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(label + (" (lower is better)" if lower_better else ""))
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------- confusion

def confusion_matrices(runs: list[dict], ds: str, path: Path) -> Optional[Path]:
    """Row-normalised confusion matrices summed over seeds, one panel per model."""
    sums: dict[str, np.ndarray] = {}
    for j in runs:
        if j["run"]["dataset"] != ds:
            continue
        c = j["metrics"]["test"]["confusion"]
        sums.setdefault(j["run"]["model"], np.zeros((2, 2)))
        sums[j["run"]["model"]] += np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    models = [m for m in MODEL_ORDER if m in sums]
    if not models:
        return None
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 2.7))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        cm = sums[m]
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for k in range(2):
                ax.text(k, i, f"{100 * norm[i, k]:.1f}%\n({int(cm[i, k]):,})", ha="center",
                        va="center", fontsize=7.5, color="white" if norm[i, k] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Normal", "Attack"])
        ax.set_yticks([0, 1], ["Normal", "Attack"])
        ax.set_xlabel("Predicted")
        ax.grid(False)
        ax.set_title(MODEL_LABEL[m])
    axes[0].set_ylabel("Actual")
    fig.suptitle(f"{DATASET_LABEL[ds]} — confusion matrices (test, summed over seeds)",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------- curves

def _scores(scores_dir: Path, ds: str, model: str, seed: int) -> Optional[dict]:
    p = scores_dir / f"{ds}__{model}__s{seed}.npz"
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def roc_pr_curves(scores_dir: Path, ds: str, seed: int, path: Path) -> Optional[Path]:
    """ROC and precision-recall curves for every model (one seed), side by side."""
    data = {m: _scores(scores_dir, ds, m, seed) for m in MODEL_ORDER}
    data = {m: v for m, v in data.items() if v is not None}
    if not data:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for m, s in data.items():
        y, sc = s["y_test"], s["test_scores"]
        fpr, tpr, _ = roc_curve(y, sc)
        prec, rec, _ = precision_recall_curve(y, sc)
        a1.plot(fpr, tpr, DASH[m], color=COLOR[m], label=f"{MODEL_LABEL[m]} (AUC {auc(fpr, tpr):.3f})")
        a2.plot(rec, prec, DASH[m], color=COLOR[m],
                label=f"{MODEL_LABEL[m]} (AP {average_precision_score(y, sc):.3f})")
    a1.plot([0, 1], [0, 1], color=GRID, lw=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    a1.legend(loc="lower right")
    a2.legend(loc="lower left")
    fig.suptitle(f"{DATASET_LABEL[ds]} — test set, seed {seed}", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def error_histogram(scores_dir: Path, run: dict, path: Path, p_ref: str = "p95") -> Optional[Path]:
    """SAE reconstruction-error histogram, normal vs attack, with τ lines (log x)."""
    r = run["run"]
    s = _scores(scores_dir, r["dataset"], "sae", r["seed"])
    if s is None:
        return None
    y, err = s["y_test"], np.clip(s["test_scores"], 1e-12, None)
    bins = np.logspace(np.log10(err.min()), np.log10(err.max()), 70)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.hist(err[y == 0], bins=bins, color=NORMAL_C, alpha=0.55, label=f"Normal (n={int((y == 0).sum()):,})")
    ax.hist(err[y == 1], bins=bins, color=ATTACK_C, alpha=0.55, label=f"Attack (n={int((y == 1).sum()):,})")
    th = run["metrics"]["thresholds"]
    tau_j = th["val_youden"]["tau"]
    tau_p = th["percentile"][p_ref]["tau"]
    ax.axvline(tau_j, color=INK, lw=1.4, label=f"τ Youden's J (headline) = {tau_j:.3g}")
    ax.axvline(tau_p, color=INK2, lw=1.2, ls="--", label=f"τ {p_ref} of normal val = {tau_p:.3g}")
    ax.set_xscale("log")
    ax.set(xlabel="Reconstruction error (MSE, log scale)", ylabel="Test flows",
           title=f"{DATASET_LABEL[r['dataset']]} — SAE reconstruction error (seed {r['seed']})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    return _save(fig, path)


def loss_curves(runs: list[dict], ds: str, seed: int, path: Path) -> Optional[Path]:
    """Training curves: SAE (MSE incl. L1 / val-normal MSE) and LSTM (BCE), separate panels."""
    pick = {j["run"]["model"]: j for j in runs
            if j["run"]["dataset"] == ds and j["run"]["seed"] == seed
            and j["run"]["model"] in ("sae", "lstm") and j.get("history")}
    if not pick:
        return None
    fig, axes = plt.subplots(1, len(pick), figsize=(4.0 * len(pick), 3.1))
    axes = np.atleast_1d(axes)
    for ax, (m, j) in zip(axes, [(m, pick[m]) for m in ("sae", "lstm") if m in pick]):
        h = j["history"]
        ep = np.arange(1, len(h["loss"]) + 1)
        ax.plot(ep, h["loss"], "-", color=COLOR[m], label="train loss")
        val_key = "val_mse" if m == "sae" else "val_loss"
        ax.plot(ep, h[val_key], "--", color=INK2,
                label="val-normal MSE" if m == "sae" else "val loss")
        best = j["model"].get("best_epoch")
        if best:
            ax.axvline(best, color=GRID, lw=1.2)
            ax.text(best, 0.98, f" best epoch {best}", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=7, color=INK2)
        ax.set(xlabel="Epoch", ylabel="MSE (+ L1 term in train loss)" if m == "sae" else "Binary cross-entropy",
               title=f"{MODEL_LABEL[m]}")
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle(f"{DATASET_LABEL[ds]} — training curves (seed {seed})", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------- features

def rf_importance_plot(features_json: Path, path: Path, cumulative: float = 0.95,
                       max_show: int = 40) -> Optional[Path]:
    """RF importance figure from a results/features/<tag>.json report."""
    if not features_json.is_file():
        return None
    rep = json.loads(features_json.read_text())
    title = (f"{DATASET_LABEL.get(rep['dataset'], rep['dataset'])} — RF importance "
             f"(seed {rep['seed']})")
    return rf_importance_bars(pd.Series(rep["rf_importances"]), len(rep["after_stage2_rf"]),
                              title, path, cumulative, max_show)


def rf_importance_bars(importances: pd.Series, k: int, title: str, path: Path,
                       cumulative: float = 0.95, max_show: int = 40) -> Path:
    """Two stacked panels (no dual axis): MDI importance bars, then cumulative importance."""
    apply_style()
    imp = importances.sort_values(ascending=False)
    show = imp.iloc[:max_show]
    cum = (imp.cumsum() / imp.sum()).iloc[:max_show]
    x = np.arange(len(show))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(7.0, 0.22 * len(show)), 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.bar(x, show.to_numpy(), width=0.75,
           color=[COLOR["sae"] if i < k else "#c3c2b7" for i in range(len(show))])
    a1.set_ylabel("MDI importance")
    a1.grid(axis="x", visible=False)
    a1.set_title(f"{title}: kept {k} of {len(imp)} (blue)")
    a2.plot(x, cum.to_numpy(), "-", color=INK2, marker="o", ms=3)
    a2.axhline(cumulative, color=COLOR["rf"], lw=1, ls="--")
    a2.text(0, cumulative - 0.03, f"{cumulative:.0%} target", ha="left", va="top",
            fontsize=7, color=INK2)
    a2.axvline(k - 0.5, color=INK2, lw=0.8, ls=":")
    a2.set_ylim(0, 1.02)
    a2.set_ylabel("Cumulative")
    a2.set_xticks(x, show.index, rotation=90, fontsize=6.5)
    fig.tight_layout()
    return _save(fig, path)


def pearson_heatmap(corr: pd.DataFrame, dropped: list[str], ds: str, seed: int,
                    threshold: float, path: Path) -> Path:
    """Correlation heatmap (diverging red/blue, white at r = 0) with dropped features marked."""
    n = len(corr)
    size = min(max(6.0, 0.16 * n), 16.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    labels = [f"✕ {c}" if c in dropped else c for c in corr.columns]
    fs = max(3.5, 8 - n / 14)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=fs)
    ax.set_yticks(range(n), labels, fontsize=fs)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Pearson r")
    ax.set_title(f"{DATASET_LABEL.get(ds, ds)} — Pearson correlation of encoded train features "
                 f"(seed {seed}); ✕ = dropped at |r| ≥ {threshold}")
    fig.tight_layout()
    return _save(fig, path)
