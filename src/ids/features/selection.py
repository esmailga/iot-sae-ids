"""Two-stage feature optimisation, fitted on the training split only.

Stage 1 — :class:`PearsonFilter`: for each pair with |r| >= threshold, drop the feature with
the higher mean absolute correlation with all other (remaining) features.

Stage 2 — :class:`RFImportanceSelector`: rank features by Random Forest MDI importance and
keep the smallest top-k reaching the cumulative-importance target (default 95%).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class PearsonFilter(BaseEstimator, TransformerMixin):
    """Remove highly correlated features (stage 1).

    The procedure is greedy and order-independent. While some remaining pair has
    |r| >= ``threshold``, take the pair with the largest |r| and drop the member whose mean
    |r| against the other remaining features is higher (ties: drop the later column). This
    is the standard ``findCorrelation`` heuristic with means recomputed after each drop.

    Constant columns have undefined correlation; their correlations are treated as 0.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: object = None) -> "PearsonFilter":
        """Compute the correlation matrix on ``X`` (train) and decide which columns to drop."""
        cols = list(X.columns)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X.to_numpy(dtype="float64"), rowvar=False)
        corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0))
        self.corr_ = pd.DataFrame(corr, index=cols, columns=cols)
        a = np.abs(corr)
        np.fill_diagonal(a, 0.0)

        remaining = np.ones(len(cols), dtype=bool)
        dropped: list[tuple[str, str, float]] = []  # (dropped, kept, |r|)
        while True:
            sub = a[np.ix_(remaining, remaining)]
            if sub.size == 0 or sub.max() < self.threshold:
                break
            idx = np.flatnonzero(remaining)
            i_loc, j_loc = np.unravel_index(np.argmax(sub), sub.shape)
            i, j = sorted((idx[i_loc], idx[j_loc]))
            denom = max(remaining.sum() - 1, 1)
            mean_i = a[i, remaining].sum() / denom
            mean_j = a[j, remaining].sum() / denom
            drop, keep = (i, j) if mean_i > mean_j else (j, i)
            remaining[drop] = False
            dropped.append((cols[drop], cols[keep], float(a[i, j])))

        self.dropped_ = dropped
        self.selected_ = [c for c, r in zip(cols, remaining) if r]
        self.n_features_in_ = len(cols)
        logger.info("PearsonFilter(|r|>=%.2f): %d -> %d features (dropped %d)",
                    self.threshold, len(cols), len(self.selected_), len(dropped))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep only the selected columns."""
        return X[self.selected_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Selected feature names."""
        return np.asarray(self.selected_, dtype=object)


class RFImportanceSelector(BaseEstimator, TransformerMixin):
    """Keep the smallest set of top-ranked RF (MDI) features reaching ``cumulative`` importance.

    Args:
        cumulative: Target cumulative importance, e.g. 0.95.
        n_estimators: Trees in the ranking forest.
        max_rows: If > 0 and the train split is larger, fit on a stratified subsample of this
            many rows (keeps the ranking cheap on large datasets; still train-only).
        random_state: Seed for the forest and the subsample.
        n_jobs: Parallel jobs for the forest.
    """

    def __init__(self, cumulative: float = 0.95, n_estimators: int = 100, max_rows: int = 0,
                 random_state: int = 0, n_jobs: int = -1) -> None:
        self.cumulative = cumulative
        self.n_estimators = n_estimators
        self.max_rows = max_rows
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "RFImportanceSelector":
        """Fit the ranking forest on (a stratified subsample of) the training data."""
        y = np.asarray(y)
        Xf = X
        if self.max_rows and len(X) > self.max_rows:
            idx, _ = train_test_split(np.arange(len(X)), train_size=self.max_rows,
                                      random_state=self.random_state, stratify=y)
            Xf, y = X.iloc[np.sort(idx)], y[np.sort(idx)]
            logger.info("RF selector: fitting on stratified subsample of %d / %d rows",
                        len(Xf), len(X))
        rf = RandomForestClassifier(n_estimators=self.n_estimators, criterion="gini",
                                    random_state=self.random_state, n_jobs=self.n_jobs)
        rf.fit(Xf.to_numpy(dtype="float32"), y)
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        cum = imp.cumsum()
        total = cum.iloc[-1] if len(cum) else 0.0
        if total <= 0:  # degenerate: no split anywhere -> keep everything
            k = len(imp)
        else:
            k = int(np.searchsorted(cum.to_numpy() / total, self.cumulative - 1e-12) + 1)
            k = min(max(k, 1), len(imp))
        self.importances_ = imp
        self.cumulative_ = cum / (total if total > 0 else 1.0)
        self.selected_ = imp.index[:k].tolist()
        self.n_features_in_ = X.shape[1]
        logger.info("RFImportanceSelector(cum>=%.2f): %d -> %d features",
                    self.cumulative, X.shape[1], k)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep only the selected columns (in importance order)."""
        return X[self.selected_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Selected feature names."""
        return np.asarray(self.selected_, dtype=object)


# ----------------------------------------------------------------------------- plots

def plot_correlation_heatmap(corr: pd.DataFrame, path: str | Path, title: str = "",
                             threshold: float | None = None) -> None:
    """Save a Pearson correlation heatmap (300 dpi)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(corr)
    size = min(max(6.0, 0.22 * n), 22.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    if n <= 80:
        ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=max(4, 9 - n // 15))
        ax.set_yticks(range(n), corr.index, fontsize=max(4, 9 - n // 15))
    else:
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    t = title + (f"  (filter |r| ≥ {threshold})" if threshold is not None else "")
    ax.set_title(t)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_rf_importance(importances: pd.Series, n_selected: int, path: str | Path,
                       title: str = "", cumulative: float = 0.95, max_show: int = 40) -> None:
    """Save the RF importance figure (bars + cumulative curve as two stacked panels)."""
    from ids.figures import rf_importance_bars

    rf_importance_bars(importances, n_selected, title, Path(path), cumulative, max_show)
