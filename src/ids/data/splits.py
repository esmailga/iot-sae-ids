"""Seeded, stratified train/val/test splitting and subsampling (done *before* any fitting)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ids.data.loaders import LoadedData

logger = logging.getLogger(__name__)

MIN_STRATUM = 10  # multiclass strata smaller than this fall back to the binary label


@dataclass
class Split:
    """Train/val/test partitions. Indices refer to the frames held here, not the raw file."""

    X_train: pd.DataFrame
    y_train: pd.Series
    m_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    m_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    m_test: pd.Series

    def sizes(self) -> dict[str, int]:
        """Row counts per partition."""
        return {"train": len(self.X_train), "val": len(self.X_val), "test": len(self.X_test)}


def stratify_key(y_bin: pd.Series, y_multi: pd.Series, min_count: int = MIN_STRATUM) -> np.ndarray:
    """Stratify on the multiclass label; rare classes (< ``min_count``) fall back to binary.

    This keeps every attack category represented in each split where possible, without
    crashing ``train_test_split`` on classes with one or two members.
    """
    counts = y_multi.value_counts()
    rare = y_multi.map(counts) < min_count
    key = y_multi.astype(str).where(~rare, "__rare_" + y_bin.astype(str))
    return key.to_numpy()


def stratified_subsample(
    data: LoadedData, fraction: float, seed: int
) -> LoadedData:
    """Return a stratified ``fraction`` of the dataset (and of its official test set)."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1:
        return data

    def _sub(X: pd.DataFrame, yb: pd.Series, ym: pd.Series) -> tuple:
        idx, _ = train_test_split(
            np.arange(len(X)), train_size=fraction, random_state=seed,
            stratify=stratify_key(yb, ym),
        )
        idx = np.sort(idx)
        return (X.iloc[idx].reset_index(drop=True), yb.iloc[idx].reset_index(drop=True),
                ym.iloc[idx].reset_index(drop=True))

    X, yb, ym = _sub(data.X, data.y_bin, data.y_multi)
    test = _sub(*data.test) if data.test is not None else None
    logger.info("Subsampled %.3f -> %d rows%s", fraction, len(X),
                f" (+{len(test[0])} official test)" if test else "")
    return LoadedData(X, yb, ym, test, data.dropped_columns)


def _take(X: pd.DataFrame, yb: pd.Series, ym: pd.Series, ix: np.ndarray) -> tuple:
    ix = np.sort(ix)
    return (X.iloc[ix].reset_index(drop=True), yb.iloc[ix].reset_index(drop=True),
            ym.iloc[ix].reset_index(drop=True))


def make_split(data: LoadedData, cfg: dict, seed: int) -> Split:
    """Split per ``cfg['dataset']['split_mode']``.

    * ``stratified``: train/val/test = cfg.split.train/val/test (default 70/10/20).
    * ``official``: ``data.X`` (official train) -> train/val by ``unsw_val_fraction``;
      ``data.test`` is used unchanged as the test set.
    """
    mode = cfg["dataset"]["split_mode"]
    sp = cfg["split"]
    X, yb, ym = data.X, data.y_bin, data.y_multi
    idx = np.arange(len(X))

    if mode == "stratified":
        if not np.isclose(sp["train"] + sp["val"] + sp["test"], 1.0):
            raise ValueError("split fractions must sum to 1")
        trval, te = train_test_split(idx, test_size=sp["test"], random_state=seed,
                                     stratify=stratify_key(yb, ym))
        val_frac = sp["val"] / (sp["train"] + sp["val"])
        tr, va = train_test_split(trval, test_size=val_frac, random_state=seed,
                                  stratify=stratify_key(yb.iloc[trval], ym.iloc[trval]))
        parts = {"train": tr, "val": va, "test": te}
        tr_xy, va_xy, te_xy = (_take(X, yb, ym, parts[k]) for k in ("train", "val", "test"))
    elif mode == "official":
        if data.test is None:
            raise ValueError("split_mode 'official' requires an official test set")
        tr, va = train_test_split(idx, test_size=sp["unsw_val_fraction"], random_state=seed,
                                  stratify=stratify_key(yb, ym))
        tr_xy, va_xy = _take(X, yb, ym, tr), _take(X, yb, ym, va)
        te_xy = data.test
    else:
        raise ValueError(f"Unknown split_mode {mode!r}")

    split = Split(*tr_xy, *va_xy, *te_xy)
    logger.info("Split (%s, seed=%d): %s", mode, seed, split.sizes())
    return split
