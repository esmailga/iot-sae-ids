"""Cleaning, encoding and scaling. Everything is *fitted on the train split only*.

Order (spec §5):
    1. exact-duplicate removal         -> :func:`deduplicate` (dataset- or train-scope)
    2. missing values                  -> :class:`Preprocessor` (drop >5% missing, impute)
    3. impossible values               -> :func:`impossible_row_mask` removes train rows;
                                          :class:`Preprocessor` clips val/test and logs counts
    4. categorical encoding            -> :class:`Preprocessor` (label <=5 cats, else top-k one-hot)
    5. MinMax scaling to [0, 1]        -> :class:`Preprocessor` (val/test clipped to [0, 1])

Row-removing steps (1, 3) are functions applied to the training rows *before* fitting, because
sklearn transformers must preserve the number of rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

PORT_MAX = 65535
OTHER = "__other__"


# ----------------------------------------------------------------------------- row filters

def deduplicate(
    X: pd.DataFrame, y_bin: pd.Series, y_multi: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.Series, int]:
    """Drop exact duplicate rows (identical features *and* multiclass label); keep the first.

    Returns:
        ``(X, y_bin, y_multi, n_removed)`` with reset indices.
    """
    dup = X.assign(__label__=y_multi.values).duplicated(keep="first").to_numpy()
    n = int(dup.sum())
    keep = ~dup
    return (X.loc[keep].reset_index(drop=True), y_bin.loc[keep].reset_index(drop=True),
            y_multi.loc[keep].reset_index(drop=True), n)


def _matches(col: str, patterns: list[str]) -> bool:
    lc = col.lower()
    return any(p.lower() in lc for p in patterns)


def impossible_value_rules(
    X: pd.DataFrame, nonneg_patterns: list[str], port_patterns: list[str]
) -> dict[str, tuple[float, float]]:
    """Return ``{column: (lower, upper)}`` bounds for numeric columns covered by the rules."""
    rules: dict[str, tuple[float, float]] = {}
    for c in X.select_dtypes(include=[np.number]).columns:
        if _matches(c, port_patterns):
            rules[c] = (0.0, float(PORT_MAX))
        elif _matches(c, nonneg_patterns):
            rules[c] = (0.0, np.inf)
    return rules


def impossible_row_mask(X: pd.DataFrame, rules: dict[str, tuple[float, float]]) -> np.ndarray:
    """Boolean mask of rows violating any bound (NaN is *not* a violation)."""
    bad = np.zeros(len(X), dtype=bool)
    for c, (lo, hi) in rules.items():
        v = X[c].to_numpy(dtype="float64")
        bad |= (v < lo) | (v > hi)
    return bad


def clean_train_rows(
    X: pd.DataFrame, y_bin: pd.Series, y_multi: pd.Series, pp_cfg: dict
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Apply train-only row removal: dedup (if ``dedup_scope == 'train'``) + impossible values.

    Returns:
        Filtered ``(X, y_bin, y_multi)`` and a dict of counts for logging/reporting.
    """
    stats: dict = {"train_rows_in": len(X)}
    if pp_cfg.get("dedup_scope", "dataset") == "train":
        X, y_bin, y_multi, n = deduplicate(X, y_bin, y_multi)
        stats["train_duplicates_removed"] = n
        logger.info("Removed %d exact duplicates from train", n)
    rules = impossible_value_rules(X, pp_cfg["nonnegative_patterns"], pp_cfg["port_patterns"])
    bad = impossible_row_mask(X, rules)
    stats["train_impossible_rows_removed"] = int(bad.sum())
    if bad.any():
        logger.info("Removed %d train rows with impossible values", int(bad.sum()))
        keep = ~bad
        X, y_bin, y_multi = (X.loc[keep].reset_index(drop=True),
                             y_bin.loc[keep].reset_index(drop=True),
                             y_multi.loc[keep].reset_index(drop=True))
    stats["train_rows_out"] = len(X)
    return X, y_bin, y_multi, stats


# ----------------------------------------------------------------------------- transformer

@dataclass
class _FitState:
    dropped_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    impute_values: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)
    label_maps: dict = field(default_factory=dict)       # col -> {category: code}
    onehot_levels: dict = field(default_factory=dict)    # col -> [kept categories] (+ OTHER)
    feature_names: list[str] = field(default_factory=list)
    data_min: np.ndarray | None = None
    data_range: np.ndarray | None = None


class Preprocessor(BaseEstimator, TransformerMixin):
    """Missing-value handling, impossible-value clipping, encoding and MinMax scaling.

    Fit on the training split only; ``transform`` is row-preserving and stateless.

    Args:
        missing_drop_threshold: Drop columns with a larger missing fraction on train.
        label_encode_max_categories: Columns with at most this many train categories are
            label-encoded; others are one-hot encoded.
        onehot_top_k: Keep this many most-frequent categories per one-hot column; the rest
            (and unseen categories) map to an ``other`` indicator.
        nonnegative_patterns / port_patterns: Impossible-value rules (see module docstring).
        clip_eval_to_unit: Clip transformed output to [0, 1] (affects val/test only, since
            train is in [0, 1] by construction).
    """

    def __init__(
        self,
        missing_drop_threshold: float = 0.05,
        label_encode_max_categories: int = 5,
        onehot_top_k: int = 10,
        nonnegative_patterns: tuple[str, ...] = ("dur", "byte", "len", "pkt"),
        port_patterns: tuple[str, ...] = ("port",),
        clip_eval_to_unit: bool = True,
    ) -> None:
        self.missing_drop_threshold = missing_drop_threshold
        self.label_encode_max_categories = label_encode_max_categories
        self.onehot_top_k = onehot_top_k
        self.nonnegative_patterns = nonnegative_patterns
        self.port_patterns = port_patterns
        self.clip_eval_to_unit = clip_eval_to_unit

    @classmethod
    def from_config(cls, pp_cfg: dict) -> "Preprocessor":
        """Build from the ``preprocess`` config section."""
        return cls(
            missing_drop_threshold=pp_cfg["missing_drop_threshold"],
            label_encode_max_categories=pp_cfg["label_encode_max_categories"],
            onehot_top_k=pp_cfg["onehot_top_k"],
            nonnegative_patterns=tuple(pp_cfg["nonnegative_patterns"]),
            port_patterns=tuple(pp_cfg["port_patterns"]),
            clip_eval_to_unit=pp_cfg.get("clip_eval_to_unit", True),
        )

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: object = None) -> "Preprocessor":
        """Learn all statistics from the training frame ``X``."""
        st = _FitState()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        miss = X.isna().mean()
        st.dropped_missing = miss[miss > self.missing_drop_threshold].index.tolist()
        Xk = X.drop(columns=st.dropped_missing)
        nun = Xk.nunique(dropna=False)
        st.dropped_constant = nun[nun <= 1].index.tolist()
        Xk = Xk.drop(columns=st.dropped_constant)

        st.numeric_cols = [c for c in Xk.columns if _is_numeric(Xk[c])]
        st.categorical_cols = [c for c in Xk.columns if c not in st.numeric_cols]

        for c in st.numeric_cols:
            med = Xk[c].astype("float64").median()
            st.impute_values[c] = 0.0 if pd.isna(med) else float(med)
        for c in st.categorical_cols:
            mode = Xk[c].astype(str).where(Xk[c].notna()).mode(dropna=True)
            st.impute_values[c] = str(mode.iloc[0]) if len(mode) else "missing"

        st.bounds = impossible_value_rules(
            Xk[st.numeric_cols], list(self.nonnegative_patterns), list(self.port_patterns)
        )

        Xi = self._impute(Xk, st)
        for c in st.categorical_cols:
            vc = Xi[c].value_counts()
            if len(vc) <= self.label_encode_max_categories:
                st.label_maps[c] = {cat: i for i, cat in enumerate(sorted(vc.index))}
            else:
                kept = vc.index[: self.onehot_top_k].tolist()
                st.onehot_levels[c] = kept + ([OTHER] if len(vc) > len(kept) else [])

        self.state_ = st
        enc = self._encode(Xi, st, log_unseen=False)
        st.feature_names = list(enc.columns)
        arr = enc.to_numpy(dtype="float64")
        st.data_min = np.nanmin(arr, axis=0)
        rng = np.nanmax(arr, axis=0) - st.data_min
        st.data_range = np.where(rng > 0, rng, 1.0)
        logger.info(
            "Preprocessor fit: %d cols in -> dropped %d (>%.0f%% missing) + %d (constant); "
            "%d numeric, %d label-encoded, %d one-hot -> %d features",
            X.shape[1], len(st.dropped_missing), 100 * self.missing_drop_threshold,
            len(st.dropped_constant), len(st.numeric_cols), len(st.label_maps),
            len(st.onehot_levels), len(st.feature_names),
        )
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame, name: str = "eval") -> pd.DataFrame:
        """Apply the fitted steps; returns a float32 DataFrame with ``get_feature_names_out``."""
        st = self.state_
        missing = [c for c in st.numeric_cols + st.categorical_cols if c not in X.columns]
        if missing:
            raise ValueError(f"transform: columns missing from input: {missing}")
        Xk = X[st.numeric_cols + st.categorical_cols]
        Xi = self._impute(Xk, st)
        n_clipped = 0
        for c, (lo, hi) in st.bounds.items():
            v = Xi[c]
            out = (v < lo) | (v > hi)
            if out.any():
                n_clipped += int(out.sum())
                Xi[c] = v.clip(lo, hi)
        if n_clipped:
            logger.info("[%s] clipped %d impossible values (no rows dropped)", name, n_clipped)
        enc = self._encode(Xi, st, log_unseen=True, name=name)
        arr = (enc.to_numpy(dtype="float64") - st.data_min) / st.data_range
        if self.clip_eval_to_unit:
            n_out = int(((arr < 0) | (arr > 1)).sum())
            if n_out:
                logger.info("[%s] clipped %d scaled values to [0, 1]", name, n_out)
            arr = np.clip(arr, 0.0, 1.0)
        return pd.DataFrame(arr.astype("float32"), columns=st.feature_names, index=X.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Names of output features."""
        return np.asarray(self.state_.feature_names, dtype=object)

    def summary(self) -> dict:
        """JSON-serialisable description of what was fitted."""
        st = self.state_
        return {
            "n_in": int(self.n_features_in_),
            "dropped_missing": st.dropped_missing,
            "dropped_constant": st.dropped_constant,
            "numeric": st.numeric_cols,
            "label_encoded": list(st.label_maps),
            "onehot": {c: v for c, v in st.onehot_levels.items()},
            "n_out": len(st.feature_names),
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _impute(X: pd.DataFrame, st: _FitState) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in st.numeric_cols:
            out[c] = X[c].astype("float64").fillna(st.impute_values[c])
        for c in st.categorical_cols:
            s = X[c]
            out[c] = s.astype(str).where(s.notna(), st.impute_values[c])
        return out

    def _encode(self, X: pd.DataFrame, st: _FitState, log_unseen: bool,
                name: str = "eval") -> pd.DataFrame:
        parts: list[pd.DataFrame] = [X[st.numeric_cols].astype("float64")]
        for c, mapping in st.label_maps.items():
            codes = X[c].map(mapping)
            unseen = codes.isna()
            if unseen.any():
                fallback = mapping[st.impute_values[c]] if st.impute_values[c] in mapping else 0
                if log_unseen:
                    logger.info("[%s] %s: %d unseen categories -> mode", name, c, int(unseen.sum()))
                codes = codes.fillna(fallback)
            parts.append(codes.astype("float64").to_frame(c))
        for c, levels in st.onehot_levels.items():
            kept = [lv for lv in levels if lv != OTHER]
            col = X[c].where(X[c].isin(kept), OTHER)
            if log_unseen and OTHER not in levels and (col == OTHER).any():
                logger.info("[%s] %s: %d unseen categories (all-zero one-hot)", name, c,
                            int((col == OTHER).sum()))
            block = pd.DataFrame(
                {f"{c}={lv}": (col == lv).astype("float64") for lv in levels}, index=X.index
            )
            parts.append(block)
        return pd.concat(parts, axis=1)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
