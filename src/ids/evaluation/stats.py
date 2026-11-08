"""Aggregation over seeds and significance tests (SAE vs each baseline).

Pairs are (dataset, seed) results. With 3 datasets x 5 seeds there are 15 pairs per
comparison. Per-dataset tests (5 pairs) are also reported, but note that a two-sided
Wilcoxon test with n = 5 cannot reach p < 0.0625, so those are descriptive only.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


def mean_std(df: pd.DataFrame, by: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    """Mean and sample std of each metric per group; columns ``<metric>_mean``/``_std``/``n``."""
    metrics = list(metrics)
    g = df.groupby(by)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out["n"] = g.size()
    cols = [c for m in metrics for c in (f"{m}_mean", f"{m}_std")] + ["n"]
    return out[cols].reset_index()


def cohens_dz(diff: np.ndarray) -> float:
    """Cohen's d_z for paired samples: mean(diff) / sd(diff)."""
    diff = np.asarray(diff, dtype="float64")
    if len(diff) < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0 if diff.mean() == 0 else float(np.sign(diff.mean()) * np.inf)
    return float(diff.mean() / sd)


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (W+ - W-) / (W+ + W-), zero diffs dropped."""
    d = np.asarray(diff, dtype="float64")
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def paired_tests(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of ``a - b`` (two-sided) + effect sizes."""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    diff = a - b
    res: dict = {"n_pairs": int(len(diff)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                 "mean_diff": float(diff.mean()), "cohens_dz": cohens_dz(diff),
                 "rank_biserial": rank_biserial(diff)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(diff) >= 2 and diff.std(ddof=1) > 0:
            t = stats.ttest_rel(a, b)
            res.update(t_stat=float(t.statistic), t_p=float(t.pvalue))
        else:
            res.update(t_stat=float("nan"), t_p=float("nan") if diff.any() else 1.0)
        if np.any(diff != 0) and len(diff) >= 2:
            w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            res.update(w_stat=float(w.statistic), w_p=float(w.pvalue))
        else:
            res.update(w_stat=float("nan"), w_p=1.0)
    res["t_significant"] = bool(res["t_p"] < alpha) if not np.isnan(res["t_p"]) else False
    res["w_significant"] = bool(res["w_p"] < alpha)
    return res


def compare_to_reference(
    df: pd.DataFrame,
    metrics: Iterable[str],
    reference: str = "sae",
    alpha: float = 0.05,
    per_dataset: bool = True,
    pair_keys: tuple[str, ...] = ("dataset", "seed"),
    baselines: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compare ``reference`` against every other model on paired (dataset, seed) results.

    Args:
        df: Long-format results with columns ``model``, ``dataset``, ``seed`` and metrics.

    Returns:
        One row per (scope, baseline, metric) with test statistics, p-values and effect sizes.
        ``mean_diff`` > 0 means the reference (SAE) is higher.
    """
    rows = []
    models = sorted(set(df["model"]) - {reference}) if baselines is None else list(baselines)
    scopes: list[tuple[str, pd.DataFrame]] = [("all", df)]
    if per_dataset:
        scopes += [(ds, g) for ds, g in df.groupby("dataset")]
    for scope, sub in scopes:
        ref = sub[sub["model"] == reference].set_index(list(pair_keys))
        for base in models:
            other = sub[sub["model"] == base].set_index(list(pair_keys))
            common = ref.index.intersection(other.index)
            if len(common) == 0:
                continue
            for m in metrics:
                r = paired_tests(ref.loc[common, m].to_numpy(),
                                 other.loc[common, m].to_numpy(), alpha)
                rows.append({"scope": scope, "reference": reference, "baseline": base,
                             "metric": m, **r})
    return pd.DataFrame(rows)
