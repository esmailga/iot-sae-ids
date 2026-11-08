"""Tests for the Pearson filter, RF importance selector and the cached prepare() pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.data.prepare import prepare
from ids.features.selection import PearsonFilter, RFImportanceSelector
from tests.conftest import cfg_for


def _corr_frame(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = (a + 0.5 * b > 0).astype(int)
    X = pd.DataFrame({
        "a": a,
        "a_copy": a * 2 + 1e-3 * rng.normal(size=n),   # |r| ~ 1 with a
        "a_copy2": -a + 1e-3 * rng.normal(size=n),     # |r| ~ 1 with a
        "b": b,
        "noise": rng.normal(size=n),
        "const": np.ones(n),
    })
    return X, y


def test_pearson_drops_one_of_each_correlated_group() -> None:
    X, _ = _corr_frame()
    pf = PearsonFilter(0.95).fit(X)
    kept_a = [c for c in pf.selected_ if c.startswith("a")]
    assert len(kept_a) == 1
    assert {"b", "noise", "const"} <= set(pf.selected_)
    assert len(pf.dropped_) == 2
    out = pf.transform(X)
    assert list(out.columns) == pf.selected_


def test_pearson_drops_higher_mean_correlation() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    z = rng.normal(size=n)
    hub = z + 0.01 * rng.normal(size=n)
    X = pd.DataFrame({
        "x": z + 0.25 * rng.normal(size=n),          # correlated with hub (~0.97) only
        "hub": hub,
        "y1": z + 0.4 * rng.normal(size=n),          # moderately correlated with hub
        "y2": z + 0.4 * rng.normal(size=n),
    })
    pf = PearsonFilter(0.95).fit(X)
    assert "hub" not in pf.selected_ and "x" in pf.selected_


def test_pearson_shapes_consistent_on_new_data() -> None:
    Xtr, _ = _corr_frame(seed=0)
    Xte, _ = _corr_frame(n=100, seed=1)
    pf = PearsonFilter(0.95).fit(Xtr)
    assert pf.transform(Xte).shape == (100, len(pf.selected_))


def test_rf_selector_cumulative_and_shapes() -> None:
    X, y = _corr_frame()
    sel = RFImportanceSelector(cumulative=0.95, n_estimators=50, random_state=0).fit(X, y)
    k = len(sel.selected_)
    assert 1 <= k <= X.shape[1]
    cum = sel.cumulative_.to_numpy()
    assert cum[k - 1] >= 0.95 - 1e-9
    if k > 1:
        assert cum[k - 2] < 0.95                    # smallest such k
    assert sel.selected_[0] in {"a", "a_copy", "a_copy2", "b"}  # informative first
    assert "const" not in sel.selected_
    assert sel.transform(X).shape == (len(X), k)


def test_rf_selector_subsample_is_seeded() -> None:
    X, y = _corr_frame(n=1000)
    s1 = RFImportanceSelector(n_estimators=20, max_rows=300, random_state=3).fit(X, y)
    s2 = RFImportanceSelector(n_estimators=20, max_rows=300, random_state=3).fit(X, y)
    assert s1.selected_ == s2.selected_


@pytest.mark.parametrize("name", ["ton_iot", "unsw_nb15", "rt_iot2022"])
def test_prepare_pipeline_consistent_and_cached(raw_root: Path, name: str) -> None:
    cfg = cfg_for(name, ["features.rf_n_estimators=20"])
    p = prepare(cfg, seed=0, plots=False)
    d = p.d
    for part in ("train", "val", "test"):
        assert p.X[part].shape[1] == d
        assert p.X[part].dtype == np.float32
        assert len(p.X[part]) == len(p.y[part]) == len(p.m[part])
        assert p.X[part].min() >= 0 and p.X[part].max() <= 1
    c = p.counts
    assert c["raw"] >= c["after_leakage_drop"]
    assert c["after_preprocess"] >= c["after_stage1_pearson"] >= c["after_stage2_rf"] == d
    # The near-duplicate synthetic dst_bytes column should be removed by the Pearson filter.
    assert "dst_bytes" not in p.pearson.selected_ or "src_bytes" not in p.pearson.selected_
    assert (Path(cfg["paths"]["features_dir"]) / f"{name}__s0.json").is_file()
    # Second call hits the cache and returns identical arrays and features.
    q = prepare(cfg, seed=0, plots=False)
    assert q.feature_names == p.feature_names
    np.testing.assert_array_equal(q.X["test"], p.X["test"])
    np.testing.assert_array_equal(q.m["val"], p.m["val"])
