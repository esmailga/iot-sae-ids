"""Tests for preprocessing: train-only fitting, missing values, encoding, scaling, clipping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ids.data.preprocess import (
    OTHER,
    Preprocessor,
    clean_train_rows,
    deduplicate,
    impossible_row_mask,
    impossible_value_rules,
)

PP_CFG = {
    "dedup_scope": "train",
    "missing_drop_threshold": 0.05,
    "label_encode_max_categories": 5,
    "onehot_top_k": 3,
    "nonnegative_patterns": ["dur", "byte"],
    "port_patterns": ["port"],
    "clip_eval_to_unit": True,
}


def _frame(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "duration": rng.exponential(1.0, n) + shift,
        "src_bytes": rng.integers(0, 1000, n).astype(float) + shift,
        "mostly_missing": np.where(rng.random(n) < 0.5, np.nan, 1.0),
        "few_missing": np.where(np.arange(n) % 50 == 0, np.nan, rng.normal(size=n)),
        "constant": 7.0,
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "service": rng.choice(["-", "http", "dns", "ssl", "ftp", "ssh", "smtp"], n),
    })


def test_fit_uses_train_only() -> None:
    tr, te = _frame(500, 0), _frame(300, 1, shift=50.0)
    pp = Preprocessor.from_config(PP_CFG).fit(tr)
    st = pp.state_
    # Imputation and scaling statistics equal those computed from train alone.
    assert st.impute_values["few_missing"] == pytest.approx(tr["few_missing"].median())
    i = st.feature_names.index("duration")
    assert st.data_min[i] == pytest.approx(tr["duration"].min())
    assert st.data_min[i] + st.data_range[i] == pytest.approx(tr["duration"].max())
    before = (st.data_min.copy(), st.data_range.copy())
    out_te = pp.transform(te)
    # transform() must not change fitted state.
    np.testing.assert_array_equal(before[0], pp.state_.data_min)
    np.testing.assert_array_equal(before[1], pp.state_.data_range)
    # Test values shifted far above train max are clipped to 1.
    assert out_te["duration"].max() == pytest.approx(1.0)
    assert out_te.to_numpy().min() >= 0.0 and out_te.to_numpy().max() <= 1.0


def test_train_output_in_unit_interval() -> None:
    tr = _frame(400, 0)
    out = Preprocessor.from_config(PP_CFG).fit_transform(tr)
    arr = out.to_numpy()
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_missing_and_constant_columns() -> None:
    tr = _frame(500, 0)
    pp = Preprocessor.from_config(PP_CFG).fit(tr)
    assert pp.state_.dropped_missing == ["mostly_missing"]
    assert pp.state_.dropped_constant == ["constant"]
    out = pp.transform(tr)
    assert "few_missing" in out.columns and not out["few_missing"].isna().any()


def test_encoding_label_vs_onehot_topk() -> None:
    tr = _frame(500, 0)
    pp = Preprocessor.from_config(PP_CFG).fit(tr)
    assert "proto" in pp.state_.label_maps                  # 3 categories -> label
    levels = pp.state_.onehot_levels["service"]             # 7 categories -> one-hot
    assert len(levels) == 3 + 1 and levels[-1] == OTHER
    cols = [c for c in pp.state_.feature_names if c.startswith("service=")]
    out = pp.transform(tr)
    np.testing.assert_allclose(out[cols].sum(axis=1), 1.0)  # exactly one level active


def test_unseen_categories() -> None:
    tr = _frame(300, 0)
    te = _frame(50, 1)
    te.loc[:9, "proto"] = "sctp"
    te.loc[:9, "service"] = "never_seen"
    pp = Preprocessor.from_config(PP_CFG).fit(tr)
    out = pp.transform(te)
    assert out.notna().all().all()
    assert (out.loc[:9, f"service={OTHER}"] == 1).all()


def test_impossible_values_train_removed_eval_clipped() -> None:
    tr = _frame(200, 0)
    tr.loc[:4, "duration"] = -1.0
    tr["dst_port"] = 80.0
    tr.loc[5, "dst_port"] = 70000.0
    rules = impossible_value_rules(tr, PP_CFG["nonnegative_patterns"], PP_CFG["port_patterns"])
    assert rules["dst_port"] == (0.0, 65535.0)
    assert impossible_row_mask(tr, rules).sum() == 6
    y = pd.Series(np.zeros(len(tr), dtype=int))
    Xc, yc, mc, stats = clean_train_rows(tr, y, y.astype(str), PP_CFG)
    assert stats["train_impossible_rows_removed"] == 6 and len(Xc) == 194
    pp = Preprocessor.from_config(PP_CFG).fit(Xc)
    te = _frame(20, 1)
    te["dst_port"] = 80.0
    te.loc[0, "duration"] = -5.0
    out = pp.transform(te)
    assert len(out) == 20  # eval rows never dropped
    assert out.loc[0, "duration"] == pytest.approx(0.0)


def test_deduplicate() -> None:
    X = pd.DataFrame({"a": [1, 1, 1, 2], "b": [0, 0, 0, 0]})
    yb = pd.Series([0, 0, 1, 1])
    ym = pd.Series(["normal", "normal", "dos", "dos"])
    Xd, ybd, ymd, n = deduplicate(X, yb, ym)
    assert n == 1                      # rows 0/1 identical incl. label; row 2 differs in label
    assert len(Xd) == 3 and ymd.tolist() == ["normal", "dos", "dos"]
