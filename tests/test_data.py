"""Tests for dataset loaders and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.data.loaders import (
    LabelMappingNotConfirmed,
    infer_numeric_columns,
    load_dataset,
    propose_rt_iot_normal_classes,
)
from ids.data.splits import make_split, stratified_subsample
from ids.utils.config import DATASETS, load_config
from tests.conftest import cfg_for


@pytest.mark.parametrize("name", DATASETS)
def test_loader_contract(raw_root: Path, name: str) -> None:
    cfg = cfg_for(name)
    data = load_dataset(cfg)
    assert isinstance(data.X, pd.DataFrame)
    assert len(data.X) == len(data.y_bin) == len(data.y_multi)
    assert set(data.y_bin.unique()) <= {0, 1}
    # Leakage control: identifiers, timestamps, ports and labels are gone.
    forbidden = {"ts", "src_ip", "dst_ip", "src_port", "dst_port", "id", "id.orig_p",
                 "id.resp_p", "label", "type", "attack_cat", "Attack_type"}
    assert not forbidden & set(data.X.columns)
    assert set(data.dropped_columns) <= set(cfg["dataset"]["drop_cols"])


def test_unsw_has_official_test(raw_root: Path) -> None:
    data = load_dataset(cfg_for("unsw_nb15"))
    assert data.test is not None
    assert len(data.test[0]) == 300
    assert list(data.test[0].columns) == list(data.X.columns)


def test_rt_iot_requires_confirmation(raw_root: Path) -> None:
    unconfirmed = load_config("rt_iot2022", ["dataset.label_mapping_confirmed=false",
                                             "dataset.normal_classes=[]"])
    with pytest.raises(LabelMappingNotConfirmed):
        load_dataset(unconfirmed)
    data = load_dataset(unconfirmed, allow_unconfirmed=True)
    assert data.y_bin.sum() > 0


def test_rt_iot_mapping_values(raw_root: Path) -> None:
    data = load_dataset(cfg_for("rt_iot2022"))
    normal = data.y_multi.isin(["MQTT_Publish", "Thing_Speak", "Wipro_bulb"])
    assert (data.y_bin[normal] == 0).all()
    assert (data.y_bin[~normal] == 1).all()


def test_propose_normal_classes() -> None:
    labels = pd.Series(["MQTT_Publish", "DOS_SYN_Hping", "Thing_Speak", "NMAP_TCP_scan",
                        "ARP_poisioning", "Metasploit_Brute_Force_SSH", "DDOS_Slowloris"])
    assert propose_rt_iot_normal_classes(labels) == ["MQTT_Publish", "Thing_Speak"]


def test_infer_numeric_columns_dash() -> None:
    df = pd.DataFrame({"num_dash": ["1", "-", "3"], "cat": ["-", "a", "b"]})
    out = infer_numeric_columns(df, dash_is_category=True)
    assert pd.api.types.is_float_dtype(out["num_dash"])
    assert out["num_dash"].isna().sum() == 1
    assert out["cat"].tolist() == ["-", "a", "b"]


@pytest.mark.parametrize("name", ["ton_iot", "rt_iot2022"])
def test_stratified_split_fractions_and_disjoint(raw_root: Path, name: str) -> None:
    cfg = cfg_for(name)
    data = load_dataset(cfg)
    s = make_split(data, cfg, seed=0)
    n = len(data.X)
    sizes = s.sizes()
    assert sum(sizes.values()) == n
    assert abs(sizes["test"] / n - 0.2) < 0.01
    assert abs(sizes["val"] / n - 0.1) < 0.01
    # Attack ratio preserved approximately in each split.
    p = data.y_bin.mean()
    for y in (s.y_train, s.y_val, s.y_test):
        assert abs(y.mean() - p) < 0.05


def test_split_is_seeded(raw_root: Path) -> None:
    cfg = cfg_for("ton_iot")
    data = load_dataset(cfg)
    a, b, c = make_split(data, cfg, 0), make_split(data, cfg, 0), make_split(data, cfg, 1)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)
    assert not a.X_test.equals(c.X_test)


def test_unsw_official_split(raw_root: Path) -> None:
    cfg = cfg_for("unsw_nb15")
    data = load_dataset(cfg)
    s = make_split(data, cfg, seed=0)
    pd.testing.assert_frame_equal(s.X_test, data.test[0])  # official test untouched
    assert len(s.X_train) + len(s.X_val) == len(data.X)
    assert abs(len(s.X_val) / len(data.X) - 0.125) < 0.01


def test_subsample(raw_root: Path) -> None:
    cfg = cfg_for("unsw_nb15")
    data = load_dataset(cfg)
    sub = stratified_subsample(data, 0.5, seed=0)
    assert len(sub.X) == pytest.approx(len(data.X) * 0.5, abs=2)
    assert len(sub.test[0]) == pytest.approx(len(data.test[0]) * 0.5, abs=2)
    assert np.isclose(sub.y_bin.mean(), data.y_bin.mean(), atol=0.03)


def test_resolve_file_case_insensitive_and_bom(tmp_path: Path) -> None:
    from ids.data.loaders import _read_csv, resolve_file

    (tmp_path / "train_test_network.csv").write_bytes("﻿src_ip,label\n1.2.3.4,0\n".encode())
    p = resolve_file(tmp_path, "Train_Test_Network.csv")
    assert p.name == "train_test_network.csv"
    assert list(_read_csv(p).columns) == ["src_ip", "label"]


def test_stratify_key_never_singleton() -> None:
    from ids.data.splits import stratify_key

    ym = pd.Series(["normal"] * 50 + ["dos"] * 30 + ["fin_scan"])
    yb = pd.Series([0] * 50 + [1] * 31)
    key = stratify_key(yb, ym)
    _, counts = np.unique(key, return_counts=True)
    assert counts.min() >= 2
    from sklearn.model_selection import train_test_split

    train_test_split(np.arange(len(ym)), test_size=0.2, stratify=key, random_state=0)
