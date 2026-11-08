"""Tests for config loading, overlays and overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from ids.utils.config import DATASETS, load_config, parse_overrides


def test_parse_overrides_nested_and_typed() -> None:
    out = parse_overrides(["sae.latent_dim=12", "sae.l1=1e-3", "seeds=[0,1]", "x=foo"])
    assert out["sae"]["latent_dim"] == 12
    assert out["sae"]["l1"] == pytest.approx(1e-3)
    assert out["seeds"] == [0, 1]
    assert out["x"] == "foo"


@pytest.mark.parametrize("ds", DATASETS)
def test_dataset_overlay(ds: str) -> None:
    cfg = load_config(ds, overrides=["sae.latent_dim=8"])
    assert cfg["dataset"]["name"] == ds
    assert cfg["sae"]["latent_dim"] == 8
    assert cfg["sae"]["batch_size"] == 64  # default retained
    assert Path(cfg["paths"]["raw_dir"]).is_absolute()


def test_env_var_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IDS_DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("IDS_RESULTS_ROOT", str(tmp_path / "r"))
    cfg = load_config("ton_iot")
    assert cfg["paths"]["raw_dir"] == str(tmp_path / "d" / "raw")
    assert cfg["paths"]["results_raw"] == str(tmp_path / "r" / "raw")


def test_unknown_dataset() -> None:
    with pytest.raises(ValueError):
        load_config("nope")
