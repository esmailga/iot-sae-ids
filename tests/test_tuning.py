"""Tests for the SAE search space handling (no TF) and a tiny tuning run (TF only)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ids.tuning import architecture_options, candidate_grid, select_candidates, trial_key
from ids.utils.config import load_config

SPACE = {"hidden": ["scaled", "fixed"], "latent_dim": [8, 12, 16, 20],
         "l1": [1e-4, 1e-3, 1e-2], "dropout": [0.2, 0.3, 0.4], "lr": [1e-2, 1e-3, 1e-4]}


def test_full_grid_size_and_capping() -> None:
    assert len(candidate_grid(SPACE, d=30)) == 2 * 4 * 3 * 3 * 3       # 216
    small = candidate_grid(SPACE, d=10)                                 # 12,16,20 -> 9
    assert {p["latent_dim"] for p in small} == {8, 9}
    assert len(small) == 2 * 2 * 3 * 3 * 3
    assert len({trial_key(p) for p in small}) == len(small)


def test_random_selection_seeded_and_distinct() -> None:
    a = select_candidates(SPACE, 30, "random", 20, seed=0)
    b = select_candidates(SPACE, 30, "random", 20, seed=0)
    c = select_candidates(SPACE, 30, "random", 20, seed=1)
    assert a == b and a != c and len(a) == 20
    assert len({trial_key(p) for p in a}) == 20
    assert len(select_candidates(SPACE, 30, "grid", 0, 0)) == 216
    with pytest.raises(ValueError):
        select_candidates(SPACE, 30, "bayes", 5, 0)


def test_architecture_options_param_counts() -> None:
    cfg = load_config("ton_iot")
    df = architecture_options(cfg, d=19)
    assert set(df.hidden_mode) == {"scaled", "fixed"}
    scaled = df[df.hidden_mode == "scaled"].n_params.max()
    fixed = df[df.hidden_mode == "fixed"].n_params.min()
    assert scaled < fixed


@pytest.mark.tf
def test_run_tuning_tiny(raw_root: Path) -> None:
    pytest.importorskip("tensorflow")
    from ids.tuning import run_tuning

    cfg = load_config("ton_iot", ["features.rf_n_estimators=10", "tune.epochs=2"])
    best = run_tuning(cfg, seed=0, mode="random", n_trials=2, subsample=0.5)
    path = Path(cfg["paths"]["results_root"]) / "best_sae_ton_iot.yaml"
    saved = yaml.safe_load(path.read_text())
    assert saved["params"] == best["params"] and saved["search"]["n_trials_done"] == 2
    # Resume: nothing new to run, same best.
    again = run_tuning(cfg, seed=0, mode="random", n_trials=2, subsample=0.5)
    assert again["params"] == best["params"]
    # run_experiment picks up the tuned params.
    from ids.experiment import tuned_sae_params

    assert tuned_sae_params(cfg, "ton_iot") == best["params"]
