"""Tests for the SAE helpers and models (TF tests skip without TensorFlow) and baselines."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ids.models import make_model
from ids.models.sae import capped_latent_dim, count_dense_params, hidden_widths
from ids.utils.config import load_config


def _data(n: int = 600, d: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype("int8")
    X = rng.random((n, d)).astype("float32") * 0.5
    X[y == 1, :3] += 0.4  # attacks differ in the first three features
    return np.clip(X, 0, 1), y


@pytest.fixture()
def cfg() -> dict:
    return load_config("ton_iot", ["baselines.rf.n_estimators=20", "sae.epochs=3",
                                   "baselines.lstm.epochs=2", "baselines.lstm.units=[8,4]"])


# ----------------------------------------------------------------------------- SAE (no TF)

def test_hidden_widths_scaled_and_fixed() -> None:
    assert hidden_widths(19, "scaled", [0.75, 0.5], [128, 64, 32]) == [15, 10]
    assert hidden_widths(27, "scaled", [0.75, 0.5], []) == [math.ceil(20.25), 14]
    assert hidden_widths(19, "fixed", [0.75, 0.5], [128, 64, 32]) == [128, 64, 32]
    with pytest.raises(ValueError):
        hidden_widths(10, "nope", [], [])


def test_latent_dim_capped_below_d() -> None:
    assert capped_latent_dim(31, 20) == 20
    assert capped_latent_dim(12, 16) == 11
    assert capped_latent_dim(2, 8) == 1


def test_scaled_option_compresses() -> None:
    for d in (19, 27, 31):
        h = hidden_widths(d, "scaled", [0.75, 0.5], [])
        assert all(w < d for w in h) and h == sorted(h, reverse=True)
        assert count_dense_params(d, h, 8) < count_dense_params(d, [128, 64, 32], 8)


# ----------------------------------------------------------------------------- baselines

def test_rf_interface(cfg: dict, tmp_path: Path) -> None:
    X, y = _data()
    m = make_model("rf", cfg, seed=0).fit(X[:400], y[:400], X[400:500], y[400:500])
    s = m.score(X[500:])
    assert s.shape == (100,) and s.min() >= 0 and s.max() <= 1
    assert m.predict(X[500:]).mean() > 0
    assert m.n_params() == sum(e.tree_.node_count for e in m.model.estimators_)
    assert m.save(tmp_path).stat().st_size > 0
    assert len(m.predict_batch(X[:64])) == 64


def test_svm_exact_variant(cfg: dict) -> None:
    X, y = _data()
    m = make_model("svm", cfg, seed=0).fit(X[:400], y[:400], X[400:500], y[400:500])
    assert m.info["variant"] == "exact_rbf_svc"
    assert m.n_params() == m.model.n_support_.sum() * X.shape[1]
    s = m.score(X[500:])
    assert ((s > 0.5) == (m.decision(X[500:]) > 0)).all()
    assert np.array_equal(m.predict(X[500:]), m.predict_batch(X[500:]).astype("int8"))


def test_svm_nystroem_variant(cfg: dict) -> None:
    cfg["baselines"]["svm"]["exact_max_rows"] = 100
    cfg["baselines"]["svm"]["nystroem_components"] = 50
    X, y = _data()
    m = make_model("svm", cfg, seed=0).fit(X[:400], y[:400], X[400:500], y[400:500])
    assert m.info["variant"] == "nystroem_rbf_sgd_hinge"
    assert m.n_params() == 50 * X.shape[1] + 50 + 1
    acc = (m.predict(X[500:]) == y[500:]).mean()
    assert acc > 0.7


# ----------------------------------------------------------------------------- TF models

@pytest.mark.tf
def test_sae_output_range_and_params(cfg: dict, tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    X, y = _data(d=12)
    m = make_model("sae", cfg, seed=0, sae_params={"latent_dim": 4})
    m.fit(X[:400], y[:400], X[400:500], y[400:500])
    rec = np.asarray(m.model(X[500:], training=False))
    assert rec.shape == (100, 12)
    assert rec.min() >= 0.0 and rec.max() <= 1.0
    assert m.info["hidden"] == [9, 6] and m.info["latent_dim"] == 4
    assert m.n_params() == count_dense_params(12, [9, 6], 4)
    assert m.info["n_train_normal"] == int((y[:400] == 0).sum())
    assert "val_mse" in m.history and len(m.history["val_mse"]) <= 3
    err = m.score(X[500:])
    assert err.shape == (100,) and (err >= 0).all()
    m.set_threshold(float(np.median(err)))
    assert m.predict(X[500:]).sum() == (err > np.median(err)).sum()
    assert m.save(tmp_path).stat().st_size > 0


@pytest.mark.tf
def test_sae_fixed_architecture(cfg: dict) -> None:
    pytest.importorskip("tensorflow")
    m = make_model("sae", cfg, seed=0, sae_params={"hidden": "fixed", "latent_dim": 8})
    model = m.build(19)
    assert m.info["hidden"] == [128, 64, 32]
    assert model.count_params() == count_dense_params(19, [128, 64, 32], 8)


@pytest.mark.tf
def test_lstm_interface(cfg: dict) -> None:
    pytest.importorskip("tensorflow")
    X, y = _data(d=6)
    m = make_model("lstm", cfg, seed=0).fit(X[:300], y[:300], X[300:400], y[300:400])
    s = m.score(X[400:])
    assert s.shape == (200,) and s.min() >= 0 and s.max() <= 1
    assert m.n_params() == m.model.count_params()
