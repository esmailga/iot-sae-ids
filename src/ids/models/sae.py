"""Sparse autoencoder (SAE) anomaly detector — TensorFlow/Keras, imported lazily.

Architecture (input dimension ``d`` = features after Stage 2):

    encoder:    [Dense(w, relu) -> BatchNorm -> Dropout] for w in hidden
    bottleneck: Dense(d_z, relu, activity_regularizer=L1(lambda))
    decoder:    [Dense(w, relu) -> BatchNorm] for w in reversed(hidden)   (no dropout)
    output:     Dense(d, sigmoid)

Loss = MSE + L1 activity penalty (Keras adds the regulariser to the loss).
Optimiser = Adam with ExponentialDecay. Early stopping on validation-normal *MSE*
(the ``mse`` metric, excluding the L1 term), patience 10, restoring the best weights.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ids.models.base import DetectorModel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- pure helpers

def hidden_widths(d: int, mode: str, scaled_ratios: list[float],
                  fixed_hidden: list[int]) -> list[int]:
    """Encoder hidden widths.

    * ``scaled``: ``[ceil(r * d) for r in scaled_ratios]`` — genuinely compresses for any d.
    * ``fixed``: the proposal's ``[128, 64, 32]`` (expands when d < 128; kept for comparison).
    """
    if mode == "scaled":
        return [max(1, math.ceil(r * d)) for r in scaled_ratios]
    if mode == "fixed":
        return [int(w) for w in fixed_hidden]
    raise ValueError(f"Unknown SAE hidden mode {mode!r}")


def capped_latent_dim(d: int, requested: int) -> int:
    """Bottleneck size ``min(requested, d - 1)``, i.e. capped strictly below the input dim."""
    return int(max(1, min(int(requested), d - 1)))


def count_dense_params(d: int, hidden: list[int], d_z: int) -> int:
    """Analytic parameter count (incl. BatchNorm gamma/beta/moving stats) — matches Keras."""
    total = 0
    prev = d
    for w in hidden:                       # encoder Dense + BN(4w)
        total += prev * w + w + 4 * w
        prev = w
    total += prev * d_z + d_z              # bottleneck
    prev = d_z
    for w in reversed(hidden):             # decoder Dense + BN
        total += prev * w + w + 4 * w
        prev = w
    total += prev * d + d                  # output
    return total


# ----------------------------------------------------------------------------- Keras model

def build_sae(d: int, hidden: list[int], d_z: int, l1: float, dropout: float, lr: float,
              decay_steps: int, decay_rate: float, seed: int) -> Any:
    """Build and compile the SAE Keras model."""
    from tensorflow import keras

    init = lambda k: keras.initializers.GlorotUniform(seed=seed + k)  # noqa: E731
    inp = keras.Input(shape=(d,), name="x")
    x = inp
    for i, w in enumerate(hidden):
        x = keras.layers.Dense(w, activation="relu", kernel_initializer=init(i),
                               name=f"enc_dense_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"enc_bn_{i}")(x)
        x = keras.layers.Dropout(dropout, seed=seed + 100 + i, name=f"enc_drop_{i}")(x)
    z = keras.layers.Dense(d_z, activation="relu", kernel_initializer=init(50),
                           activity_regularizer=keras.regularizers.L1(l1), name="bottleneck")(x)
    x = z
    for j, w in enumerate(reversed(hidden)):
        x = keras.layers.Dense(w, activation="relu", kernel_initializer=init(200 + j),
                               name=f"dec_dense_{j}")(x)
        x = keras.layers.BatchNormalization(name=f"dec_bn_{j}")(x)
    out = keras.layers.Dense(d, activation="sigmoid", kernel_initializer=init(300),
                             name="reconstruction")(x)
    model = keras.Model(inp, out, name="sparse_autoencoder")
    schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=schedule), loss="mse",
                  metrics=[keras.metrics.MeanSquaredError(name="mse")])
    return model


def reconstruction_errors(model: Any, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    X = np.asarray(X, dtype="float32")
    errs = np.empty(len(X), dtype="float64")
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        rec = np.asarray(model(xb, training=False))
        errs[i:i + batch_size] = np.mean((xb - rec) ** 2, axis=1)
    return errs


class SAEDetector(DetectorModel):
    """SAE wrapper with the common :class:`DetectorModel` interface.

    Hyperparameters come from ``cfg["sae"]`` (possibly overwritten by the tuned
    ``results/best_sae_<dataset>.yaml``) or from explicit ``params``.
    """

    name = "sae"
    uses_tensorflow = True

    def __init__(self, cfg: dict, seed: int, params: Optional[dict] = None,
                 verbose: int = 0) -> None:
        super().__init__(cfg, seed)
        self.params = {**cfg["sae"], **(params or {})}
        self.verbose = verbose
        self.model: Any = None
        self.threshold: Optional[float] = None
        self.default_threshold = float("inf")  # must be set from validation data

    def build(self, d: int) -> Any:
        """Build the Keras model for input dimension ``d`` and record the architecture."""
        p = self.params
        hidden = hidden_widths(d, p["hidden"], p["scaled_ratios"], p["fixed_hidden"])
        d_z = capped_latent_dim(d, p["latent_dim"])
        self.model = build_sae(d, hidden, d_z, float(p["l1"]), float(p["dropout"]),
                               float(p["lr"]), int(p["lr_decay_steps"]),
                               float(p["lr_decay_rate"]), self.seed)
        self.info.update({"d": d, "hidden": hidden, "latent_dim": d_z,
                          "hidden_mode": p["hidden"], "l1": float(p["l1"]),
                          "dropout": float(p["dropout"]), "lr": float(p["lr"]),
                          "n_params": int(self.model.count_params())})
        return self.model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "SAEDetector":
        """Train on normal train rows; early-stop on normal validation MSE."""
        from tensorflow import keras

        Xn_tr = np.asarray(X_train[np.asarray(y_train) == 0], dtype="float32")
        Xn_va = np.asarray(X_val[np.asarray(y_val) == 0], dtype="float32")
        if len(Xn_tr) == 0 or len(Xn_va) == 0:
            raise ValueError("SAE needs normal rows in both train and validation splits")
        self.build(X_train.shape[1])
        p = self.params
        cb = keras.callbacks.EarlyStopping(monitor="val_mse", mode="min",
                                           patience=int(p["patience"]),
                                           restore_best_weights=True)
        hist = self.model.fit(Xn_tr, Xn_tr, validation_data=(Xn_va, Xn_va),
                              epochs=int(p["epochs"]), batch_size=int(p["batch_size"]),
                              shuffle=True, callbacks=[cb], verbose=self.verbose)
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        best = int(np.argmin(self.history["val_mse"]))
        self.info.update({"epochs_run": len(self.history["val_mse"]), "best_epoch": best + 1,
                          "best_val_normal_mse": self.history["val_mse"][best],
                          "n_train_normal": len(Xn_tr), "n_val_normal": len(Xn_va)})
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Reconstruction error per row."""
        return reconstruction_errors(self.model, X)

    def set_threshold(self, tau: float) -> None:
        """Set the operating threshold used by :meth:`predict`."""
        self.threshold = float(tau)
        self.default_threshold = float(tau)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Full detection for one batch: reconstruct, error, compare to tau."""
        rec = np.asarray(self.model(X, training=False))
        err = np.mean((X - rec) ** 2, axis=1)
        return err > (self.threshold if self.threshold is not None else np.inf)

    def n_params(self) -> int:
        """Keras ``count_params()``."""
        return int(self.model.count_params())

    def save(self, directory: Path) -> Path:
        """Save as ``sae.keras``."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "sae.keras"
        self.model.save(path)
        return path
