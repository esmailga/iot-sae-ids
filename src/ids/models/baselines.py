"""Supervised baselines: Random Forest, SVM (exact RBF or Nystroem+SGD) and LSTM.

All are trained on *all* rows of the training split, using the same Stage 1-2 features as
the SAE, and expose the common :class:`~ids.models.base.DetectorModel` interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ids.models.base import DetectorModel

logger = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# ----------------------------------------------------------------------------- Random Forest

class RFModel(DetectorModel):
    """scikit-learn Random Forest (100 trees, Gini, unlimited depth, n_jobs=-1)."""

    name = "rf"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "RFModel":
        c = self.cfg["baselines"]["rf"]
        self.model = RandomForestClassifier(
            n_estimators=c["n_estimators"], criterion=c["criterion"], max_depth=c["max_depth"],
            n_jobs=c["n_jobs"], random_state=self.seed,
        ).fit(X_train, y_train)
        self.info.update({"variant": "random_forest", "n_estimators": c["n_estimators"],
                          "n_nodes": self.n_params()})
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """P(attack)."""
        return self.model.predict_proba(X)[:, 1]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Class predictions for one batch."""
        return self.model.predict(X)

    def n_params(self) -> int:
        """Total number of tree nodes."""
        return int(sum(est.tree_.node_count for est in self.model.estimators_))

    def save(self, directory: Path) -> Path:
        """Save with joblib."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "rf.joblib"
        joblib.dump(self.model, path)
        return path


# ----------------------------------------------------------------------------- SVM

class SVMModel(DetectorModel):
    """RBF SVM.

    * ``n_train <= exact_max_rows``: exact ``SVC(kernel="rbf", C=1, gamma="scale")``.
    * otherwise: ``Nystroem`` RBF feature map (``nystroem_components``) + linear
      ``SGDClassifier(loss="hinge")`` — an *approximation*, recorded as such in ``info``.

    Scores are ``sigmoid(decision_function)``: monotone in the SVM margin, so ROC-AUC is
    valid, but they are **not calibrated probabilities**. Prediction is ``margin > 0``.
    """

    name = "svm"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "SVMModel":
        c = self.cfg["baselines"]["svm"]
        n, d = X_train.shape
        self.d = d
        if n <= c["exact_max_rows"]:
            self.variant = "exact_rbf_svc"
            self.model = SVC(kernel="rbf", C=c["C"], gamma=c["gamma"], random_state=self.seed)
        else:
            self.variant = "nystroem_rbf_sgd_hinge"
            var = float(np.asarray(X_train, dtype="float64").var())
            gamma = 1.0 / (d * var) if var > 0 else 1.0  # same rule as gamma="scale"
            self.model = Pipeline([
                ("nystroem", Nystroem(kernel="rbf", gamma=gamma,
                                      n_components=min(c["nystroem_components"], n),
                                      random_state=self.seed)),
                ("sgd", SGDClassifier(loss="hinge", alpha=c["sgd_alpha"],
                                      max_iter=c["sgd_max_iter"], tol=c["sgd_tol"],
                                      random_state=self.seed)),
            ])
        logger.info("SVM variant: %s (n_train=%d, d=%d)", self.variant, n, d)
        self.model.fit(X_train, y_train)
        self.info.update({"variant": self.variant, "n_train": n,
                          "exact_max_rows": c["exact_max_rows"], "n_params": self.n_params()})
        if self.variant == "exact_rbf_svc":
            self.info["n_support_vectors"] = int(self.model.n_support_.sum())
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        """Signed margin (positive = attack)."""
        return self.model.decision_function(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """sigmoid(margin): monotone attack score in (0, 1), not a calibrated probability."""
        return _sigmoid(self.decision(X))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Class predictions for one batch."""
        return self.model.predict(X)

    def n_params(self) -> int:
        """Exact: n_support_vectors x d (+ dual coefs + intercept are excluded per spec).
        Nystroem: n_components x d (landmarks) + linear coefficients + intercept."""
        if self.variant == "exact_rbf_svc":
            return int(self.model.n_support_.sum() * self.d)
        ny: Nystroem = self.model.named_steps["nystroem"]
        sgd: SGDClassifier = self.model.named_steps["sgd"]
        return int(ny.components_.size + sgd.coef_.size + sgd.intercept_.size)

    def save(self, directory: Path) -> Path:
        """Save with joblib."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "svm.joblib"
        joblib.dump(self.model, path)
        return path


# ----------------------------------------------------------------------------- LSTM

def build_lstm(d: int, units: list[int], recurrent_dropout: float, lr: float, seed: int) -> Any:
    """Stacked LSTM over the feature vector treated as a length-d sequence of scalars."""
    from tensorflow import keras

    inp = keras.Input(shape=(d, 1), name="x_seq")
    x = inp
    for i, u in enumerate(units):
        x = keras.layers.LSTM(
            u, return_sequences=i < len(units) - 1, recurrent_dropout=recurrent_dropout,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + i),
            recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 10 + i),
            seed=seed + 20 + i, name=f"lstm_{i}",
        )(x)
    out = keras.layers.Dense(1, activation="sigmoid",
                             kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 99),
                             name="p_attack")(x)
    model = keras.Model(inp, out, name="lstm_classifier")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    return model


class LSTMModel(DetectorModel):
    """Two stacked LSTMs (128, 64; recurrent dropout 0.3) + sigmoid output.

    Note: ``recurrent_dropout > 0`` disables the cuDNN kernel, so GPU training is slower.
    """

    name = "lstm"
    uses_tensorflow = True

    def __init__(self, cfg: dict, seed: int, verbose: int = 0) -> None:
        super().__init__(cfg, seed)
        self.verbose = verbose

    @staticmethod
    def _seq(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        return X.reshape(X.shape[0], X.shape[1], 1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "LSTMModel":
        from tensorflow import keras

        c = self.cfg["baselines"]["lstm"]
        d = X_train.shape[1]
        self.model = build_lstm(d, list(c["units"]), float(c["recurrent_dropout"]),
                                float(c["lr"]), self.seed)
        cb = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                           patience=int(c["patience"]),
                                           restore_best_weights=True)
        hist = self.model.fit(self._seq(X_train), np.asarray(y_train, dtype="float32"),
                              validation_data=(self._seq(X_val),
                                               np.asarray(y_val, dtype="float32")),
                              epochs=int(c["epochs"]), batch_size=int(c["batch_size"]),
                              shuffle=True, callbacks=[cb], verbose=self.verbose)
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        best = int(np.argmin(self.history["val_loss"]))
        self.info.update({"variant": "stacked_lstm", "units": list(c["units"]),
                          "recurrent_dropout": float(c["recurrent_dropout"]),
                          "epochs_run": len(self.history["val_loss"]), "best_epoch": best + 1,
                          "n_params": self.n_params()})
        return self

    def score(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """P(attack)."""
        Xs = self._seq(X)
        out = np.empty(len(Xs), dtype="float64")
        for i in range(0, len(Xs), batch_size):
            out[i:i + batch_size] = np.asarray(self.model(Xs[i:i + batch_size],
                                                          training=False)).ravel()
        return out

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Class predictions for one batch."""
        return np.asarray(self.model(self._seq(X), training=False)).ravel() > 0.5

    def n_params(self) -> int:
        """Keras ``count_params()``."""
        return int(self.model.count_params())

    def save(self, directory: Path) -> Path:
        """Save as ``lstm.keras``."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "lstm.keras"
        self.model.save(path)
        return path
