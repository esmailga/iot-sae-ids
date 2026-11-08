"""Common interface shared by the SAE and the three baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np


class DetectorModel(ABC):
    """Uniform wrapper so experiments, timing and evaluation treat all models alike.

    ``score`` returns a continuous anomaly/attack score per row (higher = more attack-like):
    reconstruction error for the SAE, P(attack) for RF/LSTM, sigmoid(decision function) for SVM.
    """

    name: str = "base"
    uses_tensorflow: bool = False
    #: Default decision threshold on ``score`` for supervised models (SAE sets its own).
    default_threshold: float = 0.5

    def __init__(self, cfg: dict, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.history: dict[str, list[float]] = {}
        self.info: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "DetectorModel":
        """Train. Supervised models use all rows; the SAE filters to normal rows itself."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Continuous attack score per row, higher = more attack-like."""

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary predictions ``score > threshold`` (1 = attack)."""
        t = self.default_threshold if threshold is None else threshold
        return (self.score(X) > t).astype("int8")

    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Lowest-overhead inference call used for latency timing (one batch)."""

    @abstractmethod
    def n_params(self) -> int:
        """Parameter count as defined in the spec for this model family."""

    @abstractmethod
    def save(self, directory: Path) -> Path:
        """Serialise the fitted model into ``directory``; return the file path."""
