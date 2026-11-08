"""Model registry. TensorFlow is only imported when a TF model is fitted/built."""

from __future__ import annotations

from typing import Optional

from ids.models.base import DetectorModel

MODEL_NAMES = ("sae", "rf", "svm", "lstm")


def make_model(name: str, cfg: dict, seed: int, sae_params: Optional[dict] = None,
               verbose: int = 0) -> DetectorModel:
    """Instantiate a detector by name."""
    if name == "sae":
        from ids.models.sae import SAEDetector

        return SAEDetector(cfg, seed, params=sae_params, verbose=verbose)
    if name == "rf":
        from ids.models.baselines import RFModel

        return RFModel(cfg, seed)
    if name == "svm":
        from ids.models.baselines import SVMModel

        return SVMModel(cfg, seed)
    if name == "lstm":
        from ids.models.baselines import LSTMModel

        return LSTMModel(cfg, seed, verbose=verbose)
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
