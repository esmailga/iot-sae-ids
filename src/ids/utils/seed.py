"""Global seeding for Python, NumPy and (if installed) TensorFlow."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def tf_available() -> bool:
    """Return True if TensorFlow can be imported."""
    try:
        import tensorflow  # noqa: F401
    except Exception:  # pragma: no cover - depends on environment
        return False
    return True


def set_global_seed(seed: int, deterministic_tf: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow; enable deterministic TF ops where practical.

    sklearn estimators are seeded separately via their ``random_state`` argument.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic_tf:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    if tf_available():
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if deterministic_tf:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not enable TF op determinism: %s", exc)
