"""Logging setup shared by scripts."""

from __future__ import annotations

import logging
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logging to stderr (and optionally a file). Safe to call repeatedly."""
    root = logging.getLogger()
    root.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper()))
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    root.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
    # Quieten chatty libraries.
    for noisy in ("matplotlib", "PIL", "absl", "h5py", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
