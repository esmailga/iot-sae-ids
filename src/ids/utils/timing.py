"""Wall-clock timing helpers based on ``time.perf_counter``."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timer:
    """Context manager measuring elapsed seconds with ``time.perf_counter``.

    Example:
        >>> with Timer() as t:
        ...     pass
        >>> t.seconds >= 0
        True
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start


@contextmanager
def timed(store: dict, key: str) -> Iterator[None]:
    """Record elapsed seconds into ``store[key]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0
