"""Library-version and hardware information recorded with every run."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from typing import Any

_LIBS = ("numpy", "pandas", "sklearn", "scipy", "tensorflow", "keras", "matplotlib", "ucimlrepo")


def library_versions() -> dict[str, str | None]:
    """Return installed versions of the key libraries (None if not installed)."""
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in _LIBS:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def _gpu_names() -> list[str]:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def hardware_info() -> dict[str, Any]:
    """Return CPU, RAM, GPU and platform details."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "on_colab": "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ,
        "gpus": _gpu_names(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        info["ram_gb"] = None
    return info
