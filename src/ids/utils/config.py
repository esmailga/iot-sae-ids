"""YAML configuration loading with dataset overlays and CLI overrides.

Resolution order (later wins):
    configs/default.yaml  <-  configs/<dataset>.yaml  <-  --set key.sub=value overrides

Paths in ``paths`` are resolved relative to the repo root, unless the environment variables
``IDS_DATA_ROOT`` / ``IDS_RESULTS_ROOT`` are set (used on Colab to point at Google Drive).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

DATASETS: tuple[str, ...] = ("ton_iot", "unsw_nb15", "rt_iot2022")
MODELS: tuple[str, ...] = ("sae", "rf", "svm", "lstm")


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repo root: the first ancestor containing ``configs/default.yaml``.

    Searches upward from ``start`` (default: CWD), then from this file's location.
    """
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve())
    for base in candidates:
        for p in (base, *base.parents):
            if (p / "configs" / "default.yaml").is_file():
                return p
    raise FileNotFoundError("Could not locate repo root (configs/default.yaml not found).")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: Any) -> Any:
    """Convert strings like ``"1e-3"`` (which YAML 1.1 leaves as str) to float, recursively."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    return value


def parse_overrides(overrides: Iterable[str] | None) -> dict:
    """Turn ``["a.b=1", "c=[1,2]"]`` into a nested dict, YAML-parsing each value."""
    result: dict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = _coerce(yaml.safe_load(raw))
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dict (empty dict for an empty file)."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    dataset: str | None = None,
    overrides: Iterable[str] | None = None,
    config_path: str | Path | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Build the effective config for ``dataset``.

    Args:
        dataset: Dataset key (one of :data:`DATASETS`) or ``None`` for defaults only.
        overrides: CLI overrides like ``"sae.latent_dim=12"``.
        config_path: Optional extra YAML merged after the dataset file.
        repo_root: Repo root; auto-detected if omitted.

    Returns:
        Config dict with absolute paths under ``cfg["paths"]``.
    """
    root = repo_root or find_repo_root()
    cfg = load_yaml(root / "configs" / "default.yaml")
    if dataset is not None:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")
        cfg = deep_merge(cfg, load_yaml(root / "configs" / f"{dataset}.yaml"))
    if config_path is not None:
        cfg = deep_merge(cfg, load_yaml(Path(config_path)))
    cfg = deep_merge(cfg, parse_overrides(overrides))
    cfg["paths"] = resolve_paths(cfg.get("paths", {}), root)
    return cfg


def resolve_paths(paths: dict[str, Any], root: Path) -> dict[str, str]:
    """Resolve data/results roots to absolute paths, honouring env-var overrides."""
    data_root = os.environ.get("IDS_DATA_ROOT") or paths.get("data_root", "data")
    results_root = os.environ.get("IDS_RESULTS_ROOT") or paths.get("results_root", "results")

    def _abs(p: str) -> Path:
        pp = Path(p).expanduser()
        return pp if pp.is_absolute() else (root / pp)

    d, r = _abs(data_root), _abs(results_root)
    return {
        "repo_root": str(root),
        "data_root": str(d),
        "raw_dir": str(d / "raw"),
        "processed_dir": str(d / "processed"),
        "results_root": str(r),
        "results_raw": str(r / "raw"),
        "tables_dir": str(r / "tables"),
        "figures_dir": str(r / "figures"),
        "models_dir": str(r / "models"),
        "features_dir": str(r / "features"),
        "tuning_dir": str(r / "tuning"),
    }
