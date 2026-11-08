"""Dataset loaders: one per dataset, each returning ``(X, y_binary, y_multiclass)``.

Loaders do only *stateless* cleaning that cannot leak information across splits:
stripping whitespace, normalising label values, inferring numeric columns, and dropping the
configured identifier/timestamp columns. Everything that is *fitted* (imputation, encoding,
scaling, feature selection) lives in :mod:`ids.data.preprocess` / :mod:`ids.features`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

XY = tuple[pd.DataFrame, pd.Series, pd.Series]

NORMAL_LABEL = "normal"
DASH = "-"


class LabelMappingNotConfirmed(RuntimeError):
    """Raised when RT-IoT2022's normal/attack mapping has not been confirmed in config."""


@dataclass
class LoadedData:
    """A loaded dataset. ``test`` is set only when the dataset has an official test split."""

    X: pd.DataFrame
    y_bin: pd.Series
    y_multi: pd.Series
    test: Optional[XY] = None
    dropped_columns: tuple[str, ...] = ()


# ----------------------------------------------------------------------------- helpers

def resolve_file(base: Path, name: str) -> Path:
    """Return ``base / name``, matching the file name case-insensitively if needed.

    macOS filesystems are case-insensitive but Colab/Linux is not, and dataset mirrors differ
    in capitalisation (e.g. ``train_test_network.csv`` vs ``Train_Test_Network.csv``).
    """
    exact = base / name
    if not base.is_dir():
        return exact
    # Prefer an exact match, then a case-insensitive one, using real on-disk names
    # (on a case-insensitive filesystem `exact.is_file()` is True for any capitalisation).
    files = [c for c in base.iterdir() if c.is_file()]
    for cand in files:
        if cand.name == name:
            return cand
    for cand in files:
        if cand.name.lower() == name.lower():
            return cand
    return exact


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw file not found: {path}. See README 'Data' section for download instructions."
        )
    logger.info("Reading %s", path)
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace in object columns and column names."""
    df = df.rename(columns=lambda c: str(c).strip())
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype("string").str.strip().astype(object)
    return df


def infer_numeric_columns(df: pd.DataFrame, dash_is_category: bool = True) -> pd.DataFrame:
    """Convert object columns whose non-placeholder values are all numeric to float.

    In such columns the dash placeholder ``"-"`` (and empty strings) become NaN, because a
    numeric column cannot hold a category. Purely categorical columns keep ``"-"`` as a
    category when ``dash_is_category`` is True, otherwise it is turned into NaN.
    """
    df = df.copy()
    for c in df.columns[df.dtypes == object]:
        s = df[c]
        placeholder = s.isin([DASH, ""]) | s.isna()
        conv = pd.to_numeric(s.where(~placeholder), errors="coerce")
        non_placeholder = (~placeholder).sum()
        if non_placeholder > 0 and conv.notna().sum() == non_placeholder:
            df[c] = conv.astype("float64")
        elif not dash_is_category:
            df[c] = s.mask(s.isin([DASH, ""]))
    # Downcast numerics to save memory (8 GB dev machine).
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def _drop_columns(df: pd.DataFrame, drop_cols: list[str]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    present = [c for c in drop_cols if c in df.columns]
    missing = [c for c in drop_cols if c not in df.columns]
    if missing:
        logger.info("Configured drop columns not present (ignored): %s", missing)
    return df.drop(columns=present), tuple(present)


def _finalise(
    df: pd.DataFrame,
    y_bin: pd.Series,
    y_multi: pd.Series,
    label_cols: list[str],
    ds_cfg: dict,
    drop: bool,
) -> tuple[XY, tuple[str, ...]]:
    X = df.drop(columns=[c for c in label_cols if c in df.columns])
    dropped: tuple[str, ...] = ()
    if drop:
        X, dropped = _drop_columns(X, list(ds_cfg.get("drop_cols", [])))
    X = infer_numeric_columns(X, dash_is_category=ds_cfg.get("dash_is_category", True))
    X = X.reset_index(drop=True)
    return (X, y_bin.reset_index(drop=True).astype("int8"),
            y_multi.reset_index(drop=True).astype(str)), dropped


def _normalise_multiclass(s: pd.Series) -> pd.Series:
    """Lower-case 'normal'-like labels to a canonical ``"normal"``; keep others as-is."""
    s = s.astype(str).str.strip()
    is_normal = s.str.lower().isin(["normal", "benign", "nan", ""])
    return s.mask(is_normal, NORMAL_LABEL)


# ----------------------------------------------------------------------------- TON_IoT

def load_ton_iot(cfg: dict, drop: bool = True) -> LoadedData:
    """Load TON_IoT network data (``Train_Test_Network.csv``)."""
    ds = cfg["dataset"]
    path = resolve_file(Path(cfg["paths"]["raw_dir"]) / "ton_iot", ds["files"][0])
    df = _strip_strings(_read_csv(path))
    y_bin = pd.to_numeric(df[ds["label_col"]], errors="raise")
    y_multi = _normalise_multiclass(df[ds["multiclass_col"]])
    _check_label_consistency(y_bin, y_multi, "ton_iot")
    (X, yb, ym), dropped = _finalise(
        df, y_bin, y_multi, [ds["label_col"], ds["multiclass_col"]], ds, drop
    )
    return LoadedData(X, yb, ym, None, dropped)


# ----------------------------------------------------------------------------- UNSW-NB15

def load_unsw_nb15(cfg: dict, drop: bool = True) -> LoadedData:
    """Load UNSW-NB15 official training set (as ``X``) and official test set (as ``test``)."""
    ds = cfg["dataset"]
    base = Path(cfg["paths"]["raw_dir"]) / "unsw_nb15"
    parts: dict[str, XY] = {}
    dropped: tuple[str, ...] = ()
    for part in ("train", "test"):
        df = _strip_strings(_read_csv(resolve_file(base, ds["files"][part])))
        y_bin = pd.to_numeric(df[ds["label_col"]], errors="raise")
        y_multi = _normalise_multiclass(df[ds["multiclass_col"]])
        _check_label_consistency(y_bin, y_multi, f"unsw_nb15/{part}")
        parts[part], dropped = _finalise(
            df, y_bin, y_multi, [ds["label_col"], ds["multiclass_col"]], ds, drop
        )
    Xtr, ytr, mtr = parts["train"]
    Xte, yte, mte = parts["test"]
    # Align test columns/dtypes to train (they share a schema in the official release).
    if list(Xtr.columns) != list(Xte.columns):
        raise ValueError("UNSW-NB15 train/test column mismatch: "
                         f"{set(Xtr.columns) ^ set(Xte.columns)}")
    return LoadedData(Xtr, ytr, mtr, (Xte, yte, mte), dropped)


# ----------------------------------------------------------------------------- RT-IoT2022

_ATTACK_PATTERN = re.compile(
    r"dos|ddos|nmap|scan|arp|poison|brute|metasploit|flood|slowloris|hping|xmas|exploit|attack",
    re.IGNORECASE,
)


def propose_rt_iot_normal_classes(labels: pd.Series) -> list[str]:
    """Heuristically propose which RT-IoT2022 ``Attack_type`` values are normal traffic.

    Any label not matching a known attack keyword is proposed as a normal device class.
    The result must be confirmed by a human and written into ``configs/rt_iot2022.yaml``.
    """
    return sorted(v for v in labels.astype(str).unique() if not _ATTACK_PATTERN.search(v))


def _rt_iot_raw(cfg: dict) -> pd.DataFrame:
    """Return the raw RT-IoT2022 frame: local CSV cache first, else UCI (and cache it)."""
    ds = cfg["dataset"]
    local = resolve_file(Path(cfg["paths"]["raw_dir"]) / "rt_iot2022", ds["files"][0])
    if local.is_file():
        return _read_csv(local)
    logger.info("Local RT-IoT2022 CSV not found; fetching UCI dataset id %s", ds["uci_id"])
    try:
        from ucimlrepo import fetch_ucirepo

        repo = fetch_ucirepo(id=int(ds["uci_id"]))
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not fetch RT-IoT2022 from UCI ({exc}) and no local file at {local}."
        ) from exc
    local.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local, index=False)
    logger.info("Cached RT-IoT2022 to %s (%d rows)", local, len(df))
    return df


def load_rt_iot2022(cfg: dict, drop: bool = True, allow_unconfirmed: bool = False) -> LoadedData:
    """Load RT-IoT2022 and derive the binary label from the confirmed normal-class mapping."""
    ds = cfg["dataset"]
    df = _strip_strings(_rt_iot_raw(cfg))
    y_multi_raw = df[ds["multiclass_col"]].astype(str)
    normal = list(ds.get("normal_classes") or [])
    if not ds.get("label_mapping_confirmed", False):
        if not allow_unconfirmed:
            raise LabelMappingNotConfirmed(
                "RT-IoT2022 normal/attack mapping not confirmed. Run scripts/inspect_data.py, "
                "review the proposal, then set dataset.normal_classes and "
                "dataset.label_mapping_confirmed: true in configs/rt_iot2022.yaml."
            )
        normal = normal or propose_rt_iot_normal_classes(y_multi_raw)
        logger.warning("Using UNCONFIRMED RT-IoT2022 normal classes: %s", normal)
    unknown = set(normal) - set(y_multi_raw.unique())
    if unknown:
        raise ValueError(f"normal_classes not found in data: {sorted(unknown)}")
    is_normal = y_multi_raw.isin(normal)
    y_bin = (~is_normal).astype("int8")
    # Keep the original device-level labels: they are used for stratification, and
    # per-attack recall is computed over attack rows only (y_bin == 1).
    y_multi = y_multi_raw
    (X, yb, ym), dropped = _finalise(df, y_bin, y_multi, [ds["multiclass_col"]], ds, drop)
    return LoadedData(X, yb, ym, None, dropped)


# ----------------------------------------------------------------------------- dispatch

def _check_label_consistency(y_bin: pd.Series, y_multi: pd.Series, name: str) -> None:
    """Warn if binary label and multiclass label disagree about 'normal'."""
    mismatch = int(((y_bin == 0) != (y_multi == NORMAL_LABEL)).sum())
    if mismatch:
        logger.warning("%s: %d rows where binary label and multiclass 'normal' disagree",
                       name, mismatch)


def load_dataset(cfg: dict, drop: bool = True, allow_unconfirmed: bool = False) -> LoadedData:
    """Dispatch to the loader named by ``cfg['dataset']['name']``."""
    name = cfg["dataset"]["name"]
    if name == "ton_iot":
        return load_ton_iot(cfg, drop)
    if name == "unsw_nb15":
        return load_unsw_nb15(cfg, drop)
    if name == "rt_iot2022":
        return load_rt_iot2022(cfg, drop, allow_unconfirmed)
    raise ValueError(f"Unknown dataset {name!r}")


def raw_file_paths(cfg: dict) -> list[Path]:
    """Paths of the raw files a dataset expects (for availability checks)."""
    ds = cfg["dataset"]
    base = Path(cfg["paths"]["raw_dir"]) / ds["name"]
    files = ds["files"]
    names = list(files.values()) if isinstance(files, dict) else list(files)
    return [resolve_file(base, f) for f in names]


__all__ = [
    "LoadedData", "LabelMappingNotConfirmed", "NORMAL_LABEL", "load_dataset",
    "load_ton_iot", "load_unsw_nb15", "load_rt_iot2022", "propose_rt_iot_normal_classes",
    "infer_numeric_columns", "raw_file_paths",
]
