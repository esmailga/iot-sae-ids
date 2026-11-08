"""End-to-end data preparation shared by all models for a given (dataset, seed, sample).

load -> leakage drops -> [dataset-scope dedup] -> [subsample] -> split
     -> train-only row cleaning -> Preprocessor.fit(train) -> transform train/val/test
     -> PearsonFilter.fit(train) -> RFImportanceSelector.fit(train) -> selected features

Results are cached under ``<data_root>/processed/<dataset>/<key>/`` so every model in the
same (dataset, seed, sample) sees *identical* splits and features, and reruns are fast.
The cache key hashes the config sections that affect the data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from ids.data.loaders import LoadedData, load_dataset
from ids.data.preprocess import Preprocessor, clean_train_rows, deduplicate
from ids.data.splits import make_split, stratified_subsample
from ids.features.selection import (
    PearsonFilter,
    RFImportanceSelector,
    plot_correlation_heatmap,
    plot_rf_importance,
)

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
_PARTS = ("train", "val", "test")


@dataclass
class Prepared:
    """Model-ready arrays for one (dataset, seed, sample)."""

    dataset: str
    seed: int
    sample: Optional[float]
    X: dict[str, np.ndarray]          # part -> float32 (n, d) in [0, 1]
    y: dict[str, np.ndarray]          # part -> int8 binary labels (1 = attack)
    m: dict[str, np.ndarray]          # part -> str multiclass labels
    feature_names: list[str]
    counts: dict[str, int]            # feature counts per stage
    stats: dict[str, Any] = field(default_factory=dict)
    preprocessor: Optional[Preprocessor] = None
    pearson: Optional[PearsonFilter] = None
    rf_selector: Optional[RFImportanceSelector] = None
    cache_dir: Optional[str] = None

    @property
    def d(self) -> int:
        """Number of selected features (input dimension for every model)."""
        return len(self.feature_names)

    def normal(self, part: str) -> np.ndarray:
        """Rows of ``part`` whose binary label is normal (0)."""
        return self.X[part][self.y[part] == 0]


def run_tag(dataset: str, seed: int, sample: Optional[float]) -> str:
    """Canonical tag used in file names, e.g. ``ton_iot__s0`` or ``ton_iot__s0__frac0.02``."""
    return f"{dataset}__s{seed}" + (f"__frac{sample:g}" if sample else "")


def cache_key(cfg: dict, seed: int, sample: Optional[float]) -> str:
    """Hash of everything that changes the prepared data."""
    payload = {
        "v": CACHE_VERSION,
        "dataset": cfg["dataset"],
        "split": cfg["split"],
        "preprocess": cfg["preprocess"],
        "features": cfg["features"],
        "seed": seed,
        "sample": sample,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _dedup_dataset(data: LoadedData) -> tuple[LoadedData, dict]:
    X, yb, ym, n = deduplicate(data.X, data.y_bin, data.y_multi)
    stats = {"dataset_duplicates_removed": n}
    test = data.test
    if test is not None:
        Xt, ybt, ymt, nt = deduplicate(*test)
        test = (Xt, ybt, ymt)
        stats["official_test_duplicates_removed"] = nt
    logger.info("Dataset-scope dedup: %s", stats)
    return LoadedData(X, yb, ym, test, data.dropped_columns), stats


def prepare(
    cfg: dict,
    seed: int,
    sample: Optional[float] = None,
    use_cache: bool = True,
    plots: bool = True,
) -> Prepared:
    """Prepare (or load from cache) model-ready data. See module docstring."""
    name = cfg["dataset"]["name"]
    key = cache_key(cfg, seed, sample)
    cdir = Path(cfg["paths"]["processed_dir"]) / name / f"{run_tag(name, seed, sample)}__{key}"
    if use_cache and (cdir / "meta.json").is_file():
        logger.info("Loading prepared data from cache %s", cdir)
        return _load(cdir)

    stats: dict[str, Any] = {}
    data = load_dataset(cfg)
    counts = {"raw": data.X.shape[1] + len(data.dropped_columns),
              "after_leakage_drop": data.X.shape[1]}
    stats["dropped_leakage_columns"] = list(data.dropped_columns)
    stats["rows_loaded"] = len(data.X) + (len(data.test[0]) if data.test is not None else 0)

    if cfg["preprocess"].get("dedup_scope", "dataset") == "dataset":
        data, s = _dedup_dataset(data)
        stats.update(s)
    if sample:
        data = stratified_subsample(data, sample, seed)

    split = make_split(data, cfg, seed)
    stats["split_sizes_before_cleaning"] = split.sizes()
    Xtr, ytr, mtr, s = clean_train_rows(split.X_train, split.y_train, split.m_train,
                                        cfg["preprocess"])
    stats.update(s)

    pp = Preprocessor.from_config(cfg["preprocess"]).fit(Xtr)
    P = {"train": pp.transform(Xtr, "train"),
         "val": pp.transform(split.X_val, "val"),
         "test": pp.transform(split.X_test, "test")}
    counts["after_preprocess"] = P["train"].shape[1]
    stats["preprocessor"] = pp.summary()

    fcfg = cfg["features"]
    pf = PearsonFilter(threshold=fcfg["pearson_threshold"]).fit(P["train"])
    S1 = pf.transform(P["train"])
    counts["after_stage1_pearson"] = S1.shape[1]
    rf = RFImportanceSelector(
        cumulative=fcfg["rf_cumulative_importance"], n_estimators=fcfg["rf_n_estimators"],
        max_rows=fcfg.get("rf_max_rows", 0), random_state=seed,
    ).fit(S1, ytr.to_numpy())
    feats = rf.selected_
    counts["after_stage2_rf"] = len(feats)

    prep = Prepared(
        dataset=name, seed=seed, sample=sample,
        X={k: P[k][feats].to_numpy(dtype="float32") for k in _PARTS},
        y={"train": ytr.to_numpy("int8"), "val": split.y_val.to_numpy("int8"),
           "test": split.y_test.to_numpy("int8")},
        m={"train": mtr.to_numpy(str), "val": split.m_val.to_numpy(str),
           "test": split.m_test.to_numpy(str)},
        feature_names=feats, counts=counts, stats=stats,
        preprocessor=pp, pearson=pf, rf_selector=rf, cache_dir=str(cdir),
    )
    prep.stats["split_sizes"] = {k: int(len(v)) for k, v in prep.X.items()}
    prep.stats["normal_rows"] = {k: int((v == 0).sum()) for k, v in prep.y.items()}
    logger.info("Feature counts %s | split sizes %s", counts, prep.stats["split_sizes"])

    _save(prep, cdir)
    _write_feature_report(prep, cfg, plots)
    return prep


# ----------------------------------------------------------------------------- persistence

def _save(p: Prepared, cdir: Path) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k in _PARTS:
        arrays[f"X_{k}"], arrays[f"y_{k}"], arrays[f"m_{k}"] = p.X[k], p.y[k], p.m[k]
    np.savez(cdir / "arrays.npz", **arrays)
    joblib.dump({"preprocessor": p.preprocessor, "pearson": p.pearson,
                 "rf_selector": p.rf_selector}, cdir / "transformers.joblib")
    meta = {"dataset": p.dataset, "seed": p.seed, "sample": p.sample,
            "feature_names": p.feature_names, "counts": p.counts, "stats": p.stats}
    (cdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _load(cdir: Path) -> Prepared:
    meta = json.loads((cdir / "meta.json").read_text())
    with np.load(cdir / "arrays.npz", allow_pickle=False) as z:
        X = {k: z[f"X_{k}"] for k in _PARTS}
        y = {k: z[f"y_{k}"] for k in _PARTS}
        m = {k: z[f"m_{k}"] for k in _PARTS}
    tr = joblib.load(cdir / "transformers.joblib")
    return Prepared(meta["dataset"], meta["seed"], meta["sample"], X, y, m,
                    meta["feature_names"], meta["counts"], meta["stats"],
                    tr["preprocessor"], tr["pearson"], tr["rf_selector"], str(cdir))


def _write_feature_report(p: Prepared, cfg: dict, plots: bool) -> None:
    """Save selected-feature lists (results/features/) and, optionally, the two plots."""
    tag = run_tag(p.dataset, p.seed, p.sample)
    fdir = Path(cfg["paths"]["features_dir"])
    fdir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": p.dataset, "seed": p.seed, "sample": p.sample, "counts": p.counts,
        "after_preprocess": list(p.preprocessor.get_feature_names_out()),
        "pearson_dropped": [{"dropped": a, "kept": b, "abs_r": round(r, 4)}
                            for a, b, r in p.pearson.dropped_],
        "after_stage1_pearson": list(p.pearson.selected_),
        "rf_importances": {k: round(float(v), 6) for k, v in p.rf_selector.importances_.items()},
        "after_stage2_rf": p.feature_names,
        "preprocessor": p.stats.get("preprocessor"),
    }
    (fdir / f"{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    if plots:
        figdir = Path(cfg["paths"]["figures_dir"]) / "features"
        plot_correlation_heatmap(p.pearson.corr_, figdir / f"{tag}_pearson_heatmap.png",
                                 title=f"{p.dataset} (train, seed {p.seed})",
                                 threshold=cfg["features"]["pearson_threshold"])
        plot_rf_importance(p.rf_selector.importances_, len(p.feature_names),
                           figdir / f"{tag}_rf_importance.png",
                           title=f"{p.dataset} RF importance (seed {p.seed})",
                           cumulative=cfg["features"]["rf_cumulative_importance"])


def feature_counts_frame(preps: list[Prepared]) -> pd.DataFrame:
    """Tabulate feature counts per stage for several prepared datasets."""
    return pd.DataFrame([{"dataset": p.dataset, "seed": p.seed, **p.counts} for p in preps])
