#!/usr/bin/env python
"""Smoke test on real data: every dataset x model with --sample 0.02 (default), seed 0.

Runs in-process, continues past failures, and prints/writes a summary table to
results/tables/smoke_test.md. Models needing TensorFlow are reported as "needs TF" when it
is not installed (e.g. on the Intel-Mac dev machine) — run those on Colab.

Usage:
    python scripts/smoke_test.py [--models sae rf svm lstm] [--datasets ...] [--sample 0.02]
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path

import pandas as pd

from ids.experiment import run_experiment
from ids.models import MODEL_NAMES
from ids.utils.config import DATASETS, load_config
from ids.utils.logging_utils import setup_logging
from ids.utils.seed import tf_available
from ids.utils.timing import Timer

logger = logging.getLogger("smoke_test")


def _row(res: dict, seconds: float) -> dict:
    m, e = res["metrics"]["test"], res["efficiency"]
    lat = e.get("cpu", {}).get("per_flow_mean_ms")
    gpu = e.get("gpu", {}).get("per_flow_mean_ms") if "gpu" in e else None
    return {
        "status": "ok", "wall_s": round(seconds, 1), "d": res["data"]["d"],
        "n_train": res["data"]["split_sizes"]["train"], "n_test": res["data"]["split_sizes"]["test"],
        "variant": res["model"].get("variant", res["model"].get("hidden_mode", "")),
        "acc": m["accuracy"], "f1": m["f1"], "recall": m["recall"], "fpr": m["fpr"],
        "roc_auc": m.get("roc_auc"), "train_s": e["training_time_s"],
        "cpu_ms_per_flow": lat, "gpu_ms_per_flow": gpu, "params": e["n_params"],
        "size_mb": e["model_size_mb"],
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    ap.add_argument("--sample", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()
    setup_logging("INFO")
    have_tf = tf_available()

    rows = []
    for ds in args.datasets:
        for model in args.models:
            base = {"dataset": ds, "model": model}
            if model in ("sae", "lstm") and not have_tf:
                rows.append({**base, "status": "needs TF (run on Colab)"})
                continue
            try:
                with Timer() as t:
                    res = run_experiment(ds, model, args.seed, args.sample, args.overrides,
                                         force=args.force)
                rows.append({**base, **_row(res, t.seconds)})
            except Exception as exc:
                logger.error("%s/%s failed: %s\n%s", ds, model, exc, traceback.format_exc())
                rows.append({**base, "status": f"FAILED: {type(exc).__name__}: {exc}"[:120]})

    df = pd.DataFrame(rows)
    for c in ("acc", "f1", "recall", "fpr", "roc_auc"):
        if c in df:
            df[c] = df[c].astype(float).round(4)
    for c in ("train_s", "cpu_ms_per_flow", "gpu_ms_per_flow", "size_mb"):
        if c in df:
            df[c] = df[c].astype(float).round(4)
    md = df.to_markdown(index=False)
    cfg = load_config(None)
    out = Path(cfg["paths"]["tables_dir"]) / "smoke_test.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Smoke test (sample={args.sample}, seed={args.seed})\n\n{md}\n")
    print(md)  # CLI report output


if __name__ == "__main__":
    main()
