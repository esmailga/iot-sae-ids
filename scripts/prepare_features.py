#!/usr/bin/env python
"""Run preprocessing + two-stage feature selection and report feature counts per stage.

Caches the prepared arrays (data/processed/), writes selected-feature lists
(results/features/<tag>.json), the Pearson heatmap and RF importance plot
(results/figures/features/), and a counts table (results/tables/feature_counts_s<seed>.md).

Usage:
    python scripts/prepare_features.py [--dataset ...] [--seed 0] [--sample 0.02] [--no-cache]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ids.data.prepare import feature_counts_frame, prepare, run_tag
from ids.utils.config import DATASETS, load_config
from ids.utils.logging_utils import setup_logging
from ids.utils.seed import set_global_seed

logger = logging.getLogger("prepare_features")

COLUMNS = {
    "dataset": "Dataset",
    "raw": "Raw columns",
    "after_leakage_drop": "After leakage drop",
    "after_preprocess": "After preprocessing (encoded)",
    "after_stage1_pearson": "After Stage 1 (Pearson)",
    "after_stage2_rf": "After Stage 2 (RF)",
}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()
    setup_logging("INFO")

    preps = []
    for name in args.dataset:
        cfg = load_config(name, args.overrides)
        set_global_seed(args.seed)
        p = prepare(cfg, args.seed, args.sample, use_cache=not args.no_cache,
                    plots=args.sample is None)
        preps.append(p)
        logger.info("%s: %d selected features: %s", run_tag(name, args.seed, args.sample),
                    p.d, p.feature_names)

    df = feature_counts_frame(preps)[list(COLUMNS)].rename(columns=COLUMNS)
    md = df.to_markdown(index=False) if _has_tabulate() else df.to_string(index=False)
    if args.sample is None:
        out = Path(load_config(None)["paths"]["tables_dir"]) / f"feature_counts_s{args.seed}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")
        logger.info("Wrote %s", out)
    print(md)  # CLI report output


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
