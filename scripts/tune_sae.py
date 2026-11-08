#!/usr/bin/env python
"""SAE hyperparameter search per dataset (random or full grid), resumable.

Writes results/best_sae_<dataset>.yaml (used automatically by run_experiment/run_all),
results/tables/sae_tuning_<dataset>.{md,csv}, and raw trials to results/tuning/.

Examples:
    python scripts/tune_sae.py --dataset ton_iot                    # random, 20 trials
    python scripts/tune_sae.py --dataset unsw_nb15 --mode grid       # full grid (216 combos)
    python scripts/tune_sae.py --dataset rt_iot2022 --n-trials 40 --subsample 0.3
"""

from __future__ import annotations

import argparse
import logging

from ids.tuning import run_tuning
from ids.utils.config import DATASETS, load_config
from ids.utils.logging_utils import setup_logging

logger = logging.getLogger("tune_sae")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--mode", choices=["random", "grid"], default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--subsample", type=float, default=None,
                    help="stratified fraction of the train split (default tune.subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=float, default=None, help="dataset fraction (smoke only)")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()
    setup_logging("INFO")
    for ds in args.dataset:
        cfg = load_config(ds, args.overrides)
        best = run_tuning(cfg, args.seed, args.mode, args.n_trials, args.subsample,
                          args.sample, args.verbose)
        logger.info("[%s] best: %s  val ROC-AUC=%.4f  params=%d", ds, best["params"],
                    best["val_roc_auc"], best["n_params"])


if __name__ == "__main__":
    main()
