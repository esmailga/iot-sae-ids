#!/usr/bin/env python
"""Run one experiment: a model on a dataset with a seed.

Writes results/raw/<dataset>__<model>__s<seed>[__frac<f>].json (skipped if it exists,
unless --force), test scores to results/raw/scores/, and the model to results/models/.

Examples:
    python scripts/run_experiment.py --dataset ton_iot --model rf --seed 0 --sample 0.02
    python scripts/run_experiment.py --dataset unsw_nb15 --model sae --seed 1 \\
        --set sae.latent_dim=12 sae.l1=1e-3
"""

from __future__ import annotations

import argparse

from ids.experiment import run_experiment
from ids.models import MODEL_NAMES
from ids.utils.config import DATASETS, load_config
from ids.utils.logging_utils import setup_logging


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=float, default=None,
                    help="stratified fraction of the data to use (smoke tests)")
    ap.add_argument("--config", default=None, help="extra YAML merged over the dataset config")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], help="key.sub=value")
    ap.add_argument("--force", action="store_true", help="rerun even if the result exists")
    ap.add_argument("--no-tuned", action="store_true",
                    help="ignore results/best_sae_<dataset>.yaml")
    ap.add_argument("--verbose", type=int, default=0, help="Keras fit verbosity (0/1/2)")
    args = ap.parse_args()
    cfg = load_config(args.dataset, args.overrides, args.config)
    setup_logging(cfg["logging"]["level"])
    run_experiment(args.dataset, args.model, args.seed, args.sample, args.overrides,
                   args.config, force=args.force, use_tuned=not args.no_tuned,
                   verbose=args.verbose)


if __name__ == "__main__":
    main()
