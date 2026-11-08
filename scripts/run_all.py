#!/usr/bin/env python
"""Full experiment grid: models x datasets x seeds. Resumable.

Runs whose result JSON already exists are skipped, so after a Colab disconnect just run the
same command again. Failures are logged to results/raw/failures.log and the grid continues.
Order: dataset -> seed -> model, so the cached features of a (dataset, seed) are reused.

Examples:
    python scripts/run_all.py                                  # 4 models x 3 datasets x 5 seeds
    python scripts/run_all.py --models sae rf --datasets ton_iot
    python scripts/run_all.py --dry-run                        # show what is left to run
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import logging
import traceback
from pathlib import Path

from ids.experiment import result_path, run_experiment
from ids.models import MODEL_NAMES
from ids.utils.config import DATASETS, load_config
from ids.utils.logging_utils import setup_logging
from ids.utils.seed import tf_available
from ids.utils.timing import Timer

logger = logging.getLogger("run_all")


def _clear_tf() -> None:
    try:
        from tensorflow import keras

        keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="default: seeds from configs/default.yaml")
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--no-tuned", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_config(None, args.overrides)
    setup_logging(base["logging"]["level"],
                  Path(base["paths"]["results_raw"]) / "run_all.log")
    seeds = args.seeds if args.seeds is not None else list(base["seeds"])
    have_tf = tf_available()

    plan, done = [], 0
    for ds in args.datasets:
        cfg = load_config(ds, args.overrides)
        for seed in seeds:
            for model in args.models:
                if result_path(cfg, ds, model, seed, args.sample).is_file():
                    done += 1
                else:
                    plan.append((ds, seed, model))
    logger.info("Grid: %d runs, %d already done, %d to run", done + len(plan), done, len(plan))
    for ds, seed, model in plan:
        logger.info("  todo: %s / seed %d / %s", ds, seed, model)
    if args.dry_run:
        return
    if not have_tf and any(m in ("sae", "lstm") for _, _, m in plan):
        logger.warning("TensorFlow not installed: SAE/LSTM runs will be skipped here.")
    if not args.no_tuned:
        for ds in sorted({d for d, _, m in plan if m == "sae"}):
            best = Path(base["paths"]["results_root"]) / f"best_sae_{ds}.yaml"
            if not best.is_file():
                logger.warning("No %s: SAE runs on %s will use the DEFAULT hyperparameters. "
                               "Run scripts/tune_sae.py --dataset %s first.", best.name, ds, ds)

    fail_log = Path(base["paths"]["results_raw"]) / "failures.log"
    n_ok = n_fail = n_skip = 0
    for k, (ds, seed, model) in enumerate(plan, 1):
        if model in ("sae", "lstm") and not have_tf:
            n_skip += 1
            continue
        logger.info("=== [%d/%d] %s / seed %d / %s ===", k, len(plan), ds, seed, model)
        try:
            with Timer() as t:
                run_experiment(ds, model, seed, args.sample, args.overrides,
                               use_tuned=not args.no_tuned)
            n_ok += 1
            logger.info("=== done in %.1f min ===", t.seconds / 60)
        except Exception as exc:
            n_fail += 1
            logger.error("FAILED %s/%s/s%d: %s", ds, model, seed, exc)
            fail_log.parent.mkdir(parents=True, exist_ok=True)
            with open(fail_log, "a") as fh:
                fh.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {ds} {model} s{seed}\n"
                         f"{traceback.format_exc()}\n")
        finally:
            if model in ("sae", "lstm"):
                _clear_tf()
    logger.info("Finished: %d ok, %d failed, %d skipped (no TF), %d previously done",
                n_ok, n_fail, n_skip, done)


if __name__ == "__main__":
    main()
