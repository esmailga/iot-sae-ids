#!/usr/bin/env python
"""Build the result tables (CSV + Markdown) from results/raw/*.json into results/tables/.

Per dataset: detection, efficiency, feature reduction, SAE threshold variants, per-category
recall. Across datasets: summary, feature-reduction summary, significance, run coverage.
Works on a partial grid; every table shows the number of runs behind each cell.

Usage:
    python scripts/make_tables.py [--results-raw PATH]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ids.reporting import (
    DATASET_LABEL,
    DATASET_ORDER,
    coverage_table,
    detection_table,
    efficiency_table,
    features_summary,
    features_table,
    load_runs,
    per_category_frame,
    per_category_table,
    runs_frame,
    significance_md,
    significance_table,
    summary_table,
    threshold_frame,
    thresholds_table,
    write_table,
)
from ids.utils.config import load_config
from ids.utils.logging_utils import setup_logging

logger = logging.getLogger("make_tables")

TIMING_NOTE = ("Timing uses time.perf_counter. Inference: 64-flow batches, 10 warm-up + 100 timed; "
               "per-flow = batch time / 64. sklearn models are timed single-threaded (n_jobs=1). "
               "Values are mean ± std across seeds.")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-raw", default=None, help="default: cfg paths.results_raw")
    args = ap.parse_args()
    setup_logging("INFO")
    cfg = load_config(None)
    raw = Path(args.results_raw or cfg["paths"]["results_raw"])
    out = Path(cfg["paths"]["tables_dir"])
    runs = load_runs(raw)
    df = runs_frame(runs)
    if df.empty:
        raise SystemExit(f"No full-run result JSONs in {raw}")
    tf, pc = threshold_frame(runs), per_category_frame(runs)
    seeds = list(cfg["seeds"])

    cov = coverage_table(df, seeds)
    write_table(cov, out, "run_coverage", "Run coverage",
                [f"{len(df)} of {len(seeds) * 12} planned runs found."])
    write_table(summary_table(df), out, "summary", "Headline results (test set)",
                ["SAE operating point: Youden's J threshold chosen on the labelled validation split.",
                 "RF / LSTM: P(attack) > 0.5; SVM: decision function > 0.",
                 "CPU µs/flow: batch-64 inference time / 64 (single thread for sklearn)."])
    write_table(features_summary(df), out, "features_summary",
                "Feature reduction (range over seeds)")

    alpha = float(cfg["stats"]["alpha"])
    sig = significance_table(df, list(cfg["stats"]["metrics"]), alpha)
    write_table(significance_md(sig), out, "significance",
                "Significance: SAE vs each baseline",
                [f"Two-sided paired t-test and Wilcoxon signed-rank over (dataset, seed) pairs, "
                 f"α = {alpha}.",
                 "Δ > 0 means the SAE is higher. Effect sizes: Cohen's d_z (paired) and "
                 "matched-pairs rank-biserial r.",
                 "Per-dataset rows have ≤ 5 pairs: a two-sided Wilcoxon test cannot reach "
                 "p < 0.0625 with n = 5, so those rows are descriptive only.",
                 "'All datasets' pools only the (dataset, seed) pairs where both models ran; "
                 "check the Pairs column (e.g. a baseline missing on some datasets)."],
                csv_frame=sig)

    for ds in DATASET_ORDER:
        if (df.dataset == ds).sum() == 0:
            continue
        lab = DATASET_LABEL[ds]
        a, md = detection_table(df, ds)
        write_table(md, out, f"detection_{ds}", f"{lab} — detection (test set)",
                    ["Mean ± std across seeds.",
                     "SAE threshold: Youden's J on labelled validation (attack labels used for "
                     "threshold selection only)."], csv_frame=a)
        a, md = efficiency_table(df, ds)
        notes = [TIMING_NOTE,
                 "Params: Keras count_params (SAE, LSTM); total tree nodes (RF); "
                 "support vectors × d (exact SVM).",
                 "Peak RSS Δ: increase in process memory during inference (psutil).",
                 "Keras rows (SAE, LSTM) are eager `model(x)` calls, so a fixed per-call "
                 "framework overhead of roughly 10 ms per batch dominates small models. The SAE "
                 "TFLite rows (single-threaded interpreter, same batches) show the deployable "
                 "cost of the network itself."]
        if ((df.dataset == ds) & (df.model == "lstm") & (df.train_device == "CPU")).any():
            notes.append("LSTM training ran on CPU (the Colab GPU was unavailable), so its "
                         "training time is not comparable with the GPU-trained SAE.")
        write_table(md, out, f"efficiency_{ds}", f"{lab} — efficiency", notes, csv_frame=a)
        write_table(features_table(df, ds), out, f"features_{ds}",
                    f"{lab} — feature reduction per seed")
        a, md = thresholds_table(tf, ds)
        write_table(md, out, f"sae_thresholds_{ds}", f"{lab} — SAE at every threshold (test set)",
                    ["τ chosen on validation data; metrics on test. Percentile rows use only "
                     "normal validation errors (unsupervised)."], csv_frame=a)
        write_table(per_category_table(pc, ds), out, f"per_category_{ds}",
                    f"{lab} — recall per category (test set)",
                    ["Attack rows: detection rate. Normal rows: fraction kept as normal "
                     "(1 − FPR within that class).",
                     "Categories with very few test rows give noisy estimates."])
    logger.info("Tables written to %s", out)


if __name__ == "__main__":
    main()
