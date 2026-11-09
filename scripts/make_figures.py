#!/usr/bin/env python
"""Build the result figures (300-dpi PNG) from results/raw into results/figures/.

* Bar charts (small multiples per dataset): F1, ROC-AUC, FPR, CPU latency per flow
  (with SAE TFLite), parameters, model size (with SAE TFLite).
* Per dataset: confusion matrices (summed over seeds), ROC + PR curves, SAE reconstruction-error
  histogram with τ, SAE/LSTM training curves (seed --seed).
* Feature selection (seed --seed): RF importance (from results/features/<tag>.json) and the
  Pearson heatmap (needs the cached prepared data or the raw dataset locally; skipped otherwise).

Usage:
    python scripts/make_figures.py [--seed 0] [--no-heatmap]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ids.figures import (
    COLOR,
    apply_style,
    confusion_matrices,
    error_histogram,
    loss_curves,
    metric_bars,
    pearson_heatmap,
    rf_importance_plot,
    roc_pr_curves,
)
from ids.reporting import DATASET_ORDER, load_runs, runs_frame
from ids.utils.config import load_config
from ids.utils.logging_utils import setup_logging

logger = logging.getLogger("make_figures")


def _heatmap(ds: str, seed: int, figdir: Path) -> None:
    """Pearson heatmap from the cached prepared data (re-fits locally if no cache)."""
    import json

    from ids.data.prepare import prepare

    cfg = load_config(ds)
    feats = Path(cfg["paths"]["features_dir"]) / f"{ds}__s{seed}.json"
    dropped = ([d["dropped"] for d in json.loads(feats.read_text())["pearson_dropped"]]
               if feats.is_file() else [])
    try:
        prep = prepare(cfg, seed, plots=False)
    except FileNotFoundError as exc:
        logger.warning("Skipping %s heatmap (no data): %s", ds, exc)
        return
    pearson_heatmap(prep.pearson.corr_, dropped, ds, seed,
                    cfg["features"]["pearson_threshold"],
                    figdir / "features" / f"{ds}__s{seed}_pearson_heatmap.png")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="seed for per-run figures")
    ap.add_argument("--results-raw", default=None)
    ap.add_argument("--no-heatmap", action="store_true")
    args = ap.parse_args()
    setup_logging("INFO")
    apply_style()
    cfg = load_config(None)
    raw = Path(args.results_raw or cfg["paths"]["results_raw"])
    figdir = Path(cfg["paths"]["figures_dir"])
    scores = raw / "scores"
    runs = load_runs(raw)
    df = runs_frame(runs)
    if df.empty:
        raise SystemExit(f"No full-run result JSONs in {raw}")

    sae = df[df.model == "sae"].groupby("dataset", observed=True)
    tfl_lat = {ds: [("SAE TFLite", g.tflite_f32_flow_us.mean(), COLOR["sae"])]
               for ds, g in sae if g.tflite_f32_flow_us.notna().any()}
    tfl_size = {ds: [("SAE TFLite", g.tflite_f32_mb.mean(), COLOR["sae"])]
                for ds, g in sae if g.tflite_f32_mb.notna().any()}
    metric_bars(df, "f1", "F1 (test)", figdir / "bars_f1.png", ylim=(0, 1.05))
    metric_bars(df, "roc_auc", "ROC-AUC (test)", figdir / "bars_roc_auc.png", ylim=(0, 1.05))
    metric_bars(df, "fpr", "False positive rate (test)", figdir / "bars_fpr.png",
                lower_better=True, ylim=(0, 1.0))
    metric_bars(df, "cpu_flow_us", "CPU latency per flow (µs, log)",
                figdir / "bars_latency_cpu.png", log=True, lower_better=True, extra=tfl_lat)
    metric_bars(df, "params", "Parameters (log)", figdir / "bars_params.png", log=True,
                lower_better=True)
    metric_bars(df, "size_mb", "Model size on disk (MB, log)", figdir / "bars_size.png",
                log=True, lower_better=True, extra=tfl_size)

    for ds in DATASET_ORDER:
        ds_runs = [j for j in runs if j["run"]["dataset"] == ds]
        if not ds_runs:
            continue
        confusion_matrices(runs, ds, figdir / f"confusion_{ds}.png")
        roc_pr_curves(scores, ds, args.seed, figdir / f"roc_pr_{ds}__s{args.seed}.png")
        loss_curves(runs, ds, args.seed, figdir / f"loss_{ds}__s{args.seed}.png")
        sae_run = next((j for j in ds_runs if j["run"]["model"] == "sae"
                        and j["run"]["seed"] == args.seed), None)
        if sae_run:
            error_histogram(scores, sae_run, figdir / f"sae_error_hist_{ds}__s{args.seed}.png")
        rf_importance_plot(Path(cfg["paths"]["features_dir"]) / f"{ds}__s{args.seed}.json",
                           figdir / "features" / f"{ds}__s{args.seed}_rf_importance.png",
                           cfg["features"]["rf_cumulative_importance"])
        if not args.no_heatmap:
            _heatmap(ds, args.seed, figdir)
    logger.info("Figures written to %s", figdir)


if __name__ == "__main__":
    main()
