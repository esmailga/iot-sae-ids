"""Tables and figures from synthetic runs (partial grid: no TF models locally)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ids.experiment import run_experiment
from ids.figures import apply_style, confusion_matrices, metric_bars, roc_pr_curves
from ids.reporting import (
    coverage_table,
    detection_table,
    efficiency_table,
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
from tests.test_pipeline_smoke import FAST


def test_tables_and_figures_on_partial_grid(raw_root: Path, tmp_path: Path) -> None:
    for seed in (0, 1):
        for model in ("rf", "svm"):
            run_experiment("ton_iot", model, seed, overrides=FAST)
    cfg = load_config("ton_iot")
    raw = Path(cfg["paths"]["results_raw"])
    runs = load_runs(raw)
    df = runs_frame(runs)
    assert len(df) == 4 and set(df.model.astype(str)) == {"rf", "svm"}

    cov = coverage_table(df, [0, 1])
    assert (cov[(cov.Model == "SAE")]["Runs"] == "0/2").all()
    a, md = detection_table(df, "ton_iot")
    assert list(md.Model) == ["RF", "SVM"] and (md.Runs == 2).all()
    assert "±" in md.loc[0, "F1"]
    _, eff = efficiency_table(df, "ton_iot")
    assert {"CPU per flow (µs)", "Params", "Size (MB)"} <= set(eff.columns)
    assert len(features_table(df, "ton_iot")) == 2
    assert len(per_category_table(per_category_frame(runs), "ton_iot")) > 0
    t_a, t_md = thresholds_table(threshold_frame(runs), "ton_iot")      # no SAE -> empty
    assert t_md.empty
    assert significance_md(significance_table(df, ["f1"], 0.05)).empty  # no SAE -> empty
    write_table(summary_table(df), tmp_path, "summary", "Summary")
    assert (tmp_path / "summary.md").read_text().startswith("# Summary")
    assert pd.read_csv(tmp_path / "summary.csv").shape[0] == 2

    apply_style()
    assert metric_bars(df, "f1", "F1", tmp_path / "f1.png").stat().st_size > 0
    assert confusion_matrices(runs, "ton_iot", tmp_path / "cm.png").stat().st_size > 0
    assert roc_pr_curves(raw / "scores", "ton_iot", 0, tmp_path / "roc.png").stat().st_size > 0
