"""Collect per-run result JSONs into DataFrames and build the result tables.

Works on a partial grid: missing runs simply reduce ``n`` (shown in every table) and the
coverage table lists what is absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from ids.evaluation.stats import compare_to_reference

logger = logging.getLogger(__name__)

MODEL_ORDER = ["sae", "rf", "svm", "lstm"]
MODEL_LABEL = {"sae": "SAE", "rf": "RF", "svm": "SVM", "lstm": "LSTM"}
DATASET_LABEL = {"ton_iot": "TON_IoT", "unsw_nb15": "UNSW-NB15", "rt_iot2022": "RT-IoT2022"}
DATASET_ORDER = list(DATASET_LABEL)
DET_METRICS = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]
DET_LABEL = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
             "fpr": "FPR", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
THRESHOLD_ORDER = ["val_youden", "val_f1", "p90", "p95", "p97.5", "p99", "p99.5", "p99.9"]
THRESHOLD_LABEL = {"val_youden": "Val-optimised: Youden's J (headline)",
                   "val_f1": "Val-optimised: F1",
                   **{f"p{p}": f"Percentile p{p} of normal val error"
                      for p in ("90", "95", "97.5", "99", "99.5", "99.9")}}


def _get(d: dict, *path: str, default: Any = None) -> Any:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def load_runs(results_raw: Path, include_samples: bool = False) -> list[dict]:
    """Load every run JSON (full runs only unless ``include_samples``)."""
    runs = []
    for p in sorted(Path(results_raw).glob("*.json")):
        if "__frac" in p.name and not include_samples:
            continue
        try:
            runs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable %s", p)
    logger.info("Loaded %d runs from %s", len(runs), results_raw)
    return runs


def runs_frame(runs: Iterable[dict]) -> pd.DataFrame:
    """One row per run: detection metrics, efficiency, model and data facts."""
    rows = []
    for j in runs:
        r, e, m = j["run"], j["efficiency"], j["metrics"]["test"]
        is_tf = r["model"] in ("sae", "lstm")
        row = {
            "dataset": r["dataset"], "model": r["model"], "seed": r["seed"],
            **{k: m.get(k) for k in DET_METRICS},
            "train_s": e["training_time_s"],
            "train_device": ("GPU" if _get(j, "env", "hardware", "gpus") else "CPU")
            if is_tf else "CPU",
            "cpu_batch_ms": 1e3 * _get(e, "cpu", "batch_mean_s", default=np.nan),
            "cpu_batch_std_ms": 1e3 * _get(e, "cpu", "batch_std_s", default=np.nan),
            "cpu_flow_us": 1e3 * _get(e, "cpu", "per_flow_mean_ms", default=np.nan),
            "gpu_flow_us": 1e3 * _get(e, "gpu", "per_flow_mean_ms", default=np.nan),
            "params": e["n_params"], "size_mb": e["model_size_mb"],
            "rss_peak_mb": _get(e, "memory", "rss_peak_delta_mb", default=np.nan),
            "tracemalloc_mb": _get(e, "memory", "tracemalloc_peak_mb", default=np.nan),
            "tflite_f32_mb": _get(e, "tflite", "float32", "size_mb", default=np.nan),
            "tflite_f32_flow_us": 1e3 * _get(e, "tflite", "float32", "cpu", "per_flow_mean_ms",
                                             default=np.nan),
            "tflite_int8_mb": _get(e, "tflite", "dynamic_int8", "size_mb", default=np.nan),
            "tflite_int8_flow_us": 1e3 * _get(e, "tflite", "dynamic_int8", "cpu",
                                              "per_flow_mean_ms", default=np.nan),
            "variant": _get(j, "model", "variant") or _get(j, "model", "hidden_mode"),
            "latent_dim": _get(j, "model", "latent_dim"),
            "hidden": _get(j, "model", "hidden"),
            "d": j["data"]["d"],
            "epochs_run": _get(j, "model", "epochs_run"),
            **{f"count_{k}": v for k, v in j["data"]["counts"].items()},
            "n_train": _get(j, "data", "split_sizes", "train"),
            "n_test": _get(j, "data", "split_sizes", "test"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    return df.sort_values(["dataset", "model", "seed"]).reset_index(drop=True)


def threshold_frame(runs: Iterable[dict]) -> pd.DataFrame:
    """SAE only: test metrics at every threshold variant, one row per (run, variant)."""
    rows = []
    for j in runs:
        if j["run"]["model"] != "sae":
            continue
        th = j["metrics"]["thresholds"]
        variants = {k: th[k] for k in ("val_youden", "val_f1") if k in th}
        variants.update(th["percentile"])
        for name, v in variants.items():
            rows.append({"dataset": j["run"]["dataset"], "seed": j["run"]["seed"],
                         "threshold": name, "tau": v["tau"],
                         **{k: v["test"].get(k) for k in DET_METRICS if k in v["test"]}})
    return pd.DataFrame(rows)


def per_category_frame(runs: Iterable[dict]) -> pd.DataFrame:
    """Per-category test recall, one row per (run, category)."""
    rows = []
    for j in runs:
        for cat, v in j["metrics"]["per_category_test"].items():
            rows.append({"dataset": j["run"]["dataset"], "model": j["run"]["model"],
                         "seed": j["run"]["seed"], "category": cat, "n": v["n"],
                         "recall": v["recall"], "normal": v["normal"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- formatting

def pm(mean: float, std: float, digits: int = 4) -> str:
    """``mean ± std`` (std omitted when undefined, '—' when mean is missing)."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    if std is None or np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _agg(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out["n"] = g.size()
    return out.reset_index()


def coverage_table(df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Runs found per dataset x model, with missing seeds listed."""
    rows = []
    for ds in DATASET_ORDER:
        for mdl in MODEL_ORDER:
            have = sorted(df[(df.dataset == ds) & (df.model == mdl)].seed.tolist())
            rows.append({"Dataset": DATASET_LABEL[ds], "Model": MODEL_LABEL[mdl],
                         "Runs": f"{len(have)}/{len(seeds)}",
                         "Missing seeds": ", ".join(map(str, sorted(set(seeds) - set(have))))
                         or "—"})
    return pd.DataFrame(rows)


def detection_table(df: pd.DataFrame, ds: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(numeric CSV frame, formatted Markdown frame) of detection metrics for one dataset."""
    sub = df[df.dataset == ds]
    a = _agg(sub, ["model"], DET_METRICS)
    md = pd.DataFrame({"Model": a.model.map(MODEL_LABEL), "Runs": a.n})
    for k in DET_METRICS:
        md[DET_LABEL[k]] = [pm(m, s) for m, s in zip(a[f"{k}_mean"], a[f"{k}_std"])]
    return a, md


EFF_COLS = ["train_s", "cpu_batch_ms", "cpu_flow_us", "gpu_flow_us", "params", "size_mb",
            "rss_peak_mb", "tracemalloc_mb"]


def efficiency_table(df: pd.DataFrame, ds: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Efficiency for one dataset, plus SAE TFLite rows. Mean ± std across seeds."""
    sub = df[df.dataset == ds]
    a = _agg(sub, ["model"], EFF_COLS)
    dev = sub.groupby("model", observed=True)["train_device"].agg(
        lambda s: "/".join(sorted(set(s)))).reindex(a.model).to_numpy()
    md = pd.DataFrame({
        "Model": a.model.map(MODEL_LABEL), "Runs": a.n,
        "Train time (s)": [pm(m, s, 1) for m, s in zip(a.train_s_mean, a.train_s_std)],
        "Trained on": dev,
        "CPU batch-64 (ms)": [pm(m, s, 3) for m, s in zip(a.cpu_batch_ms_mean, a.cpu_batch_ms_std)],
        "CPU per flow (µs)": [pm(m, s, 2) for m, s in zip(a.cpu_flow_us_mean, a.cpu_flow_us_std)],
        "GPU per flow (µs)": [pm(m, s, 2) for m, s in zip(a.gpu_flow_us_mean, a.gpu_flow_us_std)],
        "Params": [f"{m:,.0f}" for m in a.params_mean],
        "Size (MB)": [f"{m:.4f}" for m in a.size_mb_mean],
        "Peak RSS Δ (MB)": [pm(m, s, 2) for m, s in zip(a.rss_peak_mb_mean, a.rss_peak_mb_std)],
    })
    sae = sub[sub.model == "sae"]
    extra = []
    for tag, size_col, lat_col in (("SAE · TFLite float32", "tflite_f32_mb", "tflite_f32_flow_us"),
                                   ("SAE · TFLite dynamic int8", "tflite_int8_mb",
                                    "tflite_int8_flow_us")):
        if len(sae) and sae[size_col].notna().any():
            extra.append({"Model": tag, "Runs": int(sae[size_col].notna().sum()),
                          "Train time (s)": "—", "Trained on": "—", "CPU batch-64 (ms)":
                          pm(sae[lat_col].mean() * 64 / 1e3, sae[lat_col].std() * 64 / 1e3, 3),
                          "CPU per flow (µs)": pm(sae[lat_col].mean(), sae[lat_col].std(), 2),
                          "GPU per flow (µs)": "—", "Params": f"{sae.params.mean():,.0f}",
                          "Size (MB)": f"{sae[size_col].mean():.4f}", "Peak RSS Δ (MB)": "—"})
    if extra:
        md = pd.concat([md, pd.DataFrame(extra)], ignore_index=True)
    return a, md


FEAT_COLS = ["count_raw", "count_after_leakage_drop", "count_after_preprocess",
             "count_after_stage1_pearson", "count_after_stage2_rf"]
FEAT_LABEL = ["Raw", "After leakage drop", "After encoding", "After Stage 1 (Pearson)",
              "After Stage 2 (RF)"]


def features_table(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Feature counts per seed (from the SAE run, else any run) + SAE latent dimension."""
    sub = df[df.dataset == ds].sort_values("model")
    per_seed = sub.drop_duplicates("seed").sort_values("seed")
    sae = sub[sub.model == "sae"].set_index("seed")
    out = per_seed[["seed"] + FEAT_COLS].copy()
    out["SAE hidden → latent"] = [
        ("→".join(map(str, sae.loc[s, "hidden"])) + f" → {int(sae.loc[s, 'latent_dim'])}")
        if s in sae.index else "—" for s in out.seed]
    out.columns = ["Seed"] + FEAT_LABEL + ["SAE hidden → latent"]
    return out.reset_index(drop=True)


def features_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: feature counts (min–max over seeds) and SAE latent dim."""
    rows = []
    for ds in DATASET_ORDER:
        sub = df[df.dataset == ds].drop_duplicates("seed")
        if sub.empty:
            continue
        rng = lambda c: (f"{int(sub[c].min())}" if sub[c].min() == sub[c].max()  # noqa: E731
                         else f"{int(sub[c].min())}–{int(sub[c].max())}")
        sae = df[(df.dataset == ds) & (df.model == "sae")]
        lat = sorted(set(sae.latent_dim.dropna().astype(int)))
        rows.append({"Dataset": DATASET_LABEL[ds], **{lab: rng(c) for c, lab in
                                                      zip(FEAT_COLS, FEAT_LABEL)},
                     "SAE latent": "/".join(map(str, lat)) or "—"})
    return pd.DataFrame(rows)


def thresholds_table(tf: pd.DataFrame, ds: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SAE test metrics at every threshold variant for one dataset."""
    cols = ["accuracy", "precision", "recall", "f1", "fpr"]
    if tf.empty or (tf.dataset == ds).sum() == 0:
        return pd.DataFrame(), pd.DataFrame()
    sub = tf[tf.dataset == ds]
    a = _agg(sub, ["threshold"], cols + ["tau"])
    a["order"] = a.threshold.map({k: i for i, k in enumerate(THRESHOLD_ORDER)})
    a = a.sort_values("order").drop(columns="order")
    md = pd.DataFrame({"Threshold": a.threshold.map(lambda t: THRESHOLD_LABEL.get(t, t)),
                       "Runs": a.n})
    for k in cols:
        md[DET_LABEL[k]] = [pm(m, s) for m, s in zip(a[f"{k}_mean"], a[f"{k}_std"])]
    md["τ"] = [f"{m:.3g}" for m in a.tau_mean]
    return a, md


def per_category_table(pc: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Recall per category (rows) × model (columns), mean ± std across seeds."""
    if pc.empty or (pc.dataset == ds).sum() == 0:
        return pd.DataFrame()
    sub = pc[pc.dataset == ds]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["category", "model"], observed=True)
    a = g.recall.agg(["mean", "std"]).reset_index()
    n = sub.groupby("category").n.mean().round().astype(int)
    normal = sub.groupby("category").normal.first()
    out = pd.DataFrame({"Category": n.index,
                        "Type": ["normal (1 − FPR)" if normal[c] else "attack" for c in n.index],
                        "Test rows": n.values})
    for mdl in MODEL_ORDER:
        s = a[a.model == mdl].set_index("category")
        out[MODEL_LABEL[mdl]] = [pm(s.loc[c, "mean"], s.loc[c, "std"], 3) if c in s.index
                                 else "—" for c in n.index]
    return out.sort_values(["Type", "Test rows"], ascending=[True, False]).reset_index(drop=True)


def significance_table(df: pd.DataFrame, metrics: list[str], alpha: float) -> pd.DataFrame:
    """SAE vs each baseline: paired t-test + Wilcoxon over (dataset, seed) pairs."""
    res = compare_to_reference(df.assign(model=df.model.astype(str),
                                         dataset=df.dataset.astype(str)),
                               metrics, reference="sae", alpha=alpha, per_dataset=True)
    if res.empty:
        return res
    res["scope"] = res.scope.map(lambda s: "All datasets" if s == "all"
                                 else DATASET_LABEL.get(s, s))
    res["baseline"] = res.baseline.map(MODEL_LABEL)
    res["metric"] = res.metric.map(DET_LABEL)
    return res


def significance_md(res: pd.DataFrame) -> pd.DataFrame:
    """Readable significance table."""
    if res.empty:
        return pd.DataFrame()
    def p(x: float) -> str:
        return "—" if pd.isna(x) else ("<0.001" if x < 1e-3 else f"{x:.3f}")
    return pd.DataFrame({
        "Scope": res.scope, "SAE vs": res.baseline, "Metric": res.metric, "Pairs": res.n_pairs,
        "SAE mean": res.mean_a.map(lambda v: f"{v:.4f}"),
        "Baseline mean": res.mean_b.map(lambda v: f"{v:.4f}"),
        "Δ (SAE − base)": res.mean_diff.map(lambda v: f"{v:+.4f}"),
        "t-test p": res.t_p.map(p), "Wilcoxon p": res.w_p.map(p),
        "Cohen's d_z": res.cohens_dz.map(lambda v: "—" if pd.isna(v) else f"{v:+.2f}"),
        "Rank-biserial r": res.rank_biserial.map(lambda v: f"{v:+.2f}"),
        "Significant (both tests)": [("yes" if (t and w) else "no")
                                     for t, w in zip(res.t_significant, res.w_significant)],
    })


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compact headline table: dataset × model → F1, Recall, FPR, ROC-AUC, CPU µs/flow, params."""
    a = _agg(df, ["dataset", "model"], ["f1", "recall", "fpr", "roc_auc", "cpu_flow_us",
                                        "params", "size_mb"])
    return pd.DataFrame({
        "Dataset": a.dataset.map(DATASET_LABEL), "Model": a.model.map(MODEL_LABEL), "Runs": a.n,
        "F1": [pm(m, s) for m, s in zip(a.f1_mean, a.f1_std)],
        "Recall": [pm(m, s) for m, s in zip(a.recall_mean, a.recall_std)],
        "FPR": [pm(m, s) for m, s in zip(a.fpr_mean, a.fpr_std)],
        "ROC-AUC": [pm(m, s) for m, s in zip(a.roc_auc_mean, a.roc_auc_std)],
        "CPU µs/flow": [f"{m:.2f}" for m in a.cpu_flow_us_mean],
        "Params": [f"{m:,.0f}" for m in a.params_mean],
        "Size (MB)": [f"{m:.3f}" for m in a.size_mb_mean],
    })


def write_table(frame: pd.DataFrame, out_dir: Path, stem: str, title: str,
                notes: Optional[list[str]] = None, csv_frame: Optional[pd.DataFrame] = None) -> None:
    """Write ``<stem>.md`` (title, notes, table) and ``<stem>.csv`` (numeric if given)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    body = f"# {title}\n\n"
    if notes:
        body += "\n".join(f"- {n}" for n in notes) + "\n\n"
    body += (frame.to_markdown(index=False) if len(frame) else "_No data._") + "\n"
    (out_dir / f"{stem}.md").write_text(body)
    (csv_frame if csv_frame is not None else frame).to_csv(out_dir / f"{stem}.csv", index=False)
