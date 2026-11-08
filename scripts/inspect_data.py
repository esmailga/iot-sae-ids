#!/usr/bin/env python
"""Inspect each raw dataset: shapes, columns, dtypes, missing %, label counts, planned drops.

Writes a Markdown report per dataset to ``results/tables/inspect_<dataset>.md`` and prints it.

Usage:
    python scripts/inspect_data.py [--dataset ton_iot unsw_nb15 rt_iot2022] [--set key=val ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ids.data.loaders import (
    NORMAL_LABEL,
    LoadedData,
    load_dataset,
    propose_rt_iot_normal_classes,
    raw_file_paths,
)
from ids.utils.config import DATASETS, load_config
from ids.utils.logging_utils import setup_logging

logger = logging.getLogger("inspect_data")

LENGTH_HINTS = ("dur", "byte", "len", "pkts", "pkt", "size", "duration")


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub Markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.astype(object).itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _label_table(y_bin: pd.Series, y_multi: pd.Series, title: str) -> str:
    vb = y_bin.value_counts().sort_index()
    tb = pd.DataFrame({"label": vb.index, "count": vb.values,
                       "pct": (100 * vb.values / vb.sum()).round(2)})
    vm = y_multi.value_counts()
    bin_of = pd.Series(y_bin.values, index=y_multi.values).groupby(level=0).first()
    tm = pd.DataFrame({"class": vm.index, "binary": bin_of.reindex(vm.index).values,
                       "count": vm.values, "pct": (100 * vm.values / vm.sum()).round(3)})
    return (f"#### {title} — binary\n\n{_md_table(tb)}\n\n"
            f"#### {title} — multiclass\n\n{_md_table(tm)}\n")


def _column_table(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        s = X[c]
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "missing_%": round(100 * s.isna().mean(), 3),
            "dash_%": 0.0 if is_num else round(100 * (s == "-").mean(), 3),
            "n_unique": int(s.nunique(dropna=True)),
            "negatives": int((s < 0).sum()) if is_num else "",
            "example": "" if is_num else str(s.dropna().astype(str).head(1).tolist()[:1])[:40],
        })
    return pd.DataFrame(rows)


def _impossible_values(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in X.select_dtypes(include=[np.number]).columns:
        lc = c.lower()
        if any(h in lc for h in LENGTH_HINTS):
            n = int((X[c] < 0).sum())
            if n:
                rows.append({"column": c, "rule": "negative length/duration", "rows": n})
        if "port" in lc or lc.endswith("_p"):
            n = int(((X[c] < 0) | (X[c] > 65535)).sum())
            if n:
                rows.append({"column": c, "rule": "port outside [0, 65535]", "rows": n})
    return pd.DataFrame(rows, columns=["column", "rule", "rows"])


def inspect(name: str, overrides: list[str]) -> str:
    """Build the Markdown inspection report for one dataset."""
    cfg = load_config(name, overrides)
    ds = cfg["dataset"]
    out = [f"## {name}\n"]
    try:
        data: LoadedData = load_dataset(cfg, drop=False, allow_unconfirmed=True)
    except FileNotFoundError as exc:
        out.append(f"**Not available:** {exc}\n")
        return "\n".join(out)
    finally:
        out.insert(1, "Files: " + ", ".join(
            f"`{p.name}` ({'found' if p.is_file() else 'MISSING'})" for p in raw_file_paths(cfg)
        ) + "\n")

    parts: list[tuple[str, pd.DataFrame, pd.Series, pd.Series]] = [
        ("all" if data.test is None else "official train", data.X, data.y_bin, data.y_multi)
    ]
    if data.test is not None:
        parts.append(("official test", *data.test))

    for title, X, yb, ym in parts:
        out.append(f"### {title}\n")
        out.append(f"- Rows: **{len(X):,}**; feature columns (excluding labels): **{X.shape[1]}**")
        n_num = X.select_dtypes(include=[np.number]).shape[1]
        out.append(f"- Numeric columns: {n_num}; categorical/object columns: {X.shape[1] - n_num}")
        dups = int(X.assign(__y=ym.values).duplicated().sum())
        out.append(f"- Exact duplicate rows (features + multiclass label): {dups:,} "
                   f"({100 * dups / len(X):.2f}%)")
        out.append(f"- Memory: {X.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")
        out.append(_label_table(yb, ym, title))

    X0 = parts[0][1]
    drop_cols = list(ds.get("drop_cols", []))
    present = [c for c in drop_cols if c in X0.columns]
    absent = [c for c in drop_cols if c not in X0.columns]
    out.append("### Planned column drops (leakage control)\n")
    out.append(f"- Present and dropped ({len(present)}): {present}")
    out.append(f"- Configured but not in file: {absent}")
    out.append(f"- Feature columns after drop: **{X0.shape[1] - len(present)}**\n")

    out.append("### Columns\n")
    out.append(_md_table(_column_table(X0)) + "\n")
    constant = [c for c in X0.columns if X0[c].nunique(dropna=False) <= 1]
    out.append(f"Constant columns (zero variance; dropped during preprocessing): {constant}\n")
    over5 = [c for c in X0.columns if X0[c].isna().mean() > cfg["preprocess"]["missing_drop_threshold"]
             and c not in present]
    out.append(f"Columns with > {100 * cfg['preprocess']['missing_drop_threshold']:.0f}% missing "
               f"(NaN, would be dropped by preprocessing): {over5}\n")

    imp = _impossible_values(X0)
    out.append("### Impossible values\n")
    out.append(_md_table(imp) if len(imp) else "None detected by the length/duration/port rules.")
    out.append("")

    if name == "rt_iot2022":
        proposed = propose_rt_iot_normal_classes(data.y_multi)
        attacks = sorted(set(data.y_multi.unique()) - set(proposed))
        vc = data.y_multi.value_counts()
        out.append("### Proposed RT-IoT2022 normal-vs-attack mapping (NEEDS CONFIRMATION)\n")
        out.append(f"- Normal (benign device traffic), {int(vc[proposed].sum()):,} rows: "
                   + ", ".join(f"`{c}` ({vc[c]:,})" for c in proposed))
        out.append(f"- Attack, {int(vc[attacks].sum()):,} rows: "
                   + ", ".join(f"`{c}` ({vc[c]:,})" for c in attacks))
        out.append(f"- Confirmed in config: {ds.get('label_mapping_confirmed', False)}\n")
    elif (data.y_multi == NORMAL_LABEL).sum() == 0:
        out.append("**Warning:** no rows with multiclass label 'normal'.\n")
    return "\n".join(out)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], help="key.sub=value")
    args = ap.parse_args()
    setup_logging("INFO")
    for name in args.dataset:
        report = inspect(name, args.overrides)
        cfg = load_config(name, args.overrides)
        out_dir = Path(cfg["paths"]["tables_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"inspect_{name}.md"
        path.write_text(report + "\n", encoding="utf-8")
        logger.info("Wrote %s", path)
        print(report)  # CLI report output
        print()


if __name__ == "__main__":
    main()
