"""SAE hyperparameter search (grid or random) on a stratified subsample of the train split.

* Search space: ``cfg["tune"]["space"]`` — hidden mode, latent dim (capped below d), L1 lambda,
  dropout, learning rate.
* Data: the same cached split/features as the experiments (seed ``seed``). The SAE trains on
  the normal rows of a stratified ``tune.subsample`` of train and early-stops on normal val.
* Criterion: validation ROC-AUC of reconstruction error (ties: lower val-normal MSE, then
  fewer parameters). Validation-normal MSE is logged too. The test split is never touched.
* Resumable: every finished trial is appended to ``results/tuning/<dataset>_trials.jsonl`` and
  skipped on rerun; ``results/best_sae_<dataset>.yaml`` is rewritten after each trial.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from ids.data.prepare import Prepared, prepare
from ids.data.splits import stratify_key
from ids.evaluation.metrics import threshold_free
from ids.models.sae import SAEDetector, capped_latent_dim, count_dense_params, hidden_widths
from ids.utils.seed import set_global_seed
from ids.utils.timing import Timer

logger = logging.getLogger(__name__)

PARAM_KEYS = ("hidden", "latent_dim", "l1", "dropout", "lr")


def candidate_grid(space: dict, d: int) -> list[dict]:
    """Full grid with latent dims capped below ``d``; duplicates after capping removed."""
    keys = list(PARAM_KEYS)
    seen: set[tuple] = set()
    out: list[dict] = []
    for combo in itertools.product(*(space[k] for k in keys)):
        p = dict(zip(keys, combo))
        p["latent_dim"] = capped_latent_dim(d, p["latent_dim"])
        p["l1"], p["dropout"], p["lr"] = float(p["l1"]), float(p["dropout"]), float(p["lr"])
        key = trial_key(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def trial_key(p: dict) -> tuple:
    """Hashable identity of a parameter set."""
    return tuple((k, p[k]) for k in PARAM_KEYS)


def select_candidates(space: dict, d: int, mode: str, n_trials: int, seed: int) -> list[dict]:
    """``grid``: every candidate; ``random``: ``n_trials`` distinct candidates (seeded)."""
    grid = candidate_grid(space, d)
    if mode == "grid":
        return grid
    if mode != "random":
        raise ValueError(f"tune.mode must be 'grid' or 'random', got {mode!r}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(grid))[: min(n_trials, len(grid))]
    return [grid[i] for i in idx]


def tuning_subsample(prep: Prepared, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified ``fraction`` of the train split (by multiclass label)."""
    X, y, m = prep.X["train"], prep.y["train"], prep.m["train"]
    if fraction >= 1:
        return X, y
    idx, _ = train_test_split(np.arange(len(X)), train_size=fraction, random_state=seed,
                              stratify=stratify_key(pd.Series(y), pd.Series(m)))
    idx = np.sort(idx)
    return X[idx], y[idx]


def architecture_options(cfg: dict, d: int, latent_dims: Optional[list[int]] = None) -> pd.DataFrame:
    """Parameter count of the scaled vs fixed architectures for each latent dim (spec §7)."""
    s = cfg["sae"]
    rows = []
    for mode in ("scaled", "fixed"):
        h = hidden_widths(d, mode, s["scaled_ratios"], s["fixed_hidden"])
        for z in latent_dims or cfg["tune"]["space"]["latent_dim"]:
            zc = capped_latent_dim(d, z)
            rows.append({"hidden_mode": mode, "d": d, "hidden": "→".join(map(str, h)),
                         "latent_dim": zc, "n_params": count_dense_params(d, h, zc)})
    return pd.DataFrame(rows).drop_duplicates()


def _load_trials(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _best(trials: list[dict]) -> dict:
    ok = [t for t in trials if t.get("val_roc_auc") is not None]
    if not ok:
        raise RuntimeError("No successful tuning trials")
    return sorted(ok, key=lambda t: (-t["val_roc_auc"], t["best_val_normal_mse"],
                                     t["n_params"]))[0]


def run_tuning(cfg: dict, seed: int = 0, mode: Optional[str] = None,
               n_trials: Optional[int] = None, subsample: Optional[float] = None,
               sample: Optional[float] = None, verbose: int = 0) -> dict:
    """Run (or resume) the search for ``cfg["dataset"]["name"]``; return the best trial."""
    tc = cfg["tune"]
    mode = mode or tc["mode"]
    n_trials = int(n_trials or tc["n_trials"])
    subsample = float(subsample if subsample is not None else tc["subsample"])
    name = cfg["dataset"]["name"]

    prep = prepare(cfg, seed, sample, plots=sample is None)
    d = prep.d
    X_sub, y_sub = tuning_subsample(prep, subsample, seed)
    Xva, yva = prep.X["val"], prep.y["val"]
    candidates = select_candidates(tc["space"], d, mode, n_trials, seed)

    tag = name + (f"__frac{sample:g}" if sample else "")
    tdir = Path(cfg["paths"]["tuning_dir"])
    tdir.mkdir(parents=True, exist_ok=True)
    trials_path = tdir / f"{tag}_trials.jsonl"
    done = {trial_key(t["params"]) for t in _load_trials(trials_path)}
    logger.info("[tune %s] d=%d, mode=%s, %d candidates (%d already done), train subsample "
                "%d rows (%d normal)", tag, d, mode, len(candidates),
                sum(trial_key(c) in done for c in candidates), len(X_sub), int((y_sub == 0).sum()))

    for i, params in enumerate(candidates, 1):
        if trial_key(params) in done:
            continue
        set_global_seed(seed)
        det = SAEDetector(cfg, seed, params={**params, "epochs": tc["epochs"]}, verbose=verbose)
        rec: dict[str, Any] = {"params": params, "seed": seed, "d": d}
        try:
            with Timer() as t:
                det.fit(X_sub, y_sub, Xva, yva)
            auc = threshold_free(yva, det.score(Xva))
            rec.update(val_roc_auc=auc["roc_auc"], val_pr_auc=auc["pr_auc"],
                       best_val_normal_mse=det.info["best_val_normal_mse"],
                       epochs_run=det.info["epochs_run"], n_params=det.n_params(),
                       hidden_widths=det.info["hidden"], train_time_s=t.seconds)
        except Exception as exc:  # keep searching; record the failure
            logger.exception("[tune %s] trial %s failed", tag, params)
            rec.update(error=f"{type(exc).__name__}: {exc}", val_roc_auc=None)
        finally:
            _clear_tf_session()
        with open(trials_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        logger.info("[tune %s] %d/%d %s -> val ROC-AUC=%s", tag, i, len(candidates), params,
                    rec.get("val_roc_auc"))
        _write_best(cfg, tag, _load_trials(trials_path), mode, subsample, seed, d)

    trials = _load_trials(trials_path)
    best = _write_best(cfg, tag, trials, mode, subsample, seed, d)
    _write_table(cfg, tag, trials, d)
    return best


def _write_best(cfg: dict, tag: str, trials: list[dict], mode: str, subsample: float,
                seed: int, d: int) -> dict:
    best = _best(trials)
    payload = {
        "dataset": cfg["dataset"]["name"],
        "selected_by": "max validation ROC-AUC of reconstruction error "
                       "(ties: lower val-normal MSE, fewer params)",
        "params": best["params"],
        "val_roc_auc": best["val_roc_auc"], "val_pr_auc": best["val_pr_auc"],
        "best_val_normal_mse": best["best_val_normal_mse"], "n_params": best["n_params"],
        "hidden_widths": best["hidden_widths"], "d": d,
        "search": {"mode": mode, "n_trials_done": len(trials), "subsample": subsample,
                   "seed": seed},
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    path = Path(cfg["paths"]["results_root"]) / f"best_sae_{tag}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return payload


def _write_table(cfg: dict, tag: str, trials: list[dict], d: int) -> None:
    rows = [{**t["params"], **{k: t.get(k) for k in ("val_roc_auc", "val_pr_auc",
                                                       "best_val_normal_mse", "n_params",
                                                       "epochs_run", "train_time_s", "error")}}
            for t in trials]
    df = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False, na_position="last")
    out = Path(cfg["paths"]["tables_dir"])
    out.mkdir(parents=True, exist_ok=True)
    arch = architecture_options(cfg, d)
    (out / f"sae_tuning_{tag}.md").write_text(
        f"# SAE tuning — {tag} (d = {d})\n\n## Trials (best first)\n\n"
        + df.round(5).to_markdown(index=False)
        + "\n\n## Architecture options: parameter counts\n\n"
        + arch.to_markdown(index=False) + "\n"
    )
    df.to_csv(out / f"sae_tuning_{tag}.csv", index=False)


def _clear_tf_session() -> None:
    try:
        from tensorflow import keras

        keras.backend.clear_session()
    except Exception:
        pass
