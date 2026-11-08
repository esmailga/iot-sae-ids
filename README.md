# iot-sae-ids

Experimental code for **"A Feature-Optimised Lightweight Intrusion Detection System for IoT Using Autoencoder"**.

A sparse autoencoder (SAE) is trained **only on normal traffic**. A flow is flagged as an attack when its reconstruction error exceeds a threshold τ. Before the SAE, features are reduced in two stages:

1. a Pearson correlation filter;
2. Random Forest importance ranking.

The SAE encoder then acts as a third, non-linear compression stage. The SAE is compared against Random Forest, SVM and LSTM baselines on TON_IoT, UNSW-NB15 and RT-IoT2022. Each model is scored on detection quality and on efficiency, across 5 seeds, with significance tests.

> Status: **Phase 6 — Colab runner, SAE tuning and the full grid.** The pipeline is tagged `v0.1-pipeline`. Sections marked _TODO_ are filled in during later phases.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/` | `default.yaml` holds the shared settings. Each dataset has its own YAML. |
| `src/ids/` | Installable package `ids`: data, features, models, detection, evaluation and utils. |
| `scripts/` | Command-line entry points: inspect, run, run-all, tune, tables and figures. |
| `notebooks/colab_runner.ipynb` | Colab Pro runner. It clones the repo, installs it, mounts Drive and runs the scripts. |
| `tests/` | pytest suite on small synthetic data. |
| `data/` | Git-ignored. Holds `raw/<dataset>/` and `processed/`. |
| `results/` | `tables/` and `figures/` are committed. `raw/` (per-run JSON) and `models/` are git-ignored. |

## Setup

### Local (CPU, no TensorFlow)

The development Mac is Intel x86_64, and TensorFlow publishes no Intel-macOS wheels after 2.16.2. Locally we therefore install only the scikit-learn stack:

- RF and SVM run locally.
- The SAE and LSTM run on Colab.
- TensorFlow tests are skipped automatically when TF is not installed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

### With TensorFlow (Colab, Linux, Apple Silicon)

```bash
pip install -r requirements-tf.txt
pip install -e .
```

### Paths

All paths come from `configs/default.yaml`. They are relative to the repo root unless one of these environment variables is set:

- `IDS_DATA_ROOT`: the folder that contains `raw/` and `processed/`;
- `IDS_RESULTS_ROOT`: the folder for results.

On Colab, set both to a Google Drive folder so data and results survive a disconnect.

## Data

Put the raw files under `data/raw/<dataset>/`:

| Dataset | Files | Source |
|---|---|---|
| TON_IoT | `data/raw/ton_iot/Train_Test_Network.csv` | UNSW Canberra TON_IoT page: <https://research.unsw.edu.au/projects/toniot-datasets>. Use the "Train_Test_datasets → Train_Test_Network_dataset" folder. |
| UNSW-NB15 | `data/raw/unsw_nb15/UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` | <https://research.unsw.edu.au/projects/unsw-nb15-dataset>. Use the "Training and Testing Sets" folder. |
| RT-IoT2022 | Downloaded automatically from UCI (id 942) via `ucimlrepo`. Offline fallback: `data/raw/rt_iot2022/RT_IOT2022.csv` | <https://archive.ics.uci.edu/dataset/942/rt-iot2022> |

Run `python scripts/inspect_data.py` to regenerate the per-dataset reports in `results/tables/inspect_<dataset>.md`.

### Actual sizes vs. the proposal

| Dataset | Proposal | Actual (file used here) | Note |
|---|---|---|---|
| TON_IoT | "~4M records" | **211,043 rows, 42 features** (+ `label`, `type`) → 33 after leakage drops | The multi-million figures refer to the full raw network captures, not the curated train/test file used here. This release has no `ts` column. 50,000 normal rows; 20,000 per attack type except `mitm` (1,043). 9.75% exact duplicates. The file may be named in lower case; the loader matches names case-insensitively. |
| UNSW-NB15 | — | **175,341 official train / 82,332 official test, 43 features** (+ `label`, `attack_cat`) → 42 after dropping `id` | Normal share is 31.9% in train vs 44.9% in test. `proto` has 133 categories, so it is one-hot encoded as the top 10 + `other`. |
| RT-IoT2022 | "~3.5M records" | **123,117 rows, 83 features** (+ `Attack_type`) | UCI id 942. 3 benign device classes (MQTT_Publish, Thing_Speak, Wipro_bulb), 9 attack classes. No Amazon-Alexa class is present in the UCI release. 89.8% of rows are attacks, 76.9% are DOS_SYN_Hping. |

### Dropped columns (leakage control)

| Dataset | Dropped | Reason |
|---|---|---|
| TON_IoT | `ts` | Timestamp (configured; absent from the current release) |
| TON_IoT | `src_ip`, `dst_ip` | Host identifiers |
| TON_IoT | `src_port`, `dst_port` | Flow identifiers. Ephemeral source ports in particular are near-unique per flow. |
| TON_IoT | `dns_query`, `ssl_subject`, `ssl_issuer`, `http_uri`, `http_user_agent` | Free-text, identifier-like fields: domain names, certificate subjects, URIs, user agents. _Proposed; awaiting confirmation._ |
| UNSW-NB15 | `id` | Row identifier |
| RT-IoT2022 | `id.orig_p`, `id.resp_p` | Source and destination ports |

Preprocessing also drops these columns, based on the train split only:

- columns with more than 5% missing values;
- constant (zero-variance) columns, e.g. RT-IoT2022's `bwd_URG_flag_count`.

For TON_IoT, `-` means "not applicable". It is kept as a category in categorical columns. In otherwise-numeric columns it becomes missing, so `http_trans_depth`, `http_version` and `weird_addl` (each over 99.8% missing) are dropped by the 5% rule.

The preprocessing logs list them.

### RT-IoT2022 label mapping

Binary label: rows whose `Attack_type` is a benign device class get 0, and all other rows get 1. The mapping lives in `configs/rt_iot2022.yaml` (`normal_classes`). Loading raises an error until `label_mapping_confirmed: true` is set.

### Duplicates

Exact duplicates are removed **from the train split only**: `preprocess.dedup_scope: train`, as in the original spec. Val and test keep theirs. Once identifiers and ports are dropped, many flows become identical. This happens especially with volumetric attacks, so the train-only step removes a lot of rows (seed 0):

| Dataset | Train rows before → after dedup | Notes |
|---|---|---|
| TON_IoT | 147,729 → 67,355 | |
| UNSW-NB15 | 153,423 → 95,630 | |
| RT-IoT2022 | 86,181 → 12,990 | `DOS_SYN_Hping` (77% of the raw data) collapses to a few dozen unique training flows, but keeps about 19k rows in test. |

Because val and test keep their duplicates, many test rows have exact copies in train. This tends to inflate scores, particularly for supervised models that can memorise. For a stricter evaluation, set `dedup_scope: dataset` to deduplicate before splitting, or `none` to keep everything.

## Feature reduction (seed 0)

Generated by `python scripts/prepare_features.py --seed 0`. Per-seed lists are in `results/features/`, and the plots are in `results/figures/features/`.

| Dataset | Raw columns | After leakage drop | After preprocessing (encoded) | After Stage 1 (Pearson) | After Stage 2 (RF) |
|---|---|---|---|---|---|
| TON_IoT | 42 | 33 | 72 | 67 | 19 |
| UNSW-NB15 | 43 | 42 | 70 | 62 | 27 |
| RT-IoT2022 | 83 | 81 | 89 | 63 | 31 |

## Reproducing the results

### 1. Tests and smoke test

```bash
pytest                                       # includes a <60 s end-to-end synthetic run
python scripts/inspect_data.py               # data reports -> results/tables/inspect_*.md
python scripts/prepare_features.py --seed 0  # feature counts, lists and plots
python scripts/smoke_test.py --sample 0.02   # all dataset x model combinations on a 2% sample
python scripts/run_experiment.py --dataset ton_iot --model rf --seed 0 --sample 0.02
```

Each run writes the following:
- `results/raw/<dataset>__<model>__s<seed>[__frac<f>].json`: config, metrics, thresholds, timings, efficiency, library versions and hardware;
- `results/raw/scores/<run>.npz`: test and validation scores for the ROC curves and histograms;
- the model file in `results/models/<run>/`.

A run is skipped when its JSON already exists; pass `--force` to rerun it.

### 2. On Google Colab Pro (T4)

1. Upload the raw data to Google Drive at `MyDrive/iot-sae-ids-work/data/raw/<dataset>/`, using the layout from the [Data](#data) section. RT-IoT2022 is optional, because Colab can fetch it from UCI.
2. Open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) in Colab (File → Open notebook → GitHub → `esmailga/iot-sae-ids`).
3. Choose a **T4 GPU** runtime, preferably High-RAM.
4. Run sections 0–2. They mount Drive, clone or update the code from GitHub, install it, and point `IDS_DATA_ROOT` and `IDS_RESULTS_ROOT` at Drive.

Everything the runs produce is stored on Drive, so a disconnect loses nothing. After one, rerun sections 0–2 and then the step you were on. Finished tuning trials and finished runs are skipped.

The notebook can also load the code from a git bundle instead of GitHub (`REPO_SOURCE = "bundle"`). Create one with `git bundle create iot-sae-ids.bundle --all`.

### 3. SAE hyperparameter tuning (notebook section 5)

```bash
python scripts/tune_sae.py --dataset ton_iot unsw_nb15 rt_iot2022      # random search, 20 trials each
python scripts/tune_sae.py --dataset ton_iot --mode grid                # full grid, 216 combinations
```

How the search works:
- **Data:** each trial trains on the normal rows of a stratified 20% subsample of the train split (`tune.subsample`). It uses the seed-0 split and features.
- **Selection:** by validation ROC-AUC of the reconstruction error. Ties go to the lower validation-normal MSE, then to fewer parameters. The test split is never used.
- **Search space:** hidden architecture (scaled or fixed), `d_z` ∈ {8, 12, 16, 20} (capped below d), λ ∈ {1e-4, 1e-3, 1e-2}, dropout ∈ {0.2, 0.3, 0.4} and learning rate ∈ {1e-2, 1e-3, 1e-4}.

Outputs:
- `results/best_sae_<dataset>.yaml`: the best configuration, used automatically by every later SAE run;
- `results/tables/sae_tuning_<dataset>.md` and `.csv`: all trials, plus the parameter count of the scaled and fixed architectures;
- `results/tuning/`: raw per-trial logs, for resuming.

The hyperparameters are chosen on seed 0 and then reused for all 5 seeds. This is standard practice. Rows in seed 0's validation split can fall in another seed's test split, so this is a mild optimistic bias, and it is noted as a limitation.

### 4. Full grid (notebook section 6)

```bash
python scripts/run_all.py --dry-run                 # what is left (60 runs = 4 models x 3 datasets x 5 seeds)
python scripts/run_all.py --models sae rf svm       # 6a
python scripts/run_all.py --models lstm             # 6b (slowest)
```

- `run_all.py` is resumable. It skips runs whose JSON exists, logs failures to `results/raw/failures.log` and keeps going.
- It warns if SAE runs would start before tuning has produced `best_sae_<dataset>.yaml`.
- To move the results to the local repo, download the zip from notebook section 8 and unzip it in the repo root.

_TODO (Phase 7):_ `make_tables.py`, `make_figures.py` and `results/README.md`.

## Method notes

- **Splits:** data is split before any fitting.
  - TON_IoT and RT-IoT2022: stratified 70/10/20.
  - UNSW-NB15: the official training set is split 87.5/12.5 into train/val, and the official test set is kept as the test set.
- **Fitting:** every transformer is fitted on the training split only.
- **SAE "fixed" architecture:** hidden layers `128 → 64 → 32`, followed by the tuned bottleneck `d_z`. This is the proposal's architecture, kept as a comparison point against the input-scaled `[⌈0.75d⌉, ⌈0.5d⌉] → d_z` option.
- **SAE threshold:** the headline τ maximises Youden's J = TPR − FPR on the labelled validation split. It uses attack labels for threshold selection only. Attacks are the majority of every validation split, so maximising F1 (the proposal's choice) pushes τ down and accepts false-positive rates of 33–69% (smoke test). J does not depend on that ratio. The validation-F1 threshold, the percentile thresholds (p90–p99.9 of normal validation errors) and the threshold-free ROC-AUC/PR-AUC are all reported as well.
- **LSTM training batch size:** 512. Recurrent dropout disables the cuDNN kernel, and at batch size 64 the full grid was estimated at about 28 T4 GPU-hours. The architecture is unchanged, and latency is still timed on 64-row batches.
- **SVM:** when the training set has more than `svm.exact_max_rows` rows (default 100k), the SVM is a Nystroem RBF approximation followed by a linear hinge-loss SGD classifier, not an exact RBF SVM. Each result JSON records which variant ran.
