# iot-sae-ids

Experimental code for **"A Feature-Optimised Lightweight Intrusion Detection System for IoT Using Autoencoder"**.

A sparse autoencoder (SAE) is trained **only on normal traffic**. A flow is flagged as an attack when its reconstruction error exceeds a threshold τ. Before the SAE, features are reduced in two stages:

1. a Pearson correlation filter;
2. Random Forest importance ranking.

The SAE encoder then acts as a third, non-linear compression stage. The SAE is compared against Random Forest, SVM and LSTM baselines on TON_IoT, UNSW-NB15 and RT-IoT2022. Each model is scored on detection quality and on efficiency, across 5 seeds, with significance tests.

> Status: **Phase 2 — data loaders (CHECKPOINT 1).** Sections marked _TODO_ are filled in during later phases.

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
| TON_IoT | "~4M records" | _pending file_ (the `Train_Test_Network.csv` release is documented at ≈461k rows) | The ~22M-row / multi-million figures refer to the full raw network captures, not the train/test file used here. |
| UNSW-NB15 | — | _pending files_ (official split documented as 175,341 train / 82,332 test) | |
| RT-IoT2022 | "~3.5M records" | **123,117 rows, 83 features** (+ `Attack_type`) | UCI id 942. 3 benign device classes (MQTT_Publish, Thing_Speak, Wipro_bulb), 9 attack classes. No Amazon-Alexa class is present in the UCI release. 89.8% of rows are attacks, 76.9% are DOS_SYN_Hping. |

### Dropped columns (leakage control)

| Dataset | Dropped | Reason |
|---|---|---|
| TON_IoT | `ts` | Timestamp |
| TON_IoT | `src_ip`, `dst_ip` | Host identifiers |
| TON_IoT | `src_port`, `dst_port` | Flow identifiers. Ephemeral source ports in particular are near-unique per flow. |
| TON_IoT | `dns_query`, `ssl_subject`, `ssl_issuer`, `http_uri`, `http_user_agent` | Free-text, identifier-like fields: domain names, certificate subjects, URIs, user agents. _Proposed; awaiting confirmation._ |
| UNSW-NB15 | `id` | Row identifier |
| RT-IoT2022 | `id.orig_p`, `id.resp_p` | Source and destination ports |

Preprocessing also drops these columns, based on the train split only:

- columns with more than 5% missing values;
- constant (zero-variance) columns, e.g. RT-IoT2022's `bwd_URG_flag_count`.

The preprocessing logs list them.

### RT-IoT2022 label mapping

Binary label: rows whose `Attack_type` is a benign device class get 0, and all other rows get 1. The mapping lives in `configs/rt_iot2022.yaml` (`normal_classes`). Loading raises an error until `label_mapping_confirmed: true` is set.

## Reproducing the results

_TODO (Phases 5–7):_ the smoke test, SAE tuning, the full grid, and the tables and figures.

## Method notes

- **Splits:** data is split before any fitting.
  - TON_IoT and RT-IoT2022: stratified 70/10/20.
  - UNSW-NB15: the official training set is split 87.5/12.5 into train/val, and the official test set is kept as the test set.
- **Fitting:** every transformer is fitted on the training split only.
- **SAE "fixed" architecture:** hidden layers `128 → 64 → 32`, followed by the tuned bottleneck `d_z`. This is the proposal's architecture, kept as a comparison point against the input-scaled `[⌈0.75d⌉, ⌈0.5d⌉] → d_z` option.
- **SVM:** when the training set has more than `svm.exact_max_rows` rows (default 100k), the SVM is a Nystroem RBF approximation followed by a linear hinge-loss SGD classifier, not an exact RBF SVM. Each result JSON records which variant ran.
