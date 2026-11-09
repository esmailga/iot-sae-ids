# Results

Status: **50 of 60 planned runs complete** (2026-11-09; full detail in [`tables/run_coverage.md`](tables/run_coverage.md)). The LSTM has run on TON_IoT only. Its 10 runs on UNSW-NB15 and RT-IoT2022 are still missing, because the Colab GPU became unavailable. Every table and figure here is generated from `results/raw/*.json` by `scripts/make_tables.py` and `scripts/make_figures.py`. Regenerate them after the remaining runs finish.

Values are mean ± std over 5 seeds. The seed controls the split, the model initialisation and any subsampling. All metrics are on the test set: the official test set for UNSW-NB15, and the stratified 20% split for the other two datasets.

## Headline results

From [`tables/summary.md`](tables/summary.md):

| Dataset | Model | F1 | Recall | FPR | ROC-AUC | Params | Size (MB) |
|---|---|---|---|---|---|---|---|
| TON_IoT | **SAE** | 0.823 ± 0.074 | 0.758 ± 0.102 | 0.253 ± 0.144 | 0.699 ± 0.155 | 1,391 | 0.089 |
| TON_IoT | RF | 0.998 ± 0.000 | 0.999 ± 0.000 | 0.006 ± 0.001 | 1.000 ± 0.000 | 93,682 | 7.534 |
| TON_IoT | SVM | 0.904 ± 0.001 | 0.870 ± 0.001 | 0.178 ± 0.003 | 0.867 ± 0.016 | 116,892 | 1.060 |
| TON_IoT | LSTM | 0.904 ± 0.001 | 0.870 ± 0.001 | 0.178 ± 0.003 | 0.855 ± 0.025 | 116,033 | 1.427 |
| UNSW-NB15 | **SAE** | 0.789 ± 0.003 | 0.922 ± 0.002 | 0.510 ± 0.007 | 0.806 ± 0.012 | 2,495 | 0.101 |
| UNSW-NB15 | RF | 0.892 ± 0.001 | 0.984 ± 0.001 | 0.270 ± 0.002 | 0.975 ± 0.001 | 966,674 | 77.374 |
| UNSW-NB15 | SVM | 0.852 ± 0.000 | 0.999 ± 0.001 | 0.423 ± 0.001 | 0.890 ± 0.002 | 503,415 | 4.408 |
| RT-IoT2022 | **SAE** | 0.939 ± 0.034 | 0.893 ± 0.061 | 0.057 ± 0.037 | 0.943 ± 0.021 | 3,418 | 0.112 |
| RT-IoT2022 | RF | 0.997 ± 0.003 | 0.996 ± 0.006 | 0.014 ± 0.003 | 1.000 ± 0.000 | 32,569 | 2.645 |
| RT-IoT2022 | SVM | 0.973 ± 0.006 | 0.954 ± 0.012 | 0.052 ± 0.008 | 0.961 ± 0.008 | 119,266 | 1.031 |

How each model's predictions are made:
- **SAE:** reconstruction error > τ, where τ maximises Youden's J on the labelled validation split.
- **RF and LSTM:** P(attack) > 0.5.
- **SVM:** exact RBF SVC on all three datasets, predicting attack when the margin is > 0.

## Findings

1. **The SAE detects worse than every supervised baseline.**
   - Against RF and SVM, over 15 (dataset, seed) pairs: F1, recall and accuracy are all significantly lower on both the paired t-test and Wilcoxon (p < 0.001). Effect sizes are large: |d_z| 1.2–1.9, rank-biserial r ≈ −1.
   - Against the LSTM: only the 5 TON_IoT pairs exist so far. The SAE is lower (ΔF1 = −0.08), but this is not significant, because Wilcoxon with n = 5 cannot go below p = 0.0625.
   - Source: [`tables/significance.md`](tables/significance.md).
2. **The SAE comes closest on RT-IoT2022:** F1 0.939 and FPR 0.057, against SVM at 0.973 and 0.052. It is weakest on UNSW-NB15. There, every model has a high FPR (0.27–0.51), consistent with the different class mix of the official test set (45% normal) and train set (32%).
3. **The SAE is far smaller than the baselines.**
   - Parameters: 1.4k–3.4k for the SAE, against 33k–967k tree nodes for RF and 117k–503k for SVM (support vectors × d).
   - Model size: about 0.1 MB in Keras format and 0.010–0.018 MB as TFLite, against 2.6–77 MB for RF.
   - Source: [`figures/bars_params.png`](figures/bars_params.png) and [`figures/bars_size.png`](figures/bars_size.png).
4. **Latency depends on the runtime more than on the model.**
   - Measured through Keras, the SAE takes 216–248 µs per flow. That is fixed per-call overhead; the network itself needs roughly as many multiply-adds per flow as it has parameters (1.4k–3.4k).
   - Exported to TFLite with a single thread, the same SAE takes **0.15–0.25 µs per flow**.
   - Single-threaded sklearn takes 106–141 µs per flow for RF and 238–1,185 µs per flow for SVM. The LSTM takes about 5,100 µs per flow on CPU.
   - These runtimes differ, so compare within one runtime, or state clearly that the TFLite figure is the deployment path.
   - Source: [`tables/efficiency_*.md`](tables/).
5. **The SAE's accuracy varies a lot between seeds on TON_IoT:** ROC-AUC 0.70 ± 0.15. Recall on `injection` and `password` swings by about ±0.4 across seeds, and `scanning` is mostly missed. SVM and LSTM also miss `scanning` (recall 0.23); RF does not. The two models agree on 99.5% of their TON_IoT test predictions. On RT-IoT2022 the SAE misses most `ARP_poisioning` and `NMAP_UDP_SCAN` flows. Source: [`tables/per_category_*.md`](tables/).
6. **The threshold choice matters.** The unsupervised percentile thresholds (p90–p99.9 of normal validation error) hold FPR at or below 3.5% on TON_IoT, but recall collapses there (0.18 at p95). The validation-F1 threshold reaches F1 0.894, but at FPR 0.56. Youden's J sits between the two, at F1 0.823 and FPR 0.25. Source: [`tables/sae_thresholds_*.md`](tables/).
7. **Feature reduction is 42/43/83 raw columns → 19/25–27/31–33 after Stage 2 → an SAE latent of 12/16/20** (TON_IoT/UNSW-NB15/RT-IoT2022). In tuning, the input-scaled architecture `[⌈0.75d⌉, ⌈0.5d⌉] → d_z` beat the proposal's fixed `128 → 64 → 32` on all three datasets. Sources: [`tables/features_summary.md`](tables/features_summary.md) and [`tables/sae_tuning_*.md`](tables/).

## Caveats

- **The LSTM is incomplete:** 0/5 runs on UNSW-NB15 and RT-IoT2022. Any "SAE vs LSTM" statement rests on TON_IoT alone.
- **LSTM training time isn't comparable.** The TON_IoT LSTM runs trained on CPU, because the GPU was unavailable, while the SAE trained on GPU. Inference latency for every model is measured on CPU, so those numbers are comparable.
- **Duplicates were removed from the train split only.** Val and test keep theirs, so some test flows have exact copies in train. This favours models that memorise, especially RF. Rerunning with `preprocess.dedup_scope: dataset` would give a stricter estimate.
- **SAE hyperparameters were tuned on seed 0's train and validation splits** and then reused for all seeds. This gives a mild optimistic bias for the SAE.
- **The SVM rows are an exact RBF SVC.** Every training set had ≤ 100k rows, so the Nystroem approximation never ran.
- **Two machines produced slightly different seed-0 Stage 1 lists for UNSW-NB15.** The local Mac (scikit-learn 1.7) and Colab (1.6) disagree on which of two near-tied correlated features to keep. The final Stage 2 feature set is identical. The Colab files in `features/` are the reference.

## Finishing the grid

On Colab, when a GPU is available, or on a CPU runtime: each TON_IoT LSTM run took about 11 minutes on CPU. Run:

```bash
python scripts/run_all.py --models lstm      # runs only the 10 missing runs
```

Then download the results zip, unzip it into the repo root, and run:

```bash
python scripts/make_tables.py && python scripts/make_figures.py
```

## Files

| Path | Content |
|---|---|
| `tables/summary.md` | headline table above |
| `tables/detection_<dataset>.md` | Accuracy, Precision, Recall, F1, FPR, ROC-AUC, PR-AUC |
| `tables/efficiency_<dataset>.md` | training time, batch/flow latency CPU (+GPU), params, size, memory, TFLite |
| `tables/features_<dataset>.md`, `features_summary.md` | feature counts per stage, SAE architecture |
| `tables/significance.md` | t-test, Wilcoxon, effect sizes |
| `tables/sae_thresholds_<dataset>.md` | SAE at every threshold variant |
| `tables/per_category_<dataset>.md` | recall per attack category |
| `tables/sae_tuning_<dataset>.md` | all tuning trials + architecture parameter counts |
| `best_sae_<dataset>.yaml` | tuned SAE hyperparameters |
| `features/<dataset>__s<seed>.json` | selected features, Pearson drops, RF importances |
| `figures/bars_*.png` | F1, ROC-AUC, FPR, latency, params, size |
| `figures/confusion_<dataset>.png` | confusion matrices (summed over seeds) |
| `figures/roc_pr_<dataset>__s0.png` | ROC and PR curves |
| `figures/sae_error_hist_<dataset>__s0.png` | reconstruction error, normal vs attack, τ marked |
| `figures/loss_<dataset>__s0.png` | SAE / LSTM training curves |
| `figures/features/*` | Pearson heatmap, RF importance |
