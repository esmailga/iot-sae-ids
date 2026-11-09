# Significance: SAE vs each baseline

- Two-sided paired t-test and Wilcoxon signed-rank over (dataset, seed) pairs, α = 0.05.
- Δ > 0 means the SAE is higher. Effect sizes: Cohen's d_z (paired) and matched-pairs rank-biserial r.
- Per-dataset rows have ≤ 5 pairs: a two-sided Wilcoxon test cannot reach p < 0.0625 with n = 5, so those rows are descriptive only.
- 'All datasets' pools only the (dataset, seed) pairs where both models ran; check the Pairs column (e.g. a baseline missing on some datasets).

| Scope        | SAE vs   | Metric   |   Pairs |   SAE mean |   Baseline mean |   Δ (SAE − base) | t-test p   | Wilcoxon p   |   Cohen's d_z |   Rank-biserial r | Significant (both tests)   |
|:-------------|:---------|:---------|--------:|-----------:|----------------:|-----------------:|:-----------|:-------------|--------------:|------------------:|:---------------------------|
| All datasets | LSTM     | F1       |       5 |     0.8231 |          0.9039 |          -0.0808 | 0.070      | 0.062        |         -1.1  |             -1    | no                         |
| All datasets | LSTM     | Recall   |       5 |     0.7583 |          0.8704 |          -0.1121 | 0.070      | 0.125        |         -1.1  |             -0.87 | no                         |
| All datasets | LSTM     | Accuracy |       5 |     0.7556 |          0.8588 |          -0.1032 | 0.067      | 0.062        |         -1.12 |             -1    | no                         |
| All datasets | RF       | F1       |      15 |     0.8503 |          0.9627 |          -0.1124 | <0.001     | <0.001       |         -1.7  |             -1    | yes                        |
| All datasets | RF       | Recall   |      15 |     0.8576 |          0.9927 |          -0.1352 | <0.001     | <0.001       |         -1.33 |             -1    | yes                        |
| All datasets | RF       | Accuracy |      15 |     0.7938 |          0.954  |          -0.1602 | <0.001     | <0.001       |         -1.89 |             -1    | yes                        |
| All datasets | SVM      | F1       |      15 |     0.8503 |          0.9097 |          -0.0594 | <0.001     | <0.001       |         -1.21 |             -1    | yes                        |
| All datasets | SVM      | Recall   |      15 |     0.8576 |          0.9408 |          -0.0833 | <0.001     | <0.001       |         -1.19 |             -0.98 | yes                        |
| All datasets | SVM      | Accuracy |      15 |     0.7938 |          0.8736 |          -0.0799 | <0.001     | <0.001       |         -1.27 |             -1    | yes                        |
| RT-IoT2022   | RF       | F1       |       5 |     0.9391 |          0.9972 |          -0.0581 | 0.020      | 0.062        |         -1.68 |             -1    | no                         |
| RT-IoT2022   | RF       | Recall   |       5 |     0.8926 |          0.996  |          -0.1035 | 0.020      | 0.062        |         -1.68 |             -1    | no                         |
| RT-IoT2022   | RF       | Accuracy |       5 |     0.8977 |          0.9951 |          -0.0973 | 0.015      | 0.062        |         -1.83 |             -1    | no                         |
| RT-IoT2022   | SVM      | F1       |       5 |     0.9391 |          0.9733 |          -0.0342 | 0.128      | 0.062        |         -0.86 |             -1    | no                         |
| RT-IoT2022   | SVM      | Recall   |       5 |     0.8926 |          0.9537 |          -0.0611 | 0.131      | 0.062        |         -0.85 |             -1    | no                         |
| RT-IoT2022   | SVM      | Accuracy |       5 |     0.8977 |          0.9531 |          -0.0554 | 0.117      | 0.062        |         -0.89 |             -1    | no                         |
| TON_IoT      | LSTM     | F1       |       5 |     0.8231 |          0.9039 |          -0.0808 | 0.070      | 0.062        |         -1.1  |             -1    | no                         |
| TON_IoT      | LSTM     | Recall   |       5 |     0.7583 |          0.8704 |          -0.1121 | 0.070      | 0.125        |         -1.1  |             -0.87 | no                         |
| TON_IoT      | LSTM     | Accuracy |       5 |     0.7556 |          0.8588 |          -0.1032 | 0.067      | 0.062        |         -1.12 |             -1    | no                         |
| TON_IoT      | RF       | F1       |       5 |     0.8231 |          0.9984 |          -0.1752 | 0.006      | 0.062        |         -2.38 |             -1    | no                         |
| TON_IoT      | RF       | Recall   |       5 |     0.7583 |          0.9985 |          -0.2403 | 0.006      | 0.062        |         -2.35 |             -1    | no                         |
| TON_IoT      | RF       | Accuracy |       5 |     0.7556 |          0.9975 |          -0.2419 | 0.004      | 0.062        |         -2.61 |             -1    | no                         |
| TON_IoT      | SVM      | F1       |       5 |     0.8231 |          0.9037 |          -0.0805 | 0.071      | 0.062        |         -1.09 |             -1    | no                         |
| TON_IoT      | SVM      | Recall   |       5 |     0.7583 |          0.8699 |          -0.1116 | 0.071      | 0.125        |         -1.09 |             -0.87 | no                         |
| TON_IoT      | SVM      | Accuracy |       5 |     0.7556 |          0.8585 |          -0.1029 | 0.068      | 0.062        |         -1.11 |             -1    | no                         |
| UNSW-NB15    | RF       | F1       |       5 |     0.7887 |          0.8924 |          -0.1038 | <0.001     | 0.062        |        -44.64 |             -1    | no                         |
| UNSW-NB15    | RF       | Recall   |       5 |     0.9218 |          0.9836 |          -0.0618 | <0.001     | 0.062        |        -32    |             -1    | no                         |
| UNSW-NB15    | RF       | Accuracy |       5 |     0.728  |          0.8694 |          -0.1415 | <0.001     | 0.062        |        -41.73 |             -1    | no                         |
| UNSW-NB15    | SVM      | F1       |       5 |     0.7887 |          0.8523 |          -0.0636 | <0.001     | 0.062        |        -20.25 |             -1    | no                         |
| UNSW-NB15    | SVM      | Recall   |       5 |     0.9218 |          0.9989 |          -0.0771 | <0.001     | 0.062        |        -26.43 |             -1    | no                         |
| UNSW-NB15    | SVM      | Accuracy |       5 |     0.728  |          0.8093 |          -0.0814 | <0.001     | 0.062        |        -18.64 |             -1    | no                         |
