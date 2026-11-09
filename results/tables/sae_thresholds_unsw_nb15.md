# UNSW-NB15 — SAE at every threshold (test set)

- τ chosen on validation data; metrics on test. Percentile rows use only normal validation errors (unsupervised).

| Threshold                            |   Runs | Accuracy        | Precision       | Recall          | F1              | FPR             |      τ |
|:-------------------------------------|-------:|:----------------|:----------------|:----------------|:----------------|:----------------|-------:|
| Val-optimised: Youden's J (headline) |      5 | 0.7280 ± 0.0042 | 0.6891 ± 0.0034 | 0.9218 ± 0.0021 | 0.7887 ± 0.0030 | 0.5095 ± 0.0071 | 0.0374 |
| Val-optimised: F1                    |      5 | 0.7248 ± 0.0011 | 0.6798 ± 0.0063 | 0.9464 ± 0.0263 | 0.7910 ± 0.0047 | 0.5467 ± 0.0315 | 0.0328 |
| Percentile p90 of normal val error   |      5 | 0.6636 ± 0.0472 | 0.8154 ± 0.0073 | 0.5026 ± 0.1101 | 0.6168 ± 0.0779 | 0.1393 ± 0.0306 | 0.0687 |
| Percentile p95 of normal val error   |      5 | 0.6019 ± 0.0143 | 0.8269 ± 0.0112 | 0.3499 ± 0.0255 | 0.4915 ± 0.0271 | 0.0894 ± 0.0016 | 0.0735 |
| Percentile p97.5 of normal val error |      5 | 0.5597 ± 0.0025 | 0.8818 ± 0.0081 | 0.2315 ± 0.0080 | 0.3666 ± 0.0093 | 0.0381 ± 0.0043 | 0.0849 |
| Percentile p99 of normal val error   |      5 | 0.5083 ± 0.0041 | 0.9421 ± 0.0100 | 0.1141 ± 0.0092 | 0.2033 ± 0.0145 | 0.0087 ± 0.0022 | 0.107  |
| Percentile p99.5 of normal val error |      5 | 0.4980 ± 0.0047 | 0.9598 ± 0.0083 | 0.0922 ± 0.0095 | 0.1681 ± 0.0159 | 0.0048 ± 0.0014 | 0.112  |
| Percentile p99.9 of normal val error |      5 | 0.4616 ± 0.0076 | 0.9907 ± 0.0037 | 0.0223 ± 0.0139 | 0.0433 ± 0.0268 | 0.0002 ± 0.0001 | 0.155  |
