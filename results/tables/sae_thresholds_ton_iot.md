# TON_IoT — SAE at every threshold (test set)

- τ chosen on validation data; metrics on test. Percentile rows use only normal validation errors (unsupervised).

| Threshold                            |   Runs | Accuracy        | Precision       | Recall          | F1              | FPR             |       τ |
|:-------------------------------------|-------:|:----------------|:----------------|:----------------|:----------------|:----------------|--------:|
| Val-optimised: Youden's J (headline) |      5 | 0.7556 ± 0.0927 | 0.9062 ± 0.0522 | 0.7583 ± 0.1022 | 0.8231 ± 0.0738 | 0.2531 ± 0.1436 | 0.0311  |
| Val-optimised: F1                    |      5 | 0.8286 ± 0.0258 | 0.8532 ± 0.0628 | 0.9478 ± 0.0689 | 0.8942 ± 0.0113 | 0.5551 ± 0.2959 | 0.00686 |
| Percentile p90 of normal val error   |      5 | 0.3705 ± 0.1694 | 0.9087 ± 0.0759 | 0.1858 ± 0.2200 | 0.2730 ± 0.2582 | 0.0346 ± 0.0284 | 0.045   |
| Percentile p95 of normal val error   |      5 | 0.3650 ± 0.1731 | 0.9078 ± 0.0777 | 0.1760 ± 0.2261 | 0.2564 ± 0.2690 | 0.0260 ± 0.0102 | 0.0473  |
| Percentile p97.5 of normal val error |      5 | 0.3563 ± 0.1764 | 0.9294 ± 0.0344 | 0.1621 ± 0.2318 | 0.2336 ± 0.2769 | 0.0183 ± 0.0035 | 0.0493  |
| Percentile p99 of normal val error   |      5 | 0.2697 ± 0.0123 | 0.9572 ± 0.0206 | 0.0448 ± 0.0165 | 0.0852 ± 0.0307 | 0.0058 ± 0.0027 | 0.0626  |
| Percentile p99.5 of normal val error |      5 | 0.2676 ± 0.0123 | 0.9700 ± 0.0132 | 0.0413 ± 0.0161 | 0.0788 ± 0.0299 | 0.0036 ± 0.0010 | 0.0698  |
| Percentile p99.9 of normal val error |      5 | 0.2441 ± 0.0066 | 0.7804 ± 0.4364 | 0.0096 ± 0.0088 | 0.0189 ± 0.0173 | 0.0006 ± 0.0004 | 0.0922  |
