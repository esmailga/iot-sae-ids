# TON_IoT — recall per category (test set)

- Attack rows: detection rate. Normal rows: fraction kept as normal (1 − FPR within that class).
- Categories with very few test rows give noisy estimates.

| Category   | Type             |   Test rows | SAE           | RF            | SVM           | LSTM          |
|:-----------|:-----------------|------------:|:--------------|:--------------|:--------------|:--------------|
| backdoor   | attack           |        4000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| ddos       | attack           |        4000 | 0.826 ± 0.134 | 0.992 ± 0.001 | 0.981 ± 0.002 | 0.976 ± 0.005 |
| dos        | attack           |        4000 | 0.910 ± 0.042 | 0.999 ± 0.000 | 0.955 ± 0.002 | 0.956 ± 0.002 |
| injection  | attack           |        4000 | 0.661 ± 0.433 | 0.999 ± 0.000 | 0.997 ± 0.001 | 0.996 ± 0.001 |
| password   | attack           |        4000 | 0.690 ± 0.364 | 1.000 ± 0.000 | 0.985 ± 0.002 | 0.985 ± 0.001 |
| ransomware | attack           |        4000 | 0.924 ± 0.003 | 1.000 ± 0.001 | 0.914 ± 0.006 | 0.920 ± 0.005 |
| scanning   | attack           |        4000 | 0.213 ± 0.016 | 1.000 ± 0.000 | 0.230 ± 0.007 | 0.230 ± 0.007 |
| xss        | attack           |        4000 | 0.853 ± 0.033 | 1.000 ± 0.000 | 0.898 ± 0.006 | 0.897 ± 0.006 |
| mitm       | attack           |         209 | 0.553 ± 0.387 | 0.973 ± 0.009 | 0.881 ± 0.010 | 0.960 ± 0.003 |
| normal     | normal (1 − FPR) |       10000 | 0.747 ± 0.144 | 0.994 ± 0.001 | 0.822 ± 0.003 | 0.822 ± 0.002 |
