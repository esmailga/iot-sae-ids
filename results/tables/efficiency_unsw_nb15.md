# UNSW-NB15 — efficiency

- Timing uses time.perf_counter. Inference: 64-flow batches, 10 warm-up + 100 timed; per-flow = batch time / 64. sklearn models are timed single-threaded (n_jobs=1). Values are mean ± std across seeds.
- Params: Keras count_params (SAE, LSTM); total tree nodes (RF); support vectors × d (exact SVM).
- Peak RSS Δ: increase in process memory during inference (psutil).
- Keras rows (SAE, LSTM) are eager `model(x)` calls, so a fixed per-call framework overhead of roughly 10 ms per batch dominates small models. The SAE TFLite rows (single-threaded interpreter, same batches) show the deployable cost of the network itself.

| Model                     |   Runs | Train time (s)   | Trained on   | CPU batch-64 (ms)   | CPU per flow (µs)   | GPU per flow (µs)   |   Params |   Size (MB) | Peak RSS Δ (MB)   |
|:--------------------------|-------:|:-----------------|:-------------|:--------------------|:--------------------|:--------------------|---------:|------------:|:------------------|
| SAE                       |      5 | 115.0 ± 27.5     | GPU          | 15.888 ± 1.536      | 248.25 ± 23.99      | 269.51 ± 34.40      |    2,495 |      0.1014 | 0.00 ± 0.00       |
| RF                        |      5 | 21.2 ± 0.4       | CPU          | 9.020 ± 1.906       | 140.94 ± 29.78      | —                   |  966,674 |     77.3738 | 0.00 ± 0.00       |
| SVM                       |      5 | 191.0 ± 36.5     | CPU          | 75.857 ± 7.615      | 1185.27 ± 118.99    | —                   |  503,415 |      4.4075 | 0.00 ± 0.00       |
| SAE · TFLite float32      |      5 | —                | —            | 0.011 ± 0.003       | 0.17 ± 0.04         | —                   |    2,495 |      0.014  | —                 |
| SAE · TFLite dynamic int8 |      5 | —                | —            | 0.013 ± 0.006       | 0.21 ± 0.09         | —                   |    2,495 |      0.014  | —                 |
