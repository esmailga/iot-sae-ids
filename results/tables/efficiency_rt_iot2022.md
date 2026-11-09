# RT-IoT2022 — efficiency

- Timing uses time.perf_counter. Inference: 64-flow batches, 10 warm-up + 100 timed; per-flow = batch time / 64. sklearn models are timed single-threaded (n_jobs=1). Values are mean ± std across seeds.
- Params: Keras count_params (SAE, LSTM); total tree nodes (RF); support vectors × d (exact SVM).
- Peak RSS Δ: increase in process memory during inference (psutil).
- Keras rows (SAE, LSTM) are eager `model(x)` calls, so a fixed per-call framework overhead of roughly 10 ms per batch dominates small models. The SAE TFLite rows (single-threaded interpreter, same batches) show the deployable cost of the network itself.

| Model                     |   Runs | Train time (s)   | Trained on   | CPU batch-64 (ms)   | CPU per flow (µs)   | GPU per flow (µs)   |   Params |   Size (MB) | Peak RSS Δ (MB)   |
|:--------------------------|-------:|:-----------------|:-------------|:--------------------|:--------------------|:--------------------|---------:|------------:|:------------------|
| SAE                       |      5 | 145.1 ± 34.8     | GPU          | 14.010 ± 2.030      | 218.91 ± 31.72      | 240.57 ± 10.47      |    3,418 |      0.1123 | 0.01 ± 0.01       |
| RF                        |      5 | 2.6 ± 0.7        | CPU          | 7.165 ± 1.450       | 111.95 ± 22.65      | —                   |   32,569 |      2.6454 | 0.00 ± 0.00       |
| SVM                       |      5 | 2.8 ± 0.3        | CPU          | 15.201 ± 1.984      | 237.52 ± 31.00      | —                   |  119,266 |      1.0313 | 0.00 ± 0.00       |
| SAE · TFLite float32      |      5 | —                | —            | 0.016 ± 0.004       | 0.25 ± 0.06         | —                   |    3,418 |      0.0175 | —                 |
| SAE · TFLite dynamic int8 |      5 | —                | —            | 0.016 ± 0.004       | 0.25 ± 0.07         | —                   |    3,418 |      0.0175 | —                 |
