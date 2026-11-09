# TON_IoT — efficiency

- Timing uses time.perf_counter. Inference: 64-flow batches, 10 warm-up + 100 timed; per-flow = batch time / 64. sklearn models are timed single-threaded (n_jobs=1). Values are mean ± std across seeds.
- Params: Keras count_params (SAE, LSTM); total tree nodes (RF); support vectors × d (exact SVM).
- Peak RSS Δ: increase in process memory during inference (psutil).
- Keras rows (SAE, LSTM) are eager `model(x)` calls, so a fixed per-call framework overhead of roughly 10 ms per batch dominates small models. The SAE TFLite rows (single-threaded interpreter, same batches) show the deployable cost of the network itself.
- LSTM training ran on CPU (the Colab GPU was unavailable), so its training time is not comparable with the GPU-trained SAE.

| Model                     |   Runs | Train time (s)   | Trained on   | CPU batch-64 (ms)   | CPU per flow (µs)   | GPU per flow (µs)   |   Params |   Size (MB) | Peak RSS Δ (MB)   |
|:--------------------------|-------:|:-----------------|:-------------|:--------------------|:--------------------|:--------------------|---------:|------------:|:------------------|
| SAE                       |      5 | 82.4 ± 44.6      | GPU          | 13.817 ± 1.624      | 215.90 ± 25.37      | 266.52 ± 41.74      |    1,391 |      0.0885 | 0.00 ± 0.00       |
| RF                        |      5 | 3.8 ± 0.4        | CPU          | 6.790 ± 1.165       | 106.10 ± 18.21      | —                   |   93,682 |      7.5345 | 0.00 ± 0.00       |
| SVM                       |      5 | 42.6 ± 25.9      | CPU          | 22.742 ± 6.352      | 355.34 ± 99.25      | —                   |  116,892 |      1.0598 | 0.00 ± 0.00       |
| LSTM                      |      5 | 635.2 ± 265.7    | CPU/GPU      | 326.226 ± 14.575    | 5097.27 ± 227.73    | 5979.08             |  116,033 |      1.4273 | 0.60 ± 1.06       |
| SAE · TFLite float32      |      5 | —                | —            | 0.010 ± 0.002       | 0.15 ± 0.03         | —                   |    1,391 |      0.0099 | —                 |
| SAE · TFLite dynamic int8 |      5 | —                | —            | 0.012 ± 0.004       | 0.19 ± 0.06         | —                   |    1,391 |      0.0099 | —                 |
