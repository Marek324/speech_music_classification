# Streaming inference latency — CPU only

- date: 2026-04-25 16:35:29 UTC
- cpu: Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz
- python: 3.12.12
- torch: 2.8.0+cu128 (cpu-only forced via `CUDA_VISIBLE_DEVICES=""`)
- torch threads: 4 (default — sklearn classics still single-threaded; comparison favours NN models)
- warmup pushes: 10, trials: 5 × 200 timed pushes (1000 pooled per model)
- input: random Gaussian noise scaled to ±0.3, seed 0
- gc disabled inside the timed loop; `gc.collect()` between trials
- between-model cleanup: 3× `gc.collect()` + glibc `malloc_trim(0)` + 0.5 s sleep
- max absolute RSS observed across the whole run: **1211 MB**

| model | load_ms | chunk_ms | median_ms | p99_ms | cv% | ms/s_audio | rtf | peak_rss_mb | Δrss_mb | residual_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DT | 784.5 | 10.00 | 4.181 | 6.797 | 1.9 | 418.1 | 2.4× | 1211 | 371 | 6 |
| GMM | 11.1 | 15.00 | 4.041 | 10.873 | 5.9 | 269.4 | 3.7× | 848 | 3 | 2 |
| SVM | 18.5 | 15.00 | 7.310 | 14.829 | 3.5 | 487.4 | 2.1× | 856 | 9 | 1 |
| SmallTCN | 24.6 | 23.22 | 5.648 | 10.741 | 13.8 | 243.2 | 4.1× | 923 | 75 | 65 |
| TCN | 11.4 | 23.22 | 4.877 | 10.249 | 17.2 | 210.0 | 4.8× | 926 | 13 | 5 |
| TCN+LSTM | 13.9 | 23.22 | 7.197 | 19.085 | 6.1 | 309.9 | 3.2× | 945 | 26 | 11 |

**ms/s_audio** is the fair cross-model metric: each model's chunk
covers a different audio duration (DT 10 ms, GMM/SVM 15 ms, NN family
23.2 ms), so raw `median_ms` is **not** comparable. **rtf > 1×** means
faster than real time; rtf < 1× cannot keep up with a live mic.

**cv%** is the coefficient of variation across the trial medians —
a quick reliability check. Numbers under ~10% mean the median is
stable; higher suggests background load on the machine.

**peak_rss_mb** is the absolute peak RSS during the model's run, but
is **run-order-dependent** — a model that ran after a heavy cleanup
starts from a lower baseline, so its absolute peak under-reports its
true standalone footprint. **Δrss_mb** is the more stable per-model
metric: the marginal cost on top of whatever baseline was present.
**residual_mb** is what survived the deep cleanup — modules and
pickle pools that don't get released between models. The header line
above (max absolute RSS observed) reflects the realistic ceiling
for a long-running session that visits every model.

Caveats: random noise undertests sklearn DT/SVM data-dependent paths;
the bias is expected to be small but not zero. Numbers reflect this
machine + this PyTorch build only.
