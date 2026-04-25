# Streaming inference latency — CPU only

- date: 2026-04-25 17:17:29 UTC
- cpu: Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz
- python: 3.12.12
- torch: 2.8.0+cu128 (cpu-only forced via `CUDA_VISIBLE_DEVICES=""`)
- torch threads: 1 (intra-op, applied via `torch.set_num_threads`)
- warmup pushes: 10, trials: 5 × 200 timed pushes (1000 pooled per model)
- input: random Gaussian noise scaled to ±0.3, seed 0
- gc disabled inside the timed loop; `gc.collect()` between trials
- between-model cleanup: 3× `gc.collect()` + glibc `malloc_trim(0)` + 0.5 s sleep
- max absolute RSS observed across the whole run: **1212 MB**

| model | load_ms | chunk_ms | median_ms | p99_ms | cv% | ms/s_audio | rtf | peak_rss_mb | Δrss_mb | residual_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DT | 520.5 | 10.00 | 4.055 | 5.793 | 0.7 | 405.5 | 2.5× | 1212 | 371 | 6 |
| GMM | 2.1 | 15.00 | 3.744 | 6.109 | 0.6 | 249.6 | 4.0× | 849 | 3 | 2 |
| SVM | 4.9 | 15.00 | 6.980 | 10.339 | 0.9 | 465.3 | 2.1× | 857 | 9 | 1 |
| SmallTCN | 11.4 | 23.22 | 5.812 | 8.058 | 16.8 | 250.3 | 4.0× | 913 | 64 | 54 |
| TCN | 9.6 | 23.22 | 6.134 | 8.674 | 19.8 | 264.2 | 3.8× | 913 | 10 | 0 |
| TCN+LSTM | 12.7 | 23.22 | 10.091 | 13.427 | 19.2 | 434.6 | 2.3× | 928 | 24 | 13 |

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
