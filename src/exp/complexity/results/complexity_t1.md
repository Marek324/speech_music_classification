# CPU complexity analysis — front-of-pipeline classifiers

- date: 2026-04-29 13:00:09 UTC
- cpu: Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz
- python: 3.12.12
- torch: 2.8.0+cu128 (cpu-only forced via `CUDA_VISIBLE_DEVICES=""`)
- torch threads: 1 (intra-op, applied via `torch.set_num_threads`)
- warmup pushes: 10, trials: 5 × 200 timed pushes (1000 pooled per model)
- input: random Gaussian noise scaled to ±0.3, seed 0
- gc disabled inside the timed loop; `gc.collect()` between trials
- between-model cleanup: 3× `gc.collect()` + glibc `malloc_trim(0)` + 0.5 s sleep
- max absolute RSS observed across the whole run: **1208 MB**



## Empirical (1-thread CPU)

| model | load_ms | chunk_ms | median_ms | p99_ms | cv% | ms/s_audio | rtf | peak_rss_mb | Δrss_mb | residual_mb | F1_macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DT | 991.5 | 10.00 | 4.145 | 7.467 | 0.9 | 414.5 | 2.4× | 1208 | 368 | 3 | 0.8376 |
| GMM | 3.3 | 15.00 | 3.939 | 8.428 | 10.0 | 262.6 | 3.8× | 845 | 3 | 2 | 0.8250 |
| SVM | 10.8 | 15.00 | 7.662 | 14.514 | 3.2 | 510.8 | 2.0× | 853 | 9 | 1 | 0.8750 |
| SmallTCN | 37.4 | 23.22 | 6.876 | 11.686 | 16.2 | 296.1 | 3.4× | 909 | 64 | 52 | 0.9766 |
| SmallerTCN | 16.2 | 23.22 | 4.380 | 8.725 | 21.5 | 188.6 | 5.3× | 910 | 13 | 3 | 0.9722 |
| TCN | 14.3 | 23.22 | 7.181 | 14.425 | 19.1 | 309.3 | 3.2× | 908 | 8 | -1 | 0.9752 |
| TCN+LSTM | 48.4 | 23.22 | 11.067 | 31.354 | 19.8 | 476.6 | 2.1× | 922 | 23 | 6 | 0.9846 |

## Symbolic complexity (per output frame)

| model | params | MACs/frame | RF_ms | buf_ms | T∞ | T₁/T∞ |
|---|---:|---:|---:|---:|---:|---:|
| DT | 4.34 M | 178 | 300.0 | 305.6 | 88 | 2 |
| GMM | 474 | 216 | 1000.0 | 1024.8 | 3 | 72 |
| SVM | 520.8 k | 468.7 k | 1000.0 | 1007.5 | 16 | 29293 |
| SmallTCN | 10.0 k | 3.47 M | 8382.4 | 8382.4 | 26 | 133626 |
| SmallerTCN | 4.6 k | 545.0 k | 2809.6 | 2809.6 | 10 | 54498 |
| TCN | 32.8 k | 11.57 M | 8382.4 | 8382.4 | 26 | 444974 |
| TCN+LSTM | 388.9 k | 137.26 M | 8382.4 | 8385.3 | 27 | 5083749 |

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

**MACs/frame** is the streaming cost (one push). For the TCN family
this includes the receptive-field re-run that the current
`StreamingInference` performs; an idealised stateful cache would
drop this by `recept_frames`× to the offline number.

**RF_ms** is the audio context the model needs to compute one output
frame; **buf_ms** is the audio-equivalent of all persistent streaming
state (audio ring buffer + LSTM h/c + smoothing windows), divided by
the model's own sample rate. They're nearly equal because the audio
ring buffer dominates: the LSTM h/c (~128 floats) and the classic
smoothing buffers (n_smooth × n_classes) add only a few ms each.

**T∞** is the critical-path depth (longest serial chain of ops);
**T₁/T∞ = MACs/frame ÷ T∞** is the algorithmic max parallelism.
DT is dominated by tree-depth comparisons (intrinsically serial,
T₁/T∞ ≈ 2 once smoothing is included); SVM is a parallel-friendly
kernel sum; the TCN family scores 10⁵–10⁶ in theory but BLAS at
batch=1 only realises a fraction of that — see the scale sweep.

Caveats: random noise undertests sklearn DT/SVM data-dependent paths;
the bias is expected to be small but not zero. Numbers reflect this
machine + this PyTorch build only.
