# CPU complexity analysis — front-of-pipeline classifiers

- date: 2026-04-26 10:34:42 UTC
- cpu: Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz
- python: 3.12.12
- torch: 2.8.0+cu128 (cpu-only forced via `CUDA_VISIBLE_DEVICES=""`)
- torch threads: 1 (intra-op, applied via `torch.set_num_threads`)
- warmup pushes: 10, trials: 5 × 200 timed pushes (1000 pooled per model)
- input: random Gaussian noise scaled to ±0.3, seed 0
- max absolute RSS observed across the whole run: **1154 MB**

## Empirical (1-thread CPU)


| model    | load_ms | chunk_ms | median_ms | p99_ms | cv%  | ms/s_audio | rtf  | peak_rss_mb | Δrss_mb | residual_mb | F1_macro |
| -------- | ------- | -------- | --------- | ------ | ---- | ---------- | ---- | ----------- | ------- | ----------- | -------- |
| DT       | 1635.2  | 10.00    | 4.014     | 6.268  | 0.4  | 401.4      | 2.5× | 1154        | 315     | -51         | 0.8376   |
| GMM      | 3.5     | 15.00    | 3.732     | 5.860  | 0.3  | 248.8      | 4.0× | 791         | 3       | 2           | 0.8250   |
| SVM      | 11.4    | 15.00    | 6.902     | 10.204 | 0.5  | 460.1      | 2.2× | 799         | 9       | 1           | 0.8750   |
| SmallTCN | 44.5    | 23.22    | 6.282     | 10.828 | 15.7 | 270.6      | 3.7× | 852         | 61      | 52          | 0.9766   |
| TCN      | 12.3    | 23.22    | 6.601     | 10.873 | 19.2 | 284.3      | 3.5× | 852         | 8       | 0           | 0.9752   |
| TCN+LSTM | 16.6    | 23.22    | 10.473    | 13.814 | 17.3 | 451.0      | 2.2× | 867         | 24      | 14          | 0.9846   |


## Symbolic complexity (per output frame)


| model    | params  | MACs/frame | RF_ms  | buf_ms | T∞  | T₁/T∞   |
| -------- | ------- | ---------- | ------ | ------ | --- | ------- |
| DT       | 4.34 M  | 178        | 300.0  | 305.6  | 88  | 2       |
| GMM      | 474     | 216        | 1000.0 | 1024.8 | 3   | 72      |
| SVM      | 520.8 k | 468.7 k    | 1000.0 | 1007.5 | 16  | 29293   |
| SmallTCN | 10.0 k  | 3.47 M     | 8382.4 | 8382.4 | 26  | 133626  |
| TCN      | 32.8 k  | 11.57 M    | 8382.4 | 8382.4 | 26  | 444974  |
| TCN+LSTM | 388.9 k | 137.26 M   | 8382.4 | 8385.3 | 27  | 5083749 |


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