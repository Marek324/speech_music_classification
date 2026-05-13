# CPU complexity analysis — results notes

Comparison of all seven classifiers (decision tree, GMM, SVM, TCN, TCN+LSTM,
SmallTCN, SmallerTCN) on the metrics relevant to using each as a **front-of-pipeline
gate**: per-frame compute, persistent state, wall-clock latency, RTF, RSS,
and quality (F1_macro) — combined into a single Pareto picture. Drives each
model through the uniform `runner.push(chunk)` interface from
`src/demo/runner.py`, the same path the live demo uses.

The CPU-only methodology is by design: classics live on sklearn (CPU only),
and torch's default intra-op threading would otherwise hand the NN family
a 4-thread head-start. We pin to 1 thread for the fair head-to-head, then
sweep N ∈ {1, 2, 4, 8} for the multi-thread story.

Outputs:
- `complexity_t1.{md,svg}` — fair single-thread comparison, `torch.set_num_threads(1)`
- `complexity_scale.{md,svg}` — thread-scaling sweep with Amdahl `p` per model

## Methodology

Each model is benchmarked in isolation with deep cleanup between models so
peak RSS is bounded by `max(model)` rather than `sum(model)`.

Per-model loop:
1. Build the runner; record `load_ms` (one-shot model construction time).
2. Warmup with 10 untimed `push()` calls (covers torch lazy init, sklearn
   first-call overhead, streaming receptive-field fill).
3. Run 5 trials × 200 timed pushes (1000 pooled samples). `gc.disable()`
   inside each trial; `gc.collect()` between trials.
4. Tear down: `runner.close()`, `del runner`, then `gc.collect()` × 3 +
   glibc `malloc_trim(0)` + 0.5 s sleep.

Inputs are pre-generated random Gaussian noise (σ=0.3, seeded), allocated
before the timed region so RNG cost stays out of the timer. `time.perf_counter_ns`
gives nanosecond resolution. CUDA is disabled at process start via
`CUDA_VISIBLE_DEVICES=""`. `torch.set_num_threads(N)` is applied immediately
after torch import, before the first runner is constructed.

Empirical metrics per model:
- `load_ms` — one-shot construction + weight load.
- `chunk_ms` — audio duration per push (varies: DT 10 ms, GMM/SVM 15 ms,
  NN family 23.2 ms).
- `median_ms` / `p99_ms` — pooled across 1000 timed pushes.
- `cv%` — coefficient of variation across the 5 trial medians; reliability
  check.
- `ms/s_audio` — fair cross-model metric, `median_ms / chunk_ms × 1000`.
- `rtf` — real-time factor, `1000 / ms/s_audio`. `>1×` means the model
  keeps up with a live mic.
- `peak_rss_mb` — absolute peak RSS during the model's run.
- `Δrss_mb` — marginal RSS, peak minus pre-load baseline. The stable
  per-model footprint metric.
- `residual_mb` — what survived the deep cleanup.
- `F1_macro` — joined from `results/<model>.eval` for the Pareto plot.

Symbolic metrics per model (closed-form from the loaded model object,
no measurement involved):
- `params` — fitted classifier parameters (tree nodes, GMM means/covs,
  support vectors, NN weights — all counted as floats).
- `MACs/frame` — multiply-accumulates per output frame in **streaming**
  mode (one push). For the TCN family this includes the receptive-field
  re-run that `StreamingInference` performs on each call.
- `RF_ms` — receptive field in milliseconds: 300 ms for DT, 1000 ms for
  GMM/SVM (set by `lt_len_ms` in the `FeatExtractor`), ≈ 8.4 s for the
  full TCN family, ≈ 2.8 s for SmallerTCN (its single-stack RF is
  121 frames vs 361 for the others).
- `buf_ms` — audio-equivalent duration of all persistent streaming state
  (signal/feature ring buffer + LSTM h/c + smoothing windows), divided by
  the model's own sample rate. Nearly equal to `RF_ms` because the audio
  ring buffer dominates; the small remainder (h/c, smoothing) adds only a
  few ms.
- `T∞` — critical path: the longest serial chain of ops per frame.
- `T₁/T∞` — algorithmic max parallelism. DT scores ≈ 2 (intrinsically
  serial); the TCN family scores 10⁵–10⁶ (huge in theory) but BLAS at
  batch=1 only realises a fraction of it (see thread-scaling sweep).

Hardware: Intel Core i5-8300H @ 2.30 GHz, Linux, governor `powersave`
(observed boosting to 3.9 GHz during the run). Python 3.12, PyTorch 2.8.0
+cu128 (CUDA forced off; GTX 1050 sm_61 is not supported by this wheel).

## Results — single-thread (fair)

`torch.set_num_threads(1)`. CV% on classics drops to 0.9–10% — single-thread
runs are far more deterministic. NN CV stays around 16–22%, normal for
torch CPU ops.

| Model | median_ms | p99_ms | cv% | ms/s_audio | rtf | Δrss_mb | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMM | 3.94 | 8.43 | 10.0 | 262.6 | 3.8× | **3** | 0.825 |
| DT | 4.15 | 7.47 | 0.9 | 414.5 | 2.4× | 368 | 0.838 |
| SmallerTCN | 4.38 | 8.73 | 21.5 | 188.6 | **5.3×** | 13 | 0.972 |
| SmallTCN | 6.88 | 11.69 | 16.2 | 296.1 | 3.4× | 64 | 0.977 |
| TCN | 7.18 | 14.43 | 19.1 | 309.3 | 3.2× | 8 | 0.975 |
| SVM | 7.66 | 14.51 | 3.2 | 510.8 | 2.0× | 9 | 0.875 |
| TCN+LSTM | 11.07 | 31.35 | 19.8 | 476.6 | 2.1× | 23 | **0.985** |

Max absolute RSS observed across the run: **1208 MB** (driven by DT — its
sklearn pickle adds 368 MB on top of the torch baseline).

## Results — thread scaling (1 → 8)

`torch.set_num_threads(N)` per model, N ∈ {1, 2, 4, 8}.

| Model | t1_ms | t2_ms | t4_ms | t8_ms | S(8) | Amdahl p |
|---|---:|---:|---:|---:|---:|---:|
| DT | 4.32 | 4.48 | 4.49 | 4.29 | 1.01× | −0.04 |
| GMM | 4.00 | 4.11 | 4.19 | 4.27 | 0.94× | −0.07 |
| SVM | 7.48 | 7.50 | 8.29 | 7.35 | 1.02× | −0.04 |
| SmallTCN | 7.25 | 5.87 | 5.34 | 7.17 | 1.01× | 0.25 |
| SmallerTCN | 4.73 | 3.79 | 3.69 | 3.88 | 1.22× | 0.30 |
| TCN | 7.35 | 5.80 | 5.25 | 6.18 | 1.19× | 0.33 |
| TCN+LSTM | 12.24 | 8.80 | 7.50 | 8.69 | 1.41× | 0.47 |

**S(N)** = t(1) / t(N) — observed parallel speedup at thread count N.
**amdahl_p** is the parallel fraction fitted to all (N, S(N)) pairs;
≈0 means single-threaded by construction, ≈1 means linear scaling,
**negative** means anti-scaling (threading overhead beats any parallel
gain, as happens with sklearn classifiers that aren't BLAS-bound during
predict).

## Analysis

### SmallerTCN is a Pareto winner

Cutting `n_stacks` from 3 to 1 in the SmallTCN recipe collapses receptive
field from 8.4 s to 2.8 s, but on the test set this costs 0.005 macro F1
(0.972 vs 0.977) — and in exchange:

- **5.3× real-time** at 1 thread, the highest of any model.
- **13 MB Δrss** vs 64 MB for SmallTCN (delta² preprocessor module imports
  dominate; SmallerTCN inherits less of that overhead because of stack=1
  weight allocation patterns — see `peak_rss` 909 vs 910 MB which is
  effectively tied, but the residual gap survives cleanup).
- **545 K MACs/frame** vs 3.47 M for SmallTCN — ~6.4× cheaper streaming
  compute, which scales linearly with the stack count drop.

If the application can absorb the smaller receptive field, SmallerTCN is the
fastest neural option.

### Threading impact is strongly model-size-dependent

| Model | speedup from 1 → 4 threads |
|---|---:|
| TCN+LSTM | −4.74 ms (−39%) |
| TCN | −2.10 ms (−29%) |
| SmallTCN | −1.91 ms (−26%) |
| SmallerTCN | −1.04 ms (−22%) |
| classics | ≈ unchanged (DT ±0.2, GMM +5%, SVM +11% slowdown) |

TCN+LSTM's 64×64 LSTM matmuls on top of TCN body benefit most from
parallelism. TCN's 16×16 inner matmuls also split productively. SmallTCN/
SmallerTCN's 8×8 inner matmuls are smaller but still benefit modestly.
Sklearn classifiers are not BLAS-bound during prediction (DT branches,
GMM/SVM kernel evaluations) and either don't move or slightly regress with
threading (oversubscription overhead).

### Memory is dominated by sklearn pickles, not weights

`Δrss_mb` (marginal cost on top of the pre-load baseline):

- DT: 368 MB — sklearn pickle is by far the heaviest single artefact in
  the experiment.
- SmallTCN: 64 MB — variant module imports (delta² compute, SmallTCN
  weights) + model construction.
- TCN+LSTM: 23 MB — LSTM head adds small overhead on top of TCN.
- SmallerTCN: 13 MB — same delta² preprocessor as SmallTCN but a leaner
  body (stack=1).
- SVM: 9 MB.
- TCN: 8 MB — paper baseline, lightest NN.
- GMM: 3 MB.

TCN's *weights* file is small (~250 KB safetensors), but module imports
(torch, numpy, librosa, etc.) sit at a fixed ~600 MB baseline. NN models
all share that baseline; classics avoid it (GMM and SVM don't import
torch at all in the runner path). The absolute RSS observed across the
whole 7-model run is 1208 MB, set by DT's pickle, not by any NN model.

### Tail latency

p99 is ~1.5–3× the median for most models. The gap is widest on TCN+LSTM
(p99 = 31.4 ms vs median 11.1 ms — torch BLAS at batch=1 introduces
sporadic contention spikes). For real-time streaming where the per-chunk
budget is `chunk_ms` (23.2 ms for NN family), every model except TCN+LSTM
clears p99 well under that budget. TCN+LSTM at p99 = 31.4 ms exceeds
`chunk_ms` — under sustained mic input it would occasionally fall a chunk
behind, which the pipeline absorbs via a small jitter buffer but is the
worst-case among the seven.

### Real-time headroom

All seven models clear 2× real-time on this CPU at 1 thread. Ranking by
RTF: SmallerTCN (5.3×) > GMM (3.8×) > SmallTCN (3.4×) > TCN (3.2×) > DT
(2.4×) > TCN+LSTM (2.1×) > SVM (2.0×). SVM and TCN+LSTM share the smallest
margin and should be avoided where the CPU is shared with other workloads.

## Limitations

1. **Single hardware target**. All numbers are from one i5-8300H. Server
   CPUs with larger caches and AVX-512 would shift absolute timings; the
   ordering effects (threading dependence, frontend-vs-body tradeoff)
   should generalise.

2. **Single benchmark invocation**. Internal CV is reported per-model,
   but inter-process variance is unmeasured. Five trials within a single
   process is the disclosed scope.

3. **Random Gaussian inputs**. Sklearn DT branching and to a lesser
   extent SVM kernel evaluation are data-dependent. Real audio would
   exercise different code paths; bias is expected to be small but
   non-zero. NN models are not affected (their compute is input-shape
   dependent only).

4. **GC disabled inside the timed loop**. This gives best-case latency.
   The live demo runs with GC enabled, so the demo's actual p99 will be
   somewhat worse than what the bench reports.

5. **CPU governor `powersave`, not `performance`**. Frequency was observed
   boosting to 3.9 GHz during the run, but a longer thermally-loaded
   workload could downclock. The bench is short (~30 s per threading
   configuration) so this likely did not occur.

6. **One trained checkpoint per model**. DT depth and SVM support-vector
   count depend on training seed; a different DT could have different
   latency. This experiment measures the deployed checkpoints, not the
   model families in general.

## Conclusion

All seven classifiers run comfortably faster than real time (≥2.0×) on a
mid-range laptop CPU. The choice of model under tight memory or
deterministic-latency constraints is now informed:

- **SmallerTCN** is the new RTF winner — 5.3× RT at 1 thread, 13 MB
  marginal RSS, 0.972 macro F1 (within 0.005 of SmallTCN). The 2.8 s
  receptive field is the only caveat.
- **GMM** remains the footprint winner — 3 MB marginal RSS, 3.8× RT,
  10% CV. Tiny, fast, predictable; the F1 ceiling (0.83) is the cost.
- **TCN** is the lightest of the full-RF NN family in marginal RSS
  (8 MB) and runs at 3.2× RT.
- **SmallTCN** sits between SmallerTCN and TCN on the cost axis (64 MB
  / 3.4× RT) with 0.977 macro F1.
- **TCN+LSTM** is the F1 leader (0.985) but the slowest among NN models
  (2.1× RT, p99 31 ms exceeding chunk budget) and adds 23 MB marginal
  RSS.
- **DT** has the highest memory cost by a large margin (368 MB pickle)
  and should be picked only when its accuracy is needed.
- **SVM** is the slowest tested (2.0× RT) and should be avoided where
  CPU is shared with other workloads.

Threading configuration matters: switching the demo to
`torch.set_num_threads(1)` removes the torch-BLAS advantage that
distorts the comparison and slows TCN/TCN+LSTM by 26–39%. For a
deployment where the CPU is shared, single-threaded torch gives
predictable per-chunk latency at a real-world cost worth knowing about.
