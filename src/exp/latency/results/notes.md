# CPU streaming-inference latency — results notes

Comparison of per-chunk wall-clock latency, real-time factor, and resident-set
memory for all six classifiers (decision tree, GMM, SVM, TCN, TCN+LSTM,
SmallTCN) under streaming inference on CPU. Drives each model through the
uniform `runner.push(chunk)` interface from `src/demo/runner.py`, the same
path the live demo uses.

Two configurations are reported because torch's default intra-op threading
makes the comparison unfair: NN models get 4 BLAS threads by default while
sklearn classifiers are effectively single-threaded. The single-thread run
is the fair comparison; the four-thread run shows what the demo actually
sees in practice.

Outputs:
- `latency_cpu_t1.{md,svg}` — fair comparison, `torch.set_num_threads(1)`
- `latency_cpu_t4.{md,svg}` — what the demo defaults to, `torch.set_num_threads(4)`

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
   `glibc malloc_trim(0)` + 0.5 s sleep.

Inputs are pre-generated random Gaussian noise (σ=0.3, seeded), allocated
before the timed region so RNG cost stays out of the timer. `time.perf_counter_ns`
gives nanosecond resolution. CUDA is disabled at process start via
`CUDA_VISIBLE_DEVICES=""`. `torch.set_num_threads(N)` is applied immediately
after torch import, before the first runner is constructed.

Reported per model:
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

Hardware: Intel Core i5-8300H @ 2.30 GHz, Linux, governor `powersave`
(observed boosting to 3.9 GHz during the run). Python 3.12, PyTorch 2.8.0
+cu128 (CUDA forced off; GTX 1050 sm_61 is not supported by this wheel).

## Results — single-thread (fair)

`torch.set_num_threads(1)`. CV% on classics drops to 0.6–0.9% — single-thread
runs are far more deterministic. NN CV stays around 17–20%, normal for
torch CPU ops.

| Model | median_ms | p99_ms | cv% | ms/s_audio | rtf | Δrss_mb |
|---|---:|---:|---:|---:|---:|---:|
| GMM | 3.74 | 6.11 | 0.6 | 249.6 | **4.0×** | **3** |
| SmallTCN | 5.81 | 8.06 | 16.8 | 250.3 | 4.0× | 64 |
| TCN | 6.13 | 8.67 | 19.8 | 264.2 | 3.8× | 10 |
| DT | 4.06 | 5.79 | 0.7 | 405.5 | 2.5× | 371 |
| TCN+LSTM | 10.09 | 13.43 | 19.2 | 434.6 | 2.3× | 24 |
| SVM | 6.98 | 10.34 | 0.9 | 465.3 | 2.1× | 9 |

Max absolute RSS observed across the run: **1212 MB** (driven by DT — its
sklearn pickle adds 371 MB on top of the torch baseline).

## Results — four threads (default torch)

`torch.set_num_threads(4)`. Matches what the demo currently sees out of
the box.

| Model | median_ms | p99_ms | cv% | ms/s_audio | rtf | Δrss_mb |
|---|---:|---:|---:|---:|---:|---:|
| TCN | 4.88 | 10.25 | 17.2 | 210.0 | **4.8×** | 13 |
| SmallTCN | 5.65 | 10.74 | 13.8 | 243.2 | 4.1× | 75 |
| GMM | 4.04 | 10.87 | 5.9 | 269.4 | 3.7× | 3 |
| TCN+LSTM | 7.20 | 19.09 | 6.1 | 309.9 | 3.2× | 26 |
| DT | 4.18 | 6.80 | 1.9 | 418.1 | 2.4× | 371 |
| SVM | 7.31 | 14.83 | 3.5 | 487.4 | 2.1× | 9 |

## Analysis

### Threading impact is strongly model-size-dependent

| Model | speedup from 1 → 4 threads |
|---|---:|
| TCN | −1.26 ms (−21%) |
| TCN+LSTM | −2.89 ms (−29%) |
| SmallTCN | −0.16 ms (−3%) |
| classics | ≈ unchanged |

TCN's 16×16 inner matmuls are large enough that BLAS productively splits
the work across cores. SmallTCN's 8×8 matmuls are below the threshold
where threading helps — parallelization overhead roughly cancels the gain.
Sklearn classifiers are not BLAS-bound during prediction (DT branches,
GMM/SVM kernel evaluations) and are unaffected.

### TCN vs SmallTCN ordering flips with threading

| Configuration | Faster |
|---|---|
| 4 threads | TCN faster by 0.77 ms (4.88 vs 5.65) |
| 1 thread | SmallTCN faster by 0.32 ms (5.81 vs 6.13) |

The two architectures have opposing cost profiles:

- **SmallTCN frontend tax (fixed)**: `log_mel_delta2` runs `compute_deltas`
  twice per push (Δ and Δ²); the first conv then projects 240 input
  channels into 8 filters. First-conv FLOPs ≈ 240·8·5·T = 9600·T vs TCN's
  80·16·5·T = 6400·T — 50% wider despite fewer filters.
- **TCN body tax (parallelizable)**: 12 inner residual blocks at 16×16 =
  256 ops/element/layer; SmallTCN's 12 inner blocks are 8×8 = 64
  ops/element/layer — 4× heavier per layer.

At 4 threads, TCN's body tax is cut ~25% by parallelism, so the
frontend tax dominates and SmallTCN looks slower. At 1 thread, TCN can't
parallelize, so the body tax dominates and SmallTCN's lighter body
finally wins out — barely, since the frontend tax still eats most of the
savings. The 1-thread comparison reflects what each architecture
actually does per push.

### Memory is dominated by sklearn pickles, not weights

`Δrss_mb` (marginal cost on top of the pre-load baseline):

- DT: 371 MB — sklearn pickle is by far the heaviest single artefact in
  the experiment.
- SmallTCN: 64–75 MB — variant module imports + model construction.
- TCN+LSTM: 24–26 MB — LSTM tail adds small overhead on top of TCN.
- TCN: 10–13 MB — paper baseline, lightest NN.
- SVM: 9 MB.
- GMM: 3 MB.

TCN's *weights* file is small (~250 KB safetensors), but module imports
(torch, numpy, librosa, etc.) sit at a fixed ~600 MB baseline. NN models
all share that baseline; classics avoid it (GMM and SVM don't import
torch at all in the runner path). The absolute RSS observed across the
whole 6-model run is 1212 MB, set by DT's pickle, not by any NN model.

### Tail latency

p99 is ~1.5–2× the median for most models. The gap is widest on the NN
side at 4 threads (TCN+LSTM p99 = 19.1 ms vs median 7.2 ms — torch BLAS
threading introduces sporadic contention spikes). On 1 thread the median–p99
gap narrows for all models, including a 6.6 ms p99 reduction on TCN+LSTM.
For real-time streaming where the per-chunk budget is `chunk_ms`
(23.2 ms for NN family), p99 well under that budget on every model means
no missed deadlines under any tested configuration.

### Real-time headroom

All six models clear 2× real-time on this CPU under both threading
configurations. The slowest configuration tested is SVM and TCN+LSTM at
~2.1–2.3× — still comfortable headroom for live capture but the smallest
margin if a CPU is shared with other workloads. TCN, SmallTCN, and GMM
all sit near 4× under fair comparison — three usable picks at very
different memory footprints (GMM 3 MB / SmallTCN 64 MB / TCN 10 MB).

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

All six classifiers run comfortably faster than real time (≥2.1×) on a
mid-range laptop CPU. The choice of model under tight memory or
deterministic-latency constraints is now informed:

- **GMM** is the clear footprint winner — 3 MB marginal RSS, 4.0× RT,
  and 0.6% CV. Tiny, fast, predictable.
- **TCN** is the fastest NN under fair comparison (3.8× RT) and the
  lightest of the NN family in marginal RSS (10 MB).
- **SmallTCN** is essentially tied with TCN on 1 thread (4.0× vs 3.8×
  RT). Its parameter-count advantage doesn't translate into a streaming
  speed advantage — the Δ² frontend cancels most of the body savings.
- **TCN+LSTM** and **SVM** are the slowest tested (2.3× and 2.1× RT
  respectively) and should be avoided where CPU is shared with other
  workloads.
- **DT** has the highest memory cost by a large margin (371 MB pickle)
  and should be picked only when its accuracy advantage is needed.

Threading configuration matters: switching the demo to
`torch.set_num_threads(1)` removes the torch-BLAS advantage that
distorts the comparison and slows TCN/TCN-LSTM by 21–29%. For a
deployment where the CPU is shared, single-threaded torch gives
predictable per-chunk latency at a real-world cost worth knowing about.
