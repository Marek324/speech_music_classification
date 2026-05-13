# Switch latency — switching crit clips

## Methodology

Each model streams every switching clip (2-class and 3-class pooled)
via its `src/demo/runner.py` wrapper, after a per-clip reset and a
silence prefill so feature rings and smoothing buffers are saturated
before measurement. **Single-thread CPU** (`torch.set_num_threads(1)`
+ `OMP_NUM_THREADS=1`) so every model sees the same compute budget.

- Silence prefill: **1500 ms** of zeros at
  the model's native sample rate.
- Consecutive-match window: **K = 3** hops.
  A transition is flipped at the smallest `k ≥ 0` such that
  `y_pred[t+k : t+k+K]` all equal the new GT class. Search halts at
  the next GT change.
- **Latency** (ms) = `k × hop_ms` using each model's native hop
  (DT 10 ms, GMM/SVM 15 ms, NN family 23.22 ms).
- **Miss** (%) — fraction of GT transitions that never flipped before
  the next GT change. Excluded from the median.
- **Drift** (%) — fraction where the streamed prediction at the hop
  just before the transition was not the *old* GT class. High drift
  means the model wasn't on the right answer to begin with, and the
  latency number is biased low for those events.

Cells read `med (p90) [miss%, drift%]`. Miss% omitted under 0.5%,
drift% omitted under 5%.

## Pooled across directions (2-class + 3-class)

_Clips per cadence: 500ms × 16, 1000ms × 16, 2000ms × 16, 4000ms × 16._

| model | 500 ms | 1000 ms | 2000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| DT | 115 (p90 340), miss 18%, drift 33% | 120 (p90 360), miss 14%, drift 35% | 150 (p90 396), miss 9%, drift 40% | 140 (p90 890), miss 3%, drift 35% |
| GMM | 0 (p90 135), miss 58%, drift 62% | 0 (p90 867), miss 50%, drift 53% | 825 (p90 1328), miss 35%, drift 46% | 908 (p90 2031), miss 34%, drift 34% |
| SVM | 0 (p90 60), miss 57%, drift 57% | 300 (p90 855), miss 33%, drift 37% | 360 (p90 1290), miss 16%, drift 30% | 555 (p90 1725), miss 10%, drift 19% |
| TCN | 0 (p90 325), miss 44%, drift 44% | 255 (p90 627), miss 34%, drift 37% | 511 (p90 1379), miss 16%, drift 20% | 604 (p90 1588), miss 12%, drift 30% |
| TCN-S | 70 (p90 302), miss 40%, drift 42% | 302 (p90 697), miss 19%, drift 28% | 464 (p90 882), miss 15%, drift 26% | 511 (p90 1335), miss 9%, drift 30% |
| TCN-L | 23 (p90 302), miss 44%, drift 46% | 70 (p90 627), miss 37%, drift 42% | 395 (p90 1609), miss 24%, drift 35% | 998 (p90 2192), miss 21%, drift 30% |

## Per direction (median ms)

2-class clips contribute to `speech↔music` only; 3-class clips contribute to all six directions.

| model | cadence | background → speech | music → background | music → speech | speech → music |
|---|---:|---:|---:|---:|---:|
| DT | 500 ms | 20 | 240 | 40 | 240 |
| DT | 1000 ms | 15 | 270 | 40 | 210 |
| DT | 2000 ms | 15 | 310 | 50 | 250 |
| DT | 4000 ms | 0 | 890 | 60 | 250 |
| GMM | 500 ms | 0 | 0 | 0 | 0 |
| GMM | 1000 ms | 405 | 180 | 0 | 592 |
| GMM | 2000 ms | 968 | 0 | 795 | 1102 |
| GMM | 4000 ms | 915 | 0 | 840 | 1688 |
| SVM | 500 ms | 0 | 345 | 0 | 0 |
| SVM | 1000 ms | 270 | 150 | 120 | 810 |
| SVM | 2000 ms | 218 | 0 | 278 | 1050 |
| SVM | 4000 ms | 210 | 0 | 278 | 1080 |
| TCN | 500 ms | 0 | 0 | 0 | 325 |
| TCN | 1000 ms | 290 | 0 | 0 | 557 |
| TCN | 2000 ms | 360 | 0 | 406 | 1091 |
| TCN | 4000 ms | 360 | 0 | 418 | 1207 |
| TCN-S | 500 ms | 116 | 46 | 46 | 255 |
| TCN-S | 1000 ms | 279 | 46 | 232 | 511 |
| TCN-S | 2000 ms | 348 | 23 | 337 | 557 |
| TCN-S | 4000 ms | 348 | 58 | 372 | 662 |
| TCN-L | 500 ms | 116 | 0 | 0 | 0 |
| TCN-L | 1000 ms | 348 | 0 | 46 | 580 |
| TCN-L | 2000 ms | 418 | 0 | 348 | 1300 |
| TCN-L | 4000 ms | 418 | 46 | 685 | 1765 |

