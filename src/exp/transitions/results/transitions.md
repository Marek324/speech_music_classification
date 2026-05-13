# Transition analysis — switching recordings

## Methodology

### Source

Reads `_scores.npz` files written by `smclassifier exp critical eval`
(`src/exp/critical/results/`). Switching clips are identified by the
`subclass` column with prefix `speech_switching_`:

- **2-class** (Speech ↔ Music) — `speech_switching_<cadence>ms`
- **3-class** (Speech ↔ Music ↔ Background) — `speech_switching_3class_<cadence>ms`

Cadences analysed: 500 ms, 1000 ms, 2000 ms, 4000 ms. Models: TCN, TCN+LSTM, SmallTCN, SmallerTCN, DT, GMM, SVM.

Each model has its own per-frame y_true / y_pred at its own hop
(NN: 23.22 ms; SVM/GMM: 15 ms; DT: 10 ms). Latency and flicker are
computed in each model's frame grid and reported in ms / Hz so they
are directly comparable.

### Latency

For each GT change at frame `t` with new label `B`, find the smallest
`k ≥ 0` such that `y_pred[t + k] == B`; the latency is `k · hop_ms`.
Search halts at the next GT change so a delayed match cannot leak
into the next stable segment. If the model never matches `B` before
the next change, the event is recorded as a **miss** and excluded
from the median — capping miss latencies at the segment length
would compress slow models toward the cadence and hide their real
cost. The miss column reports the fraction of GT changes that the
model failed to follow within one segment.

### Flicker

Inside each stable GT segment, the first **200 ms**
are excluded as a settling margin (latency-related transients are
not flicker). In the remainder, every `y_pred[i] != y_pred[i-1]` is
counted; the rate is reported in Hz of stable audio. Segments
shorter than the settling margin contribute nothing.

### Outputs

- `transitions.md` — this report
- `graphs/transition_summary.svg` — grouped-bar median latency &
  flicker, per cadence, split by 2-class vs 3-class
- `graphs/latency_cdf.svg` — latency CDF per cadence (2-class and
  3-class pooled)

## Results

## Speech ↔ Music

_Clips per cadence: 500ms × 1, 1000ms × 1, 2000ms × 1, 4000ms × 1._

**Median latency (ms) — `med (p90), miss%`**

| model | 500 ms | 1000 ms | 2000 ms | 4000 ms |
|---|---|---|---|---|
| TCN | 46 (p90 376), miss 12% | 70 (p90 464), miss 5% | 139 (p90 868), miss 4% | 70 (p90 794), miss 6% |
| TCN+LSTM | 0 (p90 70), miss 40% | 0 (p90 632), miss 29% | 0 (p90 1087) | 23 (p90 1156), miss 6% |
| SmallTCN | 23 (p90 279), miss 18% | 46 (p90 627), miss 5% | 58 (p90 557), miss 4% | 23 (p90 511), miss 6% |
| SmallerTCN | 46 (p90 302), miss 3% | 46 (p90 418) | 46 (p90 348) | 23 (p90 418) |
| DT | 30 (p90 175), miss 3% | 40 (p90 129) | 30 (p90 150) | 35 (p90 135) |
| GMM | 0 (p90 168), miss 44% | 210 (p90 825), miss 13% | 180 (p90 930), miss 9% | 180 (p90 885), miss 6% |
| SVM | 0 (p90 0), miss 51% | 322 (p90 840), miss 16% | 345 (p90 878), miss 22% | 345 (p90 819), miss 25% |

**Stable-region flicker (Hz)**

| model | 500 ms | 1000 ms | 2000 ms | 4000 ms |
|---|---|---|---|---|
| TCN | 2.77 | 2.87 | 0.68 | 0.45 |
| TCN+LSTM | 0.74 | 0.91 | 1.12 | 1.02 |
| SmallTCN | 1.60 | 1.32 | 1.28 | 0.76 |
| SmallerTCN | 3.45 | 1.96 | 1.28 | 0.87 |
| DT | 13.49 | 12.04 | 11.68 | 12.30 |
| GMM | 0.99 | 1.23 | 1.12 | 1.10 |
| SVM | 0.80 | 1.37 | 0.72 | 0.76 |

## Speech ↔ Music ↔ Background

_Clips per cadence: 500ms × 1, 1000ms × 1, 2000ms × 1, 4000ms × 1._

**Median latency (ms) — `med (p90), miss%`**

| model | 500 ms | 1000 ms | 2000 ms | 4000 ms |
|---|---|---|---|---|
| TCN | 93 (p90 302), miss 35% | 313 (p90 706), miss 37% | 197 (p90 1103), miss 20% | 70 (p90 636), miss 13% |
| TCN+LSTM | 46 (p90 174), miss 27% | 23 (p90 808), miss 23% | 70 (p90 1751), miss 10% | 0 (p90 1110), miss 13% |
| SmallTCN | 128 (p90 320), miss 33% | 186 (p90 580), miss 46% | 244 (p90 694), miss 30% | 12 (p90 511), miss 20% |
| SmallerTCN | 70 (p90 251), miss 32% | 232 (p90 553), miss 31% | 46 (p90 627), miss 20% | 23 (p90 460), miss 13% |
| DT | 35 (p90 280), miss 17% | 65 (p90 415), miss 9% | 50 (p90 370) | 40 (p90 238) |
| GMM | 0 (p90 105), miss 59% | 420 (p90 820), miss 3% | 600 (p90 954) | 165 (p90 900) |
| SVM | 0 (p90 405), miss 51% | 585 (p90 807), miss 9% | 652 (p90 857), miss 10% | 338 (p90 754), miss 20% |

**Stable-region flicker (Hz)**

| model | 500 ms | 1000 ms | 2000 ms | 4000 ms |
|---|---|---|---|---|
| TCN | 2.13 | 1.99 | 0.55 | 0.45 |
| TCN+LSTM | 1.71 | 1.28 | 1.09 | 0.82 |
| SmallTCN | 2.07 | 2.35 | 1.44 | 0.67 |
| SmallerTCN | 1.77 | 1.99 | 1.05 | 0.52 |
| DT | 15.85 | 13.73 | 13.23 | 13.74 |
| GMM | 0.95 | 2.08 | 1.29 | 0.79 |
| SVM | 2.48 | 2.74 | 1.25 | 0.86 |

