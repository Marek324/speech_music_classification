# Transition analysis — methodology & results

## 1. Objective

Compare the six classifiers on behaviour *around class changes*, where per-frame macro-F1 does not tell the full story: a model can score well on stable-label frames yet react sluggishly when the true label flips, or produce a stable per-frame accuracy by flipping its prediction constantly. Three qualities are isolated:

- **How fast** does each model respond to a true class change?
- **How noisy** is each model inside stable regions (spurious flips)?
- **How much accuracy is lost in the neighbourhood of transitions** versus stable regions?

## 2. Data

The saved `results/{model}_scores.npz` files from the headline evaluation (test split of `Marek324/speech-music-classification`, full tier). Each file carries `y_true` and the `(N, 3)` score matrix; the patched evaluator (`src/evaluator.py:379–388`) additionally writes `clip_ids`, `subclasses`, and `y_pred`. Predictions are reconstructed via `argmax` over `[speech, music, inactive]` if `y_pred` is absent.

Frame rates are read from this experiment's `config.toml`:

| Model | Frame period |
|---|---|
| DT | **10.00 ms** |
| GMM, SVM | **15.00 ms** |
| TCN, TCN+LSTM, SmallTCN | **23.22 ms** |

All latencies are reported in milliseconds so that models on different frame grids are directly comparable.

### Mode tag — `v1` vs `v2`

Each row of every results table carries a `Mode` column:
- **v1 (no clip_ids)** — the model's `_scores.npz` predates the evaluator patch. Transitions include concatenation boundaries between consecutive test clips alongside true within-clip label changes. Latencies are inflated slightly because some I→S/I→M events are concatenation artefacts that resolve at the first frame.
- **v2 (within-clip)** — the npz carries `clip_ids`. Transitions are restricted to within-clip changes; the latency search halts at each clip boundary so unresolved events cannot leak into the next clip.

The current run is **all v1** because no `_scores.npz` has been refreshed since the patch. See §9 for the re-evaluation procedure.

## 3. Transition definition

A **true transition** is any frame `t ≥ 1` where `y_true[t] ≠ y_true[t-1]`. The `(y_true[t-1], y_true[t])` pair is the *transition type*. Only four pairs occur on this test split — Speech↔Music are absent, because different-class clips are always separated by an Inactive segment in the dataset construction.

| Transition | TCN grid count | DT grid count |
|---|---:|---:|
| Speech → Inactive | 3 231 | 3 231 |
| Inactive → Speech | 3 231 | 3 231 |
| Music → Inactive | 1 094 | 1 281 |
| Inactive → Music | 1 094 | 1 281 |

(The discrepancy comes from the different frame grids — a short segment that rounds to zero frames on the 23.22 ms grid can produce a detectable pair on the 10 ms grid.)

## 4. Metrics

### 4.1 Detection latency — end-of-frame convention

For a transition at frame `t` with true label `B`, define `k` as the smallest non-negative integer such that `y_pred[t+k] = B`. The transition is at the boundary `t · hop_ms` in the clip's timeline, and the model emits its prediction for frame `t+k` at the *end* of that frame, namely `(t+k+1) · hop_ms`. The exact wall-clock latency is therefore:

```
latency_ms = (k + 1) · hop_ms
```

Pictorially, with `*` marking the transition and `▼` marking when the matching prediction becomes available:

```
   frame t-1   |   frame t    |  frame t+1   |  frame t+2  ...
─────────A────*│──────B───────│──────B───────│──────B─────  y_true
              │   pred=A      │   pred=A     │   pred=B    y_pred
              │               │              ▼
              └─ transition   └─ frame end   └─ first matching emit
                   (t·hop)         (t+1)·hop      (t+k+1)·hop, k=2

                        ◄────────  latency  ────────►
                                   (k+1)·hop_ms
```

This is what a listener experiences in the streaming demo. The earlier write-up of this experiment used `k · hop_ms` (start of the matching frame), which produced an artefactual "0 ms median" for the NN family. The numbers below have been recomputed under the correct convention.

The latency search is right-censored at the configured cap (default 2000 ms). With end-of-frame, the search depth is `k_max = floor(cap_ms / hop_ms) - 1` so the largest non-missed latency is exactly `(k_max + 1) · hop_ms ≤ cap_ms`. Unresolved events are reported as missed; the **miss rate** is the fraction hitting the cap. In v2 mode, the search additionally halts at the next clip boundary; events whose clip ends before resolution are also counted as missed (with the recorded latency = clip-end window size, ≤ cap_ms).

### 4.2 False transition rate

Inside **stable regions** — frames at least `stable_guard_ms` (default 500 ms) away from any true transition — every frame where `y_pred[t] ≠ y_pred[t-1]` is counted as a spurious flip. The rate is normalised to flips per minute of stable audio, both overall and conditioned on the stable true label.

The guard zone uses *all* true transitions, not just within-clip ones, so v1 and v2 modes share the same definition of "stable region".

### 4.3 Near-vs-stable accuracy

Frame accuracy is computed over two disjoint masks defined relative to *all* true transitions:

- **Near-transition:** frames within ±`boundary_window_ms` (default 500 ms) of any transition.
- **Stable:** frames at least `stable_guard_ms` away from any transition.

The gap `Δ = acc_near − acc_stable` is the per-model boundary penalty.

### 4.4 Transition-type breakdown

All three metrics are also broken down per `(from, to)` pair to expose directional asymmetry (e.g. music-offset is harder than music-onset).

## 5. Results — current run (v1, end-of-frame)

### 5.1 Summary

| Model | Mode | Transitions | Median latency | p90 | p99 | Miss % | False trans./min | Acc. near | Acc. stable | Δ acc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TCN+LSTM | v1 | 8 650 | 23 ms | 1 997 | 1 997 | 11.9 % | 2.86 | 0.9497 | 0.9944 | −0.0447 |
| SmallTCN | v1 | 8 650 | 23 ms | 1 997 | 1 997 | 11.6 % | 8.04 | 0.9284 | 0.9898 | −0.0614 |
| TCN | v1 | 8 650 | 46 ms | 1 997 | 1 997 | 11.1 % | 7.84 | 0.9146 | 0.9918 | −0.0772 |
| **DT** | v1 | 9 024 | **20 ms** | 90 | 2 000 | **2.3 %** | 1 004.38 | 0.7671 | 0.8486 | −0.0815 |
| SVM | v1 | 9 024 | 120 ms | 1 995 | 1 995 | 21.4 % | 29.24 | 0.6987 | 0.9188 | −0.2201 |
| GMM | v1 | 9 024 | 120 ms | 1 995 | 1 995 | 14.5 % | 41.56 | 0.6538 | 0.8682 | −0.2144 |

### 5.2 Latency by transition type (median, ms)

| Model | S → I | M → I | I → S | I → M |
|---|---:|---:|---:|---:|
| TCN+LSTM |  23 | 1 997 |  23 |  23 |
| SmallTCN |  23 | 1 997 |  23 |  23 |
| TCN |  23 | 1 997 |  70 |  23 |
| DT |  20 |    60 |  20 |  20 |
| SVM | 615 | 1 995 |  15 |  15 |
| GMM | 285 | 1 995 |  45 |  15 |

### 5.3 False transitions per minute (stable regions, ±500 ms guard)

| Model | Overall | During speech | During music | During inactive |
|---|---:|---:|---:|---:|
| TCN+LSTM |  2.86 |  6.63 |  1.90 | 0.31 |
| SmallTCN |  8.04 | 11.00 |  8.69 | 3.48 |
| TCN |  7.84 | 16.18 |  6.21 | 1.25 |
| DT | 1 004.38 | 1 518.39 | 914.88 | 568.60 |
| SVM | 29.24 | 34.12 | 35.63 | 12.13 |
| GMM | 41.56 | 61.19 | 47.88 |  7.58 |

## 6. Interpretation

**Decision Tree wins on detection latency once frame resolution is honoured.** DT's 10 ms hop gives it a 20 ms end-of-frame median — faster than any other model, including the TCN family at 23 ms. The earlier "0 ms median" headline for the NN family was an artefact of the start-of-frame convention; it disappears here. DT's miss rate of 2.3 % is also the lowest of the six. The catch is its catastrophic false-transition rate — over 1 000 spurious flips per minute — which rules it out for streaming use without a smoothing layer.

**LSTM smoothing is the decisive ingredient for stability among the NN family.** TCN+LSTM more than halves the false-transition rate of the base TCN (2.86 vs 7.84 per minute) and has the smallest near-boundary accuracy drop (−4.5 pp vs −7.7 pp). The improvement is concentrated in the hardest condition — during speech (16.18 → 6.63 flips/min) and during music (6.21 → 1.90 flips/min) — exactly where the TCN's 181-frame receptive field alone is not enough to stabilise the decision.

**SmallTCN remains surprisingly competitive on latency.** Its medians match TCN+LSTM (23 ms) on every active pair; its false-transition rate (8.04/min) is essentially on par with the base TCN. The capacity reduction costs almost nothing at transitions — only stable-region accuracy suffers slightly.

**TCN's I→S median is one hop slower than the rest.** 70 ms (3 hops) versus 23 ms (1 hop) for the LSTM and Small variants. The base TCN takes longer to commit to "speech now starting"; LSTM smoothing and SmallTCN's parameter reduction both apparently encourage earlier commitment, perhaps because they are less able to model the slow-onset speech-rise pattern that the base TCN learns to wait through.

**SVM's miss rate (21.4 %) is the notable outlier.** Almost twice the TCN family's. Its S→I median of 615 ms is also the worst of the six. Both are consistent with the 1 s moving-window feature extraction — the SVM cannot see a speech-offset as a change until the 1 s window has rolled past it.

**GMM tracks SVM closely on every axis** (median 120 ms, false-trans/min 41.6, Δ acc. −21.4 pp). Both classic-density classifiers sit at roughly the same operating point; the SVM's higher miss rate is the one place they meaningfully differ.

## 7. The M → I anomaly

Every model has its 90th-percentile M→I latency pinned at the cap. Two non-exclusive explanations:

1. **The test set does not give models enough time to resolve the offset.** Inactive gaps following music clips are likely shorter than 2 s on average, so the `M → I → M` cycle completes before any of the models switch out of music.
2. **All models genuinely struggle on music offsets** (reverb tails, decaying notes, sustained instrumental pauses are acoustically music-like).

The miss rate (~12 % for neural models, ~14–21 % for SVM/GMM, but only 2.3 % for DT) aligns with a bimodal distribution: easy M→I events resolve quickly, hard ones never resolve before the next music starts. DT is the exception only because its per-frame noise floor is high enough that *some* frame will land on the new label by chance well before any of the other models commit to the change — that pulls its M→I miss rate down without reflecting any real understanding of the transition. Distinguishing the two explanations requires within-clip analysis with the latency search censored at clip boundaries — i.e. v2 mode. Once the evaluations are re-run with `clip_ids`, this report's §7 will quantify how many of the M→I caps were "clip ended too soon" vs. "model genuinely never committed".

## 8. Threats to validity

- **All numbers are v1**, with concatenation boundaries treated as transitions. Inflates the I→S / I→M "23 ms" medians slightly (the fresh clip starts already containing the new class), but it affects all models identically. Run §9 to upgrade to v2.
- **Right-censored latency.** A 2 000 ms cap prevents a single never-resolved transition from dominating the means; medians and p90 are the preferred statistics. The miss rate column makes censoring explicit.
- **Single test split.** No bootstrap CIs on transition-level metrics yet; the existing bootstrap in `evaluator.py` operates on per-frame F1 and would need to resample clips (also requiring `clip_ids`).
- **Latency convention.** End-of-frame `(k + 1) · hop_ms` is exact under the assumption that the transition occurs at the boundary between two y_true frames (which it does, definitionally). Sub-frame-level uncertainty in *when audio actually changed* within the labelling window of frame `t` is a separate question, not addressable from frame-level scores alone.

## 9. Reproduction

### Initial run (current state — v1 mode, no eval refresh)

```bash
uv run smclassifier exp transitions run
```

Reads `_scores.npz` from the project-root `results/` directory and writes the markdown table, text report, and five SVGs into `src/exp/transitions/results/`.

### Refresh to v2 (within-clip-only)

The evaluator was patched (`src/evaluator.py:379–388`) to persist `clip_ids`/`subclasses`/`y_pred` into every `_scores.npz` it writes. All six current npz files predate the patch and lack those fields, so all six need re-evaluation.

```bash
# Classics — already CPU; CUDA_VISIBLE_DEVICES is harmless
CUDA_VISIBLE_DEVICES="" uv run smclassifier classic decision_tree eval
CUDA_VISIBLE_DEVICES="" uv run smclassifier classic gmm eval
CUDA_VISIBLE_DEVICES="" uv run smclassifier classic svm eval

# NN — CUDA_VISIBLE_DEVICES="" forces CPU since the local GPU
# (GTX 1050, sm_61) isn't supported by the installed PyTorch wheel.
CUDA_VISIBLE_DEVICES="" uv run smclassifier nn tcn eval
CUDA_VISIBLE_DEVICES="" uv run smclassifier nn tcn-lstm eval
CUDA_VISIBLE_DEVICES="" uv run smclassifier nn small-tcn eval

# Then refresh the analysis
uv run smclassifier exp transitions run
```

After refresh, every Mode column should read `v2`, and the M→I anomaly section should be updatable with the within-clip-only miss rate.

**Prerequisite — HuggingFace dataset cache.** The local `cache/` directory is absent on this checkout; the first eval will trigger `datasets.load_dataset(...)` to populate it (full tier ≈ 32 GB). If that's expensive, `scripts/download_artifacts.py` can pull a pre-staged cache from `Marek324/butfit-bp-artifacts` on HF.

**Sanity check after one refresh** (single-model spot test before committing to all six):

```python
import numpy as np
d = np.load("results/decision_tree_scores.npz")
assert "clip_ids" in d.files and "y_pred" in d.files
```

Then re-running `exp transitions run` should report `mode=v2` for DT only; mixed-mode runs are still well-defined and the report flags them prominently.
