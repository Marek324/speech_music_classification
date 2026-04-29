# Decision Tree baseline vs. Lavner & Ruinskiy (EURASIP 2009)

Reference: *A decision-tree-based algorithm for speech/music classification
and segmentation*, Yizhar Lavner and Dima Ruinskiy, EURASIP Journal on
Audio, Speech, and Music Processing 2009.

This document tracks every deviation between our DT implementation
(`src/classic/feat_extractor.py::_extract_decision_tree`,
`src/classic/decisiontree.py`) and the paper.

## What matches the paper

| Paper | This implementation |
|---|---|
| Time-domain + frequency-domain features computed per frame | `_extract_decision_tree()` returns a flat feature vector combining time-domain (STE, ZCR, autocorr) and frequency-domain (MFCC, BER, spectral rolloff / centroid / spread / flux) per-frame measurements |
| Long-term smoothing applied to per-segment decisions, "averaging the decision on each segment with past segment decisions" | `decisiontree.py::predict()` keeps a 30-decision rolling deque and averages probability vectors with exponential forgetting before argmax |
| Designed for real-time / consumer-audio operation | Streaming-compatible: `extract()` consumes one frame at a time, `predict()` returns one label per frame; `StreamingClassifier` wraps both |
| Frame-level energy, ZCR, MFCC features | `_short_time_energy`, `_zero_crossing_rate`, `_mfcc` (10 coefficients via librosa) |
| Spectral feature family (centroid, rolloff) | `_spectrum_centroid`, `_spectral_rolloff_point` (85th percentile, configurable) |

## Deviations

### 1. Classifier: sklearn `DecisionTreeClassifier` vs. paper's 3-stage Bayesian + rule-based sieve

**Paper.** A *hand-crafted* three-stage sieve. Each stage applies a different
subset of features through Bayesian thresholds estimated from per-feature PDFs,
plus rule-based decisions. Features that pass each stage are routed to the
next. The "decision tree" structure is the cascade itself, not a single
learned tree.

**Implementation.** `decisiontree.py` uses
`sklearn.tree.DecisionTreeClassifier` — a single learned binary tree fit by
CART on the full feature vector at once. `Pipeline` wraps:
1. `StandardScaler` — z-normalize features.
2. `SelectKBest(k=10, score_func=f_classif)` — pick the 10 best features by
   one-way ANOVA F-statistic.
3. `DecisionTreeClassifier(random_state=RAND_SEED)` — default sklearn tree
   (Gini, no depth/leaf limits, no pruning).

**Why.** A learned tree is the natural sklearn-side equivalent of "decision
tree". Replicating the paper's three-stage sieve would require manually fitting
per-feature Bayesian thresholds and authoring the rule-based escape branches —
out of scope for this implementation. The learned tree captures the same
high-level idea (a sequence of feature thresholds) but lets sklearn pick the
splits and order automatically. Per-stage Bayesian thresholds are not exposed.

### 2. Output classes: 3-class vs. paper's 2-class

**Paper.** Binary speech vs. music. Silence and noise frames are explicitly
filtered out of the training set ("silent and noisy frames must therefore be
eliminated"; see `docs/chapters/03_classification.tex` §3.36).

**Implementation.** Three-class — speech / music / inactive — with the
inactive frames carrying their own training labels rather than being filtered
out. The tree's output classes are `[-1, 1, 2]` per `src/common.py`.

**Why.** The project label space is fixed across all classifiers
(`src/common.py`), so the DT trains and predicts in the same 3-class
vocabulary as the GMM, SVM, and TCN family. Reducing to 2-class would require
a separate DT-only label remapping path.

### 3. Sample rate and frame schedule: 16 kHz / 20 ms / 10 ms vs. paper's 16 kHz unspecified

**Paper.** 16 kHz audio (table not parsed in detail here). Frame length and
hop are reported in `docs/chapters/03_classification.tex` discussion but the
exact values come from the paper's experimental section.

**Implementation.** `[buffers.decision_tree]` in `config.toml` sets
`frame_length_ms = 20`, `hop_length_ms = 10`, `lt_len_ms = 300` — i.e. 20 ms
frames at 10 ms hop with a 300 ms long-term smoothing window for feature
statistics.

**Why.** Empirically tuned. 20 ms / 10 ms is a common speech-recognition
default that gives ~30 frames per 300 ms window — enough to compute stable
mean/std/skew aggregates without forcing a long latency. The 300 ms window is
shorter than the GMM/SVM 1 s window because the DT's per-frame features are
already smoothed via the exponential-decay decision buffer (§7).

### 4. Feature set: explicitly enumerated vs. paper's "automatic feature selection"

**Paper.** "An automatic procedure is employed to select the best features for
separation."

**Implementation.** A fixed 19-component per-frame vector
(`_extract_decision_tree`):
- 3 time-domain: STE, ZCR, autocorrelation peak
- 11 cepstral: 10 MFCC coefficients + L2 norm of MFCC delta
- 5 frequency-domain: band energy ratio, spectral rolloff, centroid, spread,
  flux

This vector is then expanded to a longer "feature + statistics" vector by
appending mean/std of the per-frame buffer, mean/std of the per-frame *diff*
buffer, skew of ZCR, skew of spectral-centroid diff, and LSTER. The
`SelectKBest(k=10)` step picks the 10 most discriminative components by
ANOVA F-statistic at fit time.

**Why.** Hand-picking the feature set lets us share most of the extractor
between the DT and the GMM/SVM streaming path (`feat_extractor.py` is the
common backbone). `SelectKBest` plays the role of the paper's automatic
feature selection — different mechanism, same effect.

### 5. Smoothing: exponential-decay weighted vs. paper's simple averaging

**Paper.** "Smoothing technique is applied, averaging the decision on each
segment with past segment decisions" — implies a uniform-weight average over a
fixed past horizon.

**Implementation.** `decisiontree.py::predict()` keeps a 30-deep deque of
per-frame probability vectors and combines them with exponentially-decaying
weights:

```python
weights = exp(-arange(30) / decision_forget_factor)  # default forget = 0.9
weights /= weights.sum()
smoothed = weights @ recent_probs[::-1]   # most recent frame gets weight w[0]
```

`decision_forget_factor = 0.9` is the smaller-is-faster forgetting time
constant. The first decision sees weight `1/Σ`, the next `exp(-1/0.9)/Σ ≈
0.33/Σ`, falling exponentially.

**Why.** Exponential forgetting reacts faster to genuine class transitions
than a uniform window of equal length (the same total horizon, but recent
frames count more). On the crit-tier `speech_switching_*` clips this is
visible in the latency / flicker analysis (`src/exp/transitions/results/`).
The 30-frame deque covers 300 ms at 10 ms hop, matching the
`lt_len_ms = 300` long-term feature window.

### 6. MFCC count: 10 vs. paper's unspecified

**Paper.** Reports "MFCC features" without committing to an exact coefficient
count.

**Implementation.** `_mfcc(frame)` calls `librosa.feature.mfcc(..., n_mfcc=10)`
and additionally records the L2 distance `‖mfcc_t − mfcc_{t−1}‖` as a single
extra feature.

**Why.** 10 coefficients is a common compromise between coverage and
dimensionality for speech work at 16 kHz. The MFCC delta norm captures
spectral change without doubling the coefficient count.

### 7. Decision smoothing buffer separate from feature smoothing buffer

**Paper.** A single smoothing stage on the *output* labels.

**Implementation.** Two separate buffers operating at different stages:
- `feat_buffer` (size derived from `lt_len_ms / hop_length_ms − 1` ≈ 29
  frames at 300 ms / 10 ms): aggregates per-frame features into mean / std /
  diff-mean / diff-std before they reach the classifier.
- `last_decisions` (deque of 30, in `decisiontree.py`): smooths
  *post-classifier* probability vectors via exponential decay.

**Why.** Feature-level smoothing is part of the *input* representation to the
DT; decision-level smoothing damps frame-by-frame label flips. The two stages
are independent and tuned separately. The paper folds both into a single
smoothing on the output; here we split them so the feature aggregator can be
reused across other classifiers.

### 8. Real-time per-frame inference vs. paper's per-segment

**Paper.** Each segment (length unspecified in this audit, but implied to be
~1 s) is classified once, with smoothing across consecutive segments.

**Implementation.** Frame-level inference: every 10 ms hop produces one
`predict(frame)` call → one feature vector → one tree decision → one entry
in the smoothing deque → one output label. The `StreamingClassifier` wrapper
in `src/classic/streaming.py` consumes raw audio samples and emits per-hop
labels for the demo / mic path.

**Why.** Project requirement is streaming broadcast monitoring at the frame
level (per `src/exp/critical/results/notes.md` and the transitions experiment
which measures per-event latency in milliseconds). Aggregating to 1 s would
add a second of inherent latency to the speech↔music transition response.

### 9. Evaluation protocol

**Paper.** Speech database (>12 h) and music database (>22 h) reported, with
correct identification rates 99.4 % / 97.8 % respectively. Evaluation
methodology not summarised here.

**Implementation.** `Marek324/speech-music-classification` HF dataset
(full tier, 100 h). Fixed train/val/test splits. Macro F1 across 3 classes
on the test split. The deployed checkpoint reports
**macro F1 = 0.8376** [`results/decision_tree.eval` test split], well below
the TCN family but the cheapest model on a per-frame basis.

**Why.** Different research context (3-class, longer dataset, different
metric) — see also gmm_svm.md §13 and tcn.md §8.

## Summary

| Aspect | Paper | Implementation | Match? |
|---|---|---|---|
| Classifier structure | 3-stage Bayesian + rule-based sieve | sklearn `DecisionTreeClassifier` (single learned tree) | Different (§1) |
| Output classes | 2 (speech / music) | 3 (speech / music / inactive) | Different (§2) |
| Frame length / hop | Paper unspecified | 20 ms / 10 ms | Different (§3) |
| Long-term window | Paper-implicit "segment" | 300 ms (`lt_len_ms`) | §3 |
| Feature selection | Automatic per paper | Fixed list + `SelectKBest(k=10)` ANOVA-F | §4 |
| MFCC coefficients | Paper unspecified | 10 + delta-norm | §6 |
| Decision smoothing | Uniform average over past segments | 30-frame deque, exponential decay (factor 0.9) | §5 |
| Feature smoothing | Implicit in segments | 29-frame `feat_buffer` (mean/std/diff stats) | §7 |
| Inference granularity | Per-segment | Per-frame (streaming) | §8 |
| Evaluation | >12 h speech + >22 h music, custom metric | HF full tier, 3-class macro F1 | §9 |
