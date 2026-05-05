# Decision Tree baseline vs. Lavner & Ruinskiy (EURASIP 2009)

Reference: *A decision-tree-based algorithm for speech/music classification
and segmentation*, Yizhar Lavner and Dima Ruinskiy, EURASIP Journal on
Audio, Speech, and Music Processing 2009.

This document tracks every deviation between our DT implementation
(`src/classic/feat_extractor.py::_extract_decision_tree`,
`src/classic/decisiontree.py`) and the paper.

## What matches the paper

- Paper
+ This implementation

- Time-domain + frequency-domain features computed per frame
+ `_extract_decision_tree()` returns a flat feature vector combining time-domain (STE, ZCR, autocorr) and frequency-domain (MFCC, BER, spectral rolloff / centroid / spread / flux) per-frame measurements 

- Exponentially-decaying smoothing of decisions over a past horizon (paper §3.3, $D_s(t) = (1/F)\sum_{k=0}^{K} D_i(t-k)\, e^{-k/\tau}$)
+ `decisiontree.py::predict()` keeps a 30-decision rolling deque and combines probability vectors with $w_k = e^{-k/\tau} / \sum_j e^{-j/\tau}$ before argmax — same math, different granularity (see §5)

- Designed for real-time / consumer-audio operation 
+ Streaming-compatible: `extract()` consumes one frame at a time, `predict()` returns one label per frame; `StreamingClassifier` wraps both 

- Frame-level energy, ZCR, MFCC features (paper §2.3 specifies the first 10 MFC coefficients)
+ `_short_time_energy`, `_zero_crossing_rate`, `_mfcc` (10 coefficients via librosa) + L2 norm of the MFCC delta as one extra scalar — matches the paper's $\Delta\text{MFCC}$ feature 

- Spectral feature family (centroid, rolloff) 
+ `_spectrum_centroid`, `_spectral_rolloff_point` (85th percentile, configurable) 



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

### 3. Sample rate and frame schedule: 16 kHz / 20 ms / 10 ms vs. paper's 44 kHz / 40 ms / 20 ms

**Paper.** §2.3 specifies $N = 40$ ms frames at hop $h_f = 20$ ms, confirmed
again in §3.1 ("$N = 40$ milliseconds, $h_f = 20$ milliseconds, $S = 4$
seconds"). Sample rate is 44 kHz, named in the band-energy-ratio definition
(§2.3: "the energy above 11 KHz and the total energy, where the sampling
frequency is 44 KHz"). Long segment $S = 4$ s with segment hop $h_s = 100$ ms.

**Implementation.** `[buffers.decision_tree]` in `config.toml` sets
`frame_length_ms = 20`, `hop_length_ms = 10`, `lt_len_ms = 300`, and inherits
the global `sample_rate = 16000` — i.e. 20 ms frames at 10 ms hop on 16 kHz
audio with a 300 ms long-term feature-statistics window. There is no
4 s segment buffer; aggregation is per-frame.

**Why.** Empirically tuned for the project's 16 kHz pipeline (shared with the
GMM/SVM streaming path). 20 ms / 10 ms is a common speech-recognition default
that gives ~30 frames per 300 ms window — enough to compute stable
mean/std/skew aggregates without the 4 s of latency the paper's segment-level
schedule would impose. The 300 ms window is shorter than the GMM/SVM 1 s
window because the DT's per-frame features feed the exponential-decay
decision buffer downstream (§6).

### 4. Feature set and selection: 19-feature vector + ANOVA-F vs. paper's 20-feature pool + multi-stage selection

**Paper.** §2.3 enumerates a 20-feature per-frame pool: STE, ZCR, **two**
band-energy ratios (low-band $\le 70$ Hz / total and high-band $\ge 11$ kHz /
total — both as ratios to the total spectral energy, on a log scale),
autocorrelation peak in the 3–16 ms lag range, the first 10 MFCCs + the
$\Delta\text{MFCC}$ Euclidean-distance feature, spectrum rolloff, centroid,
spread, and spectral flux. §2.4 then aggregates per feature with mean / std /
mean of $|\Delta|$ / std of $|\Delta|$, plus skewness + skewness-of-$|\Delta|$
on ZCR specifically and LSTER on energy.

Feature selection (§2.6) is **per-threshold-type**, picking a separate subset
for each of the five threshold types (extreme speech / music, high-probability
speech / music, separation):
- **Extreme thresholds:** rank by inclusion fraction $C = I$, then greedy
  pick with a correlation penalty
  $i_k = \arg\max_j\{\alpha\, C(j) - (\beta/(k-1))\sum_{r<k} |\rho_{i_r,j}|\}$
  (typically $\alpha = \beta = 0.5$).
- **High-probability thresholds:** same greedy, but with separation power
  $C = I^2/Er$.
- **Separation threshold:** Sequential Floating Forward Selection over the
  scatter-matrix criterion $J_2 = |S_m|/|S_w|$ (the paper notes this gave a
  small improvement over their earlier FDR + greedy attempt).

The optimal cardinality per group is picked by cross-validation on the
training set (Table 2: 4–11 features per group).

**Implementation.** A fixed 19-component per-frame vector
(`_extract_decision_tree`):
- 3 time-domain: STE, ZCR, autocorrelation peak
- 11 cepstral: 10 MFCC coefficients + L2 norm of MFCC delta
- 5 frequency-domain: band energy ratio (single combined log-ratio
  $\log(E_\text{low}/E_\text{high})$, **not** the two paper-style
  band-vs-total ratios), spectral rolloff, centroid, spread, flux

This vector is then expanded to a longer "feature + statistics" vector by
appending mean/std of the per-frame buffer, mean/std of the per-frame *diff*
buffer, skew of ZCR, skew of ZCR-diff, and LSTER — mirroring the paper's
§2.4 segment-level statistics, just at a 300 ms / 10 ms granularity instead
of 4 s / 20 ms. The `SelectKBest(k=10)` step picks the 10 most discriminative
components by ANOVA F-statistic at fit time.

**Why.** Hand-picking the per-frame feature set lets us share most of the
extractor between the DT and the GMM/SVM streaming path
(`feat_extractor.py` is the common backbone), and collapsing BER to one log
ratio drops one feature without losing the high/low-band contrast.
`SelectKBest(k=10)` is a single-stage ANOVA-F approximation of the paper's
multi-stage scheme — the paper's scheme requires per-threshold PDFs and
correlation/scatter bookkeeping that don't apply to a single learned tree
(§1).

### 5. Smoothing: per-frame at 10 ms vs. paper's per-segment at 100 ms; threshold adaptation dropped

**Paper.** §3.3 specifies an exponentially-decaying weighted average of past
**segment-level** decisions $D_i(t)$:

$$D_s(t) = \frac{1}{F}\sum_{k=0}^{K} D_i(t-k)\, e^{-k/\tau}, \quad F = \sum_{k=0}^{K} e^{-k/\tau}$$

Segments are 4 s with $h_s = 100$ ms hop, so each $k$ step is 100 ms.
Discretization to a binary label is gated by an **adaptive threshold**
$T_h(t)$: on agreement with the previous decision the threshold is decayed
toward $T_\text{min}$ ($T_h \Leftarrow \max(M\,T_h, T_\text{min})$, $0 < M <
1$); on a flip it resets to $T_\text{init}$. A four-level "weakly speech /
weakly music" output is offered for ambiguous values inside $(-T_h, T_h)$.

**Implementation.** `decisiontree.py::predict()` keeps a 30-deep deque of
**per-frame** probability vectors and combines them with the same
exponential-decay form:

```python
weights = exp(-arange(30) / decision_forget_factor)  # default forget = 0.9
weights /= weights.sum()
smoothed = weights @ recent_probs[::-1]   # most recent frame gets weight w[0]
```

`decision_forget_factor = 0.9` is the forgetting time constant. The first
decision sees weight `1/Σ`, the next `exp(-1/0.9)/Σ ≈ 0.33/Σ`, falling
exponentially. Final label is `argmax(smoothed)`; there is no adaptive
threshold and no four-level output mode.

**Why.** The smoothing math matches the paper. The deviations are:
1. **Granularity.** Frame-level at 10 ms hop instead of segment-level at
   100 ms hop, so the 30-deep deque covers 300 ms instead of the paper's
   ~$K \cdot 100$ ms horizon. This matches the `lt_len_ms = 300`
   feature-statistics window and avoids the 4 s segment latency the paper
   incurs.
2. **No threshold adaptation.** The classifier is `sklearn`'s
   `DecisionTreeClassifier` (§1), which already commits to a hard class via
   `predict_proba` argmax — no scalar $D_i \in [-1, 1]$ for an adaptive $T_h$
   to gate. The four-level output mode is similarly inapplicable.

### 6. Two-buffer architecture: shared with paper, different granularities

**Paper.** Has both a **feature-statistics buffer** (§2.4: per-feature mean,
std, mean of $|\Delta|$, std of $|\Delta|$ over a 4 s segment, plus skew /
skew-of-$|\Delta|$ on ZCR and LSTER on energy) and a **decision-smoothing
buffer** (§3.3: exponentially-decaying weighted average over past segment
decisions $D_i$).

**Implementation.** Two separate buffers, same architecture as the paper but
at finer granularity:
- `feat_buffer` (size `lt_len_ms / hop_length_ms − 1` = 29 frames at
  300 ms / 10 ms): aggregates per-frame features into mean / std / diff-mean
  / diff-std + ZCR skew + LSTER before they reach the classifier.
- `last_decisions` (deque of 30, in `decisiontree.py`): smooths
  *post-classifier* probability vectors via exponential decay.

**Why.** Architecture matches the paper. The deviation is granularity (§3,
§5) and the fact that the feature aggregator is shared with the GMM/SVM
streaming path through `feat_extractor.py`.

### 7. Real-time per-frame inference vs. paper's per-segment

**Paper.** Each 4 s segment (with 100 ms segment hop, §3.1) is classified
once, with smoothing across consecutive segment decisions.

**Implementation.** Frame-level inference: every 10 ms hop produces one
`predict(frame)` call → one feature vector → one tree decision → one entry
in the smoothing deque → one output label. The `StreamingClassifier` wrapper
in `src/classic/streaming.py` consumes raw audio samples and emits per-hop
labels for the demo / mic path.

**Why.** Project requirement is streaming broadcast monitoring at the frame
level (per `src/exp/critical/results/notes.md` and the transitions experiment
which measures per-event latency in milliseconds). Aggregating to the paper's
4 s segments — even with the 100 ms segment hop — would add inherent latency
to the speech↔music transition response.

### 8. Evaluation protocol

**Paper.** Speech database (>12 h) and music database (>22 h) reported, with
correct identification rates 99.4 % / 97.8 % respectively. Evaluation
methodology not summarized here.

**Implementation.** `Marek324/speech-music-classification` HF dataset
(full tier, 100 h). Fixed train/val/test splits. Macro F1 across 3 classes
on the test split. The deployed checkpoint reports
**macro F1 = 0.8376** [`results/decision_tree.eval` test split], well below
the TCN family but the cheapest model on a per-frame basis.

**Why.** Different research context (3-class, longer dataset, different
metric) — see also gmm_svm.md §13 and tcn.md §8. Note also that the paper's
results are not directly comparable: the paper's "correct identification
rate" is per-segment accuracy on a 2-class task with silence/noise excluded,
whereas our macro F1 is per-frame across 3 classes including the inactive
class on a different dataset.

## Summary

| Aspect | Paper | Implementation | Match? |
|---|---|---|---|
| Classifier structure | 3-stage Bayesian + rule-based sieve | sklearn `DecisionTreeClassifier` (single learned tree) | Different (§1) |
| Output classes | 2 (speech / music) | 3 (speech / music / inactive) | Different (§2) |
| Sample rate | 44 kHz | 16 kHz | Different (§3) |
| Frame length / hop | 40 ms / 20 ms | 20 ms / 10 ms | Different (§3) |
| Segment / long-term window | 4 s (segment hop 100 ms) | 300 ms `feat_buffer` | Different (§3, §6) |
| Per-frame feature pool | 20 features (BER × 2) | 19 features (single combined BER) | Different (§4) |
| Per-feature statistics | mean / std / mean-of-$\|\Delta\|$ / std-of-$\|\Delta\|$ + skew & skew-$\|\Delta\|$ on ZCR + LSTER | Same statistic family at 300 ms granularity | §4 |
| MFCC coefficients | First 10 + $\Delta$MFCC L2 norm | 10 + delta-norm | Match |
| Feature selection | Per-threshold multi-stage (incl. fraction / $I^2/Er$ / SFFS-$J_2$) + correlation penalty | Single-stage `SelectKBest(k=10)` ANOVA-F | Different (§4) |
| Decision smoothing math | $D_s(t) = (1/F)\sum_k D_i(t-k)\,e^{-k/\tau}$ | Same exponential-decay form, $\tau = 0.9$ over 30-deep deque | Match (§5) |
| Decision smoothing granularity | Per-segment (100 ms steps) | Per-frame (10 ms steps) | Different (§5) |
| Threshold adaptation $T_h(t)$ + 4-level output | Yes | None — argmax over `predict_proba` | Different (§5) |
| Inference granularity | Per-segment (4 s, 100 ms hop) | Per-frame (10 ms hop) | Different (§7) |
| Evaluation | >12 h speech + >22 h music, per-segment 2-class accuracy | HF full tier 100 h, per-frame 3-class macro F1 | Different (§8) |
