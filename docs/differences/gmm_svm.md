# GMM/SVM baseline vs. Khonglah & Prasanna (DSP 2016)

Reference: *Speech / music classification using speech-specific features*,
Banriskhem K. Khonglah and S.R. Mahadeva Prasanna, Digital Signal Processing
2016.

This document tracks every deviation between our GMM/SVM implementation
(`src/classic/feat_extractor.py`, `src/classic/gmm.py`, `src/classic/svm.py`)
and the paper.

## What matches the paper

| Paper | This implementation |
|---|---|
| 30 ms frame, 1 ms shift for NAPS/log-mel; PSR coarsened to 10 ms (§13) | `extract_segment()` inner loop + vectorized `_batched_naps_of_zffs` / `_batched_log_mel_energy` |
| 1 s window for statistics | `extract_segment()` on 1 s (batch path); streaming `_extract_gmm_svm()` aggregates rolling 1 s rings on every 15 ms hop (§14) |
| Existing features: ZCR variance, spectral centroid variance, spectral flux variance, spectral roll-off variance, % low energy frames | `extract_segment()` step 1: per-frame features → variance + LSTER |
| Speech-specific statistics: mean(NAPS), mean(PSR), var(log-mel energy), mean(modulation spectrum) | `extract_segment()` steps 2–3 |
| NAPS of ZFFS: differenced signal → cascade of two zero-frequency resonators → two-pass trend removal → autocorrelation → first-peak / central-peak | `_naps_of_zffs()` / `_batched_naps_of_zffs()`: `lfilter([1],[1,-4,6,-4,1], diff)` + two `uniform_filter1d` passes + `correlate` / FFT autocorr + `find_peaks` |
| PSR of HE of LP residual: 10th-order LP → inverse filter → Hilbert envelope → peak / sidelobe variance | `_psr_he_lp_residual()`: `lpc(order=10)` + `lfilter` + `hilbert` + peak finding |
| Log mel spectrum energy: DFT → 22 mel filters → sum of log of first 18 | `_log_mel_spectrum_energy()` / `_batched_log_mel_energy()`: 22-band mel (fmin=0, fmax=4000) → `log(M[:18])` |
| SVM: RBF kernel, C=1, gamma=3 (libSVM) | `SVC(kernel='rbf', C=1, gamma=3)` via sklearn |
| GMM: 8 components, diagonal covariance | `GaussianMixture(n_components=8, covariance_type='diag')` |
| 4-fold cross-validation scheme | Not replicated (train/val/test split from HF dataset instead) |

## Deviations

### 1. Sample rate path: 16 kHz → 8 kHz vs. paper's 22050 → 8 kHz

**Paper.** Audio sampled at 22050 Hz, downsampled to 8000 Hz for all
feature extraction (§4).

**Implementation.** HF dataset loaded at 16 kHz (global `sample_rate` in
the root `config.toml`), resampled to 8 kHz in
`InputHandler._process_row` via `librosa.resample` before being fed
frame-by-frame to `FeatExtractor.extract()`. Streaming microphone input
follows the same path.

**Status.** Target rate matched (8 kHz). The downsampling source rate differs
(16 → 8 vs paper's 22050 → 8) but the polyphase resampler in librosa is
quality-equivalent for this conversion ratio.

### 2. LP order: 10 at 8 kHz

**Paper.** 10th-order LP analysis at Fs = 8 kHz (§2.1.2).

**Implementation.** `lpc(frame, order=10)` at 8 kHz.

**Status.** Matched. LP order 10 at 8 kHz models up to 4 kHz (Nyquist),
consistent with the rule of thumb `order ≈ sr_kHz + 2`.

### 3. NAPS trend-removal window: fixed 10 ms vs. paper's "average pitch period"

**Paper.** The trend-removal window `2N+1` corresponds to "the average
pitch period over a longer segment of speech" (§2.1.1, eq. 3–4).

**Implementation.** `N = int(sr * 0.01)` — fixed at 10 ms regardless of
actual pitch.

**Why.** The average male pitch period is ~8 ms (125 Hz), female ~5 ms
(200 Hz). Using 10 ms is a reasonable fixed approximation that avoids
the need for a pitch tracker. The paper's reference [14] (Murthy &
Yegnanarayana 2008) also uses a fixed window of 1–2 pitch periods.

### 4. PSR peak location: `find_peaks` vs. paper's epoch-guided search

**Paper.** Peaks of the Hilbert envelope are located "by searching around
the epoch locations obtained from the ZFFS" within a 3 ms frame around
each epoch (§2.1.2).

**Implementation.** `find_peaks(he, distance=5ms)` on the Hilbert
envelope directly, without using ZFFS epoch locations.

**Why.** Avoids coupling the two features and the overhead of a separate
epoch detector. The dominant peak found by `find_peaks` with a minimum
distance constraint is functionally equivalent for the PSR ratio. Minor
deviation unlikely to affect classification.

### 5. PSR sidelobe window: fixed 10 ms vs. paper's "one pitch period"

**Paper.** Sidelobe variance computed over "a frame size of one pitch
period" centered on the peak (§2.1.2).

**Implementation.** `half = int(sr * 0.01) // 2` — fixed 10 ms total
window (5 ms each side), excluding 4 samples nearest the peak. Matches
the paper's description for a typical pitch period.

**Why.** Same rationale as §3: avoids requiring a pitch tracker. 10 ms
approximates one pitch period for most speech.

### 6. Modulation spectrum: mel-spectrogram proxy vs. paper's critical-band filterbank

**Paper.** 18 trapezoidal critical-band filters (0–4 kHz) → half-wave
rectification → 28 Hz LP filter → decimate to 80 Hz → normalize per band
over entire clip → 250 ms Hamming DFT (12.5 ms shift) → sum 4 Hz bin
across bands (§2.3.1, eq. 11–12).

**Implementation.** `_compute_mod_mel_cols_batch()` manually computes a
(18, T) power-mel column matrix with `center=False`, hop `sr // 80 = 100`,
windowed with a Hann over the full `n_fft=512` analysis frame and
multiplied by a precomputed mel filterbank (`fmin=0, fmax=sr/2=4000`).
`_compute_mod_energy_from_cols()` then normalizes each band by its mean
over the 1 s window, slides the 250 ms (20-sample) Hamming window at a
1-sample hop, and sums the `|FFT[·, 1]|²` (= 4 Hz) energy across bands.
Shared by both the batch `extract_segment()` path and the streaming
`_extract_gmm_svm()` path (streaming keeps a ring of mel columns).

**Differences in detail:**
- Mel-scale triangular filters vs. Bark-scale trapezoidal filters.
  Mel and Bark scales are nearly identical below 4 kHz; 18 bands in
  either scale cover similar frequency ranges.
- The mel spectrogram inherently applies windowing + power computation,
  which approximates the half-wave rectification + LP filtering step.
- Normalization is per 1 s segment rather than per entire clip, since
  we process non-overlapping 1 s segments independently. The paper
  normalizes over the full audio file. This may reduce the feature's
  discriminative power for short segments but is consistent with the
  1 s windowed classification framework.

### 7. n_fft: 512 at 8 kHz (64 ms) — matches paper

**Paper.** 512-point DFT at 8 kHz (§2.2.1), giving 64 ms analysis window
and ~16 Hz frequency resolution.

**Implementation.** Global `n_fft = 512` with `sr = 8000` → 64 ms.

**Status.** Matched.

### 8. Feature scaling: RobustScaler / StandardScaler vs. paper unspecified

**Paper.** Does not describe feature normalization beyond the per-band
normalization for modulation spectrum. The raw feature statistics are
fed to GMM/SVM directly.

**Implementation.** GMM uses `RobustScaler` (in `gmm.py`); SVM pipeline
uses `StandardScaler` (in `svm.py`). Both are applied before classification.
The GMM additionally clips the scaled values to ±10 (`_SCALE_CLIP`) to
keep a stray outlier from blowing up the diagonal-covariance log-likelihood.

**Why.** Standard practice for sklearn classifiers. The paper's libSVM
likely performed internal scaling; sklearn's SVC does not auto-scale.
Without scaling, the RBF kernel would be dominated by high-magnitude
features (spectral centroid variance, flux variance).

### 9. Three-way GMM density classifier vs. paper's two-GMM speech-vs-music

**Paper.** Two GMMs — one fitted to speech feature vectors, one to music —
log-likelihood difference thresholded to decide speech vs music (§3.2,
binary).

**Implementation.** Three GMMs in `gmm.py` (`gmm_speech`, `gmm_music`,
`gmm_inactive`), each fit to its class on the same 9-D feature vector.
Inference computes per-class log-likelihoods, takes argmax, and remaps
to the project label space `{-1, 1, 2}`.

**Why.** The project is 3-class (speech / music / inactive). The
two-GMM threshold rule cannot represent the third class without an
auxiliary detector. Adding a third GMM keeps the density-classifier
structure of the paper while extending it to the project's label set.

### 10. GMM smoothing: 66-frame log-likelihood rolling buffer vs. paper's 1 s mean

**Paper.** Does not describe a specific smoothing method for the GMM
*classifier* output; the threshold-based approach (§3.1) uses 1 s mean
smoothing on the *raw features* before non-linear mapping.

**Implementation.** `gmm.py` keeps a 66-frame rolling buffer of the
per-class log-likelihood vector (`self.ll_buf = deque(maxlen=66)` ≈ 1 s at
15 ms hop). `predict()` averages over the buffer before argmax. The batch
prediction path (`predict_batch`) does not use the rolling buffer.

**Why.** Provides temporal smoothing consistent with the 1 s window
size used throughout the paper.

### 11. SVM smoothing and 3-class extension: 20-frame decision-function rolling buffer + sklearn OvO vs. paper's binary single-shot

**Paper.** Binary speech vs. music. Single 1 s feature vector per
non-overlapping window → SVM decision → label. No frame-level smoothing
because the segmentation is already coarse-grained.

**Implementation.** `svm.py` trains a single multi-class `SVC` on the full
3-class label set (`{-1, 1, 2}`); sklearn handles 3-class natively via OvO,
producing a 3-element decision-function vector per frame. The `_subsample_balanced`
helper subsamples 50 k frames *per class* (including the inactive class), so
no frame is filtered out of training. At inference, `predict()` keeps a
20-frame rolling buffer of the decision vector (`self.dec_buf = deque(maxlen=20)`
≈ 300 ms at 15 ms hop), averages it, and takes `argmax` over the smoothed
3-vector. The batch prediction path (`predict_batch`) uses `svm.predict()`
directly without smoothing.

**Why.** Streaming inference fires on every 15 ms hop, so the raw
decision sequence is noisy. 300 ms smoothing damps single-frame flips
without delaying the response significantly. The 20-frame buffer is
shorter than the GMM 66-frame buffer because SVM decision functions are
already discriminatively scaled and need less averaging to stabilize.

### 12. Non-linear mapping (paper §3.1): not implemented

**Paper.** Individual features are smoothed and passed through a sigmoid
non-linear mapping function (eq. 13) before combining by summation.
This is the threshold-based classification approach.

**Implementation.** Not implemented. We use only the classifier-based
approach (GMM/SVM on concatenated feature statistics, paper §3.2).

**Why.** The paper reports that classifiers outperform the threshold-based
approach (74 % → 96 % on S&S database). The non-linear mapping is
primarily of interest as a classifier-free baseline.

### 13. Dataset and evaluation protocol

**Paper.** Evaluated on three databases: Scheirer & Slaney (S&S),
GTZAN, and Indian Broadcast News. 4-fold cross-validation. Binary
classification (speech vs. music). Audio: 5–30 s clips.

**Implementation.** `Marek324/speech-music-classification` HF dataset
(mid/full tier). Fixed train/val/test splits. 3-class classification
(speech / music / inactive). Clips range from <1 s to several minutes.

**Why.** Different research context — our dataset includes an inactive
class and is designed for streaming broadcast monitoring rather than
replicating the paper's exact evaluation. The feature extraction follows
the paper; the evaluation framework differs.

### 14. SVM training scale: subsampled vs. paper's small scale

**Paper.** ~2400 feature vectors from 160 clips of 15 s each (1 s
non-overlapping windows → ~15 vectors/clip × 160 clips).

**Implementation.** Frame-level features subsampled to 50 k per class
via `_subsample_balanced()` in `svm.py`. Since the streaming refactor
(§16), training sees one 9-D vector per 15 ms hop (same thing inference
sees), so the 50 k cap does trigger on mid/full tier and subsampling
matters again.

### 15. PSR subframe shift: 10 ms vs. paper's 1 ms

**Paper.** All speech-specific features (NAPS, PSR, log-mel) share the
30 ms frame / 1 ms shift schedule (§2.1, §3.2).

**Implementation.** NAPS and log-mel are still computed at 1 ms shift
(batched via `_batched_naps_of_zffs` / `_batched_log_mel_energy`). PSR
is coarsened to 10 ms shift because LPC analysis is a recursive
Levinson-Durbin recursion that cannot be vectorized across subframes
without leaving Python, and 1 ms PSR dominated `extract_segment` cost.

**Why.** A shift sweep (`scripts/parity_shift.py`) showed that PSR and
var(log-mel) stay > 0.99 correlated with the 1 ms reference up to a 10 ms
shift — PSR's dominant peak is stable across tens of ms — while NAPS
drops to corr ≈ 0.29 at 10 ms and so must be kept at 1 ms. 10 ms PSR
still produces 97 subframes per 1 s window, well above the CLT floor
for a stable mean.

### 16. Sliding-window streaming adaptation

**Paper.** Classifier is fed one 9-D vector per 1 s non-overlapping
window (§3.2). No online/streaming description.

**Implementation.** `_extract_gmm_svm(frame)` is a rolling 1 s sliding
window: every call consumes the next 15 ms hop and re-aggregates a 9-D
vector over the most recent 1 s of audio. Internal state lives in
fixed-size ring buffers sized to exactly one window:

  - `feat_buffer` — 65 base frames (ZCR/centroid/flux/rolloff/STE)
    at 15 ms hop
  - `_stream_naps_ring` / `_stream_logmel_ring` — ~1000 entries at
    1 ms shift
  - `_stream_psr_ring` — ~100 entries at 10 ms shift (§15)
  - `_stream_mod_ring` — ~80 mel columns at 100-sample hop for the
    4 Hz modulation DFT

Each new hop appends O(new) entries to each ring (e.g. 15 new NAPS/logmel
subframes, 1–2 new PSR, 1 new mel col) via the batched kernels from
`extract_segment`. Aggregation (var / mean / mod-energy DFT over the
80-col ring) runs on every hop.

**Parity.** `scripts/parity_streaming.py` verifies the streaming output
against `extract_segment()` on the same 1 s clip. Most features agree to
< 1 % relative error; the remaining drift comes from a 10-subframe
boundary gap (streaming stops at `stream_total - fl` whereas
`extract_segment` reaches `n - fl` exactly). In continuous streaming this
boundary never appears, so `run_mic` behavior tracks the batch reference.

**Training impact.** `src/input_handler.py` routes GMM/SVM through
`_process_row` (frame-by-frame `extract()`), so training sees the
same streaming 9-D vectors that `StreamingClassifier` sees at mic time.
One 9-D vector per 15 ms hop instead of one per 1 s segment.

## Summary

| Aspect | Paper | Implementation | Match? |
|---|---|---|---|
| Sample rate | 22050 → 8 kHz | 16 → 8 kHz | Target rate matches |
| Frame / shift | 30 ms / 1 ms | 30 ms / 1 ms (NAPS, log-mel); 10 ms (PSR) | §15 |
| Statistics window | 1 s non-overlapping | 1 s sliding, 15 ms hop (streaming) | §16 |
| n_fft | 512 (64 ms @ 8 kHz) | 512 (64 ms @ 8 kHz) | Yes |
| LP order | 10 | 10 | Yes |
| NAPS algorithm | ZFFS + autocorrelation | Matched (FFT autocorr in batched path) | Yes |
| PSR algorithm | LP residual + Hilbert envelope | Peak finding differs (§4–5) | Minor |
| Log mel energy | 22 filters, sum first 18 | Matched | Yes |
| Modulation spectrum | Critical-band filterbank | Mel-spectrogram proxy (§6) | Approximate |
| SVM (C, gamma, kernel) | RBF, C=1, gamma=3 | Matched | Yes |
| GMM (k, covariance) | 8, diagonal | Matched | Yes |
| GMM structure | 2 GMMs, threshold | 3 GMMs, argmax (§9) | Extended |
| Output classes | 2 (speech / music) | 3 (speech / music / inactive) | Different |
| Feature scaling | Unspecified | RobustScaler (GMM, clipped to ±10) / StandardScaler (SVM) | Added |
| GMM smoothing | Implicit per-feature | 66-frame log-likelihood buffer (§10) | Added |
| SVM smoothing | None | 20-frame decision-function buffer (§11) | Added |
| Non-linear mapping | Implemented | Skipped (§12) | N/A |
| Evaluation protocol | 4-fold CV, 2-class | Train/test split, 3-class | Different |
