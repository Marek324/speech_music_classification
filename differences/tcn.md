# TCN baseline vs. Lemaire & Holzapfel (ISMIR 2019)

Reference: *Temporal Convolutional Networks for Speech and Music Detection in
Radio Broadcast*, Quentin Lemaire and Andre Holzapfel, ISMIR 2019.

This document tracks every deliberate deviation between our TCN implementation
(`/home/marek/bp/config.toml` → `[tcn]` section, mirrored by
`src/exp/tcn_ablation/config.toml` baseline) and the paper. Production and the
ablation baseline currently share the same recipe — the project does not run
two parallel TCN configs anymore.

## What matches the paper

| Paper §3.4–3.5 | This baseline |
|---|---|
| Resample to 22050 Hz mono | `sample_rate = 22050` |
| Hann STFT, frame length 1024, hop 512 | `n_fft = 1024`, `hop_length = 512` |
| Power spectrum → 80-mel filterbank, 27.5–8000 Hz | `n_mels = 80`, `f_min = 27.5`, `f_max = 8000.0` |
| Log scale, zero-mean unit-variance normalization over training set | `preprocess.py::LogMelSpectrogram` + `compute_and_save_preprocess_stats` |
| Spectral features pre-computed once and cached; augmentation applied to cached spectrograms (§3.4: *"saved for the training. During the training, data augmentation was applied to the saved spectrograms"*) | `training.py::_load_mel_chunks` writes `cache/nn/tcn_mel_<split>_*.pt` keyed by frontend params + `n_features`; `training.py::_iter_mel_batches` applies `augment_mel` to the cached mel in-batch |
| Binary cross-entropy loss | `loss = "bce_with_logits"` (mathematically equivalent to paper's BCE; see §2 below) |
| SGD momentum = 0.9 | `optimizer = "sgd"` (momentum=0.9 hard-coded in `training.py:351`) |
| Divide LR by 10 when validation loss does not improve for 3 epochs | `ReduceLROnPlateau(factor=0.1, patience=3)` in `training.py:354` |
| Stop training after 5 consecutive non-improving validation epochs | `patience=5` in `train_tcn` |
| Mini-batches of fixed-length chunks, batch size 32 | `batch_size = 32`, `_iter_mel_batches` in `training.py` |
| Dropout ∈ [0.05, 0.5] per block | `dropout = 0.5` (paper-prescribed upper bound) |
| Causal dilated convolutions, residual blocks, skip connections | `blocks.py::TCNResidualBlock`, `skip_connections=true` default |
| keras-tcn-style block: two dilated causal convs + WeightNorm + ReLU + dropout + residual add | `blocks.py::TCNResidualBlock` with `use_weight_norm=true`, `activation="relu"` |
| Hyperparameter search space: `n_layers ∈ 1..4`, `n_stacks ∈ 3..10`, `kernel_size ∈ {3,5,…,19}`, `n_filters ∈ {8,16,32}`, dilations `2^0…2^N_D` with `N_D=3..8` | Ablation subgroups cover the same ranges; n_filters/kernel/stacks pruned after no-signal results |

## Deviations

Each deviation is either forced by our dataset / numerical reality, or explicitly
documented as an ablation variant so the study empirically justifies it.

### 1. Output dimensionality: 3-class vs. paper's 2-class

**Paper.** Two sigmoid outputs — `[speech, music]`. A frame can be either, both,
or neither; "neither" is implicit from `(0, 0)`.

**Baseline.** Three sigmoid outputs — `[speech, music, inactive]`. The inactive
channel is an explicit class rather than the absence of the first two.

**Why.** The entire codebase (`src/common.py`, classic models, label convention
in `src/nn/dataset.py::LABEL_MAP`) is built around a 3-class vocabulary. Our
dataset has a dedicated `noise` source (`CLAUDE.md` tier table) and an explicit
inactive class gives the cross-entropy a target for those frames rather than a
flat zero vector. BCE-with-logits still works in the 3-output setup.

### 2. Loss: `BCEWithLogitsLoss` on raw logits vs. paper's `BCELoss(sigmoid(x))`

**Paper.** Binary cross-entropy after sigmoid.

**Baseline.** `nn.BCEWithLogitsLoss` directly on the TCN's raw logits. The
sigmoid is applied only at inference time in `SpeechMusicDetector.forward`;
training calls `SpeechMusicDetector.forward_logits_from_mel`.

**Why.** `BCELoss(sigmoid(x))` is the standard PyTorch numerical trap. If any
upstream activation produces NaN (which happens with SGD+WeightNorm; see §4
below), sigmoid propagates the NaN, and `BCELoss` aborts training with
`/pytorch/aten/src/ATen/native/cuda/Loss.cu:90: input_val >= zero && input_val <= one`.
`BCEWithLogitsLoss` uses the fused log-sigmoid formulation and is finite for
any logit value, so the same underlying instability becomes a recoverable loss
spike instead of a crash.

**Empirical evidence.** The `[tcn.ablations.loss.*]` subgroup (now commented
out in `config.toml`) reverts to `nn.BCELoss(sigmoid(x))` and other loss
families. All five variants landed within ±0.0003 of baseline in earlier runs;
they have been pruned from the active ablation set but the fallback codepath
remains in `training.py::build_loss`.

### 3. Initial learning rate: 1e-3 vs. paper's unspecified

**Paper.** The paper prescribes LR scheduling but does **not** specify an
initial value. The reference implementation (keras-tcn) does not fix one either.

**Baseline.** `lr = 1e-3`.

**Why.** We initially tried `lr = 1e-2` (a common SGD default with momentum
0.9). On earlier dataset revisions this drove WeightNorm's `v` parameter
toward zero over many updates, producing unbounded normalised weights
`g * v / ‖v‖` and NaN activations. Combined with grad-norm clipping
(`max_norm=1.0`) plus `BCEWithLogitsLoss` it eventually stabilised at
`lr = 1e-2`, but `lr = 1e-3` was conservatively chosen as the production
default to keep the recipe robust to data churn.

**Empirical evidence.** The `[tcn.ablations.optimizer.sgd_lr_1e-2]` variant
keeps every other baseline setting and only raises LR back to `1e-2`. On the
current full-tier dataset it **converges and outperforms baseline** (0.9787
vs 0.9752, +0.0035 — second-best variant in the ablation, behind only
`adam_batchnorm` at 0.9792). The `‖v‖→0` failure mode that motivated the
lower LR is therefore knife-edge rather than guaranteed; the safety margin
of `lr = 1e-3` is kept because the original divergence mode is data-shape
dependent and may resurface on tier rebuilds.

### 4. Gradient clipping: added, not in paper

**Baseline.** `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` in
`training.py:257`, with an in-code comment referencing the WeightNorm `‖v‖→0`
failure mode.

**Why.** Additional safety against rare spikes that would otherwise wipe out a
run. It is not sufficient on its own (see §3), but in combination with
BCEWithLogits + lr=1e-3 it stabilises every variant in the ablation study.
Plus the `torch.isfinite(loss)` skip-guard in `_train_epoch` that silently
drops a batch rather than crashing the whole run.

### 5. Chunk length: 128 frames vs. paper's 270

**Paper.** `seq_len = 270` (~6.3 s at 512 hop / 22050 sr).

**Baseline.** `seq_len = 128` (~3.0 s).

**Why.** `CLAUDE.md` records that 270 silently dropped 61 % of our speech
clips (they were shorter than the chunk and the chunker rejected them),
producing a dataset that was almost exclusively music. Reducing to 128
preserves class balance at the cost of a receptive-field mismatch: the
baseline TCN has `RF = 3 × 15 × 4 + 1 = 181` frames, so with a 128-frame chunk
the model never sees its full receptive-field context during training. The
`seq_len_256` and `seq_len_270` ablation variants (now commented out in
`config.toml`) explored the opposite direction; both landed within ±0.002 of
baseline. The `small_tcn_stacks` experiment exploits the gap from the other
side: `stacks_1` reduces RF to 121 frames, fitting inside `seq_len = 128`,
which is the only TCN variant in the project where train-time RF matches
inference-time RF (`SmallerTCN`, see `src/exp/small_tcn_stacks/results/notes.md`).

### 6. Post-processing duration thresholds (paper §3.6): not implemented

**Paper.** Minimum speech/music event durations and minimum break durations,
tuned on the training set, used to smooth frame-level predictions. Large
improvement on event-level evaluation, marginal on segment-level.

**Baseline.** Not implemented. Out of scope for the ablation study, which uses
segment-level macro F1 as its primary metric.

### 7. Hyperparameter search: OFAT coordinate-ascent vs. paper's TPE Bayesian

**Paper.** Tree-of-Parzen-Estimators Bayesian optimisation over the full
hyperparameter grid, restricted to 1 M parameters in Phase 1.

**Baseline.** One-factor-at-a-time (OFAT) ablations rooted at a paper-faithful
baseline, optionally composed via coordinate-ascent
(`tcn-ablation coord-ascent`). OFAT covers every hyperparameter range in
Table 2 of the paper but does not explore the interactions TPE would catch.
The composition is then exercised by the dedicated `tcn_combined` experiment
which stacks the source-experiment winners in a 2² factorial.

**Why.** Compute budget. OFAT also gives clean per-axis plots for the thesis.

### 8. Dataset: our HF dataset vs. paper's compiled collection

**Paper.** MUSAN + GTZAN + SSMSC + OFAI + MuSpeak + ESC + Sveriges Radio,
~156 h combined, with a low-/high-quality split and a pre-training
on LQ → fine-tune on HQ strategy.

**Baseline.** `Marek324/speech-music-classification`, built from Hugging Face
sources per `scripts/dataset/sources.toml`. Currently running on the full
tier (~100 h). Single-stage training (no LQ pre-training strategy).

**Why.** The paper's sources are partially closed / partially difficult to
redistribute, and the research question here is architectural rather than
a replication of the paper's dataset pipeline.

### 9. Framework: PyTorch `nn.utils.weight_norm` vs. paper's keras-tcn

**Paper.** `keras-tcn` (Philippe Rémy), TensorFlow/Keras.

**Baseline.** Native PyTorch: `nn.Conv1d` with `nn.utils.weight_norm` applied
manually in `blocks.py::TCNResidualBlock`. Block layout mirrors keras-tcn:
two dilated causal convolutions, each followed by WeightNorm and ReLU +
dropout, with a residual add and a final ReLU.

**Why.** The rest of the codebase is PyTorch, and a native implementation
lets us own the training loop, streaming inference (`streaming.py`), and the
ablation harness without cross-framework bridges.

### 10. Augmentation: gain only in log-mel space vs. paper's Schlüter & Grill 2015 pipeline

**Paper.** §3.7 cites the Schlüter & Grill (2015) pipeline: time stretching,
pitch shifting, Gaussian *filtering* (smoothing), loudness manipulation, and
block mixing — all applied on the saved power-spectrum features.

**Baseline.** Only loudness manipulation is implemented. `augment_mel` in
`augmentation.py` picks a per-example gain in ±6 dB with p=0.5 and adds it to
the normalized log-mel as `gain_dB · ln(10)/10 / norm_std`. This is
mathematically equivalent to a waveform gain followed by the existing log-mel
+ normalization pipeline, so from the model's perspective it matches
"loudness manipulation on spectrograms" from §3.7.

Time/pitch stretch, Gaussian filtering, and block mixing are not implemented.
A prior revision additionally applied waveform-domain additive Gaussian noise
at 30 dB SNR; this had no paper counterpart (Schlüter & Grill uses Gaussian
*filtering*, not *noise*) and was dropped in the mel-precompute refactor.

**Why.** Time/pitch stretching and block mixing require operating on
pre-mel power spectra; the refactor caches post-mel features for performance
and simplicity. Adding them would need a second cache tier (pre-mel) and a
GPU-side resampling kernel — out of scope for the current study. The
`augmentation.no_augment` ablation variant (now commented out — landed within
0.001 of baseline) measured the gain's contribution as effectively zero.

### 11. Validation loss measured as chunk-level mean

**Paper.** §3.5 says LR schedule and early stopping are triggered by
"validation loss did not improve". The unit is not specified.

**Baseline.** The mel-precompute refactor batches validation the same way it
batches training: mean BCE over fixed-size mel chunks. The previous revision
computed validation loss per full clip (one forward pass per clip, each clip
weighted equally regardless of length); the new chunk-level mean weights
longer clips more. Either definition is monotone in model quality, so the
`ReduceLROnPlateau` and early-stopping semantics are unchanged; only the
absolute numerical scale shifts.

### 12. Production recipe = ablation baseline (paper-faithful)

**Production TCN** (`/home/marek/bp/config.toml`): `optimizer = "sgd"` (m=0.9),
`lr = 1e-3`, `use_weight_norm = true` → WeightNorm, `dropout = 0.5`. Identical
to the ablation baseline (`src/exp/tcn_ablation/config.toml [tcn]`).

**Reported metric.** The deployed `weights/tcn.safetensors` checkpoint
evaluates to **macro F1 = 0.9752** [0.9687, 0.9793] on the full-tier test
split (`results/tcn.eval`). This clears the project target of 0.85 by 12 pp.

**Why one recipe and not two.** Earlier revisions of this project ran a
divergent production stack (Adam + BatchNorm + dropout=0.1) on top of the
paper-literal ablation baseline. Two findings collapsed that split:

1. The `regularization.dropout_low` ablation winner (dropout=0.1, +0.0030)
   was small enough to land inside baseline's CI under the current data
   pipeline. Adopting it in production would have shifted absolute numbers
   without changing the qualitative story.
2. The `optimizer.adam_batchnorm` variant tops the ablation board (+0.0040)
   but couples optimizer + normalization in a way that breaks the
   paper-comparability of the rest of the recipe. Keeping the production
   recipe paper-faithful preserves the ablation study as a *direct* test of
   each deviation in this document.

**Frozen vs configurable.** The deployed checkpoint is the paper-faithful
recipe. Adam+BN, dropout=0.1, and lr=1e-2 are all available as ablation
variants and produce slightly higher F1 in isolation, but none have been
promoted to production because the gains do not compose cleanly (see
`tcn_combined`, where the architectural winners F+P+A recover only ~74 %
of additivity).

## Summary table of ablation subgroups

Only the five subgroups that produced signal are kept active in
`src/exp/tcn_ablation/config.toml`. The rest were executed once, landed
within noise of baseline, and have been commented out so they don't clutter
future re-runs or plots — their eval files remain in `results/` for reference.

### Active subgroups (real signal)

| Subgroup | Paper range / default | Variants | Outcome |
|---|---|---|---|
| `optimizer` | SGD m=0.9 | `adam`, `batch_norm`, `adam_batchnorm`, `sgd_batchnorm`, `sgd_lr_1e-2` | bare `adam` collapses to a trivial predictor (0.172, speech-only); `adam_batchnorm` rescues it and tops the board (0.9792, +0.0040); `sgd_lr_1e-2` survived and lands second (0.9787, +0.0035); `sgd_batchnorm` regresses (0.9706, −0.0046) |
| `layers` | `n_layers ∈ 1..4` | `layers_1`, `layers_2`, `layers_3` | clear depth effect: `layers_1` drops to 0.9573 (RF=13 too small), `layers_2` to 0.9723 (RF=37), `layers_3` to 0.9736 (RF=85) |
| `activation` | (ReLU in keras-tcn) | `leaky_relu`, `elu`, `gelu` | `elu` drops to 0.9639 (negative saturation hurts); ReLU/LeakyReLU/GELU tied within 0.0004 |
| `regularization` | dropout ∈ [0.05, 0.5] | `dropout_low` (0.1), `dropout_medium` (0.25) | `dropout_low` beats baseline by +0.0030 (0.9782) — real but inside CI; not adopted in production to keep the recipe paper-literal (see §12) |
| `skip_connections` | true/false | `no_skip` | catastrophic without skips: collapses to music-only (0.1884) |

### Pruned subgroups (no signal, kept as commented-out blocks in config.toml)

| Subgroup | Reason pruned |
|---|---|
| `loss` | all 5 variants within ±0.0003 F1 of baseline |
| `capacity` (`n_filters`) | `filters_8`/`filters_32` within ±0.001 |
| `stacks` | `stacks_5` within 0.001 |
| `kernel` | all kernel widths within ±0.003 |
| `training` (`seq_len`) | `seq_len_256`/`seq_len_270` within ±0.002 |
| `batch_size` | `batch_16`/`batch_64` within ±0.002 |
| `augmentation` | `no_augment` within 0.001 |
| `n_mels` | eval crashes — cached preprocess stats are baked to 80 mels; needs per-mel-count stats to test properly. Note: the cache-key fix (`_nf{n_features}` baked into the path, `training.py:108`) addresses the *MFCC* case in the `tcn_frontend` experiment, where `mfcc_20` and `mfcc_40` now train cleanly to 0.9752 / 0.9764. The `n_mels` variants here remain blocked because they need separate **preprocess** stats files, not just a different cache key. |
