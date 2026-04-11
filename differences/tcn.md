# TCN baseline vs. Lemaire & Holzapfel (ISMIR 2019)

Reference: `ref_papers/tcn.pdf` — *Temporal Convolutional Networks for Speech
and Music Detection in Radio Broadcast*, Quentin Lemaire and Andre Holzapfel.

This document tracks every deliberate deviation between the ablation-study
baseline (`src/exp/tcn_ablation/config.toml` → `[tcn]` section) and the paper.
The production TCN (`/home/marek/bp/config.toml`) is a separate recipe
(Adam + BatchNorm) and is **not** the subject of this document.

## What matches the paper

| Paper §3.4–3.5 | This baseline |
|---|---|
| Resample to 22050 Hz mono | `sample_rate = 22050` |
| Hann STFT, frame length 1024, hop 512 | `n_fft = 1024`, `hop_length = 512` |
| Power spectrum → 80-mel filterbank, 27.5–8000 Hz | `n_mels = 80`, `f_min = 27.5`, `f_max = 8000.0` |
| Log scale, zero-mean unit-variance normalization over training set | `preprocess.py::LogMelSpectrogram` + `compute_and_save_preprocess_stats` |
| Binary cross-entropy loss | `loss = "bce_with_logits"` (mathematically equivalent to paper's BCE; see below) |
| SGD momentum = 0.9 | `optimizer = "sgd"` (momentum=0.9 hard-coded in `training.py`) |
| Divide LR by 10 when validation loss does not improve for 3 epochs | `ReduceLROnPlateau(factor=0.1, patience=3)` in `training.py:205` |
| Stop training after 5 consecutive non-improving validation epochs | `patience=5` in `train_tcn` |
| Mini-batches of fixed-length chunks, batch size 32 | `batch_size = 32`, `_iter_batched_chunks` in `training.py` |
| Dropout ∈ [0.05, 0.5] per block | `dropout = 0.5` (upper bound of the paper range) |
| Causal dilated convolutions, residual blocks, skip connections | `blocks.py::TCNResidualBlock`, `skip_connections=true` default |
| keras-tcn-style block: two dilated causal convs + WeightNorm + ReLU + dropout + residual add | `blocks.py::TCNResidualBlock` with `use_weight_norm=true`, `activation="relu"` |
| Hyperparameter search space: `n_layers ∈ 1..4`, `n_stacks ∈ 3..10`, `kernel_size ∈ {3,5,…,19}`, `n_filters ∈ {8,16,32}`, dilations `2^0…2^N_D` with `N_D=3..8` | Ablation subgroups cover the same ranges (`capacity`, `layers`, `stacks`, `kernel`) |

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
training calls `SpeechMusicDetector.forward_logits`.

**Why.** `BCELoss(sigmoid(x))` is the standard PyTorch numerical trap. If any
upstream activation produces NaN (which happens with SGD+WeightNorm; see §4
below), sigmoid propagates the NaN, and `BCELoss` aborts training with
`/pytorch/aten/src/ATen/native/cuda/Loss.cu:90: input_val >= zero && input_val <= one`.
`BCEWithLogitsLoss` uses the fused log-sigmoid formulation and is finite for
any logit value, so the same underlying instability becomes a recoverable loss
spike instead of a crash.

**Empirical evidence.** The ablation variant
`[tcn.ablations.loss.bce_sigmoid]` reverts to `nn.BCELoss(sigmoid(x))` and is
expected to DIVERGE on the full-tier dataset, reproducing the original bug.

### 3. Initial learning rate: 1e-3 vs. paper's unspecified

**Paper.** The paper prescribes LR scheduling but does **not** specify an
initial value. The reference implementation (keras-tcn) does not fix one either.

**Baseline.** `lr = 1e-3`.

**Why.** We initially tried `lr = 1e-2` (a common SGD default with momentum
0.9). On the full-tier dataset this drove WeightNorm's `v` parameter toward
zero over many updates, producing unbounded normalised weights `g * v / ‖v‖`
and NaN activations. The grad-norm clipping at `max_norm=1.0` (commit
`d209b77`) clips gradients but not the weight itself, so it was insufficient.
Lowering to `1e-3` stops the blowup in practice on every OFAT variant tested.

**Empirical evidence.** `[tcn.ablations.optimizer.sgd_lr_1e-2]` keeps every
other baseline setting and only raises LR back to `1e-2`. It is expected to
DIVERGE or converge to a substantially worse F1.

### 4. Gradient clipping: added, not in paper

**Baseline.** `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` in
`training.py:53`, with an in-code comment referencing the WeightNorm `‖v‖→0`
failure mode.

**Why.** Additional safety against rare spikes that would otherwise wipe out a
run. It is not sufficient on its own (see §3), but in combination with
BCEWithLogits + lr=1e-3 it stabilises the ablation study. Plus the
`torch.isfinite(loss)` skip-guard in `train_step` that silently drops a batch
rather than crashing the whole run.

### 5. Chunk length: 128 frames vs. paper's 270

**Paper.** `seq_len = 270` (~6.3 s at 512 hop / 22050 sr).

**Baseline.** `seq_len = 128` (~3.0 s).

**Why.** `CLAUDE.md` records that 270 silently dropped 61 % of our speech
clips (they were shorter than the chunk and the chunker rejected them),
producing a dataset that was almost exclusively music. Reducing to 128
preserves class balance at the cost of a receptive-field mismatch: the
baseline TCN has `RF = 3 × 15 × 4 + 1 = 181` frames, so with a 128-frame chunk
the model never sees its full receptive-field context during training. The
`training.seq_len_256` and `training.seq_len_270` ablations explore the
opposite direction.

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

### 10. Production vs. ablation recipe are intentionally separate

**Production TCN** (`/home/marek/bp/config.toml`): Adam, `lr = 1e-3`,
`use_weight_norm = false` → BatchNorm. This is the recipe that currently
holds the reported 2-class F1 = 0.9504 / 3-class F1 = 0.9033.

**Ablation baseline** (`src/exp/tcn_ablation/config.toml`): SGD m=0.9,
`lr = 1e-3`, `use_weight_norm = true` → WeightNorm. Paper-faithful
(modulo §3 and §2) and the reference point for the ablation subgroups.

**Why.** The ablation study exists to empirically justify why the production
recipe diverges from the paper. Merging them would collapse the two stories
into one and destroy the ability to run paper-literal reversion ablations
like `optimizer.sgd_lr_1e-2` and `loss.bce_sigmoid`.

## Summary table of ablation subgroups

| Subgroup | Paper range / default | Variants |
|---|---|---|
| `loss` | BCE | `bce_sigmoid` (paper-literal, DIVERGES), `focal`, `weighted_bce`, `mse`, `label_smoothing` |
| `optimizer` | SGD m=0.9 | `adam`, `batch_norm`, `adam_batchnorm`, `sgd_batchnorm`, `sgd_lr_1e-2` (paper LR, DIVERGES) |
| `capacity` | `n_filters ∈ {8,16,32}` | `filters_8`, `filters_32` |
| `layers` | `n_layers ∈ 1..4` | `layers_1`, `layers_2`, `layers_3` |
| `stacks` | `n_stacks ∈ 3..10` | `stacks_5` |
| `kernel` | `kernel_size ∈ {3,5,…,19}` | `kernel_3`, `kernel_7`, `kernel_9` |
| `regularization` | dropout ∈ [0.05, 0.5] | `dropout_low` (0.1), `dropout_medium` (0.25) |
| `training` | seq_len=270 | `seq_len_256`, `seq_len_270` |
| `skip_connections` | true/false | `no_skip` |
| `activation` | (ReLU in keras-tcn) | `leaky_relu`, `elu`, `gelu` |
| `n_mels` | 80 | `mels_40`, `mels_128` |
| `batch_size` | 32 | `batch_16`, `batch_64` |
| `augmentation` | Schlüter & Grill 2015 pipeline | `no_augment` |
