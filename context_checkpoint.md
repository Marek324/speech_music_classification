# Context Checkpoint — 2026-03-12

## What We Were Trying to Do
Train a causal TCN (Lemaire & Holzapfel, ISMIR 2019) for online speech/music detection to beat the existing DT/GMM/SVM baselines (≥ 0.88 macro F1). The model must remain causal (usable for streaming/online inference) and stay within the paper's architectural bounds.

---

## Bugs Fixed

### 1. `cli.py::_eval_tcn_on_n_rows` — stale clip-level label logic
**Problem:** Used the old `targets[1, 0].item()` (one label per clip) instead of per-frame labels.
**Fix:** Replaced with the same per-frame `speech_mask`/`music_mask` logic used in `evaluation.py`.

### 2. `dataset.py::_timestamps_to_frame_labels` — wrong label keys
**Problem:** Code used `entry["start_ms"]` / `entry["end_ms"]` but the HF dataset schema uses `entry["start"]` / `entry["end"]` (values still in ms).
**Fix:** Updated keys to `start` / `end`.

### 3. STFT center-padding frame count mismatch
**Problem:** `torchaudio.MelSpectrogram` defaults to `center=True`, giving `T = N // hop + 1` output frames. `dataset.py` computed `n_frames = (N - n_fft) // hop + 1` (center=False formula) — off by ~2 frames per clip. Caused `y_true` / `y_pred` length mismatch in evaluation.
**Fix:** Added `T_actual = tgt.shape[-1]` after slicing targets in `evaluation.py`, `cli.py`, and `training.py`. All arrays now trimmed to the overlap.

### 4. GPU inference not wired up
**Problem:** `evaluation.py::get_predictions` and `cli.py::_eval_tcn_on_n_rows` loaded weights to CPU and never moved the model to GPU.
**Fix:** Added `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` after `model.eval()` in both; `wav.to(device)` in the inference loops; `.cpu()` before `.numpy()` calls.

### 5. `streaming.py::StreamingInference` — buffer on wrong device
**Problem:** Buffer was always created on CPU (`torch.zeros(1, 0)`), causing device mismatch when model is on GPU.
**Fix:** `device = next(model.parameters()).device`; buffer created on that device; incoming chunks moved to `self.buffer.device` before concat.

### 6. SEQ_LEN=270 silently discarded 61% of speech clips
**Problem:** The training chunk iterator uses `SEQ_LEN=270` frames ≈ 6.3 seconds. Speech clips have a **median length of ~165 frames** (≈ 3.8s). 1696 of 2795 speech clips (61%) were shorter than one chunk and contributed **zero training data**. Music clips are all ~1290 frames so all contributed. Training became almost exclusively music → model either always predicted speech (no weighting) or always predicted music (with 4.5× music weighting).
**Fix:** Reduced `SEQ_LEN` to **128 frames** (≈ 3 seconds). Most speech clips now yield at least one chunk.

---

## Other Changes Made

### Model config — brought within paper bounds (`config.toml`)
The original config was outside the paper's hyperparameter search space:

| Param | Before | After | Paper range |
|-------|--------|-------|-------------|
| n_filters | 32 | 16 | 8, 16, 32 |
| n_layers | **6** (out of bounds) | 4 | 1–4 |
| n_stacks | **2** (out of bounds) | 3 | 3–10 |
| kernel_size | 5 | 5 | 3, 5, 7, ... 19 |
| dropout | 0.3 | 0.3 | 0.05–0.5 |

### Training loop overhaul (`training.py`)
- **Chunk-based mini-batches** (paper §3.5): clips are sliced into fixed-length windows, accumulated into batches of 32. `stride = seq_len * hop` for contiguous non-overlapping target coverage.
- **Optimizer**: switched from Adam lr=1e-4 to **Adam lr=1e-3** with **ReduceLROnPlateau** (patience=3, factor=0.1) per paper §3.5. (SGD with momentum=0.9 was tried but caused training instability at this data scale — see Failed Experiments.)
- **Augmentation** (`augmentation.py`): random ±6dB gain + Gaussian noise (p=0.5 each), applied to training batches only.
- **Best-checkpoint saving**: already present before — saves the state with lowest val loss.
- **LR logged** to wandb each epoch.

---

## Failed Experiments

### SGD + momentum=0.9 (paper §3.5)
Paper uses SGD. Tried it with lr=0.01 → epoch-2 train loss *increased* (1.24 → 2.10), val loss never stabilized. Tried lr=0.001 with weighted loss → val loss spiked to 23 at epoch 1. Root cause: SGD with momentum amplifies gradient overshoots; Adam's adaptive rates are more robust at this data scale (16h vs paper's 125h).

### WeightedBCELoss (music_pos_weight=4.5)
Motivated by 4:1 speech:music clip ratio. Caused catastrophic overcorrection: model predicted music for 95% of speech frames. Root cause: combined with the chunk-imbalance bug (§6 above), music was already over-represented in training data. Removing the weighted loss and fixing SEQ_LEN is the correct approach.

---

## Current State of the Code

### What's in place and working
- Full pipeline: `tcn train` → `tcn eval` → `results/tcn.eval`
- Smoke test: `uv run smclassifier tcn smoke-test-online` passes clean
- GPU inference wired throughout (training, evaluation, streaming)
- Frame-level labels used everywhere (no clip-level label shortcuts)
- Augmentation active during training
- Best val-loss checkpoint saved (not final-epoch weights)

### Last eval result (2026-03-12 18:48 UTC)
This was produced by a model trained with the **buggy SEQ_LEN=270 + WeightedBCELoss(4.5)** — worst result of the session. It's not representative of the current code.

```
2-class F1:  0.209   (target: ≥ 0.88)
Accuracy:    0.242
Confusion:   [[ 3458 78018]   ← model predicted music for 95% of speech frames
              [63994 41796]]
```

### Current training config
```toml
[tcn]
sample_rate = 22050
n_fft = 1024
hop_length = 512
n_mels = 80

[tcn.model]
n_filters = 16
kernel_size = 5
n_layers = 4
n_stacks = 3
dropout = 0.3
n_classes = 2
```

```python
# training.py
SEQ_LEN   = 128        # frames per chunk (~3s) — fits median speech clip
BATCH_SIZE = 32
optimizer = Adam(lr=1e-3)
scheduler = ReduceLROnPlateau(patience=3, factor=0.1)
loss      = BCELoss()  # unweighted
augment   = random_gain(±6dB) + gaussian_noise  # p=0.5 each
```

---

## Dataset Facts

| Split | Speech | Music | Noise |
|-------|--------|-------|-------|
| train | 2795 | 673 | 231 |
| val | 349 | 79 | 25 |
| test | 353 | 82 | 25 |

- **Training data: ~16 hours** (paper used ~125h — 8× more)
- Speech clips: median **165 frames** (≈ 3.8s), avg 294, max 855
- Music clips: median **1290 frames** (≈ 30s), all ≥ 1289
- Labels: frame-level timestamps, keys `start`/`end` in milliseconds
- Inactive frames (neither speech nor music) are present within clips

---

## Next Steps (Priority Order)

### 1. Run a full `tcn train` with the current config
The SEQ_LEN fix is the most important change. A clean training run will show whether the model now learns both speech and music. Check wandb for stable, converging train and val loss on both classes.

### 2. If still overfitting: increase dropout or add weight decay
Current dropout=0.3 is mid-range for the paper. Try 0.4 or 0.5. Adam weight_decay=1e-4 is another low-cost knob.

### 3. If model capacity is a bottleneck: try n_filters=32
We reduced from 32→16 to fight overfitting. If the SEQ_LEN fix resolves training stability and the model is underfitting, move n_filters back to 32 (still within paper bounds).

### 4. If still stuck below 0.88: implement spec augmentation
The paper (§3.4) uses time stretching, pitch shifting, Gaussian filtering on spectrograms, loudness, and block mixing. Currently only gain + noise are implemented. SpecAugment-style frequency/time masking would be the next augmentation to add.

### 5. Longer-term: pre-training on low-quality clip-level data (paper Strategy 4)
The paper's best result (val loss 0.070 vs 0.096 without) came from pre-training on low-quality (clip-level labelled) data then fine-tuning on frame-level HQ data. If the dataset has clip-level labelled data available, this is the single highest-impact remaining lever.
