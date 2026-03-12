# Project: butfit-bp — Speech/Music Classifier

## Goal
Beat the existing DT/GMM/SVM baselines (≥ **0.88 macro F1**) with a causal TCN that can run online (streaming). Currently overfitting; training is ongoing.

## CLI
```
uv run smclassifier tcn <command>
```
Commands: `train`, `eval`, `smoke-test-online`

## Architecture
Causal TCN — **must stay as close to Lemaire & Holzapfel ISMIR 2019 as possible** (`src_papers/tcn.pdf`).

### Paper-prescribed hyperparameter bounds
| Parameter | Paper search space | Current config |
|-----------|-------------------|----------------|
| n_filters | 8, 16, 32 | 16 |
| n_layers  | 1–4 | 4 |
| n_stacks  | 3–10 | 3 |
| kernel_size | 3, 5, 7, ..., 19 | 5 |
| dropout | 0.05–0.5 | 0.3 |

### Paper-prescribed training (§3.5)
- Optimizer: SGD, momentum=0.9 *(currently using Adam due to stability issues with 16h dataset)*
- LR schedule: ÷10 when val loss doesn't improve for 3 epochs (`ReduceLROnPlateau`)
- Early stopping: 5 epochs without improvement
- Batch: 32 × fixed-length chunks
- Loss: binary cross-entropy

## Dataset
HuggingFace: `Marek324/speech-music-classification`

| Split | Speech | Music | Noise | Total |
|-------|--------|-------|-------|-------|
| train | 2795 | 673 | 231 | 3699 |
| val | 349 | 79 | 25 | 453 |
| test | 353 | 82 | 25 | 460 |

- Training split ≈ **16 hours** (paper used ~125h)
- **Speech clips are short** (median ~165 frames ≈ 3.8s), music clips are long (~1290 frames ≈ 30s)
- Frame-level labels (`start`/`end` keys in ms, not `start_ms`/`end_ms`)

## Critical Known Issues

### Chunk-length mismatch (fixed)
`SEQ_LEN=270` (6.3s) silently dropped 61% of speech clips — they were too short to yield a single chunk. Training became almost exclusively music → model predicted music for everything. **Fixed by reducing `SEQ_LEN` to 128 (3s)**.

### STFT center-padding offset (fixed)
`torchaudio.MelSpectrogram` uses `center=True` (default), giving `T = N//hop + 1` frames. `dataset.py` computes `n_frames = (N - n_fft) // hop + 1` (center=False formula) — off by ~2 frames. Fixed in `evaluation.py`, `cli.py`, and `training.py` using `T_actual = tgt.shape[-1]`.

### Dataset label keys (fixed)
Dataset uses `start`/`end` keys (in ms), not `start_ms`/`end_ms`. Fixed in `dataset.py`.

## File Map
| File | Purpose |
|------|---------|
| `src/tcn/training.py` | Training loop, chunk iterator, optimizer |
| `src/tcn/evaluation.py` | Inference + metrics for full test split |
| `src/tcn/cli.py` | Click CLI: train, eval, smoke-test-online |
| `src/tcn/model.py` | `SpeechMusicDetector` (waveform → probs) + `CausalTCN` |
| `src/tcn/dataset.py` | HF dataset loader, frame-label builder |
| `src/tcn/preprocess.py` | `LogMelSpectrogram` — 22050Hz, 1024/512 STFT, 80-mel, z-norm |
| `src/tcn/streaming.py` | `StreamingInference` — online chunk-by-chunk inference |
| `src/tcn/augmentation.py` | `augment()` — random gain ±6dB + Gaussian noise |
| `src/tcn/config.py` | Config loader; reads `[tcn]` section of `config.toml` |
| `config.toml` | All hyperparameters — `[tcn]` and `[tcn.model]` sections |
| `weights/tcn.safetensors` | Trained model weights |
| `weights/tcn_preprocess_stats.pt` | Log-mel normalization mean/std (computed from train set) |
| `results/tcn.eval` | Last evaluation report |

## Inference
Model needs **two files** to run:
1. `weights/tcn_preprocess_stats.pt` — loaded automatically by `LogMelSpectrogram.__init__`
2. `weights/tcn.safetensors` — loaded explicitly by eval/CLI code

Labels: `-1` = speech, `1` = music, `2` = inactive
