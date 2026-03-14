# Project: butfit-bp — Speech/Music Classifier

## Goal
Beat the existing DT/GMM/SVM baselines (≥ **0.88 macro F1 on 2-class evaluation**) and achieve >= **.85 F1 macro on 3-class evaluation** with a causal TCN that can run online (streaming). Currently overfitting; training is ongoing.

## CLI
```
uv run smclassifier tcn <command>          # TCN model
uv run smclassifier classic <model> <command>  # classic models (decision_tree, gmm, svm)
```
TCN commands: `train`, `eval`, `smoke-test-online`
Classic commands: `train`, `eval`, `smoke-test`, `mic` (placeholder)

## Architecture
Causal TCN — **must stay as close to Lemaire & Holzapfel ISMIR 2019 as possible** (`src_papers/tcn.pdf`).

### Paper-prescribed hyperparameter bounds
| Parameter | Paper search space | Current config |
|-----------|-------------------|----------------|
| n_filters | 8, 16, 32 | 16 |
| n_layers  | 1–4 | 4 |
| n_stacks  | 3–10 | 3 |
| kernel_size | 3, 5, 7, ..., 19 | 5 |
| dropout | 0.05–0.5 | 0.5 |

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

### Shared
| File | Purpose |
|------|---------|
| `src/evaluator.py` | `run_evaluation()`, `format_report()`, metrics dataclasses — shared by all models |
| `src/input_handler.py` | HF dataset loader → frame extraction → `X, y, subclasses` arrays (classic models) |
| `src/common.py` | Label mapping: speech=−1, music=1, inactive/noise=2; `frame_label_str()` |
| `src/config.py` | Global config singleton; `init_config(model_name)` selects per-model buffer settings |
| `config.toml` | All hyperparameters — `[buffers.*]`, `[features.*]`, `[tcn]`, `[tcn.model]` |
| `results/*.eval` | Evaluation reports for each model |

### TCN (`src/tcn/`)
| File | Purpose |
|------|---------|
| `blocks.py` | `CausalConv1d` + `TCNResidualBlock` — causal dilated conv building blocks |
| `model.py` | `SpeechMusicDetector` (waveform → probs) + `CausalTCN` |
| `preprocess.py` | `LogMelSpectrogram` — 22050Hz, 1024/512 STFT, 80-mel, z-norm |
| `dataset.py` | HF dataset loader, frame-label builder |
| `training.py` | Training loop, chunk iterator, optimizer |
| `evaluation.py` | Inference + metrics for full test split |
| `streaming.py` | `StreamingInference` — online chunk-by-chunk inference |
| `augmentation.py` | `augment()` — random gain ±6dB + Gaussian noise |
| `cli.py` | Click CLI: train, eval, smoke-test-online |
| `config.py` | Config loader; reads `[tcn]` section of `config.toml` |
| `weights/tcn.safetensors` | Trained model weights |
| `weights/tcn_preprocess_stats.pt` | Log-mel normalization mean/std (computed from train set) |

### Classic models (`src/classic/`)
| File | Purpose |
|------|---------|
| `modelclass.py` | Abstract base: `fit()`, `save()`, `load()` via joblib |
| `feat_extractor.py` | `FeatExtractor` — stateful frame-level feature extraction (30ms/15ms hop); **call `reset()` between clips** |
| `gmm.py` | Two-GMM density classifier (speech GMM vs music GMM); 8 components, diag covariance |
| `decisiontree.py` | Decision tree with exponential-forgetting smoothing on last decisions |
| `svm.py` | `SGDClassifier` with hinge loss (linear SVM) |
| `evaluation.py` | `eval_classic()` — loads model + test features, runs `run_evaluation()` |
| `cli.py` | Click CLI per model: train, eval, smoke-test |
| `weights/gmm`, `weights/decision_tree`, `weights/svm` | Joblib-serialized model weights |

### Scripts (`scripts/`) — uv subproject
| File | Purpose |
|------|---------|
| `visualize_results.py` | Parses all `results/*.eval` files, plots macro F1 comparison + per-subclass breakdown → `results/results.png` |

### Dataset scripts (`dataset_scripts/`) — uv subproject
| File | Purpose |
|------|---------|
| `build.py` | Builds and uploads the HF dataset |
| `augment_over_music.py` | Generates speech-over-music augmented samples |

## Paper vs Implementation Differences

| Aspect | Paper (Lemaire & Holzapfel 2019) | This implementation |
|--------|----------------------------------|---------------------|
| Optimizer | SGD, momentum=0.9 | Adam, lr=1e-3, weight_decay=1e-4 |
| Normalization | WeightNorm (keras-tcn reference) | `BatchNorm1d` after each conv |
| Post-processing | Duration thresholds (§3.6) to smooth predictions | Not implemented |
| Receptive field vs chunk | RF = 3×(1+2+4+8)×(5−1)+1 = **181 frames**; chunks = 128 frames | Model never sees its full receptive-field context during training |

## Inference

### TCN
Needs two files:
1. `weights/tcn_preprocess_stats.pt` — loaded automatically by `LogMelSpectrogram.__init__`
2. `weights/tcn.safetensors` — loaded explicitly by eval/CLI code

### Classic models
Weights loaded via `model.load()` from `weights/<model_name>` (joblib pickle).
Feature extraction is stateful — `FeatExtractor.reset()` must be called between clips.

Labels: `-1` = speech, `1` = music, `2` = inactive

## Status
- **TCN**: done — 2-class F1=0.9504, 3-class F1=0.9033 (both targets met)
- **Classic baselines**: done — DT/SVM/GMM evaluated; GMM was broken (buffer not reset between clips, now fixed; needs retraining)
- **Next**: new module (TBD)
