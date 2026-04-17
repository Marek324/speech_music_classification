# Project: butfit-bp — Speech/Music Classifier

## Goal
Beat the existing DT/GMM/SVM baselines and achieve >= **.85 F1 macro on 3-class evaluation** with a causal TCN that can run online (streaming). Currently overfitting; training is ongoing.

## CLI
```
uv run smclassifier nn tcn <command>          # TCN model
uv run smclassifier nn own <command>          # custom model (placeholder)
uv run smclassifier classic <model> <command>  # classic models (decision_tree, gmm, svm)
```
TCN commands: `train`, `eval`, `smoke-test`, `smoke-test-online`
Own commands: `train`, `eval`, `smoke-test` (all NotImplementedError placeholders)
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
HuggingFace: `Marek324/speech-music-classification`. TCN is configured for the **full** tier in `config.toml` → `[dataset] name = "full"`. The mid-tier tables below describe a previously uploaded snapshot and are kept for historical reference; full-tier counts differ.

### Mid-tier clip counts (historical snapshot, previous HF upload)

| Split | Rows | Minutes |
|-------|------|---------|
| train | 3226 | 951 min |
| val   | 411  | 122 min |
| test  | 405  | 121 min |

### Mid-tier minutes per subclass (historical snapshot)

| Subclass | Target | Train | Val | Test |
|----------|--------|-------|-----|------|
| speech_clean | 192 | 154.7 (741) | 18.6 (93) | 18.8 (92) |
| speech_dirty | 96 | 77.1 (390) | 9.9 (49) | 9.0 (49) |
| speech_multispeaker | 36 | 21.2 (433) | 2.6 (54) | 2.5 (54) |
| music_instrumental | 72 | 58.0 (116) | 7.5 (15) | 7.0 (14) |
| music_electronic | 72 | 58.0 (116) | 7.5 (15) | 7.0 (14) |
| music_pop | 72 | 58.0 (116) | 7.5 (15) | 7.0 (14) |
| music_rock | 72 | 58.0 (116) | 7.5 (15) | 7.0 (14) |
| music_acapella | 48 | 35.5 (29) | 5.2 (4) | 7.8 (4) |
| music_hip-hop | 72 | 58.0 (116) | 7.5 (15) | 7.0 (14) |
| music_folk | 72 | 58.0 (116) | 7.5 (15) | 7.0 (14) |
| noise | 240 | 190.0 (38) | 25.0 (5) | 25.0 (5) |
| speech_som | 36 | 28.8 (143) | 3.6 (18) | 3.6 (17) |
| speech_msom | 24 | 19.2 (390) | 2.5 (50) | 2.4 (52) |
| speech_noisy | 96 | 77.0 (366) | 9.7 (48) | 9.8 (48) |

### Full tier (6000 min / 100 hours)

40% speech / 40% music / 20% inactive. Music is **seven equal HF sources** (six FMA genres + bel_canto acapella), each **2400/7 ≈ 342.86 min**. Mid tier music is the same split at **480/7 ≈ 68.57 min** per genre.

| Subclass | Target (min) |
|----------|-------------|
| speech_clean | 960 |
| speech_dirty | 480 |
| speech_multispeaker | 180 (synthetic, LibriMix-style) |
| speech_som | 180 (synthetic) |
| speech_msom | 120 (synthetic) |
| speech_noisy | 480 (synthetic) |
| Each music genre (×7, equal) | 2400/7 each |
| noise | 1200 |

- Frame-level labels (`start`/`end` keys in ms, not `start_ms`/`end_ms`)

## Critical Known Issues

### Chunk-length mismatch (fixed)
`SEQ_LEN=270` (6.3s) silently dropped 61% of speech clips — they were too short to yield a single chunk. Training became almost exclusively music → model predicted music for everything. **Fixed by reducing `SEQ_LEN` to 128 (3s)**.

### STFT center-padding offset (fixed)
`torchaudio.MelSpectrogram` uses `center=True` (default), giving `T = N//hop + 1` frames. `dataset.py` computes `n_frames = (N - n_fft) // hop + 1` (center=False formula) — off by ~2 frames. Fixed in `evaluation.py`, `cli.py`, and `training.py` using `T_actual = tgt.shape[-1]`.

### Dataset label keys (fixed)
Dataset uses `start`/`end` keys (in ms), not `start_ms`/`end_ms`. Fixed in `dataset.py`.

### Per-epoch HF re-streaming (fixed)
`_iter_batched_chunks` in `training.py` used to call `iter_nn_rows` every epoch, re-decoding and re-chunking the entire HF train split (~5 GB on mid tier, ~32 GB on full tier) 30 times per run. GPU utilization sat at ~10% because the step was CPU/IO-bound. **Fixed** by precomputing mel chunks once in `training.py::_load_mel_chunks`, caching to `cache/nn/tcn_mel_*.pt` (keyed by dataset + frontend params + seq_len), and keeping the cached tensor resident on the training device. `augment_mel` in `augmentation.py` applies gain directly in normalized log-mel space, matching paper §3.4 ("data augmentation was applied to the saved spectrograms"). Shared across ablation variants with matching frontend params.

### Full-tier test split missing subclasses (fixed)
`process.py` previously used per-source cumulative minutes to assign clips to train/val/test sequentially. Sources that exhausted their HF data before reaching `val_cutoff_min` (92% of `target_minutes`) never wrote any clips to the test split. Affected sources in the full tier: `bel_canto` (acapella, 300 min target — small dataset), all FMA genres (750 min each — limited clips per genre on HF), and `DEMAND` noise (2400 min — dataset likely too small). Augmented subclasses (`speech_som`, `speech_msom`, `speech_noisy`) cascaded to zero test clips if their base subclasses had none. Full-tier test split had only ~3 subclasses with data.

**Fixed** by replacing sequential cutoffs and the block cycle with a Bresenham-style interleaving generator (`_bresenham_split_iter` in `process.py`). Each clip is assigned to the split with the highest error accumulator, guaranteeing val and test receive clips from the first few rows regardless of how few clips a source provides.


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

### NN models (`src/nn/`)
| File | Purpose |
|------|---------|
| `blocks.py` | `CausalConv1d` + `TCNResidualBlock` — **shared** causal dilated conv building blocks |
| `dataset.py` | **Shared** HF dataset loader, frame-label builder; params (sr/hop/n_fft) passed explicitly |
| `evaluation.py` | **Shared** `run_nn_inference()` — generic inference loop for any `(1,3,T)`-output model |
| `modelclass.py` | Abstract `NNModelClass` base: `train()`, `evaluate()`, `smoke_test()` |
| `cli.py` | `nn_group` — dispatches to `tcn` and `own` subgroups |

#### TCN (`src/nn/tcn/`)
| File | Purpose |
|------|---------|
| `model.py` | `SpeechMusicDetector` (waveform → probs) + `CausalTCN` |
| `preprocess.py` | `LogMelSpectrogram` — 22050Hz, 1024/512 STFT, 80-mel, z-norm |
| `training.py` | Training loop, mel precompute + disk cache (`_load_mel_chunks`), batch iterator, optimizer |
| `evaluation.py` | Thin wrapper: loads TCN model, calls `nn/evaluation.run_nn_inference` |
| `streaming.py` | `StreamingInference` — online chunk-by-chunk inference |
| `augmentation.py` | `augment_mel()` — random gain ±6dB applied as additive shift in normalized log-mel space (paper §3.4 aug-on-spectrogram path) |
| `cli.py` | Click CLI: train, eval, smoke-test, smoke-test-online |
| `config.py` | Config loader; reads `[tcn]` section of `config.toml`; `_MODEL_KEYS` / `_TOP_KEYS` split ablation overrides into model vs top-level buckets |
| `weights/tcn.safetensors` | Trained model weights *(not in git — download via HF)* |
| `weights/tcn_preprocess_stats.pt` | Log-mel normalization mean/std *(not in git — download via HF)* |
| `cache/nn/tcn_mel_*.pt` | Precomputed mel chunks per (dataset, split, frontend params, seq_len); shared across ablation variants. *(not in git — auto-synced via `cache/` to `Marek324/butfit-bp-artifacts` on HF)* |

#### Own (`src/nn/own/`) — placeholder
| File | Purpose |
|------|---------|
| `model.py` | `OwnModel` stub (raises `NotImplementedError`) |
| `cli.py` | Placeholder CLI: train/eval/smoke-test (all raise `NotImplementedError`) |

### Classic models (`src/classic/`)
| File | Purpose |
|------|---------|
| `modelclass.py` | Abstract base: `fit()`, `save()`, `load()` via joblib |
| `feat_extractor.py` | `FeatExtractor` — stateful frame-level feature extraction (30ms/15ms hop); **call `reset()` between clips** |
| `gmm.py` | Two-GMM density classifier (speech GMM vs music GMM); 8 components, diag covariance |
| `decisiontree.py` | Decision tree with exponential-forgetting smoothing on last decisions |
| `svm.py` | `SVC(kernel='rbf', C=1, γ=3)`; balanced subsampling to 50k/class at train time |
| `evaluation.py` | `eval_classic()` — loads model + test features, runs `run_evaluation()` |
| `cli.py` | Click CLI per model: train, eval, smoke-test |
| `weights/gmm`, `weights/decision_tree`, `weights/svm` | Joblib-serialized model weights *(not in git — download via HF)* |

### Experiments (`src/exp/tcn_ablation/`)
| File | Purpose |
|------|---------|
| `config.toml` | Ablation variants — each `[tcn.ablations.<subgroup>.<name>]` section is a flat set of overrides; top-level keys (`optimizer`, `lr`, `seq_len`, …) and model keys (`n_filters`, `n_layers`, …) are written directly without a `.model` sub-section |
| `cli.py` | `train`, `eval`, `ablation` (batch all variants in a subgroup) commands |

**Ablation config format** — keys are split automatically by `_MODEL_KEYS` / `_TOP_KEYS` in `config.py`:
```toml
[tcn.ablations.capacity.filters_8]
n_filters = 8                        # model key — no [….model] sub-section needed

[tcn.ablations.optimizer.adam_batchnorm]
optimizer = "adam"                   # top-level key
lr = 1e-3
use_weight_norm = false              # model key — mixed overrides work in one section
```

### Scripts (`scripts/`) — uv subproject
| File | Purpose |
|------|---------|
| `visualize_results.py` | Parses all `results/*.eval` files, plots macro F1 comparison + per-subclass breakdown → `results/results.png` |
| `upload_artifacts.py` | Uploads `weights/` and `cache/` to `Marek324/butfit-bp-artifacts` on HF |
| `download_artifacts.py` | Downloads `weights/`, `cache/`, and `results/` from HF (run after cloning; pass `--no-cache` to skip the large mel cache) |

### Dataset scripts (`scripts/dataset/`) — uv subproject
| File | Purpose |
|------|---------|
| `build.py` | CLI entry point: parses args, orchestrates sources → augmentation |
| `dataloader.py` | `load_source()` — streams and filters a HF dataset for one `SourceEntry` |
| `process.py` | `process_source()` — labels and writes one source to parquet shards |
| `source_config.py` | `SourceEntry` / `TierConfig` dataclasses; `AUGMENT_SOURCES`, `MULTISPEAKER_AUG_SOURCES`, `NOISE_AUG_SOURCES` per tier; FMA config; loads `sources.toml` |
| `sources.toml` | Per-tier HF source definitions (target minutes, HF IDs, filters) |
| `augmentation.py` | Synthetic augmentations: multi-speaker (LibriMix-style), speech-over-music, speech-over-noise |
| `labeling.py` | `VADLabeler`, `MusicLabeler`, `SilenceLabeler` — produce frame-level labels for each clip |
| `split_writer.py` | LibriSpeech-style path: `{staging}/{mini\|mid\|full}/{train\|validation\|test}/{speech\|music\|inactive}/part_*.parquet` |
| `notes.md` | Dataset spec: tier totals, subclass targets per tier |
| `HUB_DATASET_README.md` | Template Hub card (configs `full` first, then `mid`, `mini`) — copy to repo `README.md` before upload |

## Paper vs Implementation Differences

### TCN — Lemaire & Holzapfel ISMIR 2019

| Aspect | Paper | This implementation |
|--------|-------|---------------------|
| Optimizer | SGD, momentum=0.9 | Adam, lr=1e-3, weight_decay=1e-4 |
| Normalization | WeightNorm (keras-tcn reference) | `BatchNorm1d` after each conv |
| Spectrogram caching + aug on cached spectra (§3.4) | Power spectrum pre-saved; aug applied to saved spectra | `cache/nn/tcn_mel_*.pt` caches normalized log-mel; `augment_mel()` operates on the cached tensor ✓ |
| Augmentation pipeline (§3.7, Schlüter & Grill 2015) | Time stretch + pitch shift + Gaussian filter + loudness + block mix | Loudness only (±6 dB gain in log-mel space) — see `differences/tcn.md` §10 |
| Post-processing | Duration thresholds (§3.6) to smooth predictions | Not implemented |
| Receptive field vs chunk | RF = 3×(1+2+4+8)×(5−1)+1 = **181 frames**; chunks = 128 frames | Model never sees its full receptive-field context during training |

### GMM & SVM — Khonglah & Prasanna DSP 2016

| Aspect | Paper | This implementation |
|--------|-------|---------------------|
| SVM kernel | RBF, C=1, γ=3 (libSVM) | RBF, C=1, γ=3 (`sklearn.svm.SVC`) ✓ |
| SVM training scale | ~2400 1s-window vectors from 160 clips | Frame-level; subsampled to 50k/class |
| Feature window | 1s non-overlapping | 1s rolling (`lt_len_ms=1000`) ✓ |
| GMM components | 8, diagonal covariance | 8, diagonal ✓ |
| GMM smoothing | ~1s window over log-likelihood delta | 66-frame rolling buffer (~1s @ 15ms hop) ✓ |
| Scaling | Log-mel normalization per clip | `RobustScaler` (GMM) / `StandardScaler` (SVM) |
| Non-linear mapping | Sigmoid on log-likelihood delta before threshold | Not implemented |

## Inference

### TCN
Needs two files (download with `uv run python scripts/download_artifacts.py`):
1. `weights/tcn_preprocess_stats.pt` — loaded automatically by `LogMelSpectrogram.__init__`
2. `weights/tcn.safetensors` — loaded explicitly by eval/CLI code

### Classic models
Weights loaded via `model.load()` from `weights/<model_name>` (joblib pickle).
Feature extraction is stateful — `FeatExtractor.reset()` must be called between clips.

Labels: `-1` = speech, `1` = music, `2` = inactive

## Status
- **TCN**: done — 3-class F1=0.9033 (target met)
- **Classic baselines**: DT done; SVM and GMM retrained pending — both updated to closer match paper (SVM: SGDClassifier→SVC RBF C=1 γ=3; both: lt_len_ms 600→1000; GMM: smoothing buffer 20→66 frames)
- **Dataset (mid tier)**: historical snapshot; numbers in the mid-tier tables above reflect that snapshot, not the current HF upload
- **Dataset (full tier)**: rescaled to 6000 min (100h); AMI replaced with LibriMix-style synthetic multispeaker; all subclasses should now hit 100% of target; TCN config points here
- **Next**: rebuild full tier; retrain TCN + SVM/GMM off the new mel cache
