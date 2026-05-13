# Project: butfit-bp — Speech/Music Classifier

## Goal
Beat the existing DT/GMM/SVM baselines and achieve >= **.85 F1 macro on 3-class evaluation** with a causal TCN that can run online (streaming). Target met (TCN macro F1 = 0.9752). Project is now in the **writeup phase**: drafting docs/chapters/05–07, cleaning up code, and producing visualizations. No new experiments unless the docs reveal a gap that requires them.

## CLI
```
uv run smclassifier nn tcn <command>           # TCN model (paper baseline)
uv run smclassifier nn tcn-l <command>         # TCN-L variant (delta2 + conv1d + LSTM head)
uv run smclassifier nn tcn-s <command>         # TCN-S small-footprint variant (delta2 + n_filters=8, n_stacks=1, RF≤seq_len)
uv run smclassifier classic <model> <command>  # classic models (decision_tree, gmm, svm)
uv run smclassifier demo                       # Reflex web UI: mic/file streaming for all 6 models (auto-inits on first run)
```
TCN / TCN-L / TCN-S commands: `train`, `eval`, `smoke-test`, `smoke-test-online`
Classic commands: `train`, `eval`, `smoke-test`

## Architecture
Causal TCN — **must stay as close to Lemaire & Holzapfel ISMIR 2019 as possible**. Deviations and their justifications are tracked in `docs/differences/tcn.md`.

### Hyperparameter bounds from the paper
| Parameter | Paper search space | Current config |
|-----------|-------------------|----------------|
| n_filters | 8, 16, 32 | 16 |
| n_layers  | 1–4 | 4 |
| n_stacks  | 3–10 | 3 |
| kernel_size | 3, 5, 7, ..., 19 | 5 |
| dropout | 0.05–0.5 | 0.5 |

### Training recipe from the paper (§3.5)
- Optimizer: SGD, momentum=0.9 (production matches paper)
- LR schedule: ÷10 when val loss doesn't improve for 3 epochs (`ReduceLROnPlateau`)
- Early stopping: 5 epochs without improvement
- Batch: 32 × fixed-length chunks
- Loss: `BCEWithLogitsLoss` (numerically equivalent to paper's BCE; see `docs/differences/tcn.md` §2)

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

### Mel cache key changed — `n_features` now baked in (fixed)
`_mel_cache_path` in `training.py` used to key the cache by `frontend`+`n_mels`, which collided for MFCC variants (`mfcc_20` and `mfcc_40` share `frontend="mfcc"`, `n_mels=80`, but emit different channel counts). Now the key includes `_nf{fe.n_features}`.

**Filename anatomy** — `cache/nn/tcn_mel_{split}_{h}_fe{frontend}_nf{n_features}_sr{sr}_nfft{nfft}_hop{hop}_mels{nmels}_seq{seq}.pt`. Only `{h}` is hashed: it's `sha1(dataset_url|dataset_name|dataset_revision)[:8]`, covering the dataset tuple only. Every other slot is verbose, and the `.npz` classic-feature cache at `cache/*.npz` is a *separate* cache (fully md5-hashed over a different fingerprint) — unaffected by this change.

The legacy fallback in `_load_mel_chunks` (which read pre-`_nf` filenames) was removed once all local + HF caches were migrated. Anyone with a legacy local cache must rename in place using the recipe below, or recompute.

**Old filename:** `tcn_mel_{split}_{h}_fe{frontend}_sr{sr}_nfft{nfft}_hop{hop}_mels{nmels}_seq{seq}.pt`
**Current filename:** `tcn_mel_{split}_{h}_fe{frontend}_nf{n_features}_sr{sr}_nfft{nfft}_hop{hop}_mels{nmels}_seq{seq}.pt`

**To rename legacy caches in place** (no hash recomputation needed — `{h}` stays the same unless the dataset tuple changes), compute `n_features` from the filename:
- `log_mel` → `n_mels`
- `log_mel_delta` → `2 × n_mels`
- `log_mel_delta2` → `3 × n_mels`
- `pcen` → `n_mels`
- `mfcc` → open the file, read `data["mel"].shape[1]` (can't infer from name)

Insert `_nf{n_features}` right after `_fe{frontend}` and rename. When the user asks to "rename the legacy mel caches", this is the recipe.

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
| `cli.py` | `nn_group` — registers the canonical `tcn` group and auto-adds every variant's Click group from `VARIANTS` |
| `variant.py` | **Shared** `make_variant()` factory — builds the config loader, Click group, and `SpeechMusicDetector` subclass for each TCN variant from a pre-parsed config dict + naming/path params |
| `variants.py` | Reads `variants.toml` + repo-root `config.toml[dataset]`, calls `make_variant(...)` per entry, and auto-injects per-variant symbols (`TCN-S`, `tcn_s_group`, `get_tcn_s_config`, `_TCN_S_DEFAULT`, …) plus a `VARIANTS` registry. Adding a variant is a TOML-only change. |
| `variants.toml` | Declarative registry of all TCN variants (TCN-S, TCN-L). No `[dataset]` — inherited from repo-root `config.toml`. |

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
| `cache/nn/tcn_mel_*.pt` | Precomputed mel chunks per (dataset, split, frontend params, seq_len); shared across ablation variants. *(not in git — fetch from Marek324/butfit-bp-artifacts on HF)* |

#### Variants (`variants.py`, `variants.toml`)
TCN-S (delta² frontend + `n_filters=8`, no preprocessor/head, `n_stacks=1`, RF=121 frames ≤ training `seq_len=128`, ~4.6K params) and TCN-L (delta² + conv1d preprocessor + TCN + LSTM head, combined-experiment winner) are declared as `[variants.<name>]` blocks in `variants.toml`. `variants.py` loads each block through `make_variant(...)` (`src/nn/variant.py`) and auto-exports one class, one Click group, three getters, and one `_DEFAULT` dict per variant, plus a `VARIANTS` registry dict. The `[dataset]` block is inherited from repo-root `config.toml` — variant TOML blocks only carry `[tcn]` + `[tcn.model]`. Adding a new variant is a TOML-only edit; Python picks it up automatically (including CLI registration).

| Weights / stats | Purpose |
|------|---------|
| `weights/tcn_l/tcn_l.safetensors` | Trained TCN-L weights *(not in git — download via HF)* |
| `weights/tcn_l/tcn_l_preprocess_stats.pt` | TCN-L log-mel normalization stats *(not in git)* |
| `weights/tcn_s/tcn_s.safetensors` | Trained TCN-S weights (sourced from `small_tcn_stacks/tcn_stacks_1`) *(not in git — download via HF)* |
| `weights/tcn_s/tcn_s_preprocess_stats.pt` | TCN-S log-mel normalization stats *(not in git)* |

### Classic models (`src/classic/`)
| File | Purpose |
|------|---------|
| `modelclass.py` | Abstract base: `fit()`, `save()`, `load()` via joblib |
| `feat_extractor.py` | `FeatExtractor` — stateful frame-level feature extraction (30ms/15ms hop); **call `reset()` between clips** |
| `gmm.py` | Three-GMM density classifier (one per class: speech / music / inactive); 8 components, diag covariance; argmax over per-class log-likelihoods. Paper has 2 GMMs + threshold; we extend to 3-class — see `docs/differences/gmm_svm.md` §9 |
| `decisiontree.py` | Decision tree with exponential-forgetting smoothing on last decisions |
| `svm.py` | `SVC(kernel='rbf', C=1, γ=3)`; native 3-class via sklearn OvO; balanced subsampling to 50k/class at train time; 20-frame rolling decision-vector buffer + argmax at inference |
| `evaluation.py` | `eval_classic()` — loads model + test features, runs `run_evaluation()` |
| `cli.py` | Click CLI per model: train, eval, smoke-test |
| `weights/gmm`, `weights/decision_tree`, `weights/svm` | Joblib-serialized model weights *(not in git — download via HF)* |
| `streaming.py` | `StreamingClassifier` — push raw audio samples, get per-hop labels (consumed by `src/demo/`) |

### Streaming demo (`src/demo/`)
Reflex web UI for live speech/music classification across all six models (DT/GMM/SVM + TCN/TCN-L/TCN-S).
| File | Purpose |
|------|---------|
| `runner.py` | `ClassicRunner` / `NNRunner` — uniform streaming wrapper; remaps labels to display space `{-1, 0, +1}` |
| `source.py` | `MicSource` (sounddevice) / `FileSource` (soundfile + librosa) — async chunk producers |
| `weights_check.py` | `has_weights(name)` — filesystem probe used to badge the model picker |
| `cli.py` | `demo run` / `demo init` Click commands — launch `reflex run` via subprocess |
| `rxconfig.py` | Reflex config; `app_name = "ui"` |
| `ui/ui.py` | Reflex `DemoState` + page: model picker, mic/file tabs, recharts line chart |

### Experiments (`src/exp/tcn_ablation/`)
| File | Purpose |
|------|---------|
| `config.toml` | Ablation variants — each `[tcn.ablations.<subgroup>.<name>]` section is a flat set of overrides; top-level keys (`optimizer`, `lr`, `seq_len`, …) and model keys (`n_filters`, `n_layers`, …) are written directly without a `.model` sub-section |
| `cli.py` | `train`, `eval`, `ablation` (batch all variants in a subgroup) commands |

### Flat-variant experiments (`src/exp/{tcn_frontend, nn_architecture, nn_preprocessor, tcn_combined}/`)
Each module follows the same pattern: its own `config.toml` with a base `[tcn]` section plus `[tcn.variants.<name>]` flat overrides, and a near-identical `cli.py` (`train`, `eval`, `smoke-test`, `visualize`, `run-all`). Overrides are routed into top-level vs model buckets by `_TOP_KEYS` / `_MODEL_KEYS` in `src/nn/tcn/config.py`. `tcn_combined` stacks the winners of the other three in a 2² factorial (×architecture winner) to test whether the individual gains compose.

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

Full deviation tracking lives in `docs/differences/tcn.md`. Highlights only here:

| Aspect | Paper | This implementation |
|--------|-------|---------------------|
| Optimizer | SGD, momentum=0.9 | SGD, momentum=0.9 ✓ (Adam+BN exists as ablation variant — see `docs/differences/tcn.md` §12) |
| Normalization | WeightNorm (keras-tcn reference) | WeightNorm ✓ |
| Loss | BCE | `BCEWithLogitsLoss` (math-equivalent, numerically stable — `docs/differences/tcn.md` §2) |
| Spectrogram caching + aug on cached spectra (§3.4) | Power spectrum pre-saved; aug applied to saved spectra | `cache/nn/tcn_mel_*.pt` caches normalized log-mel; `augment_mel()` operates on the cached tensor ✓ |
| Augmentation pipeline (§3.7, Schlüter & Grill 2015) | Time stretch + pitch shift + Gaussian filter + loudness + block mix | Loudness only (±6 dB gain in log-mel space) — see `docs/differences/tcn.md` §10 |
| Post-processing | Duration thresholds (§3.6) to smooth predictions | Not implemented |
| Receptive field vs chunk | RF = 1 + 2×3×(5−1)×(1+2+4+8) = **361 frames** (factor of 2 for the two convs per residual block); chunks = 128 frames | Model never sees its full receptive-field context during training. TCN-S variant (`n_stacks=1`, RF=121) is the only checkpoint where train-time RF ≤ seq_len |

### GMM & SVM — Khonglah & Prasanna DSP 2016

Full deviation tracking lives in `docs/differences/gmm_svm.md`. Highlights only here:

| Aspect | Paper | This implementation |
|--------|-------|---------------------|
| SVM kernel | RBF, C=1, γ=3 (libSVM) | RBF, C=1, γ=3 (`sklearn.svm.SVC`) ✓ |
| SVM training scale | ~2400 1s-window vectors from 160 clips | Frame-level; subsampled to 50k/class |
| SVM smoothing | none | 20-frame rolling decision buffer — `docs/differences/gmm_svm.md` §11 |
| SVM output classes | 2 (speech / music; libSVM threshold) | 3 (speech / music / inactive) via sklearn native OvO; argmax over smoothed decision vector |
| Feature window | 1s non-overlapping | 1s rolling (`lt_len_ms=1000`) ✓ |
| GMM components | 8, diagonal covariance | 8, diagonal ✓ |
| GMM structure | 2 GMMs (speech / music) + threshold | 3 GMMs (speech / music / inactive) + argmax — `docs/differences/gmm_svm.md` §9 |
| GMM smoothing | ~1s window over log-likelihood delta | 66-frame rolling buffer (~1s @ 15ms hop) ✓ |
| Scaling | Unspecified (raw features fed to classifiers, paper §3.1) | `RobustScaler` (GMM, ±10 clip) / `StandardScaler` (SVM) — added |
| Non-linear mapping | Sigmoid on log-likelihood delta before threshold | Not implemented |

### Decision Tree — Lavner & Ruinskiy EURASIP 2009

Full deviation tracking lives in `docs/differences/dt.md`. Highlights:

| Aspect | Paper | This implementation |
|--------|-------|---------------------|
| Classifier | 3-stage Bayesian + rule-based sieve | sklearn `DecisionTreeClassifier` (single learned tree, CART) — `docs/differences/dt.md` §1 |
| Output classes | 2 (speech / music) | 3 (speech / music / inactive) |
| Feature selection | "Automatic" per paper | Fixed 19-component vector + `SelectKBest(k=10)` ANOVA-F |
| Smoothing | Per-segment exp-decay weighted average ($D_s = (1/F)\sum D_i e^{-k/\tau}$) at 100 ms hop + adaptive threshold | Same exp-decay form, per-frame at 10 ms hop over 30-deep deque; adaptive threshold dropped — `docs/differences/dt.md` §5 |
| Inference granularity | Per-segment | Per-frame (streaming, 10 ms hop) |

## Inference

### TCN
Needs two files (download from Marek324/butfit-bp-artifacts on HF):
1. `weights/tcn_preprocess_stats.pt` — loaded automatically by `LogMelSpectrogram.__init__`
2. `weights/tcn.safetensors` — loaded explicitly by eval/CLI code

### Classic models
Weights loaded via `model.load()` from `weights/<model_name>` (joblib pickle).
Feature extraction is stateful — `FeatExtractor.reset()` must be called between clips.

Labels: `-1` = speech, `1` = music, `2` = inactive

## Status

Project phase: **writeup**. Experiments are frozen; deployed checkpoints in `weights/` are the artifacts the thesis describes.

### Test-split macro F1 (full tier, deployed checkpoints)

| Model | Macro F1 | Source |
|---|---:|---|
| TCN-L | 0.9846 | `results/tcn_l.eval` |
| TCN | 0.9752 | `results/tcn.eval` |
| TCN-S | 0.9722 | `results/tcn_s.eval` |
| SVM | 0.8750 | `results/svm.eval` |
| DT | 0.8376 | `results/decision_tree.eval` |
| GMM | 0.8250 | `results/gmm.eval` |

All models clear the 0.85 project target except GMM (0.825, just below). TCN family clears it by 12+ pp.

### Other artifacts

- **Dataset (mid tier)**: historical snapshot; the mid-tier tables in this doc reflect that snapshot, not the current HF upload.
- **Dataset (full tier)**: 6000 min (100 h) on `Marek324/speech-music-classification` config `full`; all subclasses hit ~100% of target after the Bresenham split fix.
- **Critical-set evaluation**: hand-curated adversarial clips at `scripts/dataset/speech_music_dataset/crit` and via `Marek324/speech-music-classification` config `crit`. All seven models evaluated; results in `src/exp/critical/results/`.
- **Experiment notes**: every `src/exp/*/results/notes.md` was refreshed against current `.eval` data on 2026-04-29 — those are the canonical narrative source for the docs chapters.
- **Deviation docs**: `docs/differences/{tcn,gmm_svm,dt}.md` track every deliberate departure from the reference papers.

### What's left

- Code cleanup pass.
- Visualizations: `scripts/visualize_results.py` is the canonical entry point — outputs SVG to `results/`.
