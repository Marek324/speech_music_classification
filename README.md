# Speech / Music Classifier — Bachelor Thesis Code

Causal speech / music / inactive classification for streaming audio. Six trained checkpoints ship with the thesis: three classic baselines (decision tree, GMM, SVM) following Lavner & Ruinskiy (2009) and Khonglah & Prasanna (2016), and three causal TCN variants (TCN, TCN-L, TCN-S) following Lemaire & Holzapfel (2019).

## Contents

- [Setup](#setup)
- [Weights and dataset](#weights-and-dataset)
- [Reproduction](#reproduction)
- [CLI](#cli)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Demo disclosure](#demo-disclosure)
- [Dead code and orphan files](#dead-code-and-orphan-files)

## Setup

**Python 3.12** and **FFmpeg 4, 5, 6, or 7** are required. The pinned `torchcodec==0.7.0` does not support FFmpeg 8 — if your system has FFmpeg 8 (`ffmpeg -version`), install a 7.x build into the venv or via a system package manager. Distro-level FFmpeg from late 2024 onwards is usually still 6.x or 7.x; rolling-release distros (Arch, Fedora rawhide) may have 8.x.

**CUDA is optional.** Every step including training and inference runs on CPU. The pinned `torch==2.8.0+cu128` wheels target CUDA 12.8, which requires an NVIDIA driver of at least **R555** (recent driver, `nvidia-smi` reports `CUDA Version: 12.8` or higher). With an older driver `torch.cuda.is_available()` will return False and the CPU path is used automatically.

**Recommended — [uv](https://github.com/astral-sh/uv) (tested):**

```bash
uv sync
```

Resolves the dependency graph against the pinned `uv.lock`, installing every package at the exact version the thesis was developed against. This is the only setup that has been verified end-to-end.

**Alternative — `pip` (should work, not tested):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pyproject.toml` pins every direct dependency with `==`, so the resulting environment is in principle reproducible without `uv`. The PyTorch wheels (`torch`, `torchaudio`, `torchcodec`) are routed through the CUDA 12.8 index via `[tool.uv.sources]` — `pip` ignores that block and pulls the default-PyPI wheels instead, which are CPU-only on most platforms. Add `--index-url https://download.pytorch.org/whl/cu128` to the pip command if you want the CUDA wheels.

## Weights and dataset

**Trained weights** and the precomputed log-mel cache live on Hugging Face at `Marek324/butfit-bp-artifacts`. Pull them into the repo root:

```bash
huggingface-cli download Marek324/butfit-bp-artifacts \
    --local-dir . --include "weights/*"
```

Pass `--include "cache/*"` as well only if you intend to retrain (the mel cache is ~32 GB on the full tier and is reproducible from the dataset).

**Dataset** is hosted on Hugging Face at `Marek324/speech-music-classification` with two configs:

- `full` — 100 hours used by the thesis
- `crit` — hand-curated adversarial subset for stress evaluation (113 clips)

Both are loaded automatically through the `datasets` library on the first `eval` / `train` invocation. To rebuild the dataset from upstream public sources:

```bash
cd scripts/dataset
uv run python build.py full              # or: mid | --only-critical-set
uv run python upload.py                  # push to Hugging Face (requires hf token)
```

## Reproduction

Happy-path walkthrough from a clean clone:

```bash
git clone <repo-url> bp && cd bp

# 1. Install dependencies
uv sync

# 2. Confirm the pipeline works on synthetic input (~5 s per model)
uv run smclassifier classic decision_tree smoke-test
uv run smclassifier nn tcn-s smoke-test

# 3. Pull trained weights (~700 MB total across all six checkpoints)
huggingface-cli download Marek324/butfit-bp-artifacts \
    --local-dir . --include "weights/*"

# 4. Evaluate one model on the test split (auto-downloads the dataset on first run)
uv run smclassifier nn tcn-s eval        # writes results/tcn_s.eval
```

To evaluate every checkpoint and regenerate the `results/*.eval` reports, repeat step 4 with each of `nn tcn`, `nn tcn-s`, `nn tcn-l`, `classic decision_tree`, `classic gmm`, `classic svm`. The CPU benchmark and critical-set review are separate commands under `exp` (see below).

## CLI

The entry point is `smclassifier`, defined in `pyproject.toml` and registered by `uv sync`.

```bash
# TCN family
uv run smclassifier nn tcn   {train|eval|smoke-test|smoke-test-online}
uv run smclassifier nn tcn-l {train|eval|smoke-test|smoke-test-online}
uv run smclassifier nn tcn-s {train|eval|smoke-test|smoke-test-online}

# Classic baselines
uv run smclassifier classic {decision_tree|gmm|svm} {train|eval|smoke-test}

# Experiments (ablations, variants, complexity benchmark, critical-set review, …)
uv run smclassifier exp {complexity|critical|nn-architecture|nn-preprocessor|
                         small-tcn-pareto|small-tcn-stacks|tcn-ablation|
                         tcn-combined|tcn-frontend|tcn-temporal-head} ...

# Streaming demo (Reflex web UI — read the disclosure below)
uv run smclassifier demo
```

Every command exposes `--help`. Smoke-test commands run a one-second synthetic clip end-to-end and are the fastest way to confirm the pipeline works.

## Configuration

The global configuration lives in `config.toml` at the repo root. Two switches are commonly toggled by reviewers:

- **`[dataset] name`** — selects the dataset tier loaded by both classic and NN pipelines. Default is `full` (the 100-hour tier used by every result in the thesis). The other value is `mid` (the smaller mid-tier, kept for historical reference). Switching tiers invalidates the precomputed mel cache (`cache/nn/*.pt`) because the cache key fingerprints the dataset tuple — the next `train` / `eval` will rebuild it.
- **`[buffers.*]` and `[features.*]`** — per-model buffer / feature-extractor settings. Defaults match the thesis. The TCN family additionally reads `[tcn]` and `[tcn.model]`; variants override these via `src/nn/variants.toml`.

**Random seeding.** Every model in the thesis was trained with a single fixed seed. The bootstrap confidence intervals reported next to each macro F1 score capture test-split sampling variance only — not training variance. Re-training will produce a slightly different checkpoint each time because of non-determinism in parallel arithmetic and data shuffling; the resulting test F1 is expected to land within the reported CI but not to match the headline number exactly. A multi-seed reproduction is listed as future work in the conclusion.

## Repository layout

```
src/
  classic/                 decision-tree, GMM, SVM (streaming + batch)
  nn/
    tcn/                   reference TCN: model, preprocess, training, streaming
    variants.{toml,py}     TCN-S, TCN-L declared in TOML, loaded by variants.py
    blocks.py              shared CausalConv1d + TCNResidualBlock
    dataset.py             HF loader (NN side), frame-label builder
    evaluation.py          shared inference loop for any TCN-shaped output
  demo/                    Reflex web UI (see disclosure)
  exp/                     experiment modules (one subdirectory per experiment)
  evaluator.py             shared metric / report machinery
  input_handler.py         HF loader (classic side)
  common.py, config.py     label mapping, global config singleton
  main.py                  CLI entry point (top-level click group)
scripts/
  dataset/                 dataset construction pipeline (build, upload, augment)
    crit/                  critical-subset generators
    sources.toml           per-tier Hugging Face source definitions
  visualizations/          standalone authoring scripts for each thesis figure
  parity_*.py              one-shot verification scripts (streaming vs batch, …)
  download_artifacts.py, upload_artifacts.py
  migrate_weights_to_subdirs.py
  visualize_ablation.py, visualize_experiment.py, visualize_results.py
weights/                   trained checkpoints (downloaded from HF — not in git)
cache/                     precomputed mel cache (downloaded — not in git)
results/                   evaluation reports (`*.eval`) per checkpoint
docs/                      thesis source (LaTeX) + figures
config.toml                global dataset / buffer / feature / TCN config
```

The classic models read every relevant setting from `config.toml`; the TCN family reads the same file plus `src/nn/variants.toml` (one block per variant; adding a new variant is a TOML-only edit).

## Demo disclosure

> [!WARNING]
> **The Reflex demo (`src/demo/`) is not properly tested and is shipped only as a curiosity.**
>
> The demo was a vibecoded side experiment — I wrote it because I wanted to see the streaming pipeline work in a browser and never invested the time to verify it carefully. **It is intentionally not referenced in the thesis text.** Known quirks include:
>
> - The "desktop audio" capture path appears to record the **pre-attenuation** system signal rather than the post-mixer output a listener would actually hear. I never confirmed what tap the OS exposes for this.
> - I have not characterized how the demo interacts with the OS / browser microphone buffer, so audible-vs-recorded alignment may be off, and chunk-boundary timing may differ from what the classifier's streaming pipeline assumes.
> - The model swap path resets some state but not all; switching models mid-stream may give one or two hops of mixed-state predictions.
> - There is no test coverage for the demo, no validation against a known-good streaming reference, and no audit of how Reflex backgrounds the inference loop.
>
> The demo is included in the repository for anyone who wants to try it, but **do not trust its output for any quantitative claim** — every quantitative result in the thesis comes from the CLI `eval` / `smoke-test` paths, not the demo.

## Dead code and orphan files

Cleanup before submission left a few orphan files on disk. They are not referenced by any surviving code, the CLI, or the thesis prose, but they are kept because they may be useful for someone extending the work and because nothing depends on them not being there.

**From the dropped switch-latency / transitions experiment:**

- `src/exp/switch_latency/` — standalone module that streamed switching clips through every runner with silence prefill and K=3 consecutive-match latency. Implementation works; the methodology turned out to be too compromised by right-censoring + drift contamination to use in the thesis.
- `src/exp/transitions/` — older analysis of the same data (K=1 latency, no prefill, sub-label transitions). Superseded by `switch_latency/` and then both were dropped.
- `scripts/dataset/crit/make_switching.py`, `make_switching_3class.py` — generators for the 64 switching clips. Manifest entries have been stripped, so re-running `build.py --only-critical-set` no longer pulls them into the parquet. The 72 corresponding `.wav` files and 72 `.labels.txt` files remain under `scripts/dataset/crit/recordings/`.
- `scripts/visualizations/transitions_latency.py` — the heatmap figure script for the dropped experiment.
- `docs/figures/experiments/switch_latency.{pdf,svg}` and `transitions_latency.{pdf,svg}` — orphan figures.

**Other orphans:**

- `src/wandb_logger.py` — Weights & Biases hooks for training. Never imported anywhere in the current codebase; `wandb` is correspondingly absent from `pyproject.toml`. Re-enable by adding `wandb` to the project dependencies and importing the logger from the training entry point.
- `docs/figures/experiments/filter_pareto.{pdf,svg}` — a Pareto-style figure that was rendered for an earlier draft and never wired into the chapter.
- `docs/figures/experiments/lightweight_scans_placeholder.png` — placeholder that was replaced by the proper figure.
