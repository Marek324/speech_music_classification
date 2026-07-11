# Reproduction & Usage

Full setup, weights/dataset access, evaluation walkthrough, CLI reference, and configuration
for the speech/music/background classifier. For a high-level overview see the root
[README.md](README.md); for the code architecture see [src/README.md](src/README.md).

## Contents

- [Setup](#setup)
- [Weights and dataset](#weights-and-dataset)
- [Reproduction](#reproduction)
- [CLI](#cli)
- [Configuration](#configuration)
- [Scripts and tooling](#scripts-and-tooling)

## Setup

**Python 3.12** and **FFmpeg 4, 5, 6, or 7** are required. The pinned `torchcodec==0.7.0` does
not support FFmpeg 8 — if your system has FFmpeg 8 (`ffmpeg -version`), install a 7.x build into
the venv or via a system package manager. Distro-level FFmpeg from late 2024 onwards is usually
still 6.x or 7.x; rolling-release distros (Arch, Fedora rawhide) may have 8.x.

**CUDA is optional.** Every step including training and inference runs on CPU. The pinned
`torch==2.8.0+cu128` wheels target CUDA 12.8, which requires an NVIDIA driver of at least
**R555** (recent driver, `nvidia-smi` reports `CUDA Version: 12.8` or higher). With an older
driver `torch.cuda.is_available()` will return False and the CPU path is used automatically.

**Recommended — [uv](https://github.com/astral-sh/uv) (tested):**

```bash
uv sync
```

Resolves the dependency graph against the pinned `uv.lock`, installing every package at the
exact version the project was developed against. This is the only setup that has been verified
end-to-end.

**Alternative — `pip` (should work, not tested):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pyproject.toml` pins every direct dependency with `==`, so the resulting environment is in
principle reproducible without `uv`. The PyTorch wheels (`torch`, `torchaudio`, `torchcodec`)
are routed through the CUDA 12.8 index via `[tool.uv.sources]` — `pip` ignores that block and
pulls the default-PyPI wheels instead, which are CPU-only on most platforms. Add
`--index-url https://download.pytorch.org/whl/cu128` to the pip command if you want the CUDA
wheels.

## Weights and dataset

The **trained weights**, the precomputed log-mel cache, and the **dataset** all live in
**private** Hugging Face repositories and are not publicly distributed. Their locations are read
from environment variables, so you can point the code at your own copies:

- `SMC_ARTIFACTS_REPO` — HF model repo holding `weights/`, the mel `cache/`, and the score `.npz` files.
- `SMC_DATASET_REPO` — HF dataset id (or a local path) for the speech/music/background corpus.

With `SMC_ARTIFACTS_REPO` set, pull the checkpoints into the repo root:

```bash
export SMC_ARTIFACTS_REPO=<your-user>/<your-weights-repo>
python scripts/download_artifacts.py --no-cache   # weights + results; skip the ~32 GB cache
```

The dataset has two configs — `full` (100 hours, used for every headline result) and `crit`
(a 113-clip adversarial subset for stress evaluation). It is streamed through the `datasets`
library on the first `eval` / `train` invocation using `SMC_DATASET_REPO`. To rebuild it from
upstream public sources instead:

```bash
cd scripts/dataset
uv run python build.py full                                   # or: mid | --only-critical-set
SMC_DATASET_REPO=<your-user>/<your-dataset> uv run python upload.py
```

Rebuilding needs an HF token with access to the gated `pyannote/segmentation-3.0` VAD model
(accept its terms on Hugging Face and export `HF_TOKEN`), plus ~50 GB free disk and several
hours of bandwidth + CPU. No other API key is required.

## Reproduction

Happy-path walkthrough from a clean clone (steps 3–4 need access to the private data and weights):

```bash
git clone https://github.com/Marek324/speech_music_classification bp && cd bp

# 1. Install dependencies
uv sync

# 2. Confirm the pipeline works on synthetic input (~5 s per model) — no data needed
uv run smclassifier classic decision_tree smoke-test
uv run smclassifier nn tcn-s smoke-test

# 3. Point at your weights + dataset, then pull the checkpoints (~700 MB)
export SMC_ARTIFACTS_REPO=<your-user>/<your-weights-repo>
export SMC_DATASET_REPO=<your-user>/<your-dataset>
python scripts/download_artifacts.py --no-cache

# 4. Evaluate one model on the test split (streams the dataset on first run)
uv run smclassifier nn tcn-s eval        # writes results/tcn_s.eval
```

To evaluate every checkpoint and regenerate the `results/*.eval` reports, repeat step 4 with each
of `nn tcn`, `nn tcn-s`, `nn tcn-l`, `classic decision_tree`, `classic gmm`, `classic svm`. The
CPU benchmark and critical-set review are separate commands under `exp` (see below). The summary
dashboard (`results/results.svg`) is regenerated with `python scripts/visualize_results.py`.

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

# Streaming demo (Reflex web UI — see the main README)
uv run smclassifier demo
```

Every command exposes `--help`. Smoke-test commands run a one-second synthetic clip end-to-end,
with no dataset or network access.

## Configuration

The global configuration lives in `config.toml` at the repo root. Two switches are commonly
toggled:

- **`[dataset] name`** — selects the dataset tier loaded by both classic and NN pipelines.
  Default is `full` (the 100-hour tier behind every headline result). The other value is `mid`
  (a smaller tier, kept for historical reference). Switching tiers invalidates the precomputed
  mel cache (`cache/nn/*.pt`) because the cache key fingerprints the dataset tuple — the next
  `train` / `eval` will rebuild it.
- **`[buffers.*]` and `[features.*]`** — per-model buffer / feature-extractor settings. The TCN
  family additionally reads `[tcn]` and `[tcn.model]`; variants override these via
  `src/nn/variants.toml`.

**Random seeding.** Every model was trained with a single fixed seed. The bootstrap confidence
intervals reported next to each macro F1 score capture test-split sampling variance only — not
training variance. Re-training will produce a slightly different checkpoint each time because of
non-determinism in parallel arithmetic and data shuffling; the resulting test F1 is expected to
land within the reported CI but not to match the headline number exactly. A multi-seed
reproduction is left as future work.

## Scripts and tooling

Development, figure-authoring, and data-pipeline scripts live under `scripts/` (a separate
`uv` subproject with its own dependency set). They are not part of the `smclassifier` CLI —
run them directly with `python scripts/<name>.py` (or `uv run --project scripts …`).

**Result & experiment plotting** (repo root of `scripts/`):

- `visualize_results.py` — canonical entry point: parses every `results/*.eval` (+ `*_scores.npz`)
  and renders the summary dashboard `results/results.svg` plus per-panel graphs under
  `results/graphs/`.
- `visualize_experiment.py`, `visualize_ablation.py` — render the flat-variant and ablation
  experiment figures.
- `visualizations/` — one authoring script per thesis figure (dataset pie & pipeline, CPU /
  reference benchmarks, symbolic complexity, per-experiment result plots, TCN dilation /
  residual-block / receptive-field diagrams, …), sharing helpers in `visualizations/_common.py`.

**Artifacts & maintenance:**

- `download_artifacts.py` / `upload_artifacts.py` — sync trained weights and the mel cache with
  the private HF artifacts repo (`SMC_ARTIFACTS_REPO`).
- `migrate_weights_to_subdirs.py` — one-off migration of the weights directory layout.
- `parity_*.py`, `check_batched_autocorr.py` — verification scripts that assert the streaming
  path matches the batch path and that vectorized/batched code is numerically equivalent.
- `dataset_stats.py` — prints per-split / per-subclass dataset statistics.

**Dataset construction** (`scripts/dataset/`) — the reproducible build pipeline behind the
Hugging Face dataset:

- `build.py` — CLI entry point (orchestrates sources → labeling → augmentation → parquet);
  `dataloader.py`, `process.py`, `split_writer.py` — stream, label, and shard each source;
  `labeling.py`, `augmentation.py` — frame-level VAD/music/silence labeling and synthetic
  multi-speaker / speech-over-noise/music mixing.
- `source_config.py` + `sources.toml` — per-tier Hugging Face source definitions (target
  minutes, dataset IDs, filters); `crit/` — generators for the hand-curated critical subset;
  `upload.py` — push the built dataset to Hugging Face.

The Reflex streaming demo and its caveats are documented in the [main README](README.md#demo).
