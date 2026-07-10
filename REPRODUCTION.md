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
- [Demo disclosure](#demo-disclosure)

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

**Trained weights** and the precomputed log-mel cache live on Hugging Face at
`Marek324/butfit-bp-artifacts`. Pull them into the repo root:

```bash
huggingface-cli download Marek324/butfit-bp-artifacts \
    --local-dir . --include "weights/*"
```

Pass `--include "cache/*"` as well only if you intend to retrain (the mel cache is ~32 GB on the
full tier and is reproducible from the dataset).

**Dataset** is hosted on Hugging Face at `Marek324/speech-music-classification` with two configs:

- `full` — 100 hours used for every headline result
- `crit` — hand-curated adversarial subset for stress evaluation (113 clips)

Both are loaded automatically through the `datasets` library on the first `eval` / `train`
invocation. To rebuild the dataset from upstream public sources:

```bash
cd scripts/dataset
uv run python build.py full              # or: mid | --only-critical-set
uv run python upload.py                  # push to Hugging Face (requires hf token)
```

Rebuilding needs an HF token with access to the gated `pyannote/segmentation-3.0` VAD model
(accept its terms on Hugging Face and export `HF_TOKEN`), plus ~50 GB free disk and several
hours of bandwidth + CPU. No other API key is required.

## Reproduction

Happy-path walkthrough from a clean clone:

```bash
git clone https://github.com/Marek324/butfit-bp bp && cd bp

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

# Streaming demo (Reflex web UI — read the disclosure below)
uv run smclassifier demo
```

Every command exposes `--help`. Smoke-test commands run a one-second synthetic clip end-to-end
and are the fastest way to confirm the pipeline works.

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
  Hugging Face (`Marek324/butfit-bp-artifacts`).
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

## Demo disclosure

> [!WARNING]
> **The Reflex demo (`src/demo/`) is not properly tested and is shipped only as a curiosity.**
>
> The demo was a quick side experiment to see the streaming pipeline work in a browser; it was
> never carefully verified. **It is intentionally not referenced in the thesis text.** Known
> quirks include:
>
> - The "desktop audio" capture path appears to record the **pre-attenuation** system signal
>   rather than the post-mixer output a listener would actually hear. The exact OS tap was never
>   confirmed.
> - Interaction with the OS / browser microphone buffer is uncharacterized, so audible-vs-recorded
>   alignment may be off, and chunk-boundary timing may differ from what the classifier's
>   streaming pipeline assumes.
> - The model swap path resets some state but not all; switching models mid-stream may give one
>   or two hops of mixed-state predictions.
> - There is no test coverage for the demo and no validation against a known-good streaming
>   reference.
>
> The demo is included for anyone who wants to try it, but **do not trust its output for any
> quantitative claim** — every quantitative result comes from the CLI `eval` / `smoke-test`
> paths, not the demo.
