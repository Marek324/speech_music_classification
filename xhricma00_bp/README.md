# Speech / Music Classification

For reproduction instructions (running eval, rerunning experiments, contacting the author for dataset access) see `src/README.md`.

## Archive contents

```
xhricma00_bp/
  speech_music_classification.pdf
  xhricma00_speech_music_classification_print.pdf
  poster.pdf                                            assignment poster
  video_presentation.mp4                                presentation video

  src/                  source code — see src/README.md
  scripts/              dataset construction + visualization scripts
  weights/              six deployed model checkpoints
  docs/                 LaTeX source for the thesis + figures + assignment PDFs

  config.toml           global model / feature / dataset configuration
  pyproject.toml
  uv.lock
  .python-version       3.12
```

## Setup

**Python 3.12** and **FFmpeg 4–7** are required. `torchcodec==0.7.0` does not support FFmpeg 8 — if your system has FFmpeg 8, install a 7.x build via your package manager or system packages.

**CUDA is optional.** Every command including eval runs on CPU. The pinned `torch==2.8.0+cu128` wheels target CUDA 12.8; with no NVIDIA driver / an older driver the CPU path is used automatically.

### Recommended — [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

### Alternative — `pip`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pyproject.toml` pins every direct dependency with `==`. The PyTorch wheels are routed through the CUDA 12.8 index via `[tool.uv.sources]`; `pip` ignores that block and pulls CPU-only wheels by default. Add `--index-url https://download.pytorch.org/whl/cu128` for CUDA wheels.

No API keys are needed to install or to run smoke-tests.

## CLI

The entry point is `smclassifier`, registered by `uv sync`. Every command exposes `--help`.

```bash
# TCN family
uv run smclassifier nn tcn        {train|eval|smoke-test|smoke-test-online}
uv run smclassifier nn tcn-l      {train|eval|smoke-test|smoke-test-online}
uv run smclassifier nn tcn-s      {train|eval|smoke-test|smoke-test-online}

# Traditional references
uv run smclassifier classic decision_tree {train|eval|smoke-test}
uv run smclassifier classic gmm           {train|eval|smoke-test}
uv run smclassifier classic svm           {train|eval|smoke-test}

# Experiments (one Click subgroup per thesis subsection in §5)
uv run smclassifier exp {complexity|critical|nn-architecture|nn-preprocessor|
                         small-tcn-pareto|small-tcn-stacks|tcn-ablation|
                         tcn-combined|tcn-frontend|tcn-temporal-head} ...

# Streaming demo — see disclosure below
uv run smclassifier demo
```

`smoke-test` runs a synthetic forward pass through the model and is the fastest way to verify the pipeline works. It requires no network and no dataset access — see `src/README.md` for the offline-only commands.

`eval` against the test split needs Hugging Face access — also covered in `src/README.md`.

## Demo disclosure

> [!WARNING]
> **The Reflex demo (`src/demo/`) is not properly tested and is shipped only as a curiosity.**
>
> The demo was a side experiment to see the streaming pipeline work in a browser. **It is intentionally not referenced in the thesis text.** Known quirks include:
>
> - The "desktop audio" capture path appears to record the **pre-attenuation** system signal rather than the post-mixer output a listener would actually hear.
> - The model swap path resets some state but not all; switching models mid-stream may give one or two hops of mixed-state predictions.
> - There is no test coverage and no validation against a known-good streaming reference.
>
> Every quantitative result in the thesis comes from the CLI `eval` / `smoke-test` paths, not the demo.

## Author

Marek Hric - xhricma00
