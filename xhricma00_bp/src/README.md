# `src/` — Source Code & Reproduction

CLI entry point: `smclassifier` (defined in `pyproject.toml` → `src.main:cli`). The top-level Click group registers four subgroups: `classic`, `nn`, `exp`, `demo`. For the full CLI surface and setup instructions see the top-level `README.md`.

## Reproduction

### Offline — no HF access needed

The fastest way to verify the bundled pipeline works:

```bash
uv run smclassifier nn tcn                 smoke-test
uv run smclassifier nn tcn-l               smoke-test
uv run smclassifier nn tcn-s               smoke-test
uv run smclassifier classic decision_tree  smoke-test
uv run smclassifier classic gmm            smoke-test
uv run smclassifier classic svm            smoke-test
```

Each runs a synthetic forward pass through the deployed checkpoint and logs `smoke-test passed` with the output shape (and a per-class macro F1 for the classic models). Takes a few seconds per command. No network, no dataset.

### With HF access — full eval / rerunning experiments

The test split lives on Hugging Face at `Marek324/speech-music-classification` (configs `full` and `crit`). The dataset is **access-gated, not public** — running `eval` or any `exp` subcommand against the dataset will fail with a 401 without authentication.

> **If you want to rerun anything against the dataset — `eval`, the `exp` experiments, the critical-set timeline, etc. — contact the author (mhric24@gmail.com / xhricma00@stud.fit.vut.cz`) and I'll issue you a read-only HF token.** That lets you stream the dataset on-demand without rebuilding it.

With the token in place:

```bash
export HF_TOKEN=<your-read-token>          # or: huggingface-cli login

uv run smclassifier nn tcn                eval
uv run smclassifier nn tcn-l              eval
uv run smclassifier nn tcn-s              eval
uv run smclassifier classic decision_tree eval
uv run smclassifier classic gmm           eval
uv run smclassifier classic svm           eval
```

Each report matches the macro F1 numbers cited in the thesis (§5, Test-split macro F1 table).

### Rebuilding the dataset (last resort)

If you'd rather rebuild the dataset from the original public sources instead of using the prebuilt one:

```bash
cd scripts/dataset
uv run python build.py full              # or: mid | --only-critical-set
```

This requires:

- An HF token with access to the gated pyannote VAD model (`pyannote/segmentation-3.0`); accept its terms on Hugging Face and export `HF_TOKEN`.
- ~50 GB free disk and several hours of bandwidth + CPU.

The critical subset is rebuilt from the hand-curated audio under `scripts/dataset/crit/{recordings,sources}/`.

No other API key is needed anywhere.

## Layout

```
src/
  main.py                  Top-level Click group + subgroup registration
  common.py                Label mapping (Speech=-1, Music=1, Background=2)
  config.py                Global config singleton (init_config / get_config / reset_config)
  evaluator.py             Shared metric machinery: run_evaluation, format_report, CIs
  input_handler.py         HF loader → (X, y, subclasses) arrays for classic models
  logging_config.py        Logging setup
  seed.py                  Reproducible random seeding

  classic/                 Decision tree, GMM, SVM
    modelclass.py          Abstract base: fit / save / load via joblib
    feat_extractor.py      Stateful frame-level feature extractor
    decisiontree.py        DecisionTree
    gmm.py                 Three-GMM density classifier
    svm.py                 SVC
    streaming.py           StreamingClassifier — sample-by-sample inference
    evaluation.py          eval_classic: load model + test features, run evaluation
    cli.py                 Per-model Click groups: train / eval / smoke-test

  nn/                      Neural-network models
    blocks.py              CausalConv1d + TCNResidualBlock (shared)
    backbones.py           TCN / GRU / LSTM / Transformer backbones
    preprocessors.py       conv1d / conv2d / no-op preprocessors
    temporal_heads.py      LSTM / GRU / attention heads
    dataset.py             HF loader (NN side) + frame-label builder
    evaluation.py          run_nn_inference — generic inference loop
    modelclass.py          Abstract NNModelClass base: train / evaluate / smoke_test
    variant.py             make_variant() factory used by variants.py
    variants.toml          TCN-S / TCN-L declarations
    variants.py            Reads variants.toml, registers Click groups + classes
    cli.py                 nn_group — registers tcn group + auto-adds every variant

    tcn/                   Reference TCN (Lemaire & Holzapfel, ISMIR 2019)
      model.py             SpeechMusicDetector (waveform → probs) + CausalTCN
      preprocess.py        LogMelSpectrogram + Δ/Δ² + MFCC + PCEN frontends
      training.py          Training loop + mel precompute/cache + optimizer
      streaming.py         StreamingInference — online chunk-by-chunk inference
      augmentation.py      augment_mel — ±6 dB gain in normalized log-mel space
      evaluation.py        Thin wrapper around nn/evaluation.run_nn_inference
      config.py            Config loader: get_config / get_ablation_config
      cli.py               Click CLI: train / eval / smoke-test / smoke-test-online

  exp/                     Experiment modules (one per thesis subsection in §5)
    cli.py                 exp_group — registers every experiment subgroup
    complexity/            §5.4.1–5.4.2 CPU + symbolic complexity benchmark
    critical/              §5.4.3 critical-subset per-recording timelines
    nn_architecture/       §5.2.4 GRU / LSTM / Transformer backbones at two budgets
    nn_preprocessor/       §5.2.3 conv1d / conv2d learned preprocessor
    small_tcn_pareto/      §5.3 n_filters Pareto sweep
    small_tcn_stacks/      §5.3 n_stacks sweep — produced TCN-S
    tcn_ablation/          §5.2.1 categorical substitutions + numeric sensitivity
    tcn_combined/          §5.2.5 F+P / F+A / P+A / F+P+A factorial
    tcn_frontend/          §5.2.2 mel-bin count, Δ/Δ², MFCC, PCEN
    tcn_temporal_head/     §5.2.6 LSTM / GRU / attention heads — produced TCN-L

  demo/                    Reflex streaming web UI (see top-level README disclosure)
    cli.py                 demo command — launches reflex run via subprocess
    rxconfig.py            Reflex config
    runner.py              ClassicRunner / NNRunner — uniform streaming wrappers
    source.py              MicSource / FileSource — async chunk producers
    weights_check.py       Filesystem probe used to badge the model picker
    ui/ui.py               Reflex DemoState + page (model picker, recharts plot)
```
