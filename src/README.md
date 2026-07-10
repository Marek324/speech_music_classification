# `src/` — Code Architecture

This document maps the source tree. For setup, evaluation, and the full CLI surface see the
root [REPRODUCTION.md](../REPRODUCTION.md); for the project overview and results see the root
[README.md](../README.md).

## Overview

The CLI entry point is `smclassifier` (`pyproject.toml` → `src.main:cli`). The top-level Click
group registers four subgroups:

- **`classic`** — the three reference baselines (decision tree, GMM, SVM), each with
  `train` / `eval` / `smoke-test`.
- **`nn`** — the TCN family. The canonical `tcn` group is registered explicitly; every variant
  (`tcn-s`, `tcn-l`) is auto-added from `src/nn/variants.toml`.
- **`exp`** — one subgroup per experiment (each realizing a thesis subsection).
- **`demo`** — the Reflex streaming web UI.

Two design points are worth calling out:

- **Shared vs. model-specific.** Everything reusable across models lives one level up from the
  model packages: the metric/report machinery (`evaluator.py`), the causal conv building blocks
  (`nn/blocks.py`), the NN dataset loader and inference loop (`nn/dataset.py`,
  `nn/evaluation.py`), and the label mapping (`common.py`). Each model package only holds what
  is genuinely specific to it.
- **TOML-driven variants.** TCN-S and TCN-L are not hand-written subclasses — they are declared
  as TOML blocks in `src/nn/variants.toml` and materialized at import time by
  `make_variant()` (`nn/variant.py`), which builds the config loader, Click group, and model
  subclass for each entry. Adding a new variant is a TOML-only edit.

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

  demo/                    Reflex streaming web UI (see the disclosure in REPRODUCTION.md)
    cli.py                 demo command — launches reflex run via subprocess
    rxconfig.py            Reflex config
    runner.py              ClassicRunner / NNRunner — uniform streaming wrappers
    source.py              MicSource / FileSource — async chunk producers
    weights_check.py       Filesystem probe used to badge the model picker
    ui/ui.py               Reflex DemoState + page (model picker, recharts plot)
```
