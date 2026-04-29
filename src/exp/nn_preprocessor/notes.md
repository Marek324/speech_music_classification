# Preprocessor layer experiment — results

Tests whether inserting a small learned causal layer between the log-mel
frontend and the TCN backbone improves accuracy. Everything else is held at
the ablation baseline (SGD, lr=1e-3, seq_len=128, 16 filters, k=5, 4 layers,
3 stacks, dropout 0.5, weight-norm on). Dataset: `full` tier. Hardware: A6000.

## Variants
- **baseline** — no preprocessor, frontend feeds backbone directly.
- **conv1d** — causal depthwise/separable 1D conv across time.
- **conv2d** — causal 2D conv across (time, mel).

## Headline metrics

| Variant  | Macro F1 | 95% CI           | Weighted F1 | Acc    | AUROC  | ms/frame |
|----------|---------:|------------------|------------:|-------:|-------:|---------:|
| baseline | 0.9752   | [0.9687, 0.9793] | 0.9771      | 0.9771 | 0.9980 | 0.0083   |
| conv1d   | **0.9784** | [0.9728, 0.9822] | **0.9802**  | **0.9802** | **0.9985** | **0.0066** |
| conv2d   | 0.9742   | [0.9684, 0.9783] | 0.9760      | 0.9761 | 0.9978 | 0.0070   |

## Per-class F1

| Variant  | Speech | Music  | Inactive |
|----------|-------:|-------:|---------:|
| baseline | 0.9745 | 0.9899 | 0.9612   |
| conv1d   | 0.9796 | 0.9907 | 0.9649   |
| conv2d   | 0.9736 | 0.9880 | 0.9611   |

## Takeaways
- **conv1d wins on every axis**: +0.32 macro-F1 over baseline and ~20% faster
  (0.0066 vs 0.0083 ms/frame). Gains are uniform across Speech/Music/Inactive.
- **conv2d slightly regresses** vs baseline (−0.10 macro-F1) despite extra
  parameters — mel-axis convolution does not help here.
- Hardest subclass remains `speech_som` (~0.96 F1) across all variants;
  `noise` is essentially solved (>0.998 in every run).
- All variants clear the 0.85 project target by a wide margin.

## Artifacts
`results/tcn_<variant>.eval` — full report.
`results/tcn_<variant>_scores.npz` — frame-level scores for bootstrap/CI.
`results/tcn_<variant>.diverged` — failed earlier runs (channel-count mismatch with cached preprocess stats; superseded by the post-fix retrain).
