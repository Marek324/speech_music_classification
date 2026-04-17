# Preprocessor layer experiment — results

Tests whether inserting a small learned causal layer between the log-mel
frontend and the TCN backbone improves accuracy. Everything else is held at
the ablation baseline (SGD, lr=1e-3, seq_len=128, 16 filters, k=5, 4 layers,
3 stacks, dropout 0.5, weight-norm on). Dataset: `full` tier. Hardware: A100.

## Variants
- **baseline** — no preprocessor, frontend feeds backbone directly.
- **conv1d** — causal depthwise/separable 1D conv across time.
- **conv2d** — causal 2D conv across (time, mel).

## Headline metrics

| Variant  | Macro F1 | 95% CI           | Weighted F1 | Acc    | AUROC  | ms/frame |
|----------|---------:|------------------|------------:|-------:|-------:|---------:|
| baseline | 0.9739   | [0.9678, 0.9779] | 0.9758      | 0.9758 | 0.9978 | 0.0093   |
| conv1d   | **0.9786** | [0.9735, 0.9821] | **0.9803**  | **0.9803** | **0.9984** | **0.0067** |
| conv2d   | 0.9728   | [0.9668, 0.9775] | 0.9746      | 0.9746 | 0.9970 | 0.0081   |

## Per-class F1

| Variant  | Speech | Music  | Inactive |
|----------|-------:|-------:|---------:|
| baseline | 0.9738 | 0.9879 | 0.9600   |
| conv1d   | 0.9794 | 0.9909 | 0.9654   |
| conv2d   | 0.9733 | 0.9859 | 0.9592   |

## Takeaways
- **conv1d wins on every axis**: +0.47 macro-F1 over baseline and ~28% faster
  (0.0067 vs 0.0093 ms/frame). Gains are uniform across Speech/Music/Inactive.
- **conv2d slightly regresses** vs baseline (-0.11 macro-F1) despite extra
  parameters — mel-axis convolution does not help here.
- Hardest subclass remains `speech_som` (~0.95 F1) across all variants;
  `noise` is essentially solved (>0.998 in every run).
- All variants clear the 0.85 project target by a wide margin.

## Artifacts
`results/tcn_<variant>.eval` — full report.
`results/tcn_<variant>_scores.npz` — frame-level scores for bootstrap/CI.
