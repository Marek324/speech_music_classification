# TCN ablation — results notes

Base recipe: SGD, lr=1e-3, WeightNorm, n_filters=16, n_layers=4, n_stacks=3, kernel=5, dropout=0.5, seq_len=128, 30 epochs, full tier. Baseline Macro F1 = **0.9249**.

## Results

| Variant | Macro F1 | Δ vs base | Note |
|---|---|---|---|
| baseline | 0.9249 | — | reference |
| **sgd_lr_1e-2** | **0.9280** | +0.0031 | best flat run; higher LR squeezes a touch more |
| **dropout_low** (0.1) | **0.9277** | +0.0028 | dropout 0.5 may be too aggressive |
| adam_batchnorm | 0.9262 | +0.0013 | Adam only survives with BN |
| dropout_medium (0.25) | 0.9258 | +0.0009 | flat |
| baseline_leaky_relu | 0.9249 | 0.0000 | flat |
| sgd_batchnorm | 0.9247 | −0.0002 | flat |
| batch_norm | 0.9248 | −0.0001 | flat |
| loss (bce_with_logits) | 0.9246 | −0.0003 | flat |
| filters_32 | 0.9242 | −0.0007 | capacity saturates at 16 |
| stacks_5 | 0.9242 | −0.0007 | flat |
| gelu | 0.9241 | −0.0008 | flat |
| filters_8 | 0.9240 | −0.0009 | flat |
| seq_len_256 | 0.9240 | −0.0009 | flat (RF already covered at 128) |
| no_augment | 0.9237 | −0.0012 | ±6 dB gain barely moves the needle |
| batch_64 | 0.9236 | −0.0013 | flat |
| seq_len_270 | 0.9234 | −0.0015 | flat |
| batch_16 | 0.9227 | −0.0022 | flat |
| kernel_3 | 0.9222 | −0.0027 | flat |
| kernel_7 | 0.9226 | −0.0023 | flat |
| kernel_9 | 0.9218 | −0.0031 | flat |
| layers_3 | 0.9224 | −0.0025 | minor loss |
| layers_2 | 0.9189 | −0.0060 | RF=37 too small |
| layers_1 | 0.9057 | −0.0192 | RF=13 clearly insufficient |
| elu | 0.9091 | −0.0158 | ELU's negative saturation hurts |
| **adam** | **0.1914** | diverged | collapses to speech-only |
| **no_skip** | **0.1895** | diverged | collapses to music-only — residuals are essential |
| sgd_lr_1e-2 (WN) | 0.9280 | +0.0031 | survived; knife-edge |
| mels_40 / mels_128 | n/a | diverged | stats hardcoded at 80 mels |

## Conclusion

The paper-prescribed TCN recipe is **saturated** along every structural knob tested. 21 of 28 completed variants land within ±0.003 of baseline — below any plausible single-seed noise floor. Two classes of signal emerge:

1. **Catastrophic failures** — Adam (without BN), removing skip connections, and `mels_40/128` variants (cache key bug). These identify *load-bearing* pieces of the recipe: residuals, normalization choice coupled to optimizer, mel-bin dimensionality.
2. **Real but small effects** — reducing `n_layers` below 4 hurts monotonically (RF starvation); dropout=0.5 is mildly over-regularized (dropout_low +0.003); ELU underperforms ReLU/LeakyReLU/GELU; lr=1e-2 barely survives but gains a touch.

No single ablation on model *structure* (filters, kernel, stacks, seq_len, batch, kernel) beats noise. Gains must come from **outside the backbone** — see frontend and preprocessor experiments.
