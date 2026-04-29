# TCN ablation — results notes

Base recipe: SGD, lr=1e-3, WeightNorm, n_filters=16, n_layers=4, n_stacks=3, kernel=5, dropout=0.5, seq_len=128, BCE-with-logits, 30 epochs, full tier. Baseline Macro F1 = **0.9752**.

## Results

| Variant | Macro F1 | CI | Δ vs base | Note |
|---|---|---|---|---|
| **adam_batchnorm** | **0.9792** | [0.9737, 0.9829] | +0.0040 | best — Adam survives with BN |
| **sgd_lr_1e-2** | **0.9787** | [0.9729, 0.9824] | +0.0035 | survived; paper-adjacent LR is knife-edge under WeightNorm |
| **dropout_low** (0.1) | **0.9782** | [0.9723, 0.9820] | +0.0030 | dropout 0.5 is mildly over-regularized |
| dropout_medium (0.25) | 0.9774 | [0.9713, 0.9813] | +0.0022 | flat improvement |
| gelu | 0.9756 | [0.9693, 0.9801] | +0.0004 | flat |
| baseline | 0.9752 | [0.9687, 0.9794] | 0 | reference |
| leaky_relu | 0.9752 | [0.9687, 0.9793] | 0 | identical to baseline at this seed |
| batch_norm | 0.9739 | [0.9671, 0.9784] | −0.0013 | swapping WN→BN alone is neutral |
| layers_3 | 0.9736 | [0.9671, 0.9780] | −0.0016 | RF=85 frames |
| layers_2 | 0.9723 | [0.9655, 0.9768] | −0.0029 | RF=37 frames |
| sgd_batchnorm | 0.9706 | [0.9627, 0.9758] | −0.0046 | SGD+BN underperforms SGD+WN |
| elu | 0.9639 | [0.9541, 0.9704] | −0.0113 | ELU's negative saturation hurts |
| layers_1 | 0.9573 | [0.9450, 0.9662] | −0.0179 | RF=13 frames clearly insufficient |
| **no_skip** | **0.1884** | [0.1725, 0.2039] | diverged | collapses to music-only — residuals are essential |
| **adam** | **0.1720** | [0.1583, 0.1862] | diverged | Adam without BN collapses to speech-only |

## Conclusion

The paper-prescribed TCN recipe is **saturated** along most structural knobs. Three classes of signal emerge:

1. **Catastrophic failures** — Adam without BN, removing skip connections. These identify *load-bearing* pieces of the recipe: residuals are non-negotiable, and Adam requires BN to stay numerically stable on this data.
2. **Real but small effects** — dropping dropout from 0.5 to 0.1 (+0.0030) is the cleanest in-recipe win; Adam+BN (+0.0040) and SGD@1e-2 (+0.0035) win at the optimizer level but each comes with stability caveats. Reducing `n_layers` below 4 hurts monotonically as RF starves; ELU underperforms ReLU/LeakyReLU/GELU.
3. **No effect** — the activation choice between ReLU/LeakyReLU/GELU is within noise; layer count drops within the 3–4 range are within ~0.0016; switching WN→BN under SGD is neutral (BN alone) or slightly worse (SGD+BN combined).

Several knobs were tested then commented out of the config because every variant landed within ±0.003 of baseline (`config.toml` lines 33–44, 62–67, 80–83, 85–92, 101–106, 130–140): loss function (`bce`/`focal`/`weighted_bce`/`mse`/`label_smoothing_bce`), capacity (`filters_8`/`filters_32`), stacks (`stacks_5`), kernel size (3/7/9), seq_len (256/270), batch size (16/64), and augmentation (`no_augment`). `mels_40`/`mels_128` diverged — the cached preprocess stats are hardcoded at 80 mels, so those variants needed their own stats path.

**Takeaway.** No single ablation on model *structure* (filters, kernel, stacks, seq_len, batch) beats noise. The recipe-level wins (Adam+BN, dropout=0.1, lr=1e-2) compose poorly because each shifts a different piece of the numerical envelope. Gains must come from **outside the backbone** — see frontend and preprocessor experiments.
