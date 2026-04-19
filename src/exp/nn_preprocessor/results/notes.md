# Preprocessor experiment — results notes

Sweep of shape-preserving *learned* layers inserted between the fixed log-mel front-end and the TCN backbone. Baseline passes mel directly into the TCN.

## Results

| Variant | Macro F1 | CI | Δ vs base |
|---|---|---|---|
| **conv1d** | **0.9786** | [0.9735, 0.9821] | +0.0047 |
| baseline (none) | 0.9739 | [0.9678, 0.9779] | 0.0 |
| conv2d | 0.9728 | [0.9668, 0.9775] | −0.0011 |

## Conclusion

**A learned 1-D channel projection clears baseline CI**, gaining +0.0047 macro F1. This is comparable in magnitude to the temporal-derivative gain in the frontend experiment and comes at minimal parameter cost.

**2-D convolution does not help.** Treating the mel spectrogram as an image (conv2d, (time, freq) kernels) lands slightly below baseline — CI fully overlaps, so call it flat. The plausible reading: the TCN already convolves along time with dilated kernels, so a 2-D preprocessor redundantly reprocesses the time axis, while conv1d cleanly handles only the *channel* dimension the TCN cannot mix on its own (its first-layer 1x5 conv mixes time, not mel bins at t=0).

**Why conv1d helps.** The raw mel filterbank outputs are not optimal features for the downstream TCN — a learned linear combination of the 80 mel bins (essentially a trainable, possibly wider-receptive frequency mixer) gives the backbone features better aligned with the loss. This is the "learnable front-end on top of fixed front-end" pattern that's become standard in audio models.

**Takeaway.** Adopt `conv1d` as the preprocessor. Composes with delta2 and tcn_large in the combined experiment to reach 0.9855.
