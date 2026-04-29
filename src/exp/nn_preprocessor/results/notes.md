# Preprocessor experiment — results notes

Sweep of shape-preserving *learned* layers inserted between the fixed log-mel front-end and the TCN backbone. Baseline passes mel directly into the TCN.

## Results

| Variant | Macro F1 | CI | Δ vs base |
|---|---|---|---|
| **conv1d** | **0.9784** | [0.9728, 0.9822] | +0.0032 |
| baseline (none) | 0.9752 | [0.9687, 0.9793] | 0.0 |
| conv2d | 0.9742 | [0.9684, 0.9783] | −0.0010 |

## Conclusion

**A learned 1-D channel projection helps, just inside the noise floor.** `conv1d` gains +0.0032 macro F1 with CIs that overlap baseline at the lower edge ([0.9728, 0.9822] vs [0.9687, 0.9793]). The improvement is real but small — comparable in magnitude to the temporal-derivative gain in the frontend experiment, and stacks with it (combined experiment).

**2-D convolution does not help.** Treating the mel spectrogram as an image (conv2d, (time, freq) kernels) lands slightly below baseline — CI fully overlaps, so call it flat. The plausible reading: the TCN already convolves along time with dilated kernels, so a 2-D preprocessor redundantly reprocesses the time axis, while conv1d cleanly handles only the *channel* dimension the TCN cannot mix on its own (its first-layer 1×5 conv mixes time, not mel bins at t=0).

**Why conv1d helps.** The raw mel filterbank outputs are not optimal features for the downstream TCN — a learned linear combination of the 80 mel bins (essentially a trainable, possibly wider-receptive frequency mixer) gives the backbone features better aligned with the loss. This is the "learnable front-end on top of fixed front-end" pattern that's become standard in audio models.

**Earlier failed runs.** The three `.diverged` files in this directory record a prior architecture-mismatch incident: the preprocessor expected 80-channel input but received 240 (delta2 frontend remnants in the cache key). Fixed by including `n_features` in the mel cache filename — see CLAUDE.md "Mel cache key changed". The current `.eval` files are from the post-fix retrain.

**Takeaway.** Adopt `conv1d` as the preprocessor. Composes with delta2 and tcn_large in the combined experiment to reach 0.9844.
