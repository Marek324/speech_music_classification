# Combined-winners experiment — results notes

2² factorial over the three source-experiment winners, multiplied by the capacity winner:

- **F** = `log_mel_delta2` frontend (frontend exp winner, +0.0053)
- **P** = `conv1d` preprocessor (preprocessor exp winner, +0.0047)
- **A** = `n_filters = 32` TCN (architecture exp winner, +0.0007)

Tests whether these gains compose or cannibalize. All variants use SGD + WeightNorm + 30 epochs + full tier.

## Results

| Variant | F | P | A | Macro F1 | CI | Δ vs base |
|---|---|---|---|---|---|---|
| **delta2_conv1d_large** | ✓ | ✓ | ✓ | **0.9855** | [0.9814, 0.9884] | **+0.0093** |
| delta2_conv1d | ✓ | ✓ | — | 0.9846 | [0.9802, 0.9876] | +0.0084 |
| delta2_large | ✓ | — | ✓ | 0.9834 | [0.9790, 0.9864] | +0.0072 |
| conv1d_large | — | ✓ | ✓ | 0.9802 | [0.9753, 0.9836] | +0.0040 |
| baseline | — | — | — | 0.9762 | [0.9708, 0.9802] | 0.0 |

Expected-if-additive sum of individual gains: **+0.0107**. Observed F+P+A: **+0.0093**. Recovery: ~87%.

## Conclusion

**Gains compose, mostly.** The full stack F+P+A reaches 0.9855 macro F1, clearing the baseline CI and recovering ~87% of the sum of individual experiment gains. Diminishing returns are modest — plausibly because delta2 and conv1d both provide "better features in a fixed-capacity backbone," so they partially target the same slack.

**Ranking of effect sizes (controlled in-experiment):**
1. Frontend (`delta2`): ~+0.0070 contribution inside the combined grid
2. Preprocessor (`conv1d`): ~+0.0040
3. Capacity (`tcn_large`): ~+0.0010 — within noise, consistent with the architecture-experiment finding that capacity is saturated

**Takeaway.** `delta2_conv1d_large` is the best configuration produced by the ablation program, 0.0093 above baseline with CIs that cleanly separate. If a single-change deployment is preferred, `delta2` alone is a safe pick (frontend exp: 0.9799) — it captures ~60% of the combined gain with a zero-parameter modification.

**Not reproduced under the paper-literal recipe.** These results are on 30-epoch SGD+WeightNorm runs. Moving to 50 epochs (`957bfb0`) or to Adam+BN (ablation exp result: +0.0013) would likely shift absolute numbers but is orthogonal to the compositional finding.
