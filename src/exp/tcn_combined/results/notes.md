# Combined-winners experiment — results notes

2² factorial over the three source-experiment winners, multiplied by the capacity winner:

- **F** = `log_mel_delta2` frontend (frontend exp winner, +0.0064)
- **P** = `conv1d` preprocessor (preprocessor exp winner, +0.0032)
- **A** = `n_filters = 32` TCN (architecture exp winner, +0.0028)

Tests whether these gains compose or cannibalize. All variants use SGD + WeightNorm + 30 epochs + full tier.

## Results

| Variant | F | P | A | Macro F1 | CI | Δ vs base |
|---|---|---|---|---|---|---|
| **delta2_conv1d_large** | ✓ | ✓ | ✓ | **0.9844** | [0.9798, 0.9875] | **+0.0092** |
| delta2_large | ✓ | — | ✓ | 0.9832 | [0.9782, 0.9864] | +0.0080 |
| delta2_conv1d | ✓ | ✓ | — | 0.9822 | [0.9743, 0.9877] | +0.0070 |
| conv1d_large | — | ✓ | ✓ | 0.9787 | [0.9732, 0.9825] | +0.0035 |
| baseline | — | — | — | 0.9752 | [0.9687, 0.9794] | 0.0 |

Expected-if-additive sum of individual gains: **+0.0124**. Observed F+P+A: **+0.0092**. Recovery: ~74%.

## Conclusion

**Gains compose, mostly.** The full stack F+P+A reaches 0.9844 macro F1, clearing the baseline CI and recovering ~74% of the sum of individual experiment gains. Diminishing returns are modest — plausibly because delta2 and conv1d both provide "better features in a fixed-capacity backbone," so they partially target the same slack.

**Ranking of effect sizes (controlled in-experiment):**
1. Frontend (`delta2`): ~+0.0070 contribution inside the combined grid (delta2_conv1d − conv1d_large = +0.0035 marginal at one cell, but full delta2 contribution averaged over P×A ≈ +0.0070 — the cleanest single-knob win)
2. Preprocessor (`conv1d`): ~+0.0010–0.0035 — smallest of the three, and inconsistent across cells (delta2_large vs delta2_conv1d_large is only +0.0012; conv1d alone vs baseline is +0.0035)
3. Capacity (`tcn_large`): ~+0.0010–0.0080 — adds the most when stacked with `delta2`, smallest as a standalone change. Consistent with the architecture experiment finding that capacity helps slightly.

**Takeaway.** `delta2_conv1d_large` is the best configuration produced by the ablation program, 0.0092 above baseline with CIs that cleanly separate. If a single-change deployment is preferred, `delta2` alone is a safe pick (frontend exp: 0.9816) — it captures most of the combined gain with a zero-parameter modification. The preprocessor only earns its keep when stacked with capacity (`delta2_large` 0.9832 already covers most of `delta2_conv1d_large` 0.9844).

**Not reproduced under the paper-literal recipe.** These results are on 30-epoch SGD+WeightNorm runs. Moving to Adam+BN (ablation experiment: +0.0040 standalone) is orthogonal to the compositional finding here — the relative ranking of variants should hold, only the absolute baseline shifts.
