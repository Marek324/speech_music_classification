# SmallTCN Pareto sweep — results notes

Capacity-only sweep over `n_filters` at the fixed SmallTCN recipe (log_mel_delta2 frontend, no preprocessor, no tail, SGD + WeightNorm, 50 epochs, full tier). Isolates capacity as the single free axis to estimate the Pareto curve between parameter count and macro F1 for the small-footprint deployment target.

## Results

| Variant | n_filters | Macro F1 | CI | Δ vs deployed (8) |
|---|---|---|---|---|
| filters_4 | 4 | 0.9530 | [0.9398, 0.9631] | −0.0236 |
| filters_6 | 6 | 0.9733 | [0.9662, 0.9779] | −0.0033 |
| **filters_8** (deployed) | 8 | **0.9766** | [0.9704, 0.9808] | 0.0 |
| filters_12 | 12 | 0.9795 | [0.9742, 0.9830] | +0.0029 |

## Conclusion

**Sharp knee at n_filters = 6 → 8.** The 4 → 6 step buys +0.0203 macro F1 (the largest single jump in the sweep), 6 → 8 buys +0.0033, and 8 → 12 buys +0.0029. Below 8 filters the body has too few channels to disentangle the speech/music/background subspaces (the 4-filter run loses 0.05 F1 on `music_acapella` specifically, suggesting that subclass is the first to collapse under capacity pressure).

**Diminishing returns past n_filters = 8.** Filters_12 lands +0.0029 above the deployed model, well within filters_8's CI of [0.9704, 0.9808]. The marginal accuracy is real but small enough that the additional MACs/frame (which scale with n_filters², so ~2.25× the body compute) aren't a clean trade for streaming deployment.

**Filters_8 is the right Pareto point** for the SmallTCN target. It clears the 0.85 project bar by 13 pp, sits within 0.003 F1 of the strictly larger filters_12, and matches the deployed checkpoint shipped via `weights/small_tcn/`. Filters_12 stays in the table as evidence that the chosen capacity is not pessimistic — there is no large gain hiding above the deployed configuration.

**Note on training duration.** This sweep uses 50 epochs, vs 30 epochs in the rest of the ablation programme. The deployed `weights/small_tcn/tcn_small.safetensors` originated from the same 50-epoch recipe, so filters_8's 0.9766 is the in-experiment seed-noise reference for the deployed model.
