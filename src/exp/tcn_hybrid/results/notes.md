# Hybrid-backbone experiment — results notes

Stacks a stateful causal tail on top of the combined-experiment winner (delta2 frontend + conv1d preprocessor + TCN f=16). Tests whether an RNN / attention / parallel-TCN tail on the TCN feature map beats the pure-TCN stack. All variants use SGD + WeightNorm + full tier.

## Results

| Variant | Tail | Added params | Macro F1 | CI | Δ vs baseline |
|---|---|---|---|---|---|
| tcn_lstm | LSTM (h=32) | ~12K | **0.9865** | [0.9828, 0.9891] | +0.0023 |
| tcn_gru | GRU (h=32) | ~9K | 0.9862 | [0.9823, 0.9889] | +0.0020 |
| tcn_gru_wide | GRU (h=64) | ~25K | 0.9856 | [0.9819, 0.9882] | +0.0014 |
| tcn_two_branch | parallel dilated TCN (f=16) | ~30K | 0.9847 | [0.9803, 0.9876] | +0.0005 |
| **baseline** | — (delta2_conv1d) | 0 | 0.9842 | [0.9800, 0.9871] | 0.0 |
| tcn_attn | causal self-attention (h=32, 4 heads) | ~20K | 0.9828 | [0.9759, 0.9870] | −0.0014 |

## Conclusion

**No tail produces a statistically significant gain.** The best variant (LSTM) lands at +0.0023 over baseline, fully inside the baseline CI. RNNs (LSTM / GRU) tie each other within noise; widening the GRU from h=32 to h=64 doesn't help, consistent with the architecture-experiment finding that capacity is saturated on this dataset.

**Attention is again the loser** (−0.0014). Matches the architecture experiment's observation that causal self-attention underperforms at this scale and sequence length (`seq_len = 128` starves attention of the long-range context it's designed to exploit, and no positional prior matches the strong local temporal structure of speech/music frames).

**Two-branch parallel TCN is flat.** Doubling the feature-map compute path with a wider-dilation branch adds 30K params for +0.0005 — same money, no return. The TCN receptive field is already large enough at 3 stacks × 4 layers × kernel 5.

**Takeaway.** The `delta2 + conv1d + TCN` stack from the combined experiment appears to be saturated: stacking a stateful head on top buys at most 0.2 pp and never outside noise. For the thesis, `delta2_conv1d` (0.9842) is the defensible production config; hybrid tails are a negative result worth reporting but not deploying.
