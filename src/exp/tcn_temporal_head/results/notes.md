# Hybrid-backbone experiment — results notes

Stacks a stateful causal tail on top of the combined-experiment winner (delta2 frontend + conv1d preprocessor + TCN f=16). Tests whether an RNN / attention / parallel-TCN tail on the TCN feature map beats the pure-TCN stack. All variants use SGD + WeightNorm + full tier.

## Results

| Variant | Tail | Added params | Macro F1 | CI | Δ vs baseline |
|---|---|---|---|---|---|
| **tcn_gru_wide** | GRU (h=64) | ~25K | **0.9863** | [0.9820, 0.9891] | +0.0032 |
| tcn_gru | GRU (h=32) | ~9K | 0.9852 | [0.9807, 0.9882] | +0.0021 |
| tcn_lstm | LSTM (h=32) | ~12K | 0.9845 | [0.9792, 0.9882] | +0.0014 |
| **baseline** | — (delta2_conv1d) | 0 | 0.9831 | [0.9757, 0.9882] | 0.0 |
| tcn_two_branch | parallel dilated TCN (f=16) | ~30K | 0.9829 | [0.9755, 0.9879] | −0.0002 |
| tcn_attn | causal self-attention (h=32, 4 heads) | ~20K | 0.9810 | [0.9735, 0.9856] | −0.0021 |

## Conclusion

**No tail produces a statistically significant gain.** The best variant (`tcn_gru_wide`) lands at +0.0032 over baseline, fully inside the baseline CI. RNNs (GRU / LSTM) tie each other within noise; the wider GRU at h=64 edges the h=32 versions but the gap (~0.001) is below the seed noise floor.

**Attention is again the loser** (−0.0021). Matches the architecture experiment's observation that causal self-attention underperforms at this scale and sequence length (`seq_len = 128` starves attention of the long-range context it's designed to exploit, and no positional prior matches the strong local temporal structure of speech/music frames).

**Two-branch parallel TCN is flat.** Adding a wider-dilation branch alongside the main TCN buys 30K params for −0.0002 — same money, no return. The TCN receptive field is already large enough at 3 stacks × 4 layers × kernel 5.

**Takeaway.** The `delta2 + conv1d + TCN` stack from the combined experiment appears to be saturated: stacking a stateful head on top buys at most ~0.3 pp and never outside noise. For the thesis, `delta2_conv1d` (0.9831) remains the defensible production config; an RNN tail can be ramped in if a marginal win is needed but adds complexity disproportionate to the gain. The TCN-LSTM variant is kept as a packaged option in `variants.toml` for the demo despite landing inside the baseline CI here — the streaming-receptive-field argument for keeping a stateful tail is independent of macro-F1.
