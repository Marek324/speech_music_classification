# Architecture experiment — results notes

Sweep of causal sequence-model backbones at two capacity budgets (~32K and ~128K params). Everything else matches baseline recipe (SGD, WeightNorm, log-mel 80, seq_len=128, 30 epochs, full tier).

## Results

| Variant | Params | Macro F1 | CI | Δ vs TCN |
|---|---|---|---|---|
| **tcn_large** (f=32) | ~126K | **0.9762** | [0.9708, 0.9802] | +0.0007 |
| **baseline TCN** (f=16) | ~32K | **0.9755** | [0.9701, 0.9793] | 0.0 |
| gru_small | ~28K | 0.9720 | [0.9655, 0.9764] | −0.0035 |
| lstm_large | ~138K | 0.9715 | [0.9653, 0.9758] | −0.0040 |
| gru_large | ~132K | 0.9709 | [0.9624, 0.9773] | −0.0046 |
| lstm_small | ~36K | 0.9686 | [0.9618, 0.9736] | −0.0069 |
| transformer_large | ~117K | 0.8792 | [0.8316, 0.9197] | −0.0963 |
| transformer_small | ~31K | 0.8526 | [0.8114, 0.8921] | −0.1229 |

## Conclusion

**TCN is the right inductive bias for this task.** The dilated-conv baseline at 32K params beats every RNN variant at 128K params — causal attention (Transformer) trails badly even at matched budget, and GRU/LSTM land ~0.4–0.7 pp below TCN regardless of size.

**Capacity is saturated.** `tcn_large` (32 filters, ~126K params) gains +0.0007 over baseline — well inside its CI of [0.9708, 0.9802]. Neither GRU nor LSTM improves by scaling either. Quadrupling params yields no measurable win.

**Transformer finding.** Causal self-attention underperforms by ~10 pp. Likely causes: (i) short seq_len=128 starves attention of context it's designed to exploit; (ii) no positional prior matching the strong local temporal structure of speech/music frames; (iii) limited data at full tier. Either way, TCN's hard-coded dilation hierarchy is a far stronger prior here than learned attention.

**Takeaway for downstream work.** Use TCN; use 16 filters unless stacking with other gains (see combined experiment, where `tcn_large` composes with delta2+conv1d for the project's best result). No further time should be spent searching backbones or scaling capacity.
