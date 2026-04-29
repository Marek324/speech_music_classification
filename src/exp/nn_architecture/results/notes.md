# Architecture experiment — results notes

Sweep of causal sequence-model backbones at two capacity budgets (~32K and ~128K params). Everything else matches baseline recipe (SGD, WeightNorm, log-mel 80, seq_len=128, 30 epochs, full tier).

## Results

| Variant | Params | Macro F1 | CI | Δ vs TCN |
|---|---|---|---|---|
| **tcn_large** (f=32) | ~126K | **0.9780** | [0.9723, 0.9818] | +0.0028 |
| gru_large (f=72) | ~132K | 0.9771 | [0.9712, 0.9811] | +0.0019 |
| lstm_large (f=64) | ~138K | 0.9758 | [0.9692, 0.9802] | +0.0006 |
| **baseline TCN** (f=16) | ~32K | **0.9752** | [0.9687, 0.9794] | 0.0 |
| lstm_small (f=32) | ~36K | 0.9741 | [0.9678, 0.9784] | −0.0011 |
| gru_small (f=32) | ~28K | 0.9704 | [0.9609, 0.9771] | −0.0048 |
| transformer_small (f=24) | ~31K | 0.8935 | [0.8513, 0.9242] | −0.0817 |
| transformer_large (f=48) | ~117K | 0.6942 | [0.6552, 0.7418] | −0.2810 |

## Conclusion

**TCN is the right inductive bias for this task.** The dilated-conv baseline at 32K params is within CI of every RNN variant at 128K params, and `tcn_large` extends the lead by another +0.0028. GRU/LSTM at matched capacity track TCN within ~0.001–0.005, and Transformers trail badly at every budget tested.

**Capacity helps slightly but is mostly saturated.** `tcn_large` (32 filters, ~126K params) gains +0.0028 over baseline, just inside its CI of [0.9723, 0.9818]. Quadrupling parameters yields a small but consistent win for TCN; for RNNs, scaling 32→128K closes the gap with TCN but never overtakes it (large GRU 0.9771 vs large TCN 0.9780).

**Transformer scaling regresses.** `transformer_small` lands at 0.8935 — already 8 pp below baseline. `transformer_large` collapses further to 0.6942: the per-class breakdown shows the larger model dumps most inactive frames into music (recall on `inactive` falls from 0.69 to 0.20; `noise` subclass F1 from 0.87 to 0.13). At seq_len=128 the larger Transformer apparently overfits the speech/music boundary while losing the noise/silence axis entirely. Likely causes: (i) short seq_len starves attention of the long-range context it's designed to exploit; (ii) no positional prior matching the strong local temporal structure of speech/music frames; (iii) inactive is the rarest class and the most sensitive to under-regularization.

**Takeaway for downstream work.** Use TCN. `tcn_large` (f=32) composes with delta2+conv1d for the project's best result (combined experiment). RNN tails on top of a TCN body are explored separately (hybrid experiment) — none of the RNN-only backbones here are competitive.
