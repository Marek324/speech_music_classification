# SmallTCN stacks sweep — results notes

Receptive-field-only sweep over `n_stacks` at the fixed SmallTCN recipe (log_mel_delta2 frontend, no preprocessor, no tail, n_filters=8, SGD + WeightNorm, full tier). Reads off the F1-vs-RF tradeoff directly. Also serves as the source experiment that promoted `n_stacks=1` to a separately deployed variant (**SmallerTCN**, see `weights/smaller_tcn/`).

Receptive field per variant (`frames = 1 + n_stacks · n_layers · 2 · (k−1)`):

| Variant | n_stacks | RF (frames) | RF (ms) |
|---|---|---|---|
| stacks_1 (SmallerTCN) | 1 | 121 | ≈ 2810 |
| stacks_2 | 2 | 241 | ≈ 5596 |
| stacks_3 (SmallTCN deployed) | 3 | 361 | ≈ 8382 |

## Results

| Variant | Macro F1 | CI | Δ vs deployed (3) |
|---|---|---|---|
| stacks_1 (RF=2.81 s) | 0.9722 | [0.9653, 0.9769] | −0.0049 |
| stacks_2 (RF=5.60 s) | 0.9749 | [0.9682, 0.9795] | −0.0022 |
| **stacks_3** (RF=8.38 s) | **0.9771** | [0.9710, 0.9812] | 0.0 |

## Conclusion

**F1 grows monotonically with RF, but the slope is shallow.** stacks_1 → stacks_2 buys +0.0027, stacks_2 → stacks_3 buys +0.0022. All three CIs overlap. The classifier doesn't need 8 s of audio context to discriminate speech / music / background — a 2.8 s window already lands within 0.005 of the deployed model.

**stacks_1 has a property the others don't: RF ≤ seq_len.** At the training chunk length of 128 frames, only `stacks_1` (RF = 121 frames) lets the model see its full receptive field on every training chunk. `stacks_2` and `stacks_3` are *context-starved* during training — their chunks are shorter than their RF, so the early frames of every chunk see less left-context than they would at inference time. This is a known mismatch in the recipe proposed by the paper (see CLAUDE.md, "Receptive field vs chunk").

**SmallerTCN deployment.** The `stacks_1` checkpoint was promoted to a separately packaged variant (`SmallerTCN` in `variants.toml`) because:
1. **No train/inference RF mismatch** — the only checkpoint in the project where this holds.
2. **6.4× fewer streaming MACs/frame** (545 K vs 3.47 M for SmallTCN — see `complexity_t1.md`), because each push re-runs only one stack instead of three.
3. **Highest measured RTF** at 1 thread on the i5-8300H benchmark (5.3× vs 3.4× for SmallTCN).

In exchange, SmallerTCN gives up −0.005 macro F1 on the standard test split. On the crit-tier evaluation the deficit relative to SmallTCN collapses to noise (0.4298 vs 0.4294, see `src/exp/critical/results/notes.md`), so under adversarial conditions the two variants are interchangeable.

**stacks_3 remains the SmallTCN deployment.** It gives the best macro F1 of the three at the cost of the largest streaming compute. Whether to ship `stacks_3` (SmallTCN) or `stacks_1` (SmallerTCN) is a deployment-time tradeoff between accuracy on clean test data and streaming cost; the demo packages both.
