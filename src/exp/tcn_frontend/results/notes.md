# Frontend experiment — results notes

Sweep of fixed (non-learned) spectral front-ends feeding the baseline TCN (SGD, WeightNorm, seq_len=128, 30 epochs, full tier).

## Results

| Variant | Channels | Macro F1 | CI | Δ vs base |
|---|---|---|---|---|
| **log_mel_delta2** | 240 (mel+Δ+ΔΔ) | **0.9799** | [0.9748, 0.9834] | +0.0053 |
| **log_mel_delta** | 160 (mel+Δ) | **0.9795** | [0.9743, 0.9830] | +0.0049 |
| log_mel_128 | 128 | 0.9753 | [0.9698, 0.9792] | +0.0007 |
| baseline (log_mel_80) | 80 | 0.9746 | [0.9688, 0.9787] | 0.0 |
| log_mel_40 | 40 | 0.9741 | [0.9684, 0.9781] | −0.0005 |
| pcen | 80 | 0.9556 | [0.9470, 0.9623] | −0.0190 |
| mfcc_20 | 20 | diverged | — | — |
| mfcc_40 | 40 | diverged | — | — |

## Conclusion

**Temporal derivatives help; mel-bin count does not.** Going from 40 → 80 → 128 mel bins produces no measurable change (all within ±0.001 of baseline). But concatenating first-order deltas (+0.005) or first+second (+0.005) cleanly clears the baseline CI. The signal is *temporal dynamics*, not *finer frequency resolution* — unsurprising given the receptive-field gap: the TCN sees 181 frames but still benefits from pre-computed short-term derivatives that a single small conv layer apparently can't learn as cleanly.

`delta2` edges `delta` by 0.0004 — within noise. Either is a safe win; `delta2` is chosen downstream for stacking.

**PCEN loses.** Per-channel energy normalization, normally a robust choice, drops 1.9 pp. Plausibly because our log-mel stats are already z-normalized per-clip, and PCEN's AGC-like dynamics double-normalize the signal, stripping level cues the classifier was using (particularly for `inactive`/`noise`).

**MFCC diverged** at both 20 and 40 coefficients. Most likely cause: cached mean/std normalization stats are hardcoded for log-mel power; reusing them for MFCC produces out-of-distribution inputs at init and the model never recovers. Needs its own stats path to be revisited.

**Takeaway.** Adopt `log_mel_delta2` as the frontend. It stacks with preprocessor and capacity gains (see combined experiment → 0.9855).
