# Frontend experiment — results notes

Sweep of fixed (non-learned) spectral front-ends feeding the baseline TCN (SGD, WeightNorm, seq_len=128, 30 epochs, full tier).

## Results

| Variant | Channels | Macro F1 | CI | Δ vs base |
|---|---|---|---|---|
| **log_mel_delta2** | 240 (mel+Δ+ΔΔ) | **0.9816** | [0.9765, 0.9849] | +0.0064 |
| **log_mel_delta** | 160 (mel+Δ) | **0.9804** | [0.9749, 0.9840] | +0.0052 |
| mfcc_40 | 40 | 0.9764 | [0.9703, 0.9804] | +0.0012 |
| mfcc_20 | 20 | 0.9752 | [0.9688, 0.9793] | 0.0000 |
| baseline (log_mel_80) | 80 | 0.9752 | [0.9687, 0.9794] | 0.0 |
| log_mel_128 | 128 | 0.9750 | [0.9681, 0.9793] | −0.0002 |
| log_mel_40 | 40 | 0.9730 | [0.9661, 0.9774] | −0.0022 |
| pcen | 80 | 0.9532 | [0.9435, 0.9600] | −0.0220 |

## Conclusion

**Temporal derivatives help; mel-bin count does not.** Going from 40 → 80 → 128 mel bins produces no measurable change (all within ±0.002 of baseline). But concatenating first-order deltas (+0.0052) or first+second (+0.0064) cleanly clears the baseline CI. The signal is *temporal dynamics*, not *finer frequency resolution* — unsurprising given the receptive-field gap: the TCN sees 181 frames but still benefits from pre-computed short-term derivatives that a single small conv layer apparently can't learn as cleanly.

`delta2` edges `delta` by 0.0012 — within noise. Either is a safe win; `delta2` is chosen downstream for stacking.

**MFCC is essentially flat.** 20 and 40 cepstral coefficients land within 0.0012 of baseline. The DCT decorrelation isn't actively useful (TCN handles correlated channels fine) but isn't harmful either. With 4× fewer channels at `mfcc_20`, this is the cheapest frontend that doesn't regress — useful evidence that the bottleneck is not at the spectral representation.

**PCEN loses.** Per-channel energy normalization, normally a robust choice, drops 2.2 pp. Plausibly because our log-mel stats are already z-normalized per-clip, and PCEN's AGC-like dynamics double-normalize the signal, stripping level cues the classifier was using (particularly for `inactive`/`noise` — F1 falls from 0.96 to 0.93 there).

**Takeaway.** Adopt `log_mel_delta2` as the frontend. It stacks with preprocessor and capacity gains in the combined experiment.
