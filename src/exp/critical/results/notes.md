# Critical-set evaluation — results notes

Evaluates every classifier (DT, GMM, SVM, TCN, TCN+LSTM, SmallTCN, SmallerTCN) against the hand-curated `crit` tier — a deliberately adversarial set of clips drawn from outside the train/val/test splits, plus synthetic switching clips at multiple cadences. Designed to expose failure modes the standard test split hides.

## Methodology

The crit tier is staged by `scripts/dataset/build.py --only-critical-set` and stored under `scripts/dataset/speech_music_dataset/crit`. Each clip is loaded through the canonical dataset path (`src/nn/dataset.py` for NN models, `src/input_handler.py` for classics), so this experiment is a config-+-CLI shim — it does not fork the inference loop.

Per-clip outputs:
- `<model>.eval` — full evaluation report (per-class + per-subclass F1/P/R, confusion matrix).
- `<model>_scores.npz` — frame-level y_true / y_pred / subclass arrays at each model's native hop. Consumed by the transitions experiment for switching-cadence analysis.
- `graphs/clip<NNN>_<subclass>.svg` — per-recording timeline visualization comparing every model's prediction stream to ground truth.

## Results

| Model | Macro F1 | CI | Speech F1 | Music F1 | Background F1 |
|---|---|---|---|---|---|
| **TCN+LSTM** | **0.4861** | [0.3704, 0.5782] | 0.5512 | 0.4409 | 0.4661 |
| GMM | 0.4375 | [0.3436, 0.5181] | 0.5058 | 0.4445 | 0.3622 |
| SmallerTCN | 0.4298 | [0.3187, 0.5202] | 0.5131 | 0.4280 | 0.3470 |
| SmallTCN | 0.4294 | [0.3187, 0.5188] | 0.5161 | 0.4234 | 0.3497 |
| TCN | 0.4282 | [0.3178, 0.5181] | 0.5331 | 0.4280 | 0.3234 |
| DT | 0.3807 | [0.2999, 0.4475] | 0.5310 | 0.4448 | 0.1663 |
| SVM | 0.3763 | [0.2872, 0.4599] | 0.4593 | 0.3842 | 0.2853 |

All models lose 0.5–0.6 absolute macro F1 vs the standard test split (0.83–0.98 there → 0.38–0.49 here). The crit tier is much harder by construction.

## Where models break down

Per-subclass F1 reveals four consistent failure modes across every model:

1. **Babble / machinery / outdoor noise**. `noise_babble`, `noise_machinery`, and `noise_outdoor` all land near zero F1 across every classifier (recall 0.0000–0.0577). These conditions are categorically absent from the train/val/test noise distribution (which is dominated by `noise_ambient` and `noise_wildlife`), and no model has learned to label them as background. GMM is the partial exception — `noise_ambient` recall 0.84, vs 0.39–0.65 for everything else — because GMM's background-vs-speech log-likelihood threshold happens to fire on lower-frequency hum.
2. **Whispered speech**. `speech_whisper` collapses everywhere (TCN 0.21, SmallTCN 0.04, SmallerTCN 0.03, TCN+LSTM 0.00, GMM 0.43, SVM 0.57, DT 0.73). The NN family is worst — whisper has no voicing cues, and the energy envelope alone confuses everything that depends on the dynamic range of normal speech. Classics survive because they fall back on long-term feature averages rather than specific spectral patterns.
3. **Speech with strong noise (≤ 0 dB SNR)**. `speech_noisy_0db` and `speech_noisy_neg3db` recall stays around 0.25–0.40 for every model. The training data uses milder SNR distributions; pushing noise above the speech level fully exposes that gap.
4. **Multi-speaker on music (`speech_msom`)**. Recall 0.05–0.47 across the board. Two speakers + music is rare in training, and no model resolves it well. NN models lead (0.16–0.47) but still leave half the frames misclassified.

By contrast, all models score near-perfectly on **clean music subclasses** (acapella, electronic, instrumental, pop, rock, hip-hop, folk, beatbox, extreme): F1 ≥ 0.99 for the NN family, ≥ 0.88 for classics. Music classification is essentially solved on this set.

## Switching cadences

The crit tier contains 8 switching clips: 2-class (speech↔music) and 3-class (speech↔music↔background) at 500/1000/2000/4000 ms cadences. Subclass F1 captures the static-frame view of these (e.g., `speech_switching_3class_500ms`); the *temporal* view (per-event latency, stable-region flicker) is computed by the transitions experiment from these same `_scores.npz` files — see `src/exp/transitions/results/transitions.md`.

## Conclusion

**TCN+LSTM is the clear winner** under adversarial conditions, +0.05 macro F1 over the next-best model and the only one with `background` F1 above 0.45. The LSTM head's persistent state apparently smooths the decision in noisy / multi-talker frames in a way the pure-conv models cannot match.

**The TCN family is essentially tied** in the middle of the table (0.428–0.430). SmallTCN and SmallerTCN match TCN within 0.002 F1 — the model-size differences that show up on the standard test split do not generalise to crit-tier conditions, where every TCN variant is bottlenecked by the same out-of-distribution failure modes (whisper, ≤0 dB noise, msom).

**Classics (DT, SVM) trail** — DT by its inability to label background at all (F1 0.17), SVM by a uniform 0.04 F1 deficit on every class. GMM is unexpectedly strong on noise specifically (`noise_ambient` recall 0.84) which lifts its macro F1 above the smaller TCNs.

**Takeaway.** The crit tier is the right artefact to inform deployment choices: the standard test split says all NN models are within 0.01 F1 of each other, while crit shows TCN+LSTM is meaningfully more robust (+0.05) and that the entire model family has consistent blind spots (whisper, very-low-SNR speech, babble) that future training data should target.
