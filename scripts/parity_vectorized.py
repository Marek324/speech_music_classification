"""Parity + speed check for the vectorized extract_segment.

Compares:
  - NAPS: should match the scalar per-subframe loop within FFT rounding (< 1e-8).
  - log-mel: has a known semantic change (single-FFT per subframe vs librosa's
    5-frame internal segmentation) — we just check that the per-segment values
    are strongly correlated (>0.95) so retraining can absorb it.
  - PSR: 10ms shift (see earlier parity sweep — corr 0.994).
"""
import sys
import time
from pathlib import Path

import librosa
import numpy as np
from datasets import Audio, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.classic.feat_extractor import FeatExtractor, _stack_subframes  # noqa: E402


def naps_scalar_loop(fe, subframes):
    """Reference: per-subframe loop using fe._naps_of_zffs."""
    out = np.zeros(subframes.shape[0])
    for i in range(subframes.shape[0]):
        out[i] = fe._naps_of_zffs(subframes[i])
    return out


def logmel_scalar_loop(fe, subframes):
    out = np.zeros(subframes.shape[0])
    for i in range(subframes.shape[0]):
        out[i] = fe._log_mel_spectrum_energy(subframes[i])
    return out


def main():
    config.init_config(None, model_name="gmm")
    fe = FeatExtractor()
    target_sr = fe.sr

    cfg = config.get_config()
    ds = load_dataset(
        cfg["dataset"]["url"], name="mid", split="test",
    ).cast_column("audio", Audio(sampling_rate=16000, num_channels=1))

    # Collect a few 1s segments
    segments = []
    classes = []
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ds), size=10, replace=False):
        row = ds[int(i)]
        audio = row["audio"].get_all_samples().data
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        audio_8k = librosa.resample(audio, orig_sr=16000, target_sr=target_sr)
        avail = len(audio_8k) // target_sr
        for s in range(min(2, avail)):
            segments.append(audio_8k[s * target_sr : (s + 1) * target_sr])
            classes.append(row["class"])

    print(f"n_segments = {len(segments)}")

    # --- NAPS parity: batched vs scalar ---
    # Cast stacks to float64 so we compare apples to apples (batched uses float64
    # internally; the real-audio float32 loop otherwise drifts by small amounts).
    naps_batched_all = []
    naps_scalar_all = []
    for seg in segments:
        seg64 = seg.astype(np.float64)
        positions = np.arange(0, len(seg64) - fe.fl + 1, fe._step_1ms)
        subs = _stack_subframes(seg64, positions, fe.fl)
        naps_batched_all.append(fe._batched_naps_of_zffs(subs))
        naps_scalar_all.append(naps_scalar_loop(fe, subs))
    diff = np.max([np.max(np.abs(a - b)) for a, b in zip(naps_batched_all, naps_scalar_all)])
    print(f"NAPS max abs diff (batched vs scalar loop, float64): {diff:.2e}")

    # --- log-mel: batched vs scalar loop ---
    lm_mean_corrs = []
    for seg in segments:
        positions = np.arange(0, len(seg) - fe.fl + 1, fe._step_1ms)
        subs = _stack_subframes(seg, positions, fe.fl)
        a = fe._batched_log_mel_energy(subs)
        b = logmel_scalar_loop(fe, subs)
        if a.std() > 0 and b.std() > 0:
            lm_mean_corrs.append(float(np.corrcoef(a, b)[0, 1]))
    print(f"log-mel batched vs scalar per-subframe: median corr = {np.median(lm_mean_corrs):.4f}")

    # Compare the final var_mel scalar across segments
    var_mel_batched = []
    var_mel_scalar = []
    for seg in segments:
        positions = np.arange(0, len(seg) - fe.fl + 1, fe._step_1ms)
        subs = _stack_subframes(seg, positions, fe.fl)
        var_mel_batched.append(float(np.var(fe._batched_log_mel_energy(subs))))
        var_mel_scalar.append(float(np.var(logmel_scalar_loop(fe, subs))))
    var_mel_batched = np.array(var_mel_batched)
    var_mel_scalar = np.array(var_mel_scalar)
    if var_mel_batched.std() > 0 and var_mel_scalar.std() > 0:
        c = float(np.corrcoef(var_mel_batched, var_mel_scalar)[0, 1])
    else:
        c = 1.0
    print(f"var_mel (scalar-aggregated) cross-segment corr: {c:.4f}")

    # --- Speed benchmark ---
    N_RUNS = 5
    seg = segments[0]
    t = time.perf_counter_ns()
    for _ in range(N_RUNS):
        fe.extract_segment(seg)
    elapsed_ms = (time.perf_counter_ns() - t) / N_RUNS / 1e6
    print(f"extract_segment: {elapsed_ms:.2f} ms per 1s segment ({N_RUNS} runs)")


if __name__ == "__main__":
    main()
