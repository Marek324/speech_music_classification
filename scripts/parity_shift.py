"""Parity test: 1ms vs 10ms subframe shift for extract_segment.

Loads a handful of test clips, extracts features with both subframe
step sizes, compares the 9-D feature distributions.
"""
import sys
from pathlib import Path

import librosa
import numpy as np
from datasets import Audio, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.classic.feat_extractor import FeatExtractor  # noqa: E402


def extract_at_step(fe: FeatExtractor, segment: np.ndarray, step_ms: float) -> np.ndarray:
    """Clone of FeatExtractor.extract_segment with configurable subframe step."""
    n = len(segment)
    fl = fe.fl
    fh = fe.fh

    # Step 1: per-frame existing features (unchanged hop)
    from librosa import feature as libfeat
    from librosa.util import fix_length

    srp_thresh = fe.cfg["features"]["spectral_rolloff_point"]["threshold"]

    frames_feats = []
    last_fft = None
    pos = 0
    while pos + fl <= n:
        frame = segment[pos : pos + fl]
        padded = fix_length(frame, size=fe.n_fft)

        ste = float(10 * np.log10(np.mean(frame**2) + 1e-10))
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / 2)

        sc = libfeat.spectral_centroid(y=padded, sr=fe.sr, n_fft=fe.n_fft)[0, 0]
        sc = 0.0 if np.isnan(sc) or np.isinf(sc) else float(sc)

        cur_fft = np.fft.fft(padded, n=fe.n_fft)
        flux = 0.0 if last_fft is None else float(np.sum(np.abs(cur_fft - last_fft) ** 2))
        last_fft = cur_fft

        rolloff = float(
            libfeat.spectral_rolloff(
                y=padded, sr=fe.sr, roll_percent=srp_thresh, n_fft=fe.n_fft
            )[0, 0]
        )
        frames_feats.append([ste, zcr, sc, flux, rolloff])
        pos += fh

    frames_arr = np.array(frames_feats)
    var_zcr = float(np.var(frames_arr[:, 1]))
    var_centroid = float(np.var(frames_arr[:, 2]))
    var_flux = float(np.var(frames_arr[:, 3]))
    var_rolloff = float(np.var(frames_arr[:, 4]))
    energies = frames_arr[:, 0]
    thr = float(np.mean(energies) / 3)
    lster = float(np.sum(energies < thr) / len(energies))

    # Step 2: per-subframe speech-specific with configurable step
    step = max(1, int(fe.sr * (step_ms / 1000.0)))
    speech_feats = []
    pos = 0
    while pos + fl <= n:
        subframe = segment[pos : pos + fl]
        speech_feats.append([
            fe._naps_of_zffs(subframe),
            fe._psr_he_lp_residual(subframe),
            fe._log_mel_spectrum_energy(subframe),
        ])
        pos += step

    speech_arr = np.nan_to_num(np.array(speech_feats), nan=0.0, posinf=0.0, neginf=0.0)
    mean_naps = float(np.mean(speech_arr[:, 0]))
    mean_psr = float(np.mean(speech_arr[:, 1]))
    var_mel = float(np.var(speech_arr[:, 2]))

    mod_energy = fe._modulation_spectrum_energy_segment(segment)

    return np.array([
        var_zcr, var_centroid, var_flux, var_rolloff, lster,
        mean_naps, mean_psr, var_mel, mod_energy,
    ])


def main():
    config.init_config(None, model_name="gmm")
    fe = FeatExtractor()
    target_sr = fe.sr

    cfg = config.get_config()
    ds = load_dataset(
        cfg["dataset"]["url"],
        name="mid",
        split="test",
    ).cast_column("audio", Audio(sampling_rate=16000, num_channels=1))

    N_CLIPS = 15
    SEG_PER_CLIP = 2

    steps_ms = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    feats_by_step = {s: [] for s in steps_ms}
    classes = []

    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=N_CLIPS, replace=False)

    for i in indices:
        row = ds[int(i)]
        audio = row["audio"].get_all_samples().data
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        audio_8k = librosa.resample(audio, orig_sr=16000, target_sr=target_sr)

        seg_samples = target_sr
        avail = len(audio_8k) // seg_samples
        if avail == 0:
            continue
        k = min(SEG_PER_CLIP, avail)
        for s in range(k):
            seg = audio_8k[s * seg_samples : (s + 1) * seg_samples]
            for step in steps_ms:
                feats_by_step[step].append(extract_at_step(fe, seg, step_ms=step))
            classes.append(row["class"])

    names = ["var_zcr", "var_centroid", "var_flux", "var_rolloff", "lster",
             "mean_naps", "mean_psr", "var_mel", "mod_energy"]
    F = {s: np.array(v) for s, v in feats_by_step.items()}
    F_ref = F[1.0]
    print(f"n_segments = {len(F_ref)}")
    print(f"class counts: {dict((c, classes.count(c)) for c in set(classes))}")
    print()

    # Focus on the 3 features affected by shift
    interesting = [5, 6, 7]  # naps, psr, mel
    print(f"{'step_ms':>8}", end=" ")
    for j in interesting:
        print(f"{names[j]+'_corr':>16}", end=" ")
    print()
    for step in steps_ms:
        if step == 1.0:
            continue
        print(f"{step:>8.1f}", end=" ")
        for j in interesting:
            a = F_ref[:, j]
            b = F[step][:, j]
            if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                corr = float(np.corrcoef(a, b)[0, 1])
            else:
                corr = 1.0
            print(f"{corr:>16.4f}", end=" ")
        print()

    print()
    print(f"{'step_ms':>8}", end=" ")
    for j in interesting:
        print(f"{names[j]+'_relerr%':>16}", end=" ")
    print()
    for step in steps_ms:
        if step == 1.0:
            continue
        print(f"{step:>8.1f}", end=" ")
        for j in interesting:
            m1 = F_ref[:, j].mean()
            m = F[step][:, j].mean()
            rel = 100 * abs(m1 - m) / max(abs(m1), 1e-10)
            print(f"{rel:>16.2f}", end=" ")
        print()


if __name__ == "__main__":
    main()
