# scripts/parity_streaming.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Parity test: streaming _extract_gmm_svm vs extract_segment.

Drives a 1s clip through the streaming path in 15ms hops and compares the
final 9-D output to extract_segment on the same clip. After full warmup
(one window worth of hops) the two should agree to FFT rounding.
"""
import sys
from pathlib import Path

import librosa
import numpy as np
from datasets import Audio, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.classic.feat_extractor import FeatExtractor


def main():
    config.init_config(None, model_name="gmm")
    fe = FeatExtractor()
    target_sr = fe.sr

    cfg = config.get_config()
    ds = load_dataset(
        cfg["dataset"]["url"], name="mid", split="test",
    ).cast_column("audio", Audio(sampling_rate=16000, num_channels=1))

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=5, replace=False)

    names = ["var_zcr", "var_centroid", "var_flux", "var_rolloff", "lster",
             "mean_naps", "mean_psr", "var_mel", "mod_energy"]

    fh = fe.fh
    fl = fe.fl

    for i in idxs:
        row = ds[int(i)]
        audio = row["audio"].get_all_samples().data
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        audio_8k = librosa.resample(audio, orig_sr=16000, target_sr=target_sr)
        if len(audio_8k) < target_sr:
            continue
        seg = audio_8k[:target_sr].astype(np.float64)

        fe.reset()
        ref = fe.extract_segment(seg)

        fe.reset()
        last_out = None
        pos = 0
        while pos + fl <= len(seg):
            frame = seg[pos : pos + fl]
            last_out = fe.extract(frame)
            pos += fh

        diff = np.abs(ref - last_out)
        print(f"clip idx={i}")
        for n, r, s, d in zip(names, ref, last_out, diff):
            print(f"  {n:>14}: ref={r: .4e}  stream={s: .4e}  |Δ|={d:.2e}")
        print(f"  max |Δ| = {diff.max():.2e}")
        print()


if __name__ == "__main__":
    main()
