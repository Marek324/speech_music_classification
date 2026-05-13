# scripts/dataset/crit/crit_mix.py
# Marek Hric

"""Mix a speech clip with a noise/music clip at a target SNR; emit manifest.

Math:
    scale = rms_speech / (rms_noise · 10^(SNR/20))
    so that rms_speech / rms_noise_scaled = 10^(SNR/20).
    Speech RMS is computed over **VAD-active frames only**, so silences in the
    speech track don't dilute the SNR you actually hear.
    Noise is tiled if shorter than speech, trimmed if longer.
    If the resulting peak > 0.99, the **whole mix** is rescaled (preserving SNR)
    to avoid hard-clipping.

Workflow:
    1. First run:
         uv run python scripts/dataset/crit/crit_mix.py \\
             sources/speech/s_clean_0.wav sources/background/cafeteria.mp3 -3 \\
             --out speech_cafeteria_neg3db --subclass low_snr
       Writes:
         recordings/speech_cafeteria_neg3db.wav         (16 kHz mono mix)
         recordings/speech_cafeteria_neg3db.labels.txt  (Audacity TSV — VAD seed)
       Prints the manifest TOML block.

    2. (Optional) Edit labels in Audacity (see crit_label.py docs).

    3. Re-run the same command — re-mixes with corrected active-frame RMS
       (so the actual SNR matches the target more accurately) and prints an
       updated manifest. Pass --force-vad to overwrite labels.txt instead.

Output name defaults to "{speech_stem}_{noise_stem}_{snr}db" (negative SNRs
become "negNdb"); pass --out to override.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from _common import (
    RECORDINGS_DIR,
    SR,
    load_mono,
    print_manifest,
    read_labels_tsv,
    vad_labels,
    write_labels_tsv,
)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) or 1e-9


def _speech_active_rms(x: np.ndarray, labels_ms: list[tuple[int, int]]) -> float:
    if not labels_ms:
        return _rms(x)
    mask = np.zeros(len(x), dtype=bool)
    for s_ms, e_ms in labels_ms:
        s = max(0, int(s_ms * SR / 1000))
        e = min(len(x), int(e_ms * SR / 1000))
        if e > s:
            mask[s:e] = True
    return _rms(x[mask]) if mask.any() else _rms(x)


def _fit_to_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n].copy()
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n].astype(np.float32, copy=False)


def _default_out_name(speech: Path, noise: Path, snr_db: float) -> str:
    snr_int = int(round(snr_db))
    snr_tag = f"neg{abs(snr_int)}db" if snr_int < 0 else f"{snr_int}db"
    return f"{speech.stem}_{noise.stem}_{snr_tag}"


def main() -> None:
    """CLI entry point: mix speech + noise at target SNR, save wav + labels TSV, print manifest snippet."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("speech_path", type=Path, help="path to speech audio file")
    p.add_argument("noise_path", type=Path, help="path to noise/music audio file")
    p.add_argument("snr_db", type=float, help="target SNR in dB (negative ⇒ noise > speech)")
    p.add_argument("--out", default=None, help="output name (default: speech_noise_<snr>db)")
    p.add_argument("--subclass", default="low_snr", help='manifest subclass (default: "low_snr")')
    p.add_argument("--force-vad", action="store_true", help="re-run VAD even if labels.txt exists")
    args = p.parse_args()

    speech_path: Path = args.speech_path
    noise_path: Path = args.noise_path
    if not speech_path.exists():
        raise SystemExit(f"speech file not found: {speech_path}")
    if not noise_path.exists():
        raise SystemExit(f"noise file not found: {noise_path}")
    out_name: str = args.out or _default_out_name(speech_path, noise_path, args.snr_db)

    speech = load_mono(speech_path)
    noise = load_mono(noise_path)
    print(f"speech: {len(speech) / SR:6.2f} s   ({speech_path.name})")
    print(f"noise : {len(noise) / SR:6.2f} s   ({noise_path.name})")

    out_wav = RECORDINGS_DIR / f"{out_name}.wav"
    out_labels = RECORDINGS_DIR / f"{out_name}.labels.txt"
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if out_labels.exists() and not args.force_vad:
        labels = read_labels_tsv(out_labels)
        print(f"reused {len(labels)} label(s) from {out_labels.name} (skip VAD)")
    else:
        labels = vad_labels(speech)
        write_labels_tsv(labels, out_labels)
        print(f"silero VAD → {len(labels)} label(s) → {out_labels}")

    speech_rms = _speech_active_rms(speech, labels)
    noise_fit = _fit_to_length(noise, len(speech))
    noise_rms = _rms(noise_fit)
    scale = speech_rms / (noise_rms * 10 ** (args.snr_db / 20))
    noise_scaled = noise_fit * scale

    mix = speech + noise_scaled
    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        factor = 0.99 / peak
        mix = mix * factor
        speech_play = speech * factor
        noise_play = noise_scaled * factor
        print(f"[peak {peak:.3f} > 0.99 → mix scaled by {factor:.4f}; SNR preserved]")
    else:
        speech_play = speech
        noise_play = noise_scaled

    actual_snr = 20 * np.log10(_speech_active_rms(speech_play, labels) / _rms(noise_play))
    print(
        f"target SNR: {args.snr_db:+.1f} dB    actual: {actual_snr:+.2f} dB    "
        f"final peak: {float(np.max(np.abs(mix))):.3f}"
    )

    sf.write(str(out_wav), mix.astype(np.float32), SR, subtype="PCM_16")
    print(f"wrote → {out_wav}")

    notes = (
        f"Speech ({speech_path.name}) + noise ({noise_path.name}) "
        f"mixed at target {args.snr_db:+.1f} dB SNR (actual {actual_snr:+.2f})."
    )
    print_manifest(out_name, args.subclass, notes, labels)


if __name__ == "__main__":
    main()
