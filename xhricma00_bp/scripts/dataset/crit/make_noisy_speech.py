# scripts/dataset/crit/make_noisy_speech.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

"""Generate noisy-speech crit clips by mixing every clean speech clip with every background/ noise.

Cartesian product: SPEECH_IDS × NOISES × SNRS. For each combo:
  1. Load both at 16 kHz mono.
  2. Reuse the manually-labelled regions from recordings/s_clean_<idx>.labels.txt.
  3. Mix at target SNR (active-frame RMS, peak-clip protected, SNR preserved on rescale).
  4. Save mix.wav + labels.txt to recordings/<out_name>.{wav,labels.txt}.

Re-runs always regenerate every noisy entry in manifest.toml: any existing
[<key>] block whose subclass starts with "noisy_" is stripped from the file
before fresh entries are appended. Other entries (clean, whisper, music) are untouched.
"""
import argparse
import re
import tomllib
from pathlib import Path

import numpy as np
import soundfile as sf

from _common import (
    RECORDINGS_DIR,
    SCRIPT_DIR,
    SR,
    load_mono,
    read_labels_tsv,
    write_labels_tsv,
)

SPEECH_IDS = [0, 1, 2, 3, 4, 5]
NOISES = ["airport", "birdsong", "cafeteria", "cooler_fan", "fireplace", "rain", "street"]
SNRS = [0.0, -3.0]


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


def _snr_tag(snr_db: float) -> str:
    snr_int = int(round(snr_db))
    return f"neg{abs(snr_int)}db" if snr_int < 0 else f"{snr_int}db"


def _format_manifest_entry(
    key: str, out_name: str, subclass: str, notes: str, labels: list[tuple[int, int]]
) -> str:
    lines = [
        f"[{key}]",
        f'file     = "recordings/{out_name}.wav"',
        'cls      = "speech"',
        f'subclass = "{subclass}"',
        f'notes    = "{notes}"',
        "labels = [",
    ]
    for s_ms, e_ms in labels:
        lines.append(f'  {{ label = "speech", start = {s_ms}, end = {e_ms} }},')
    lines.append("]")
    return "\n".join(lines)


def _strip_noisy_entries(manifest_path: Path) -> int:
    """Remove every [key] block whose subclass starts with "noisy_". Returns count removed."""
    with open(manifest_path, "rb") as f:
        existing = tomllib.load(f)
    stale = [k for k, v in existing.items() if str(v.get("subclass", "")).startswith("noisy_")]
    if not stale:
        return 0
    text = manifest_path.read_text()
    for key in stale:
        pattern = rf"\n?\[{re.escape(key)}\]\n(?:[^\n]*\n)*?(?=\n\[|\Z)"
        text = re.sub(pattern, "", text)
    manifest_path.write_text(text.rstrip() + "\n")
    return len(stale)


def main() -> None:
    """CLI entry point: regenerate every noisy-speech crit clip and append manifest entries."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()

    speech_dir = SCRIPT_DIR / "sources" / "speech"
    noise_dir = SCRIPT_DIR / "sources" / "background"
    manifest_path = SCRIPT_DIR / "manifest.toml"

    removed = _strip_noisy_entries(manifest_path)
    if removed:
        print(f"stripped {removed} stale noisy entry(ies) from {manifest_path.name}")

    speech_cache: dict[int, tuple[np.ndarray, list[tuple[int, int]]]] = {}
    noise_cache: dict[str, np.ndarray] = {}

    new_entries: list[str] = []
    for speech_idx in SPEECH_IDS:
        if speech_idx not in speech_cache:
            speech_path = speech_dir / f"s_clean_{speech_idx}.wav"
            labels_path = RECORDINGS_DIR / f"s_clean_{speech_idx}.labels.txt"
            if not speech_path.exists():
                raise SystemExit(f"missing speech: {speech_path}")
            if not labels_path.exists():
                raise SystemExit(f"missing labels: {labels_path}")
            speech_cache[speech_idx] = (load_mono(speech_path), read_labels_tsv(labels_path))
        speech, labels = speech_cache[speech_idx]

        for noise_stem in NOISES:
            if noise_stem not in noise_cache:
                noise_path = noise_dir / f"{noise_stem}.mp3"
                if not noise_path.exists():
                    raise SystemExit(f"missing noise: {noise_path}")
                noise_cache[noise_stem] = load_mono(noise_path)
            noise = noise_cache[noise_stem]

            for snr_db in SNRS:
                tag = _snr_tag(snr_db)
                out_name = f"s_clean_{speech_idx}_{noise_stem}_{tag}"
                key = f"speech_clean{speech_idx}_{noise_stem}_{tag}"
                out_wav = RECORDINGS_DIR / f"{out_name}.wav"
                out_labels = RECORDINGS_DIR / f"{out_name}.labels.txt"

                speech_rms = _speech_active_rms(speech, labels)
                noise_fit = _fit_to_length(noise, len(speech))
                noise_rms = _rms(noise_fit)
                scale = speech_rms / (noise_rms * 10 ** (snr_db / 20))
                noise_scaled = noise_fit * scale

                mix = speech + noise_scaled
                peak = float(np.max(np.abs(mix)))
                if peak > 0.99:
                    factor = 0.99 / peak
                    mix = mix * factor
                    speech_play = speech * factor
                    noise_play = noise_scaled * factor
                    clip_note = f" (mix scaled by {factor:.4f} to avoid clip)"
                else:
                    speech_play = speech
                    noise_play = noise_scaled
                    clip_note = ""

                actual_snr = 20 * np.log10(_speech_active_rms(speech_play, labels) / _rms(noise_play))

                out_wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_wav), mix.astype(np.float32), SR, subtype="PCM_16")
                write_labels_tsv(labels, out_labels)

                notes = (
                    f"Speech (s_clean_{speech_idx}.wav) + noise ({noise_stem}.mp3) "
                    f"mixed at target {snr_db:+.1f} dB SNR (actual {actual_snr:+.2f}){clip_note}."
                )
                new_entries.append(_format_manifest_entry(key, out_name, f"noisy_{tag}", notes, labels))

    with open(manifest_path, "a") as f:
        f.write("\n")
        f.write("\n\n".join(new_entries))
        f.write("\n")
    print(f"{len(new_entries)} noisy entry(ies) appended to {manifest_path.name}")


if __name__ == "__main__":
    main()
