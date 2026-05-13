# scripts/dataset/crit/make_switching_3class.py
# Marek Hric

"""Generate 3-class fast-switching crit clips with content variation.

Same diagnostic angle as make_switching.py (per-cadence switching latency) but
extended to the background class so the eval can show how each model handles
arrival/departure of all three target classes — not just speech↔music.

For each (cadence × variant), cycles speech → music → background turns with
SEG-ms duration. Within each clip, speech turns rotate through SPEECH_INDICES
(6 speakers), music turns through MUSIC_FILENAMES (10 genres), and background
turns through BACKGROUND_FILENAMES (7 ambients). Variant index offsets each
cycle's starting source.

Sources:
  speech   → s_clean_<idx>.wav with hand-curated active-span labels
  music    → MUSIC_FILENAMES, RMS-matched to speech reference
  background → BACKGROUND_FILENAMES, scaled to ~−10 dB below speech (typical noise floor)

Re-runs strip + regenerate only [<key>] blocks whose subclass starts with
"switching_3class_". The 2-class set (managed by make_switching.py) is left
untouched.
"""
from collections import Counter
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
    write_labels_tsv_named,
)

CADENCES_MS = [500, 1000, 2000, 4000]
TOTAL_DURATION_MS = 30_000
N_VARIANTS = 8
BACKGROUND_DB_BELOW_SPEECH = 10.0

SPEECH_INDICES = [0, 1, 2, 3, 4, 5]
MUSIC_FILENAMES = [
    "acapella.mp3", "beatbox.mp3", "electronic.mp3", "folk.mp3",
    "hiphop.mp3", "instrumental.mp3", "jumpup.mp3", "pop.mp3",
    "rock.mp3", "speedcore.mp3",
]
BACKGROUND_FILENAMES = [
    "airport.mp3", "birdsong.mp3", "cafeteria.mp3", "cooler_fan.mp3",
    "fireplace.mp3", "rain.mp3", "street.mp3",
]

CLASSES = ("speech", "music", "background")  # cycle order


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) or 1e-9


def _strip_3class_entries(manifest_path: Path) -> int:
    with open(manifest_path, "rb") as f:
        existing = tomllib.load(f)
    stale = [k for k, v in existing.items() if str(v.get("subclass", "")).startswith("switching_3class_")]
    if not stale:
        return 0
    text = manifest_path.read_text()
    for key in stale:
        pattern = rf"\n?\[{re.escape(key)}\]\n(?:[^\n]*\n)*?(?=\n\[|\Z)"
        text = re.sub(pattern, "", text)
    manifest_path.write_text(text.rstrip() + "\n")
    return len(stale)


def _format_manifest_entry(key: str, subclass: str, notes: str, labels: list[dict]) -> str:
    lines = [
        f"[{key}]",
        f'file     = "recordings/{key}.wav"',
        'cls      = "speech"',
        f'subclass = "{subclass}"',
        f'notes    = "{notes}"',
        "labels = [",
    ]
    for lab in labels:
        lines.append(f'  {{ label = "{lab["label"]}", start = {lab["start"]}, end = {lab["end"]} }},')
    lines.append("]")
    return "\n".join(lines)


def _load_source_speech(speech_idx: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    speech_path = SCRIPT_DIR / "sources" / "speech" / f"s_clean_{speech_idx}.wav"
    labels_path = RECORDINGS_DIR / f"s_clean_{speech_idx}.labels.txt"
    audio = load_mono(speech_path)
    labels = sorted(read_labels_tsv(labels_path))
    return audio, labels


def _speech_active_rms(audio: np.ndarray, labels_ms: list[tuple[int, int]]) -> float:
    if not labels_ms:
        return _rms(audio)
    mask = np.zeros(len(audio), dtype=bool)
    for s_ms, e_ms in labels_ms:
        s = max(0, int(s_ms * SR / 1000))
        e = min(len(audio), int(e_ms * SR / 1000))
        if e > s:
            mask[s:e] = True
    return _rms(audio[mask]) if mask.any() else _rms(audio)


def _speech_turn_sublabels(
    out_start_ms: int,
    duration_ms: int,
    src_pos_ms: int,
    src_labels_ms: list[tuple[int, int]],
) -> list[dict]:
    out: list[dict] = []
    src_end_ms = src_pos_ms + duration_ms
    out_end_ms = out_start_ms + duration_ms
    cursor = out_start_ms
    for s, e in src_labels_ms:
        if e <= src_pos_ms or s >= src_end_ms:
            continue
        overlap_s = max(s, src_pos_ms)
        overlap_e = min(e, src_end_ms)
        out_s = out_start_ms + (overlap_s - src_pos_ms)
        out_e = out_start_ms + (overlap_e - src_pos_ms)
        if out_s > cursor:
            out.append({"label": "background", "start": cursor, "end": out_s})
        out.append({"label": "speech", "start": out_s, "end": out_e})
        cursor = out_e
    if cursor < out_end_ms:
        out.append({"label": "background", "start": cursor, "end": out_end_ms})
    return out


def main() -> None:
    """CLI entry point: regenerate 3-class switching wavs and append manifest entries."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()

    manifest_path = SCRIPT_DIR / "manifest.toml"
    removed = _strip_3class_entries(manifest_path)
    if removed:
        print(f"stripped {removed} stale 3-class switching entry(ies) from {manifest_path.name}")

    # Load all sources.
    speech_data: dict[int, tuple[np.ndarray, list[tuple[int, int]]]] = {}
    for idx in SPEECH_INDICES:
        audio, labs = _load_source_speech(idx)
        speech_data[idx] = (audio, labs)
        print(f"speech s_clean_{idx}: {len(audio) / SR:.2f} s")
    speech_target_rms = float(np.mean([_speech_active_rms(a, l) for a, l in speech_data.values()]))
    print(f"speech reference rms (mean across speakers): {speech_target_rms:.4f}")

    music_data: dict[str, np.ndarray] = {}
    for fname in MUSIC_FILENAMES:
        path = SCRIPT_DIR / "sources" / "music" / fname
        if not path.exists():
            raise SystemExit(f"missing music: {path}")
        audio = load_mono(path)
        audio = (audio * (speech_target_rms / _rms(audio))).astype(np.float32, copy=False)
        music_data[fname] = audio
        print(f"music {fname}: {len(audio) / SR:.2f} s")

    background_target_rms = speech_target_rms * 10 ** (-BACKGROUND_DB_BELOW_SPEECH / 20)
    background_data: dict[str, np.ndarray] = {}
    for fname in BACKGROUND_FILENAMES:
        path = SCRIPT_DIR / "sources" / "background" / fname
        if not path.exists():
            raise SystemExit(f"missing background: {path}")
        audio = load_mono(path)
        audio = (audio * (background_target_rms / _rms(audio))).astype(np.float32, copy=False)
        background_data[fname] = audio
        print(f"background {fname}: {len(audio) / SR:.2f} s")

    print(f"background scaled to {BACKGROUND_DB_BELOW_SPEECH:.0f} dB below speech reference")

    peak = max(
        max(float(np.max(np.abs(a))) for a, _ in speech_data.values()),
        max(float(np.max(np.abs(a))) for a in music_data.values()),
        max(float(np.max(np.abs(a))) for a in background_data.values()),
    )
    if peak > 0.99:
        scale = 0.99 / peak
        for idx in SPEECH_INDICES:
            audio, labs = speech_data[idx]
            speech_data[idx] = (audio * scale, labs)
        for fname in MUSIC_FILENAMES:
            music_data[fname] = music_data[fname] * scale
        for fname in BACKGROUND_FILENAMES:
            background_data[fname] = background_data[fname] * scale
        print(f"  rescaled all sources by {scale:.4f} to keep peak ≤ 0.99")

    total_samples = int(TOTAL_DURATION_MS * SR / 1000)
    new_entries: list[str] = []

    for cadence_ms in CADENCES_MS:
        seg_samples = int(cadence_ms * SR / 1000)
        for variant in range(N_VARIANTS):
            # Relatively prime strides → variant gets a different (s, m, i) starting triple.
            speech_start = variant % len(SPEECH_INDICES)
            music_start = (variant * 3) % len(MUSIC_FILENAMES)
            background_start = (variant * 5) % len(BACKGROUND_FILENAMES)

            out = np.zeros(total_samples, dtype=np.float32)
            labels: list[dict] = []
            speech_positions = {idx: 0 for idx in SPEECH_INDICES}
            music_positions = {fname: 0 for fname in MUSIC_FILENAMES}
            background_positions = {fname: 0 for fname in BACKGROUND_FILENAMES}
            per_class_turn = {"speech": 0, "music": 0, "background": 0}

            pos = 0
            idx_cycle = 0
            while pos < total_samples:
                end = min(pos + seg_samples, total_samples)
                n = end - pos
                cls = CLASSES[idx_cycle % len(CLASSES)]
                t = per_class_turn[cls]
                if cls == "speech":
                    spk = SPEECH_INDICES[(speech_start + t) % len(SPEECH_INDICES)]
                    audio, src_labels = speech_data[spk]
                    spos = speech_positions[spk]
                    if spos + n > len(audio):
                        spos = 0
                    out[pos:end] = audio[spos : spos + n]
                    src_pos_ms = int(round(spos * 1000 / SR))
                    turn_dur_ms = int(round(n * 1000 / SR))
                    out_start_ms = int(round(pos * 1000 / SR))
                    labels.extend(_speech_turn_sublabels(out_start_ms, turn_dur_ms, src_pos_ms, src_labels))
                    speech_positions[spk] = spos + n
                elif cls == "music":
                    fname = MUSIC_FILENAMES[(music_start + t) % len(MUSIC_FILENAMES)]
                    audio = music_data[fname]
                    mpos = music_positions[fname]
                    if mpos + n > len(audio):
                        mpos = 0
                    out[pos:end] = audio[mpos : mpos + n]
                    music_positions[fname] = mpos + n
                    labels.append({
                        "label": "music",
                        "start": int(round(pos * 1000 / SR)),
                        "end": int(round(end * 1000 / SR)),
                    })
                else:
                    fname = BACKGROUND_FILENAMES[(background_start + t) % len(BACKGROUND_FILENAMES)]
                    audio = background_data[fname]
                    ipos = background_positions[fname]
                    if ipos + n > len(audio):
                        ipos = 0
                    out[pos:end] = audio[ipos : ipos + n]
                    background_positions[fname] = ipos + n
                    labels.append({
                        "label": "background",
                        "start": int(round(pos * 1000 / SR)),
                        "end": int(round(end * 1000 / SR)),
                    })
                per_class_turn[cls] = t + 1
                pos = end
                idx_cycle += 1

            key = f"switching_3class_{cadence_ms}ms_v{variant}"
            out_wav = RECORDINGS_DIR / f"{key}.wav"
            out_labels_path = RECORDINGS_DIR / f"{key}.labels.txt"
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_wav), out, SR, subtype="PCM_16")
            write_labels_tsv_named(
                [(lab["start"], lab["end"], lab["label"]) for lab in labels],
                out_labels_path,
            )
            notes = (
                f"Speech / music / background cycle at {cadence_ms} ms cadence "
                f"(variant {variant}/{N_VARIANTS - 1}). Speech rotates through "
                f"{len(SPEECH_INDICES)} speakers, music through {len(MUSIC_FILENAMES)} genres, "
                f"background through {len(BACKGROUND_FILENAMES)} ambients; variant offsets start indices. "
                f"Background is {BACKGROUND_DB_BELOW_SPEECH:.0f} dB below speech."
            )
            new_entries.append(_format_manifest_entry(key, key, notes, labels))
            cnts = Counter(lab["label"] for lab in labels)
            print(f"  {key:36s}  {dict(cnts)}  {len(labels)} segments  {TOTAL_DURATION_MS/1000:.1f} s")

    with open(manifest_path, "a") as f:
        f.write("\n")
        f.write("\n\n".join(new_entries))
        f.write("\n")
    print(f"\n{len(new_entries)} 3-class switching entry(ies) appended to {manifest_path.name}")


if __name__ == "__main__":
    main()
