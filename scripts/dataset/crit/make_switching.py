# scripts/dataset/crit/make_switching.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

"""Generate fast-switching speech↔music crit clips with content variation.

For each (cadence × variant) pair, alternates SEG-ms chunks of pure-active
speech and music (RMS-matched) for TOTAL_DURATION_MS. Within each clip, speech
turns cycle through SPEECH_INDICES (6 hand-labeled speakers) and music turns
cycle through MUSIC_FILENAMES (10 genres); the variant index shifts each cycle's
starting source so different variants emphasise different (speaker, genre)
pairings at each cadence boundary.

Use case:
- Diagnostic latency for ground-truth class switches.
- 8 variants × 4 cadences = 32 clips → enough transitions per (cadence × from→to)
  for stable median + p95 even at the slow end.

Re-runs strip + regenerate manifest blocks whose subclass starts with
"switching_" and is NOT "switching_3class_*". 3-class entries (managed by
make_switching_3class.py) and unrelated keys are untouched.
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

SPEECH_INDICES = [0, 1, 2, 3, 4, 5]
MUSIC_FILENAMES = [
    "acapella.mp3", "beatbox.mp3", "electronic.mp3", "folk.mp3",
    "hiphop.mp3", "instrumental.mp3", "jumpup.mp3", "pop.mp3",
    "rock.mp3", "speedcore.mp3",
]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) or 1e-9


def _is_two_class_switching(subclass: str) -> bool:
    return subclass.startswith("switching_") and not subclass.startswith("switching_3class_")


def _strip_switching_entries(manifest_path: Path) -> int:
    with open(manifest_path, "rb") as f:
        existing = tomllib.load(f)
    stale = [
        k for k, v in existing.items()
        if _is_two_class_switching(str(v.get("subclass", "")))
    ]
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
    """Load raw s_clean_<idx>.wav + its hand-curated active-speech label spans (in ms).

    The labels carry intra-clip background gaps (breaths, between-phrase silence) implicitly:
    any source ms not covered by an active span is background.
    """
    speech_path = SCRIPT_DIR / "sources" / "speech" / f"s_clean_{speech_idx}.wav"
    labels_path = RECORDINGS_DIR / f"s_clean_{speech_idx}.labels.txt"
    audio = load_mono(speech_path)
    labels = sorted(read_labels_tsv(labels_path))
    return audio, labels


def _speech_active_rms(audio: np.ndarray, labels_ms: list[tuple[int, int]]) -> float:
    """RMS over the labelled active spans only — silences in source don't dilute it."""
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
    """Generate sub-label spans inside one speech turn.

    Audio for the turn is a slice of source clip [src_pos_ms, src_pos_ms+duration_ms].
    Source spans labelled "speech" map to "speech"; gaps map to "background".
    """
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
    """CLI entry point: regenerate switching cadence × variant wavs and append manifest entries."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()

    manifest_path = SCRIPT_DIR / "manifest.toml"
    removed = _strip_switching_entries(manifest_path)
    if removed:
        print(f"stripped {removed} stale 2-class switching entry(ies) from {manifest_path.name}")

    speech_data: dict[int, tuple[np.ndarray, list[tuple[int, int]]]] = {}
    for idx in SPEECH_INDICES:
        audio, labs = _load_source_speech(idx)
        speech_data[idx] = (audio, labs)
        print(f"speech s_clean_{idx}: {len(audio) / SR:.2f} s  active-rms={_speech_active_rms(audio, labs):.4f}")
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
        print(f"music {fname}: {len(audio) / SR:.2f} s  rms_after_match={_rms(audio):.4f}")

    peak = max(
        max(float(np.max(np.abs(a))) for a, _ in speech_data.values()),
        max(float(np.max(np.abs(a))) for a in music_data.values()),
    )
    if peak > 0.99:
        scale = 0.99 / peak
        for idx in SPEECH_INDICES:
            audio, labs = speech_data[idx]
            speech_data[idx] = (audio * scale, labs)
        for fname in MUSIC_FILENAMES:
            music_data[fname] = music_data[fname] * scale
        print(f"  rescaled all sources by {scale:.4f} to keep peak ≤ 0.99")

    total_samples = int(TOTAL_DURATION_MS * SR / 1000)
    new_entries: list[str] = []

    for cadence_ms in CADENCES_MS:
        seg_samples = int(cadence_ms * SR / 1000)
        for variant in range(N_VARIANTS):
            speech_start = variant % len(SPEECH_INDICES)
            music_start = (variant * 3) % len(MUSIC_FILENAMES)

            out = np.zeros(total_samples, dtype=np.float32)
            labels: list[dict] = []
            speech_positions = {idx: 0 for idx in SPEECH_INDICES}
            music_positions = {fname: 0 for fname in MUSIC_FILENAMES}
            speech_turn_count = 0
            music_turn_count = 0

            pos = 0
            turn = "speech"
            while pos < total_samples:
                end = min(pos + seg_samples, total_samples)
                n = end - pos
                if turn == "speech":
                    idx = SPEECH_INDICES[(speech_start + speech_turn_count) % len(SPEECH_INDICES)]
                    audio, src_labels = speech_data[idx]
                    spos = speech_positions[idx]
                    if spos + n > len(audio):
                        spos = 0
                    out[pos:end] = audio[spos : spos + n]
                    src_pos_ms = int(round(spos * 1000 / SR))
                    turn_dur_ms = int(round(n * 1000 / SR))
                    out_start_ms = int(round(pos * 1000 / SR))
                    labels.extend(_speech_turn_sublabels(out_start_ms, turn_dur_ms, src_pos_ms, src_labels))
                    speech_positions[idx] = spos + n
                    speech_turn_count += 1
                else:
                    fname = MUSIC_FILENAMES[(music_start + music_turn_count) % len(MUSIC_FILENAMES)]
                    audio = music_data[fname]
                    mpos = music_positions[fname]
                    if mpos + n > len(audio):
                        mpos = 0
                    out[pos:end] = audio[mpos : mpos + n]
                    music_positions[fname] = mpos + n
                    music_turn_count += 1
                    labels.append({
                        "label": "music",
                        "start": int(round(pos * 1000 / SR)),
                        "end": int(round(end * 1000 / SR)),
                    })
                pos = end
                turn = "music" if turn == "speech" else "speech"

            key = f"switching_{cadence_ms}ms_v{variant}"
            out_wav = RECORDINGS_DIR / f"{key}.wav"
            out_labels_path = RECORDINGS_DIR / f"{key}.labels.txt"
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_wav), out, SR, subtype="PCM_16")
            write_labels_tsv_named(
                [(lab["start"], lab["end"], lab["label"]) for lab in labels],
                out_labels_path,
            )
            notes = (
                f"Speech↔music alternation at {cadence_ms} ms cadence (variant {variant}/"
                f"{N_VARIANTS - 1}). Speech rotates through {len(SPEECH_INDICES)} speakers, "
                f"music through {len(MUSIC_FILENAMES)} genres; variant offsets start indices."
            )
            new_entries.append(_format_manifest_entry(key, key, notes, labels))
            cnts = Counter(lab["label"] for lab in labels)
            print(f"  {key:30s}  {dict(cnts)}  {len(labels)} labels  {TOTAL_DURATION_MS/1000:.1f} s")

    with open(manifest_path, "a") as f:
        f.write("\n")
        f.write("\n\n".join(new_entries))
        f.write("\n")
    print(f"\n{len(new_entries)} switching entry(ies) appended to {manifest_path.name}")


if __name__ == "__main__":
    main()
