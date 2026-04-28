"""Generate fast-switching speech↔music crit clips at multiple cadences.

For each cadence in CADENCES_MS, alternates SEG-ms chunks of pure-active speech
and music (RMS-matched) for TOTAL_DURATION_MS, then emits manifest entry +
labels.txt + 16 kHz mono wav under recordings/.

Use case:
- DT/GMM/SVM have ~1 s rolling windows → should lag on sub-second switching.
- TCN paper RF = 181 frames ≈ 4.2 s — even slower to switch.
- SmallTCN (smaller RF) should track fastest.
This batch quantifies the per-model switching latency on diagnostic alternation.

Re-runs always strip and regenerate any [<key>] block whose subclass starts
with "switching_". Other entries are untouched.
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
    write_labels_tsv_named,
)

CADENCES_MS = [500, 1000, 2000, 4000]
TOTAL_DURATION_MS = 30_000
SPEECH_IDX = 0  # use s_clean_0 (10 labelled segments)
MUSIC_FILE = SCRIPT_DIR / "sources" / "music" / "pop.mp3"


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) or 1e-9


def _strip_switching_entries(manifest_path: Path) -> int:
    with open(manifest_path, "rb") as f:
        existing = tomllib.load(f)
    stale = [k for k, v in existing.items() if str(v.get("subclass", "")).startswith("switching_")]
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

    The labels carry intra-clip inactive gaps (breaths, between-phrase silence) implicitly:
    any source ms not covered by an active span is inactive.
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
    Source spans labelled "speech" map to "speech"; gaps map to "inactive".
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
            out.append({"label": "inactive", "start": cursor, "end": out_s})
        out.append({"label": "speech", "start": out_s, "end": out_e})
        cursor = out_e
    if cursor < out_end_ms:
        out.append({"label": "inactive", "start": cursor, "end": out_end_ms})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()

    manifest_path = SCRIPT_DIR / "manifest.toml"
    removed = _strip_switching_entries(manifest_path)
    if removed:
        print(f"stripped {removed} stale switching entry(ies) from {manifest_path.name}")

    if not MUSIC_FILE.exists():
        raise SystemExit(f"missing music: {MUSIC_FILE}")

    speech_buf, speech_src_labels = _load_source_speech(SPEECH_IDX)
    music_buf = load_mono(MUSIC_FILE)
    speech_active_rms = _speech_active_rms(speech_buf, speech_src_labels)
    print(f"speech buffer (raw):  {len(speech_buf) / SR:.2f} s  active-rms={speech_active_rms:.4f}")
    print(f"music buffer:         {len(music_buf) / SR:.2f} s")

    # Match music RMS to speech-active RMS (not whole-clip RMS — silences would dilute).
    music_buf = (music_buf * (speech_active_rms / _rms(music_buf))).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(music_buf)))
    if peak > 0.99:
        scale = 0.99 / peak
        speech_buf = (speech_buf * scale).astype(np.float32, copy=False)
        music_buf = (music_buf * scale).astype(np.float32, copy=False)
        print(f"  rescaled both by {scale:.4f} to keep mix peak ≤ 0.99")

    total_samples = int(TOTAL_DURATION_MS * SR / 1000)
    new_entries: list[str] = []

    for cadence_ms in CADENCES_MS:
        seg_samples = int(cadence_ms * SR / 1000)
        out = np.zeros(total_samples, dtype=np.float32)
        labels: list[dict] = []
        speech_pos = 0
        music_pos = 0
        pos = 0
        turn = "speech"
        while pos < total_samples:
            end = min(pos + seg_samples, total_samples)
            n = end - pos
            if turn == "speech":
                if speech_pos + n > len(speech_buf):
                    speech_pos = 0  # wrap
                out[pos:end] = speech_buf[speech_pos : speech_pos + n]
                src_pos_ms = int(round(speech_pos * 1000 / SR))
                turn_dur_ms = int(round(n * 1000 / SR))
                out_start_ms = int(round(pos * 1000 / SR))
                labels.extend(_speech_turn_sublabels(out_start_ms, turn_dur_ms, src_pos_ms, speech_src_labels))
                speech_pos += n
            else:
                if music_pos + n > len(music_buf):
                    music_pos = 0
                out[pos:end] = music_buf[music_pos : music_pos + n]
                music_pos += n
                labels.append({
                    "label": "music",
                    "start": int(round(pos * 1000 / SR)),
                    "end": int(round(end * 1000 / SR)),
                })
            pos = end
            turn = "music" if turn == "speech" else "speech"

        key = f"switching_{cadence_ms}ms"
        out_wav = RECORDINGS_DIR / f"{key}.wav"
        out_labels_path = RECORDINGS_DIR / f"{key}.labels.txt"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), out, SR, subtype="PCM_16")
        write_labels_tsv_named(
            [(lab["start"], lab["end"], lab["label"]) for lab in labels],
            out_labels_path,
        )

        notes = (
            f"Speech↔music alternation at {cadence_ms} ms cadence — diagnostic for "
            f"switching latency. Speech is concatenated active-only spans from "
            f"s_clean_{SPEECH_IDX}.wav; music is RMS-matched {MUSIC_FILE.name}."
        )
        new_entries.append(_format_manifest_entry(key, key, notes, labels))
        from collections import Counter
        cnts = Counter(lab["label"] for lab in labels)
        print(f"  {key:25s}  {dict(cnts)}  {len(labels)} labels  {TOTAL_DURATION_MS/1000:.1f} s")

    with open(manifest_path, "a") as f:
        f.write("\n")
        f.write("\n\n".join(new_entries))
        f.write("\n")
    print(f"\n{len(new_entries)} switching entry(ies) appended to {manifest_path.name}")


if __name__ == "__main__":
    main()
