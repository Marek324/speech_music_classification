"""Generate 3-class fast-switching crit clips: speech → music → inactive cycle.

Same diagnostic angle as make_switching.py (per-cadence switching latency) but
extended to the inactive class so the eval can show how each model handles
arrival/departure of all three target classes — not just speech↔music.

Sources:
  speech   → concatenated active spans of s_clean_<SPEECH_IDX>.wav
  music    → MUSIC_FILE, RMS-matched to speech
  inactive → INACTIVE_FILE (rain — broadband ambient, closest to DEMAND timbre),
             scaled to ~−10 dB below speech to mimic typical noise-floor level.

Re-runs always strip + regenerate any [<key>] block whose subclass starts with
"switching_3class_". Other entries (incl. the 2-class switching set) untouched.
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
SPEECH_IDX = 0
MUSIC_FILE = SCRIPT_DIR / "sources" / "music" / "pop.mp3"
INACTIVE_FILE = SCRIPT_DIR / "sources" / "inactive" / "rain.mp3"
INACTIVE_DB_BELOW_SPEECH = 10.0

CLASSES = ("speech", "music", "inactive")  # cycle order


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
    removed = _strip_3class_entries(manifest_path)
    if removed:
        print(f"stripped {removed} stale 3-class switching entry(ies) from {manifest_path.name}")

    for f in (MUSIC_FILE, INACTIVE_FILE):
        if not f.exists():
            raise SystemExit(f"missing source: {f}")

    speech_buf, speech_src_labels = _load_source_speech(SPEECH_IDX)
    music_buf = load_mono(MUSIC_FILE)
    inactive_buf = load_mono(INACTIVE_FILE)
    speech_rms = _speech_active_rms(speech_buf, speech_src_labels)
    music_buf = (music_buf * (speech_rms / _rms(music_buf))).astype(np.float32, copy=False)
    inactive_target_rms = speech_rms * 10 ** (-INACTIVE_DB_BELOW_SPEECH / 20)
    inactive_buf = (inactive_buf * (inactive_target_rms / _rms(inactive_buf))).astype(np.float32, copy=False)
    print(f"speech buffer (raw):  {len(speech_buf) / SR:.2f} s  active-rms={speech_rms:.4f}")
    print(f"music buffer:         {len(music_buf) / SR:.2f} s  rms={_rms(music_buf):.4f}")
    print(f"inactive buffer:      {len(inactive_buf) / SR:.2f} s  rms={_rms(inactive_buf):.4f} ({INACTIVE_DB_BELOW_SPEECH:.0f} dB below speech)")

    peak = max(float(np.max(np.abs(speech_buf))), float(np.max(np.abs(music_buf))))
    if peak > 0.99:
        scale = 0.99 / peak
        speech_buf = (speech_buf * scale).astype(np.float32, copy=False)
        music_buf = (music_buf * scale).astype(np.float32, copy=False)
        inactive_buf = (inactive_buf * scale).astype(np.float32, copy=False)
        print(f"  rescaled all by {scale:.4f} to keep peak ≤ 0.99")

    buffers = {"speech": speech_buf, "music": music_buf, "inactive": inactive_buf}
    total_samples = int(TOTAL_DURATION_MS * SR / 1000)
    new_entries: list[str] = []

    for cadence_ms in CADENCES_MS:
        seg_samples = int(cadence_ms * SR / 1000)
        out = np.zeros(total_samples, dtype=np.float32)
        labels: list[dict] = []
        positions = {c: 0 for c in CLASSES}
        pos = 0
        idx = 0
        while pos < total_samples:
            end = min(pos + seg_samples, total_samples)
            n = end - pos
            cls = CLASSES[idx % len(CLASSES)]
            buf = buffers[cls]
            if positions[cls] + n > len(buf):
                positions[cls] = 0
            out[pos:end] = buf[positions[cls] : positions[cls] + n]
            if cls == "speech":
                src_pos_ms = int(round(positions[cls] * 1000 / SR))
                turn_dur_ms = int(round(n * 1000 / SR))
                out_start_ms = int(round(pos * 1000 / SR))
                labels.extend(_speech_turn_sublabels(out_start_ms, turn_dur_ms, src_pos_ms, speech_src_labels))
            else:
                labels.append({
                    "label": cls,
                    "start": int(round(pos * 1000 / SR)),
                    "end": int(round(end * 1000 / SR)),
                })
            positions[cls] += n
            pos = end
            idx += 1

        key = f"switching_3class_{cadence_ms}ms"
        out_wav = RECORDINGS_DIR / f"{key}.wav"
        out_labels_path = RECORDINGS_DIR / f"{key}.labels.txt"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), out, SR, subtype="PCM_16")
        write_labels_tsv_named(
            [(lab["start"], lab["end"], lab["label"]) for lab in labels],
            out_labels_path,
        )
        notes = (
            f"Speech / music / inactive cycle at {cadence_ms} ms cadence — diagnostic for "
            f"3-class switching latency. Inactive is rain.mp3 scaled to {INACTIVE_DB_BELOW_SPEECH:.0f} dB "
            f"below speech (typical noise-floor level)."
        )
        new_entries.append(_format_manifest_entry(key, key, notes, labels))
        from collections import Counter
        cnts = Counter(lab["label"] for lab in labels)
        print(f"  {key:32s}  {dict(cnts)}  {len(labels)} segments  {TOTAL_DURATION_MS/1000:.1f} s")

    with open(manifest_path, "a") as f:
        f.write("\n")
        f.write("\n\n".join(new_entries))
        f.write("\n")
    print(f"\n{len(new_entries)} 3-class switching entry(ies) appended to {manifest_path.name}")


if __name__ == "__main__":
    main()
