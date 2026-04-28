"""Shared helpers for crit_label.py / crit_mix.py."""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SR = 16_000
SCRIPT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = SCRIPT_DIR / "recordings"


def load_mono(path: Path) -> np.ndarray:
    try:
        data, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except sf.LibsndfileError:
        # mp3/m4a/etc — soundfile can't decode, fall back to librosa (audioread/ffmpeg).
        data, src_sr = librosa.load(str(path), sr=None, mono=False)
        data = data.astype(np.float32, copy=False)
        if data.ndim == 2:
            data = data.T  # librosa returns (channels, n) → match soundfile's (n, channels)
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32, copy=False)
    if src_sr != SR:
        data = librosa.resample(data, orig_sr=src_sr, target_sr=SR, res_type="polyphase")
    return np.ascontiguousarray(data, dtype=np.float32)


def vad_labels(speech: np.ndarray) -> list[tuple[int, int]]:
    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(speech), model, sampling_rate=SR, return_seconds=False,
    )
    return [
        (int(round(seg["start"] * 1000 / SR)), int(round(seg["end"] * 1000 / SR)))
        for seg in ts
    ]


def write_labels_tsv(labels: list[tuple[int, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s_ms, e_ms in labels:
            f.write(f"{s_ms / 1000:.3f}\t{e_ms / 1000:.3f}\tspeech\n")


def write_labels_tsv_named(labels: list[tuple[int, int, str]], path: Path) -> None:
    """Like write_labels_tsv, but each tuple carries its own label name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s_ms, e_ms, name in labels:
            f.write(f"{s_ms / 1000:.3f}\t{e_ms / 1000:.3f}\t{name}\n")


def read_labels_tsv(path: Path) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            s_sec, e_sec = float(parts[0]), float(parts[1])
            out.append((int(round(s_sec * 1000)), int(round(e_sec * 1000))))
    return out


def print_manifest(out_name: str, subclass: str, notes: str, labels: list[tuple[int, int]]) -> None:
    print()
    print(f"[{out_name}]")
    print(f'file     = "recordings/{out_name}.wav"')
    print('cls      = "speech"')
    print(f'subclass = "{subclass}"')
    print(f'notes    = "{notes}"')
    print("labels = [")
    for s_ms, e_ms in labels:
        print(f'  {{ label = "speech", start = {s_ms}, end = {e_ms} }},')
    print("]")
