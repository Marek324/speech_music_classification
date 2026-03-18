"""
Synthetic data generator
========================

Reads already-written speech files from data/ and produces two new subclasses:

    speech_over_music
        → clean speech  +  FMA music at MUSIC_RELATIVE_DB below speech RMS
    speech_multispeaker_over_music
        → multispeaker  +  FMA music at MUSIC_RELATIVE_DB below speech RMS

Duration targets (% of speech total = 45% of total_hours):
    speech_over_music               7.5%  →  0.03375 × total_hours
    speech_multispeaker_over_music  5.0%  →  0.02250 × total_hours

Each split (train/val/test) is filled to 80/10/10 of target seconds by cycling
through available source files.

Can be run standalone or called from build.py.
"""

import argparse
import io
import itertools
import random
from pathlib import Path

import jsonlines
import numpy as np
import soundfile as sf
from datasets import Audio, IterableDataset, load_dataset
from pydub import AudioSegment
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAND_SEED         = 381
MAX_FILES_PER_FOLDER = 9000
SR                = 16000
MUSIC_RELATIVE_DB = -10.0
MUSIC_POOL_SIZE   = 200   # FMA clips to pre-fetch (evenly across 6 genres)

FMA_GENRES = ["Electronic", "Folk", "Hip-Hop", "Instrumental", "Pop", "Rock"]
GENRE_MAP  = {"Electronic": 0, "Folk": 2, "Hip-Hop": 3, "Instrumental": 4, "Pop": 6, "Rock": 7}

# Fractions of total_hours
AUG_FRACTIONS: dict[str, dict] = {
    "speech_over_music": {
        "speech_subclass": "speech_clean",
        "fraction": 0.45 * 0.075,   # 45% speech × 7.5% share = 3.375%
    },
    "speech_multispeaker_over_music": {
        "speech_subclass": "speech_multispeaker",
        "fraction": 0.45 * 0.050,   # 45% speech × 5.0% share = 2.25%
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_label(label: str, start: int, end: int) -> dict:
    return {"label": label, "start": start, "end": end}


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2)) + 1e-9)


def mix(speech: np.ndarray, music: np.ndarray) -> np.ndarray:
    target_rms = rms(speech) * (10 ** (MUSIC_RELATIVE_DB / 20))
    if len(music) == 0:
        music_adj = np.zeros(len(speech), dtype=np.float32)
    else:
        repeats = -(-len(speech) // len(music))
        music_adj = np.tile(music, repeats)[: len(speech)]
        music_adj = music_adj * (target_rms / (rms(music_adj)))
    mixed = speech + music_adj
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


def pydub_to_numpy(seg: AudioSegment) -> np.ndarray:
    arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
    arr /= float(2 ** (8 * seg.sample_width - 1))
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# VAD labeler
# ─────────────────────────────────────────────────────────────────────────────


def _ms(n: int) -> int:
    return int(n * 1000 / SR)


class VADLabeler:
    def __init__(self):
        self.model = load_silero_vad()

    def label(self, audio: np.ndarray) -> list[dict]:
        import torch
        tensor = torch.from_numpy(audio).unsqueeze(0)
        timestamps = get_speech_timestamps(tensor, self.model)
        if not timestamps:
            return [make_label("inactive", 0, _ms(len(audio)))]
        labels, prev_end = [], 0
        for ts in timestamps:
            if ts["start"] > prev_end:
                labels.append(make_label("inactive", _ms(prev_end), _ms(ts["start"])))
            labels.append(make_label("speech", _ms(ts["start"]), _ms(ts["end"])))
            prev_end = ts["end"]
        if prev_end < len(audio):
            labels.append(make_label("inactive", _ms(prev_end), _ms(len(audio))))
        return labels


# ─────────────────────────────────────────────────────────────────────────────
# SplitWriter
# ─────────────────────────────────────────────────────────────────────────────


class SplitWriter:
    def __init__(self, split: str, data_dir: Path):
        self.base = data_dir / split
        self.base.mkdir(parents=True, exist_ok=True)
        self._resume()

    def _resume(self):
        existing = sorted(
            [d for d in self.base.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if not existing:
            self.folder_idx = 0
            self.folder = self.base / "000"
            self.folder.mkdir()
            self.file_count = 0
        else:
            self.folder = existing[-1]
            self.folder_idx = int(self.folder.name)
            self.file_count = len(list(self.folder.glob("*.wav")))

    def _rotate_if_full(self):
        if self.file_count >= MAX_FILES_PER_FOLDER:
            self.folder_idx += 1
            self.folder = self.base / f"{self.folder_idx:03d}"
            self.folder.mkdir(exist_ok=True)
            self.file_count = 0

    def write(self, audio: np.ndarray, labels: list[dict], name: str, idx: int, cls: str, subclass: str):
        self._rotate_if_full()
        file_name = f"{name}_{idx:05d}.wav"
        file_path = self.folder / file_name
        if file_path.exists():
            return
        sf.write(file_path, audio, samplerate=SR)
        with jsonlines.open(self.folder / "metadata.jsonl", mode="a") as f:
            f.write({"file_name": file_name, "class": cls, "subclass": subclass, "labels": labels})
        self.file_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Music pool
# ─────────────────────────────────────────────────────────────────────────────


def build_music_pool(pool_size: int) -> list[np.ndarray]:
    print(f"Building music pool ({pool_size} clips, {len(FMA_GENRES)} genres)...")
    per_genre = pool_size // len(FMA_GENRES)
    pool: list[np.ndarray] = []
    for genre in FMA_GENRES:
        genre_val = GENRE_MAP[genre]
        ds = (
            load_dataset("rpmon/fma-genre-classification", split="train", streaming=True)
            .cast_column("audio", Audio(decode=False))
        )
        assert isinstance(ds, IterableDataset)
        ds = ds.shuffle(seed=RAND_SEED).filter(lambda row: row["genre"] == genre_val).take(per_genre)
        for row in tqdm(ds, desc=f"  FMA {genre}", total=per_genre):
            raw = row["audio"]
            try:
                seg = (
                    AudioSegment.from_file(io.BytesIO(raw["bytes"]))
                    if raw["bytes"] is not None
                    else AudioSegment.from_file(raw["path"])
                )
                pool.append(pydub_to_numpy(seg.set_frame_rate(SR).set_channels(1)))
            except Exception as e:
                print(f"  [SKIP] FMA {genre}: {e}")
    random.seed(RAND_SEED)
    random.shuffle(pool)
    print(f"Music pool ready: {len(pool)} clips\n")
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Speech file collector
# ─────────────────────────────────────────────────────────────────────────────


def collect_speech_files(subclass: str, data_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for split in result:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for folder in sorted(split_dir.iterdir()):
            meta_path = folder / "metadata.jsonl"
            if not meta_path.exists():
                continue
            with jsonlines.open(meta_path) as reader:
                for entry in reader:
                    if entry.get("subclass") == subclass:
                        wav_path = folder / entry["file_name"]
                        if wav_path.exists():
                            result[split].append(wav_path)
    for split, paths in result.items():
        print(f"  {subclass}/{split}: {len(paths)} files")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic mixer — duration-based, cycles source files as needed
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic(
    subclass_out: str,
    speech_files: dict[str, list[Path]],
    music_pool: list[np.ndarray],
    target_seconds: float,
    vad: VADLabeler,
    data_dir: Path,
):
    split_targets = {
        "train": target_seconds * 0.8,
        "val":   target_seconds * 0.1,
        "test":  target_seconds * 0.1,
    }
    rng = random.Random(RAND_SEED)

    print(f"\n{'─' * 60}")
    print(f"  Generating {subclass_out}  (target {target_seconds / 3600:.3f}h / {target_seconds / 60:.1f} min)")
    print(f"{'─' * 60}")

    global_idx = 0
    for split, split_target in split_targets.items():
        available = speech_files.get(split, [])
        if not available:
            print(f"  [WARN] No source files in {split}/ — skipping")
            continue
        if not music_pool:
            print(f"  [WARN] Empty music pool — skipping")
            continue

        writer = SplitWriter(split, data_dir)
        accumulated = 0.0
        shuffled = rng.sample(available, len(available))

        for speech_path in itertools.cycle(shuffled):
            speech_arr, _ = sf.read(speech_path, dtype="float32")
            if speech_arr.ndim > 1:
                speech_arr = speech_arr.mean(axis=1)
            mixed = mix(speech_arr, rng.choice(music_pool))
            labels = vad.label(mixed)
            writer.write(mixed, labels, subclass_out, global_idx, "speech", subclass_out)
            accumulated += len(mixed) / SR
            global_idx += 1
            if accumulated >= split_target:
                break

        print(f"  {split}: {accumulated / 60:.1f} min  (target {split_target / 60:.1f} min)")

    print(f"  Done: {subclass_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API (called from build.py)
# ─────────────────────────────────────────────────────────────────────────────


def run_augmentation(total_hours: float, data_dir: Path):
    print("\n" + "=" * 60)
    print(f"Augmentation  ({total_hours:.1f}h total → {data_dir})")
    print("=" * 60)

    print("\nLoading VAD model...")
    vad = VADLabeler()
    pool_size = max(12, min(MUSIC_POOL_SIZE, int(total_hours * 10)))
    music_pool = build_music_pool(pool_size)

    for subclass_out, cfg in AUG_FRACTIONS.items():
        target_s = total_hours * 3600 * cfg["fraction"]
        print(f"\nCollecting {cfg['speech_subclass']} for {subclass_out}...")
        speech_files = collect_speech_files(cfg["speech_subclass"], data_dir)
        has_any = any(paths for paths in speech_files.values())
        if not has_any:
            print(f"  [SKIP] No base speech files found — run build.py first")
            continue
        generate_synthetic(subclass_out, speech_files, music_pool, target_s, vad, data_dir)

    print("\nAugmentation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Speech-over-music augmentation")
    tier_group = parser.add_mutually_exclusive_group()
    tier_group.add_argument("--tier", choices=["mini", "mid", "full"], default="mid")
    tier_group.add_argument("--hours", type=float)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    from build import TIERS, TIER_DIRS
    total_hours = args.hours if args.hours is not None else TIERS[args.tier]
    tier_name   = args.tier if args.hours is None else "custom"
    data_dir    = args.out_dir or TIER_DIRS.get(tier_name, Path("data"))

    run_augmentation(total_hours=total_hours, data_dir=data_dir)


if __name__ == "__main__":
    main()
