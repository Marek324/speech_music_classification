"""
Synthetic data generator
========================

Reads already-written clean speech and multispeaker files from data/ and
produces two new synthetic subclasses:

    speech_over_music
        → clean speech  +  FMA music track  at MUSIC_RELATIVE_DB below speech RMS

    speech_multispeaker_over_music
        → multispeaker  +  FMA music track  at MUSIC_RELATIVE_DB below speech RMS

Target row counts (matching build.py targets, 80/10/10):
    speech_over_music               194 total  (155 train / 20 val / 19 test)
    speech_multispeaker_over_music  260 total  (208 train / 26 val / 26 test)

Music source: rpmon/fma-genre-classification (30s MP3 clips, 6 genres)
Music is looped or trimmed to match the speech clip length.
pydub is used for all music decoding to avoid torchcodec failing on MP3s.
"""

import io
import random
from pathlib import Path

import jsonlines
import numpy as np
import soundfile as sf
from datasets import IterableDataset, load_dataset, Audio
from pydub import AudioSegment
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAND_SEED = 381
DATA_DIR = Path("data")
MAX_FILES_PER_FOLDER = 9000
SR = 16000

# How many dB quieter the music is relative to the speech RMS.
# -10 dB = clearly audible background; -15 dB = subtle background.
MUSIC_RELATIVE_DB = -10.0


def _build_config(total_hours: int) -> tuple[int, dict[str, dict]]:
    """Return (music_pool_size, targets) scaled to total_hours."""
    pool_size = max(200, total_hours * 5)
    scale = total_hours / 20
    targets = {
        "speech_over_music": {
            "speech_subclass": "speech_clean",
            "total": int(194 * scale),
        },
        "speech_multispeaker_over_music": {
            "speech_subclass": "speech_multispeaker",
            "total": int(260 * scale),
        },
    }
    return pool_size, targets

FMA_GENRES = ["Electronic", "Folk", "Hip-Hop", "Instrumental", "Pop", "Rock"]
genre_map = {"Electronic": 0, "Folk": 2, "Hip-Hop": 3, "Instrumental": 4, "Pop": 6, "Rock": 7}



# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_label(label: str, start: int, end: int) -> dict:
    return {"label": label, "start": start, "end": end}


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2)) + 1e-9)


def adjust_to_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    return audio * (target_rms / rms(audio))


def loop_or_trim(music: np.ndarray, target_len: int) -> np.ndarray:
    """Repeat music until it covers target_len samples, then trim."""
    if len(music) == 0:
        return np.zeros(target_len, dtype=np.float32)
    repeats = -(-target_len // len(music))  # ceiling division
    return np.tile(music, repeats)[:target_len]


def mix(speech: np.ndarray, music: np.ndarray, music_relative_db: float) -> np.ndarray:
    """
    Mix speech and music.
    Music is scaled so its RMS is music_relative_db below speech RMS,
    looped/trimmed to match speech length, then peak-normalised to [-1, 1].
    """
    music_target_rms = rms(speech) * (10 ** (music_relative_db / 20))
    music_adjusted = adjust_to_rms(loop_or_trim(music, len(speech)), music_target_rms)
    mixed = speech + music_adjusted
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


def pydub_to_numpy(seg: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to a float32 numpy array normalised to [-1, 1]."""
    arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
    arr /= float(2 ** (8 * seg.sample_width - 1))
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# VAD labeler
# ─────────────────────────────────────────────────────────────────────────────


def ms(samples: int) -> int:
    return int(samples * 1000 / SR)


class VADLabeler:
    def __init__(self):
        self.model = load_silero_vad()

    def label(self, audio: np.ndarray) -> list[dict]:
        import torch

        tensor = torch.from_numpy(audio).unsqueeze(0)
        timestamps = get_speech_timestamps(tensor, self.model)

        if not timestamps:
            return [make_label("inactive", 0, ms(len(audio)))]

        labels = []
        prev_end = 0
        for ts in timestamps:
            if ts["start"] > prev_end:
                labels.append(make_label("inactive", ms(prev_end), ms(ts["start"])))
            labels.append(make_label("speech", ms(ts["start"]), ms(ts["end"])))
            prev_end = ts["end"]

        if prev_end < len(audio):
            labels.append(make_label("inactive", ms(prev_end), ms(len(audio))))

        return labels


# ─────────────────────────────────────────────────────────────────────────────
# SplitWriter (identical pattern to build.py)
# ─────────────────────────────────────────────────────────────────────────────


class SplitWriter:
    """Writes .wav + metadata.jsonl, rotates subfolders, resumes on restart."""

    def __init__(self, split: str):
        self.base = DATA_DIR / split
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

    def write(
        self,
        audio: np.ndarray,
        labels: list[dict],
        name: str,
        idx: int,
        cls: str,
        subclass: str,
    ):
        self._rotate_if_full()

        file_name = f"{name}_{idx:05d}.wav"
        file_path = self.folder / file_name

        if file_path.exists():
            return

        sf.write(file_path, audio, samplerate=SR)

        with jsonlines.open(self.folder / "metadata.jsonl", mode="a") as f:
            f.write(
                {
                    "file_name": file_name,
                    "class": cls,
                    "subclass": subclass,
                    "labels": labels,
                }
            )

        self.file_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Music pool — pre-fetch FMA clips into memory using pydub (no torchcodec)
# ─────────────────────────────────────────────────────────────────────────────


def build_music_pool(pool_size: int) -> list[np.ndarray]:
    """
    Stream FMA clips round-robin across all 6 genres.
    Audio is decoded with pydub directly from raw bytes — no cast_column,
    no torchcodec — to reliably handle FMA's MP3 files.
    Returns a shuffled list of float32 numpy arrays.
    """
    print(f"Building music pool ({pool_size} clips, {len(FMA_GENRES)} genres)...")
    per_genre = pool_size // len(FMA_GENRES)
    pool: list[np.ndarray] = []

    for genre in FMA_GENRES:
        ds = load_dataset(
            "rpmon/fma-genre-classification",
            split="train",
            streaming=True,
        ).cast_column('audio', Audio(decode=False))
        assert isinstance(ds, IterableDataset)

        genre_capture = genre_map[genre]
        ds = (
            ds.shuffle(seed=RAND_SEED)
            .filter(lambda row: row["genre"] == genre_capture)
            .take(per_genre)
            # deliberately NO cast_column — keep raw {"bytes": ..., "path": ...}
            # so the HF audio decoder (torchcodec) is never invoked
        )

        for row in tqdm(ds, desc=f"  FMA {genre}", total=per_genre):
            raw = row[
                "audio"
            ]  # raw struct: {"bytes": bytes | None, "path": str | None}
            try:
                seg = (
                    AudioSegment.from_file(io.BytesIO(raw["bytes"]))
                    if raw["bytes"] is not None
                    else AudioSegment.from_file(raw["path"])
                )
            except Exception as e:
                print(f"  [SKIP] FMA {genre} — {e}")
                continue

            seg = seg.set_frame_rate(SR).set_channels(1)
            pool.append(pydub_to_numpy(seg))

    random.seed(RAND_SEED)
    random.shuffle(pool)
    print(f"Music pool ready: {len(pool)} clips\n")
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Speech source collector — scan already-written data/ files
# ─────────────────────────────────────────────────────────────────────────────


def collect_speech_files(subclass: str) -> dict[str, list[Path]]:
    """
    Walk data/{train,val,test}/ and collect .wav paths whose metadata entry
    matches the requested subclass.
    """
    result: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for split in result:
        split_dir = DATA_DIR / split
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
        print(f"  {subclass} / {split}: {len(paths)} files")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic mixer
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic(
    subclass_out: str,
    speech_files: dict[str, list[Path]],
    music_pool: list[np.ndarray],
    total_rows: int,
    vad: VADLabeler,
):
    train_n = int(0.8 * total_rows)
    val_n = int(0.9 * total_rows) - train_n
    test_n = total_rows - train_n - val_n

    split_counts = {"train": train_n, "val": val_n, "test": test_n}
    rng = random.Random(RAND_SEED)

    print(f"\n{'─' * 60}")
    print(f"  Generating {subclass_out}  (total {total_rows})")
    print(f"  train {train_n} / val {val_n} / test {test_n}")
    print(f"{'─' * 60}")

    for split, n_needed in split_counts.items():
        available = speech_files.get(split, [])

        if not available:
            print(f"  [WARN] No source files in {split}/ — skipping")
            continue

        if n_needed > len(available):
            print(
                f"  [WARN] Need {n_needed} but only {len(available)} available — sampling with replacement"
            )
            chosen = rng.choices(available, k=n_needed)
        else:
            chosen = rng.sample(available, k=n_needed)

        writer = SplitWriter(split)
        local_idx = 0

        for speech_path in tqdm(chosen, desc=f"    {split}"):
            speech_arr, _ = sf.read(speech_path, dtype="float32")
            if speech_arr.ndim > 1:
                speech_arr = speech_arr.mean(axis=1)

            music_arr = rng.choice(music_pool)
            mixed = mix(speech_arr, music_arr, MUSIC_RELATIVE_DB)
            labels = vad.label(mixed)

            writer.write(mixed, labels, subclass_out, local_idx, "speech", subclass_out)
            local_idx += 1

    print(f"  Done: {subclass_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_augmentation(total_hours: int):
    """Entry point — called from build.py or standalone."""
    pool_size, targets = _build_config(total_hours)

    print("=" * 60)
    print(f"Synthetic data generation  ({total_hours}h scale)")
    print(f"Source : {DATA_DIR.resolve()}")
    print(f"Music  : rpmon/fma-genre-classification  ({', '.join(FMA_GENRES)})")
    print(f"Mix    : music at {MUSIC_RELATIVE_DB} dB relative to speech RMS")
    print("=" * 60)

    print("\nLoading VAD model...")
    vad = VADLabeler()

    music_pool = build_music_pool(pool_size)

    for subclass_out, cfg in targets.items():
        print(
            f"\nCollecting source files for {subclass_out}  (from {cfg['speech_subclass']})..."
        )
        speech_files = collect_speech_files(cfg["speech_subclass"])
        generate_synthetic(
            subclass_out=subclass_out,
            speech_files=speech_files,
            music_pool=music_pool,
            total_rows=cfg["total"],
            vad=vad,
        )

    print("\n" + "=" * 60)
    print("Synthetic generation complete.")
    print("=" * 60)


def main():
    import argparse

    from build import SIZE_PRESETS

    parser = argparse.ArgumentParser(description="Generate synthetic speech-over-music data")
    parser.add_argument("size", choices=SIZE_PRESETS, help="mini (1h), mid (20h), or full (200h)")
    args = parser.parse_args()
    run_augmentation(SIZE_PRESETS[args.size])


if __name__ == "__main__":
    main()
