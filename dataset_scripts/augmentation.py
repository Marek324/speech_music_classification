"""Speech-over-music mixes (see ``source_config.AUGMENT_SOURCES`` per ``TierName``)."""

import io
import itertools
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, IterableDataset, load_dataset
from pydub import AudioSegment
from tqdm import tqdm

from labeling import SR, VADLabeler
from source_config import (
    AUGMENT_OUTPUT_SPLIT_FRACTIONS,
    FMA_GENRE_MAP,
    FMA_GENRES,
    FMA_HF_ID,
    MUSIC_POOL_SIZE,
    MUSIC_RELATIVE_DB,
    RAND_SEED,
    TierName,
    augment_entries,
    seed_all,
)
from split_writer import SplitWriter


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


def build_music_pool(pool_size: int) -> list[np.ndarray]:
    print(f"Building music pool ({pool_size} clips, {len(FMA_GENRES)} genres)...")
    per_genre = max(1, pool_size // len(FMA_GENRES))
    pool: list[np.ndarray] = []
    for genre in FMA_GENRES:
        genre_val = FMA_GENRE_MAP[genre]
        ds = load_dataset(FMA_HF_ID, split="train", streaming=True).cast_column(
            "audio", Audio(decode=False)
        )
        assert isinstance(ds, IterableDataset)
        ds = (
            ds.shuffle(seed=RAND_SEED)
            .filter(lambda row: row["genre"] == genre_val)
            .take(per_genre)
        )
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


def collect_speech_refs(subclass: str, data_dir: Path) -> dict[str, list[tuple[Path, int]]]:
    """Row refs under ``{config_dir}/{split}/speech/`` (same layout as ``SplitWriter``)."""
    result: dict[str, list[tuple[Path, int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split in result:
        root = data_dir / split / "speech"
        if not root.exists():
            continue
        for path in sorted(root.glob("part_*.parquet")):
            t = pq.read_table(path, columns=["subclass"])
            subs = t["subclass"].to_pylist()
            for i, sc in enumerate(subs):
                if sc == subclass:
                    result[split].append((path, i))
    for split, refs in result.items():
        print(f"  {subclass}/{split}: {len(refs)} clips")
    return result


def _read_wav_from_parquet(
    path: Path, row_idx: int, table_cache: dict[Path, pa.Table]
) -> np.ndarray:
    if path not in table_cache:
        table_cache[path] = pq.read_table(path, columns=["audio_wav"])
    t = table_cache[path]
    raw = t["audio_wav"][row_idx].as_py()
    bio = io.BytesIO(raw)
    arr, _ = sf.read(bio, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def generate_synthetic(
    subclass_out: str,
    speech_refs: dict[str, list[tuple[Path, int]]],
    music_pool: list[np.ndarray],
    target_minutes: float,
    vad: VADLabeler,
    data_dir: Path,
):
    split_targets_min = {
        split: target_minutes * frac
        for split, frac in AUGMENT_OUTPUT_SPLIT_FRACTIONS.items()
    }
    rng = random.Random(RAND_SEED)
    table_cache: dict[Path, pa.Table] = {}

    print(f"\n{'─' * 60}")
    print(f"  Generating {subclass_out}  (target {target_minutes:.2f} min audio)")
    print(f"{'─' * 60}")

    global_idx = 0
    for split, split_target_min in split_targets_min.items():
        available = speech_refs.get(split, [])
        if not available:
            print(f"  [WARN] No source clips in {split}/speech/ — skipping")
            continue
        if not music_pool:
            print("  [WARN] Empty music pool — skipping")
            continue

        writer = SplitWriter("speech", split, data_dir)
        accumulated_min = 0.0
        shuffled = rng.sample(available, len(available))

        for path, row_idx in itertools.cycle(shuffled):
            speech_arr = _read_wav_from_parquet(path, row_idx, table_cache)
            mixed = mix(speech_arr, rng.choice(music_pool))
            labels = vad.label(mixed)
            if labels is None:
                continue
            writer.write(
                mixed, labels, f"augmented_{subclass_out}", global_idx, "speech", subclass_out
            )
            accumulated_min += (len(mixed) / SR) / 60.0
            global_idx += 1
            if accumulated_min >= split_target_min:
                break

        writer.close()
        print(
            f"  {split}: {accumulated_min:.2f} min  (target {split_target_min:.2f} min)"
        )

    print(f"  Done: {subclass_out}")


def run_augmentation(tier: TierName, data_dir: Path, *, test: bool = False) -> None:
    seed_all()

    entries = augment_entries(tier, test=test)
    total_aug_min = sum(e.target_minutes for e in entries)

    print("\n" + "=" * 60)
    print(
        f"Augmentation  [{tier.value}]  {total_aug_min:.2f} min synthetic target  →  {data_dir}"
    )
    print("=" * 60)

    print("\nLoading VAD model...")
    vad = VADLabeler()
    if test:
        pool_size = 12
    else:
        pool_size = max(12, min(MUSIC_POOL_SIZE, int(total_aug_min / 6)))
    music_pool = build_music_pool(pool_size)

    for e in entries:
        print(f"\nCollecting {e.speech_subclass} for {e.output_subclass}...")
        speech_refs = collect_speech_refs(e.speech_subclass, data_dir)
        has_any = any(refs for refs in speech_refs.values())
        if not has_any:
            print("  [SKIP] No base speech clips found — run build.py first")
            continue
        generate_synthetic(
            e.output_subclass,
            speech_refs,
            music_pool,
            e.target_minutes,
            vad,
            data_dir,
        )

    print("\nAugmentation complete.")
