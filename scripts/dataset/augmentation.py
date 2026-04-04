"""Synthetic augmentations: multi-speaker, speech-over-music, speech-over-noise."""

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
    AUGMENT_MUSIC_GENRES,
    FMA_GENRE_MAP,
    FMA_HF_ID,
    MULTISPEAKER_GAIN_RANGE_DB,
    MUSIC_POOL_SIZE,
    MUSIC_RELATIVE_DB,
    NOISE_SNR_RANGE_DB,
    RAND_SEED,
    TierName,
    MultispeakerAugEntry,
    NoiseAugEntry,
    augment_entries,
    multispeaker_aug_entries,
    noise_aug_entries,
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


def build_music_pool(pool_size: int, genres: tuple[str, ...]) -> list[np.ndarray]:
    print(f"Building music pool ({pool_size} clips, {len(genres)} genres: {', '.join(genres)})...")
    per_genre = max(1, pool_size // len(genres))
    pool: list[np.ndarray] = []
    for genre in genres:
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


def mix_with_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    if len(noise) == 0:
        return speech.copy()
    if len(noise) > len(speech):
        offset = rng.randint(0, len(noise) - len(speech))
        noise = noise[offset:offset + len(speech)]
    else:
        repeats = -(-len(speech) // len(noise))
        noise = np.tile(noise, repeats)[:len(speech)]
    noise_rms_val = rms(noise)
    if noise_rms_val < 1e-9:
        return speech.copy()
    target_noise_rms = rms(speech) / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / noise_rms_val)
    mixed = speech + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


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


def collect_noise_refs(data_dir: Path) -> dict[str, list[tuple[Path, int]]]:
    """Row refs under ``{config_dir}/{split}/inactive/``."""
    result: dict[str, list[tuple[Path, int]]] = {"train": [], "validation": [], "test": []}
    for split in result:
        root = data_dir / split / "inactive"
        if not root.exists():
            continue
        for path in sorted(root.glob("part_*.parquet")):
            t = pq.read_table(path, columns=["row_idx"])
            for i in range(len(t)):
                result[split].append((path, i))
    for split, refs in result.items():
        print(f"  noise/{split}: {len(refs)} clips")
    return result


def _read_wav_from_parquet(
    path: Path, row_idx: int, table_cache: dict[Path, pa.Table]
) -> np.ndarray:
    if path not in table_cache:
        table_cache[path] = pq.read_table(path, columns=["audio"])
    t = table_cache[path]
    raw = t["audio"][row_idx].as_py()["bytes"]
    bio = io.BytesIO(raw)
    arr, _ = sf.read(bio, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def _read_labels_from_parquet(
    path: Path, row_idx: int, labels_cache: dict[Path, pa.Table]
) -> list[dict]:
    if path not in labels_cache:
        labels_cache[path] = pq.read_table(path, columns=["labels"])
    t = labels_cache[path]
    # HF Sequence(Features) stores as struct-of-lists: {"label": [...], "start": [...], "end": [...]}
    raw = t["labels"][row_idx].as_py()
    return [{"label": l, "start": s, "end": e} for l, s, e in zip(raw["label"], raw["start"], raw["end"])]


def mix_speakers(a: np.ndarray, b: np.ndarray, gain_db: float) -> np.ndarray:
    """Sum two speech waveforms, truncating to the shorter (LibriMix min-mode)."""
    n = min(len(a), len(b))
    a, b = a[:n].copy(), b[:n].copy()
    b *= 10 ** (gain_db / 20)
    mixed = a + b
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


def generate_multispeaker(
    entry: MultispeakerAugEntry,
    speech_refs: dict[str, list[tuple[Path, int]]],
    vad: VADLabeler,
    data_dir: Path,
    split_fractions: dict[str, float],
    smoke: bool = False,
) -> dict[str, tuple[float, int]]:
    """LibriMix-style multi-speaker: mix random pairs of single-speaker clips.

    Returns ``{split: (actual_minutes, row_count)}`` for each split.
    """
    split_targets_min = {split: entry.target_minutes * frac for split, frac in split_fractions.items()}
    rng = random.Random(RAND_SEED)
    audio_cache: dict[Path, pa.Table] = {}
    split_minutes: dict[str, float] = {"train": 0.0, "validation": 0.0, "test": 0.0}
    split_rows: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    lo, hi = MULTISPEAKER_GAIN_RANGE_DB
    print(f"\n{'─' * 60}")
    print(f"  Generating {entry.output_subclass}  (target {entry.target_minutes:.2f} min, gain {lo:+.0f} to {hi:+.0f} dB)")
    print(f"{'─' * 60}")

    global_idx = 0
    for split, split_target_min in split_targets_min.items():
        available = speech_refs.get(split, [])
        if not available:
            print(f"  [WARN] No speech clips in {split}/speech/ — skipping")
            continue

        writer = SplitWriter("speech", split, data_dir)
        accumulated_min = 0.0
        n_rows = 0
        shuffled_a = rng.sample(available, len(available))
        shuffled_b = rng.sample(available, len(available))

        for (pa_, ia), (pb, ib) in itertools.cycle(zip(shuffled_a, shuffled_b)):
            if len(available) > 1 and (pa_, ia) == (pb, ib):
                continue
            arr_a = _read_wav_from_parquet(pa_, ia, audio_cache)
            arr_b = _read_wav_from_parquet(pb, ib, audio_cache)
            gain_db = rng.uniform(lo, hi)
            mixed = mix_speakers(arr_a, arr_b, gain_db)
            labels = vad.label(mixed)
            if labels is None:
                continue
            writer.write(
                mixed, labels, f"augmented_{entry.output_subclass}",
                global_idx, "speech", entry.output_subclass,
            )
            accumulated_min += (len(mixed) / SR) / 60.0
            n_rows += 1
            global_idx += 1
            if smoke or accumulated_min >= split_target_min:
                break

        writer.close()
        split_minutes[split] = accumulated_min
        split_rows[split] = n_rows
        print(f"  {split}: {accumulated_min:.2f} min ({n_rows})  (target {split_target_min:.2f} min)")

    print(f"  Done: {entry.output_subclass}")
    return {s: (split_minutes[s], split_rows[s]) for s in ("train", "validation", "test")}


def generate_noisy_speech(
    entry: NoiseAugEntry,
    speech_refs: dict[str, list[tuple[Path, int]]],
    noise_refs: dict[str, list[tuple[Path, int]]],
    data_dir: Path,
    split_fractions: dict[str, float],
    smoke: bool = False,
) -> dict[str, tuple[float, int]]:
    """Returns ``{split: (actual_minutes, row_count)}`` for each split."""
    split_targets_min = {split: entry.target_minutes * frac for split, frac in split_fractions.items()}
    rng = random.Random(RAND_SEED)
    audio_cache: dict[Path, pa.Table] = {}
    labels_cache: dict[Path, pa.Table] = {}
    split_minutes: dict[str, float] = {"train": 0.0, "validation": 0.0, "test": 0.0}
    split_rows: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    print(f"\n{'─' * 60}")
    print(f"  Generating {entry.output_subclass}  (target {entry.target_minutes:.2f} min, SNR {NOISE_SNR_RANGE_DB[0]:.0f}–{NOISE_SNR_RANGE_DB[1]:.0f} dB)")
    print(f"{'─' * 60}")

    global_idx = 0
    for split, split_target_min in split_targets_min.items():
        available_speech = speech_refs.get(split, [])
        available_noise = noise_refs.get(split, [])
        if not available_speech:
            print(f"  [WARN] No speech clips in {split}/speech/ — skipping")
            continue
        if not available_noise:
            print(f"  [WARN] No noise clips in {split}/inactive/ — skipping")
            continue

        writer = SplitWriter("speech", split, data_dir)
        accumulated_min = 0.0
        n_rows = 0
        shuffled = rng.sample(available_speech, len(available_speech))

        for path, row_idx in itertools.cycle(shuffled):
            speech_arr = _read_wav_from_parquet(path, row_idx, audio_cache)
            labels = _read_labels_from_parquet(path, row_idx, labels_cache)
            noise_path, noise_row_idx = rng.choice(available_noise)
            noise_arr = _read_wav_from_parquet(noise_path, noise_row_idx, audio_cache)
            snr_db = rng.uniform(*NOISE_SNR_RANGE_DB)
            mixed = mix_with_noise(speech_arr, noise_arr, snr_db, rng)
            writer.write(mixed, labels, f"augmented_{entry.output_subclass}", global_idx, "speech", entry.output_subclass)
            accumulated_min += (len(mixed) / SR) / 60.0
            n_rows += 1
            global_idx += 1
            if smoke or accumulated_min >= split_target_min:
                break

        writer.close()
        split_minutes[split] = accumulated_min
        split_rows[split] = n_rows
        print(f"  {split}: {accumulated_min:.2f} min ({n_rows})  (target {split_target_min:.2f} min)")

    print(f"  Done: {entry.output_subclass}")
    return {s: (split_minutes[s], split_rows[s]) for s in ("train", "validation", "test")}


def generate_synthetic(
    subclass_out: str,
    speech_refs: dict[str, list[tuple[Path, int]]],
    music_pool: list[np.ndarray],
    target_minutes: float,
    vad: VADLabeler,
    data_dir: Path,
    split_fractions: dict[str, float],
    smoke: bool = False,
) -> dict[str, tuple[float, int]]:
    """Returns ``{split: (actual_minutes, row_count)}`` for each split."""
    split_targets_min = {
        split: target_minutes * frac
        for split, frac in split_fractions.items()
    }
    rng = random.Random(RAND_SEED)
    table_cache: dict[Path, pa.Table] = {}
    split_minutes: dict[str, float] = {"train": 0.0, "validation": 0.0, "test": 0.0}
    split_rows: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

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
        n_rows = 0
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
            n_rows += 1
            global_idx += 1
            if smoke or accumulated_min >= split_target_min:
                break

        writer.close()
        split_minutes[split] = accumulated_min
        split_rows[split] = n_rows
        print(f"  {split}: {accumulated_min:.2f} min ({n_rows})  (target {split_target_min:.2f} min)")

    print(f"  Done: {subclass_out}")
    return {s: (split_minutes[s], split_rows[s]) for s in ("train", "validation", "test")}


def run_augmentation(
    tier: TierName,
    data_dir: Path,
    *,
    split_fractions: dict[str, float],
    smoke: bool = False,
) -> dict[str, dict[str, tuple[float, int]]]:
    """Run all augmentation recipes for the tier.

    Returns a dict mapping subclass name → ``{split: (actual_minutes, row_count)}``,
    e.g. ``{"speech_som": {"train": (30.2, 180), "validation": (2.9, 18), "test": (2.8, 17)}}``.

    """
    seed_all()

    ms_entries = multispeaker_aug_entries(tier, smoke=smoke)
    music_entries = augment_entries(tier, smoke=smoke)
    n_entries = noise_aug_entries(tier, smoke=smoke)
    total_aug_min = (
        sum(e.target_minutes for e in ms_entries)
        + sum(e.target_minutes for e in music_entries)
        + sum(e.target_minutes for e in n_entries)
    )

    print("\n" + "=" * 60)
    print(f"Augmentation  [{tier.value}]  {total_aug_min:.2f} min synthetic target  →  {data_dir}")
    print("=" * 60)

    aug_stats: dict[str, dict[str, float]] = {}

    # VAD is shared by multispeaker and speech-over-music steps.
    vad: VADLabeler | None = None
    if ms_entries or music_entries:
        print("\nLoading VAD model...")
        vad = VADLabeler()

    # ── multi-speaker (must run before speech_msom which reads its output) ──
    if ms_entries and vad is not None:
        for e in ms_entries:
            print(f"\nCollecting {e.speech_subclass} for {e.output_subclass}...")
            speech_refs = collect_speech_refs(e.speech_subclass, data_dir)
            if not any(refs for refs in speech_refs.values()):
                print("  [SKIP] No base speech clips found — run build.py first")
                continue
            aug_stats[e.output_subclass] = generate_multispeaker(
                e, speech_refs, vad, data_dir, split_fractions, smoke=smoke
            )

    # ── speech-over-music (speech_som, speech_msom) ──
    if music_entries and vad is not None:
        genres = AUGMENT_MUSIC_GENRES[tier]
        music_total = sum(e.target_minutes for e in music_entries)
        if smoke:
            pool_size = max(len(genres), 12)
        else:
            pool_size = max(len(genres), min(MUSIC_POOL_SIZE, int(music_total / 6)))
        music_pool = build_music_pool(pool_size, genres)

        for e in music_entries:
            print(f"\nCollecting {e.speech_subclass} for {e.output_subclass}...")
            speech_refs = collect_speech_refs(e.speech_subclass, data_dir)
            if not any(refs for refs in speech_refs.values()):
                print("  [SKIP] No base speech clips found — run build.py first")
                continue
            aug_stats[e.output_subclass] = generate_synthetic(
                e.output_subclass, speech_refs, music_pool, e.target_minutes, vad, data_dir, split_fractions, smoke=smoke
            )

    # ── speech-over-noise (speech_noisy) ──
    if n_entries:
        print("\nCollecting noise clips...")
        noise_refs = collect_noise_refs(data_dir)
        if not any(refs for refs in noise_refs.values()):
            print("  [SKIP] No inactive clips found — run build.py first")
        else:
            for e in n_entries:
                print(f"\nCollecting {e.speech_subclass} for {e.output_subclass}...")
                speech_refs = collect_speech_refs(e.speech_subclass, data_dir)
                if not any(refs for refs in speech_refs.values()):
                    print("  [SKIP] No base speech clips found — run build.py first")
                    continue
                aug_stats[e.output_subclass] = generate_noisy_speech(
                    e, speech_refs, noise_refs, data_dir, split_fractions, smoke=smoke
                )

    print("\nAugmentation complete.")
    return aug_stats
