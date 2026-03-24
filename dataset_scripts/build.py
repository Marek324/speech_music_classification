"""Build clips from Hugging Face sources (see notes.md, sources.py).

Music sources use pyannote voice activity + internal RMS gating; set HF_TOKEN and
accept https://hf.co/pyannote/voice-activity-detection .

Layout (LibriSpeech-like: **config** = tier, then HF splits, then modality):

  ``{staging}/{mini|mid|full}/{train|validation|test}/{speech|music|inactive}/part_*.parquet``

``--out-dir`` is the **staging parent** (default ``speech_music_dataset``); each run writes only its tier subfolder.

Run:
    uv run python build.py mid
    uv run python build.py mini --out-dir /path/to/staging
    uv run python build.py mid --test    # one HF row per source + small augmentation
"""

import argparse
import sys
from pathlib import Path

from datasets import Audio, IterableDataset, load_dataset
from source_config import (
    HF_DATASET_STAGING_ROOT,
    HF_SPLIT_NAMES,
    RAND_SEED,
    TEST_AUG_TOTAL_MINUTES,
    TEST_SOURCE_MAX_ROWS,
    TierName,
    SourceEntry,
    seed_all,
    tier_dataset_dir,
)
from sources import TIER_TOTAL_MINUTES, make_entries
from tqdm import tqdm

from labeling import SR, get_labeler
from split_writer import SplitWriter


def load_source(entry: SourceEntry, max_rows: int, skip: int = 0) -> IterableDataset:
    col = entry.audio_col
    ds = load_dataset(entry.hf_id, split=entry.split, streaming=True, **entry.hf_load_kwargs())
    assert isinstance(ds, IterableDataset)
    ds = ds.shuffle(seed=RAND_SEED)
    if entry.filter_col and entry.filter_val is not None:
        ds = ds.filter(lambda row: row[entry.filter_col] == entry.filter_val)
    if skip > 0:
        ds = ds.skip(skip)
    ds = ds.take(max_rows)
    ds = ds.select_columns(col)
    ds = ds.cast_column(col, Audio(sampling_rate=SR, num_channels=1, decode=entry.audio_decode))
    return ds


def process_source(
    entry: SourceEntry,
    data_dir: Path,
    writers: dict[tuple[str, str], SplitWriter],
    *,
    test: bool,
) -> bool:
    """Stream HF rows until accumulated labeled audio reaches ``entry.target_minutes``.

    With ``test=True``, read at most ``TEST_SOURCE_MAX_ROWS`` rows and keep the first
    row that produces valid labels (smoke test).
    """
    target_min = entry.target_minutes
    max_rows = TEST_SOURCE_MAX_ROWS if test else entry.max_rows_for_stream()
    col = entry.audio_col

    print(f"\n{'─' * 60}")
    print(f"  {entry.display_name}  [{entry.cls} → {entry.metadata_subclass}]")
    if test:
        print(f"  target: test mode (≤{TEST_SOURCE_MAX_ROWS} HF row(s) per source)")
    else:
        print(f"  target: {target_min:.2f} min audio")
    print(f"{'─' * 60}")

    try:
        ds = load_source(entry, max_rows)
    except Exception as e:
        print(f"  [FAIL] load_source: {e}", file=sys.stderr)
        return False

    labeler = get_labeler(entry.detector)

    train_cutoff_min = target_min * 0.8
    val_cutoff_min = target_min * 0.9
    accumulated_min = 0.0
    local_idx = 0
    any_written = False

    for row in tqdm(ds, desc=f"  {entry.display_name}", total=max_rows if test else None):
        labels = labeler.label(row[col])
        if labels is None:
            print(f"  [SKIP] row {local_idx} — invalid audio", file=sys.stderr)
            local_idx += 1
            if test:
                break
            continue

        clip_min = labels[-1]["end"] / 1000.0 / 60.0

        if test:
            split_name = "train"
        elif accumulated_min < train_cutoff_min:
            split_name = "train"
        elif accumulated_min < val_cutoff_min:
            split_name = "validation"
        else:
            split_name = "test"

        writers[(entry.cls, split_name)].write(
            row[col], labels, entry.display_name, local_idx, entry.cls, entry.metadata_subclass
        )
        accumulated_min += clip_min
        local_idx += 1
        any_written = True

        if test:
            break
        if accumulated_min >= target_min:
            break

    if test:
        print(f"  test mode: wrote {int(any_written)} clip(s)")
    else:
        pct = accumulated_min / target_min * 100 if target_min else 0
        print(f"  {accumulated_min:.2f} min written  (target {target_min:.2f} min, {pct:.0f}%)")
        if pct < 90:
            print("  [WARN] source exhausted before reaching target", file=sys.stderr)
    return any_written


def main():
    seed_all()

    tier_choices = [t.value for t in TierName]
    tier_help = ", ".join(f"{t.value}≈{TIER_TOTAL_MINUTES[t]:.0f}min" for t in TierName)
    parser = argparse.ArgumentParser(description="Speech/music dataset builder")
    parser.add_argument(
        "tier",
        choices=tier_choices,
        help=f"Tier from sources.SOURCES; totals are target minutes ({tier_help})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=f"Hub staging parent (default: {HF_DATASET_STAGING_ROOT}); files go to <out-dir>/<tier>/…",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Smoke test: ≤{TEST_SOURCE_MAX_ROWS} HF row per source; small augmentation budget",
    )
    args = parser.parse_args()

    tier_key = TierName(args.tier)
    staging = args.out_dir or HF_DATASET_STAGING_ROOT
    data_dir = tier_dataset_dir(staging, tier_key)

    entries = make_entries(tier_key)
    nominal_total_min = sum(e.target_minutes for e in entries)
    m_speech = sum(e.target_minutes for e in entries if e.cls == "speech")
    m_music = sum(e.target_minutes for e in entries if e.cls == "music")
    m_inact = sum(e.target_minutes for e in entries if e.cls == "inactive")

    print("=" * 60)
    mode = "TEST (1 row/source)" if args.test else "full"
    print(f"Dataset build  [{args.tier}]  {mode}  →  {nominal_total_min:.0f} min nominal target")
    print(f"Output  : {data_dir.resolve()}")
    print(f"Sources : {len(entries)}")
    print(f"Speech  : {m_speech:.0f} min   Music: {m_music:.0f} min   Inactive: {m_inact:.0f} min")
    print("=" * 60)

    modalities = ("speech", "music", "inactive")
    writers = {
        (m, s): SplitWriter(m, s, data_dir) for m in modalities for s in HF_SPLIT_NAMES
    }

    failed, succeeded = [], []
    for entry in entries:
        ok = process_source(entry, data_dir, writers, test=args.test)
        (succeeded if ok else failed).append(entry.display_name)

    for w in writers.values():
        w.close()

    print("\n" + "=" * 60)
    print(f"Base sources: {len(succeeded)} OK, {len(failed)} FAILED")
    if failed:
        for name in failed:
            print(f"  ✗ {name}")

    aug_minutes = TEST_AUG_TOTAL_MINUTES if args.test else nominal_total_min
    print("\nRunning augmentation...")
    from augmentation import run_augmentation

    run_augmentation(total_minutes=aug_minutes, data_dir=data_dir, test=args.test)

    print("\nBuild complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
