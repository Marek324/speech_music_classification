"""Build clips from Hugging Face sources (see notes.md, sources.py).

Music sources use pyannote voice activity + internal RMS gating; set HF_TOKEN and
accept https://hf.co/pyannote/voice-activity-detection .

Layout (LibriSpeech-like: **config** = tier, then HF splits, then modality):

  ``{staging}/{mini|mid|full}/{train|validation|test}/{speech|music|inactive}/part_*.parquet``

``--out-dir`` is the **staging parent** (default ``speech_music_dataset``); each run writes only its tier subfolder.

Run:
    uv run python build.py mid
    uv run python build.py mini --out-dir /path/to/staging
    uv run python build.py mid --smoke    # one HF row per split per source + small augmentation
"""

import argparse
import os
import shutil
from pathlib import Path

from augmentation import run_augmentation
from process import process_source
from source_config import (
    HF_DATASET_STAGING_ROOT,
    HF_SPLIT_NAMES,
    TIER_SPLIT_FRACTIONS,
    TierName,
    seed_all,
    tier_dataset_dir,
)
from sources import TIER_TOTAL_MINUTES, make_entries
from split_writer import SplitWriter


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
        "--smoke",
        action="store_true",
        help="Smoke run: 1 HF row per split per source; small augmentation budget",
    )
    args = parser.parse_args()

    tier_key = TierName(args.tier)
    staging = args.out_dir or HF_DATASET_STAGING_ROOT
    data_dir = tier_dataset_dir(staging, tier_key)
    if args.smoke:
        data_dir = data_dir.parent / (data_dir.name + "_smoke")
        if data_dir.exists():
            shutil.rmtree(data_dir)

    entries = make_entries(tier_key)
    nominal_total_min = sum(e.target_minutes for e in entries)
    m_speech = sum(e.target_minutes for e in entries if e.cls == "speech")
    m_music = sum(e.target_minutes for e in entries if e.cls == "music")
    m_inact = sum(e.target_minutes for e in entries if e.cls == "inactive")

    print("=" * 60)
    mode = "SMOKE (1 row/source)" if args.smoke else "full"
    print(f"Dataset build  [{args.tier}]  {mode}  →  {nominal_total_min:.0f} min nominal target")
    print(f"Output  : {data_dir.resolve()}")
    print(f"Sources : {len(entries)}")
    print(f"Speech  : {m_speech:.0f} min   Music: {m_music:.0f} min   Inactive: {m_inact:.0f} min")
    print("=" * 60)

    modalities = ("speech", "music", "inactive")
    writers = {
        (m, s): SplitWriter(m, s, data_dir) for m in modalities for s in HF_SPLIT_NAMES
    }

    fractions = TIER_SPLIT_FRACTIONS[tier_key]
    failed, succeeded = [], []
    for entry in entries:
        ok = process_source(entry, writers, split_fractions=fractions, smoke=args.smoke)
        (succeeded if ok else failed).append(entry.display_name)

    for w in writers.values():
        w.close()

    print("\n" + "=" * 60)
    print(f"Base sources: {len(succeeded)} OK, {len(failed)} FAILED")
    if failed:
        for name in failed:
            print(f"  ✗ {name}")

    print("\nRunning augmentation...")
    run_augmentation(tier_key, data_dir=data_dir, split_fractions=fractions, smoke=args.smoke)

    print("\nBuild complete.")
    print("=" * 60)
    # HF datasets streaming leaves background threads alive; Python's GIL
    # teardown ABORTs (exit 134) when those threads release thread-state after
    # the interpreter has already started finalizing.  os._exit skips the
    # problematic cleanup path while still flushing stdio.
    os._exit(0)


if __name__ == "__main__":
    main()
