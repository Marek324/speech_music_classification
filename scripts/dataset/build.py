"""Build clips from Hugging Face sources (see notes.md, sources.toml).

Music sources use pyannote voice activity + internal RMS gating; set HF_TOKEN and
accept https://hf.co/pyannote/voice-activity-detection .

Layout (LibriSpeech-like: **config** = tier, then HF splits, then modality):

  ``{staging}/{mid|full}/{train|validation|test}/{speech|music|inactive}/part_*.parquet``

``--out-dir`` is the **staging parent** (default ``speech_music_dataset``); each run writes only its tier subfolder.

Run:
    uv run python build.py mid
    uv run python build.py full --out-dir /path/to/staging
    uv run python build.py mid --smoke    # one HF row per split per source + small augmentation
"""

import argparse
import os
import shutil
from pathlib import Path

from augmentation import run_augmentation
from process import process_source
from source_config import (
    AUGMENT_SOURCES,
    HF_DATASET_STAGING_ROOT,
    HF_SPLIT_NAMES,
    MULTISPEAKER_AUG_SOURCES,
    NOISE_AUG_SOURCES,
    TIER_SPLIT_FRACTIONS,
    TierName,
    seed_all,
    tier_dataset_dir,
)
from source_config import TIER_TOTAL_NOMINAL_MINUTES, make_entries
from split_writer import SplitWriter


def main():
    seed_all()

    tier_choices = [t.value for t in TierName]
    tier_help = ", ".join(
        f"{t.value}≈{TIER_TOTAL_NOMINAL_MINUTES[t]:.0f}min (HF+synth)" for t in TierName
    )
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
    base_total_min = sum(e.target_minutes for e in entries)
    tier_nominal_min = TIER_TOTAL_NOMINAL_MINUTES[tier_key]
    synth_min = tier_nominal_min - base_total_min
    m_speech = sum(e.target_minutes for e in entries if e.cls == "speech")
    m_music = sum(e.target_minutes for e in entries if e.cls == "music")
    m_inact = sum(e.target_minutes for e in entries if e.cls == "inactive")

    print("=" * 60)
    mode = "SMOKE (1 row/source)" if args.smoke else "full"
    print(
        f"Dataset build  [{args.tier}]  {mode}  →  {tier_nominal_min:.0f} min tier target "
        f"({base_total_min:.0f} min HF + {synth_min:.0f} min synthetic)"
    )
    print(f"Output  : {data_dir.resolve()}")
    print(f"Sources : {len(entries)}")
    print(f"Speech  : {m_speech:.0f} min   Music: {m_music:.0f} min   Inactive: {m_inact:.0f} min")
    print("=" * 60)

    modalities = ("speech", "music", "inactive")
    writers = {
        (m, s): SplitWriter(m, s, data_dir) for m in modalities for s in HF_SPLIT_NAMES
    }

    fractions = TIER_SPLIT_FRACTIONS[tier_key]
    failed = []
    # (subclass, target_min, {split: (minutes, rows)})
    source_stats: list[tuple[str, float, dict[str, tuple[float, int]]]] = []
    for entry in entries:
        result = process_source(entry, writers, split_fractions=fractions, smoke=args.smoke)
        if result is None:
            failed.append(entry.display_name)
        else:
            source_stats.append((entry.metadata_subclass, entry.target_minutes, result))

    for w in writers.values():
        w.close()

    print("\n" + "=" * 60)
    print(f"Base sources: {len(source_stats)} OK, {len(failed)} FAILED")
    if failed:
        for name in failed:
            print(f"  ✗ {name}")

    print("\nRunning augmentation...")
    aug_stats = run_augmentation(tier_key, data_dir=data_dir, split_fractions=fractions, smoke=args.smoke)

    # ── Summary table: targets vs actuals per subclass/split ──
    aug_targets: dict[str, float] = {}
    for e in (
        list(MULTISPEAKER_AUG_SOURCES.get(tier_key, ()))
        + list(AUGMENT_SOURCES.get(tier_key, ()))
        + list(NOISE_AUG_SOURCES.get(tier_key, ()))
    ):
        aug_targets[e.output_subclass] = e.target_minutes

    def _fmt(minutes: float, rows: int) -> str:
        return f"{minutes:.2f} ({rows})"

    col_w = 26
    cell_w = 14
    print("\n" + "=" * 60)
    print("Summary: actual minutes (rows) written")
    print(f"{'Subclass':<{col_w}} {'Target':>8}  {'Train':<{cell_w}}  {'Val':<{cell_w}}  {'Test':<{cell_w}}")
    sep = "─" * (col_w + 8 + 3 * (cell_w + 2) + 4)
    print(sep)
    warnings = []
    for subclass, target, splits in source_stats:
        train_m, train_r = splits.get("train", (0.0, 0))
        val_m, val_r = splits.get("validation", (0.0, 0))
        test_m, test_r = splits.get("test", (0.0, 0))
        flag = "  ⚠" if (val_r == 0 or test_r == 0) else ""
        print(f"  {subclass:<{col_w - 2}} {target:>8.1f}  {_fmt(train_m, train_r):<{cell_w}}  {_fmt(val_m, val_r):<{cell_w}}  {_fmt(test_m, test_r):<{cell_w}}{flag}")
        if flag:
            warnings.append(subclass)
    for subclass, splits in aug_stats.items():
        target = aug_targets.get(subclass, 0.0)
        train_m, train_r = splits.get("train", (0.0, 0))
        val_m, val_r = splits.get("validation", (0.0, 0))
        test_m, test_r = splits.get("test", (0.0, 0))
        flag = "  ⚠" if (val_r == 0 or test_r == 0) else ""
        print(f"  {subclass:<{col_w - 2}} {target:>8.1f}  {_fmt(train_m, train_r):<{cell_w}}  {_fmt(val_m, val_r):<{cell_w}}  {_fmt(test_m, test_r):<{cell_w}}{flag}")
        if flag:
            warnings.append(subclass)
    print(sep)
    total_min: dict[str, float] = {"train": 0.0, "validation": 0.0, "test": 0.0}
    total_rows: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for _, _, splits in source_stats:
        for s in total_min:
            m, r = splits.get(s, (0.0, 0))
            total_min[s] += m
            total_rows[s] += r
    for _, splits in aug_stats.items():
        for s in total_min:
            m, r = splits.get(s, (0.0, 0))
            total_min[s] += m
            total_rows[s] += r
    total_target = sum(e.target_minutes for e in entries) + sum(aug_targets.values())
    print(f"  {'TOTAL':<{col_w - 2}} {total_target:>8.1f}  {_fmt(total_min['train'], total_rows['train']):<{cell_w}}  {_fmt(total_min['validation'], total_rows['validation']):<{cell_w}}  {_fmt(total_min['test'], total_rows['test']):<{cell_w}}")
    if warnings:
        print(f"\n  ⚠  {len(warnings)} subclass(es) missing val or test data: {', '.join(warnings)}")

    print("\nBuild complete.")
    print("=" * 60)
    # HF datasets streaming leaves background threads alive; Python's GIL
    # teardown ABORTs (exit 134) when those threads release thread-state after
    # the interpreter has already started finalizing.  os._exit skips the
    # problematic cleanup path while still flushing stdio.
    os._exit(0)


if __name__ == "__main__":
    main()
