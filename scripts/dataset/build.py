# scripts/dataset/build.py
# Marek Hric

"""Build clips from Hugging Face sources (see notes.md, sources.toml).

Music sources use pyannote voice activity + internal RMS gating; set HF_TOKEN and
accept https://hf.co/pyannote/voice-activity-detection .

Layout (LibriSpeech-like: **config** = tier, then HF splits, then modality):

  ``{staging}/{mid|full}/{train|validation|test}/{speech|music|background}/part_*.parquet``

The hand-curated `crit` tier ships alongside whichever HF tier is built and
writes only a test split:

  ``{staging}/crit/test/{speech|music|background}/part_*.parquet``

``--out-dir`` is the **staging parent** (default ``speech_music_dataset``); each run writes only its tier subfolder(s).

Run:
    uv run python build.py mid                       # mid + crit
    uv run python build.py full --out-dir <path>     # full + crit
    uv run python build.py mid --no-critical-set     # mid only
    uv run python build.py --only-critical-set       # crit only
    uv run python build.py mid --smoke               # smoke runs both tiers
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
    TIER_TOTAL_NOMINAL_MINUTES,
    TierName,
    make_entries,
    seed_all,
    tier_dataset_dir,
)
from split_writer import SplitWriter


# Tier names exposed on the CLI. `crit` is intentionally hidden — it's bundled
# automatically with whichever HF tier is being built and toggled with the
# --no-critical-set / --only-critical-set flags below.
_USER_TIER_CHOICES = [t.value for t in TierName if t != TierName.crit]


def _smoke_data_dir(data_dir: Path) -> Path:
    """Return the smoke-mode sibling of ``data_dir`` and clear any stale contents."""
    smoke_dir = data_dir.parent / (data_dir.name + "_smoke")
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    return smoke_dir


def _fmt(minutes: float, rows: int) -> str:
    return f"{minutes:.2f} ({rows})"


def _print_summary(
    label: str,
    *,
    source_stats: list[tuple[str, float, dict[str, tuple[float, int]]]],
    aug_stats: dict[str, dict[str, tuple[float, int]]],
    aug_targets: dict[str, float],
    extra_target: float,
    suppress_val_warning: bool = False,
) -> None:
    """Emit the per-tier summary table.

    ``label`` is the tier name shown in the header. ``suppress_val_warning``
    drops the "missing val data" ⚠ flag for the crit tier where empty
    validation/train rows are by design.
    """
    col_w = 26
    cell_w = 14
    print("\n" + "=" * 60)
    print(f"Summary [{label}]: actual minutes (rows) written")
    print(f"{'Subclass':<{col_w}} {'Target':>8}  {'Train':<{cell_w}}  {'Val':<{cell_w}}  {'Test':<{cell_w}}")
    sep = "─" * (col_w + 8 + 3 * (cell_w + 2) + 4)
    print(sep)

    warnings: list[str] = []
    def _flag(val_r: int, test_r: int) -> str:
        if suppress_val_warning:
            return "  ⚠" if test_r == 0 else ""
        return "  ⚠" if (val_r == 0 or test_r == 0) else ""

    # Aggregate by subclass: HF tiers normally have one source per subclass (no-op),
    # but the crit tier has many entries per subclass and would print one row per clip.
    agg: dict[str, tuple[float, dict[str, tuple[float, int]]]] = {}
    for subclass, target, splits in source_stats:
        prev_target, prev_splits = agg.get(subclass, (0.0, {}))
        merged_splits = dict(prev_splits)
        for split_name, (m, r) in splits.items():
            pm, pr = merged_splits.get(split_name, (0.0, 0))
            merged_splits[split_name] = (pm + m, pr + r)
        agg[subclass] = (prev_target + target, merged_splits)

    for subclass, (target, splits) in agg.items():
        train_m, train_r = splits.get("train", (0.0, 0))
        val_m, val_r = splits.get("validation", (0.0, 0))
        test_m, test_r = splits.get("test", (0.0, 0))
        flag = _flag(val_r, test_r)
        print(f"  {subclass:<{col_w - 2}} {target:>8.1f}  {_fmt(train_m, train_r):<{cell_w}}  {_fmt(val_m, val_r):<{cell_w}}  {_fmt(test_m, test_r):<{cell_w}}{flag}")
        if flag:
            warnings.append(subclass)
    for subclass, splits in aug_stats.items():
        target = aug_targets.get(subclass, 0.0)
        train_m, train_r = splits.get("train", (0.0, 0))
        val_m, val_r = splits.get("validation", (0.0, 0))
        test_m, test_r = splits.get("test", (0.0, 0))
        flag = _flag(val_r, test_r)
        print(f"  {subclass:<{col_w - 2}} {target:>8.1f}  {_fmt(train_m, train_r):<{cell_w}}  {_fmt(val_m, val_r):<{cell_w}}  {_fmt(test_m, test_r):<{cell_w}}{flag}")
        if flag:
            warnings.append(subclass)
    print(sep)

    total_min = {"train": 0.0, "validation": 0.0, "test": 0.0}
    total_rows = {"train": 0, "validation": 0, "test": 0}
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
    total_target = extra_target + sum(aug_targets.values())
    print(f"  {'TOTAL':<{col_w - 2}} {total_target:>8.1f}  {_fmt(total_min['train'], total_rows['train']):<{cell_w}}  {_fmt(total_min['validation'], total_rows['validation']):<{cell_w}}  {_fmt(total_min['test'], total_rows['test']):<{cell_w}}")
    if warnings:
        if suppress_val_warning:
            msg = f"⚠  {len(warnings)} subclass(es) missing test data"
        else:
            msg = f"⚠  {len(warnings)} subclass(es) missing val or test data"
        print(f"\n  {msg}: {', '.join(warnings)}")


def _build_hf_tier(tier_key: TierName, staging: Path, *, smoke: bool) -> None:
    """Build one of the HF-streamed tiers (mid / full)."""
    data_dir = tier_dataset_dir(staging, tier_key)
    if smoke:
        data_dir = _smoke_data_dir(data_dir)

    entries = make_entries(tier_key)
    base_total_min = sum(e.target_minutes for e in entries)
    tier_nominal_min = TIER_TOTAL_NOMINAL_MINUTES[tier_key]
    synth_min = tier_nominal_min - base_total_min
    m_speech = sum(e.target_minutes for e in entries if e.cls == "speech")
    m_music = sum(e.target_minutes for e in entries if e.cls == "music")
    m_inact = sum(e.target_minutes for e in entries if e.cls == "background")

    print("=" * 60)
    mode = "SMOKE (1 row/source)" if smoke else "full"
    print(
        f"Dataset build  [{tier_key.value}]  {mode}  →  {tier_nominal_min:.0f} min tier target "
        f"({base_total_min:.0f} min HF + {synth_min:.0f} min synthetic)"
    )
    print(f"Output  : {data_dir.resolve()}")
    print(f"Sources : {len(entries)}")
    print(f"Speech  : {m_speech:.0f} min   Music: {m_music:.0f} min   Background: {m_inact:.0f} min")
    print("=" * 60)

    modalities = ("speech", "music", "background")
    writers = {
        (m, s): SplitWriter(m, s, data_dir) for m in modalities for s in HF_SPLIT_NAMES
    }

    fractions = TIER_SPLIT_FRACTIONS[tier_key]
    failed: list[str] = []
    source_stats: list[tuple[str, float, dict[str, tuple[float, int]]]] = []
    for entry in entries:
        result = process_source(entry, writers, split_fractions=fractions, smoke=smoke)
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
    aug_stats = run_augmentation(tier_key, data_dir=data_dir, split_fractions=fractions, smoke=smoke)

    aug_targets: dict[str, float] = {}
    for e in (
        list(MULTISPEAKER_AUG_SOURCES.get(tier_key, ()))
        + list(AUGMENT_SOURCES.get(tier_key, ()))
        + list(NOISE_AUG_SOURCES.get(tier_key, ()))
    ):
        aug_targets[e.output_subclass] = e.target_minutes

    _print_summary(
        tier_key.value,
        source_stats=source_stats,
        aug_stats=aug_stats,
        aug_targets=aug_targets,
        extra_target=sum(e.target_minutes for e in entries),
    )

    print(f"\n[{tier_key.value}] build complete.")
    print("=" * 60)


def _build_crit_tier(staging: Path, *, smoke: bool) -> None:
    """Build the hand-curated critical-set test shard from crit/manifest.toml."""
    from crit.crit_config import load_manifest
    from crit.crit_process import process_crit_entry

    tier_key = TierName.crit
    data_dir = tier_dataset_dir(staging, tier_key)
    if smoke:
        data_dir = _smoke_data_dir(data_dir)

    manifest_path = Path(__file__).parent / "crit" / "manifest.toml"
    crit_entries = load_manifest(manifest_path)

    print("=" * 60)
    print(f"Critical-set build  [crit]  →  {data_dir.resolve()}")
    print(f"Manifest: {manifest_path}  ({len(crit_entries)} recording(s))")
    print("=" * 60)

    modalities = ("speech", "music", "background")
    # Crit tier is small and always rebuilt from manifest — clear any stale
    # shards from previous runs so SplitWriter doesn't append duplicates.
    for m in modalities:
        mdir = data_dir / "test" / m
        if mdir.exists():
            for stale in mdir.glob("part_*.parquet"):
                stale.unlink()
    # Test split only — no train/validation directories on disk.
    writers = {(m, "test"): SplitWriter(m, "test", data_dir) for m in modalities}

    failed: list[str] = []
    source_stats: list[tuple[str, float, dict[str, tuple[float, int]]]] = []
    for entry in crit_entries:
        result = process_crit_entry(entry, writers, smoke=smoke)
        if result is None:
            failed.append(entry.name)
        else:
            # Use the actual clip minutes as the per-source "target" for the
            # summary table — coverage is trivially 100% for hand-curated data.
            clip_min = result["test"][0]
            source_stats.append((entry.metadata_subclass, clip_min, result))

    for w in writers.values():
        w.close()

    print("\n" + "=" * 60)
    print(f"Crit recordings: {len(source_stats)} OK, {len(failed)} FAILED")
    if failed:
        for name in failed:
            print(f"  ✗ {name}")

    _print_summary(
        "crit",
        source_stats=source_stats,
        aug_stats={},
        aug_targets={},
        extra_target=sum(t for _, t, _ in source_stats),
        suppress_val_warning=True,
    )

    print("\n[crit] build complete.")
    print("=" * 60)


def main():
    """CLI entry point: parse tier flags and build the requested HF tier and/or crit tier."""
    seed_all()

    tier_help = ", ".join(
        f"{t.value}≈{TIER_TOTAL_NOMINAL_MINUTES[t]:.0f}min (HF+synth)"
        for t in TierName if t != TierName.crit
    )
    parser = argparse.ArgumentParser(description="Speech/music dataset builder")
    parser.add_argument(
        "tier",
        choices=_USER_TIER_CHOICES,
        nargs="?",
        help=f"Tier to build (required unless --only-critical-set). {tier_help}",
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
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--no-critical-set",
        action="store_true",
        help="Build only the HF tier; skip the critical set.",
    )
    grp.add_argument(
        "--only-critical-set",
        action="store_true",
        help="Build only the critical set; tier positional may be omitted.",
    )
    args = parser.parse_args()

    if not args.only_critical_set and not args.tier:
        parser.error("tier is required unless --only-critical-set is set")

    staging = args.out_dir or HF_DATASET_STAGING_ROOT

    if not args.only_critical_set:
        _build_hf_tier(TierName(args.tier), staging, smoke=args.smoke)
    if not args.no_critical_set:
        _build_crit_tier(staging, smoke=args.smoke)

    # HF datasets streaming leaves background threads alive; Python's GIL
    # teardown ABORTs (exit 134) when those threads release thread-state after
    # the interpreter has already started finalizing. os._exit skips the
    # problematic cleanup path while still flushing stdio.
    os._exit(0)


if __name__ == "__main__":
    main()
