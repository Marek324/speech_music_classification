# scripts/dataset/process.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

"""Source processing: stream HF rows, label, and write to parquet shards."""

import sys
from collections.abc import Generator

from tqdm import tqdm
from source_config import HF_SPLIT_NAMES, SMOKE_SOURCE_MAX_ROWS, SourceEntry

from dataloader import load_source
from labeling import get_labeler
from split_writer import SplitWriter


def _bresenham_split_iter(split_fractions: dict[str, float]) -> Generator[str, None, None]:
    """Yield split names interleaved proportionally using the Bresenham algorithm.

    Maintains an error accumulator per split (initialized to its fraction).
    Each step yields the split with the highest error, decrements it by 1.0,
    and increments all errors by their fractions.

    This guarantees val and test receive clips from the very first few rows,
    even when the source only provides enough data for a handful of clips.

    Example for (train=0.70, val=0.15, test=0.15):
        Clip 1 → train, Clip 2 → train, Clip 3 → val, Clip 4 → train, Clip 5 → test, ...
    """
    splits = ["train", "validation", "test"]
    fracs = [split_fractions.get(s, 0.0) for s in splits]
    errors = list(fracs)
    while True:
        idx = max(range(len(splits)), key=lambda i: errors[i])
        yield splits[idx]
        errors[idx] -= 1.0
        for i in range(len(errors)):
            errors[i] += fracs[i]


def process_source(
    entry: SourceEntry,
    writers: dict[tuple[str, str], SplitWriter],
    *,
    split_fractions: dict[str, float],
    smoke: bool,
) -> dict[str, tuple[float, int]] | None:
    """Stream HF rows until accumulated labeled audio reaches ``entry.target_minutes``.

    Clips are assigned to train/val/test via Bresenham interleaving derived from
    ``split_fractions``, so all splits receive data proportionally from the first
    few clips even when the source exhausts early.

    With ``smoke=True``, read at most ``SMOKE_SOURCE_MAX_ROWS`` rows (1 per split).

    Returns a dict of ``{split: (actual_minutes, row_count)}`` or ``None`` on failure.
    """
    target_min = entry.target_minutes
    max_rows = SMOKE_SOURCE_MAX_ROWS if smoke else entry.max_rows_for_stream()
    col = entry.audio_col

    print(f"\n{'─' * 60}")
    print(f"  {entry.display_name}  [{entry.cls} → {entry.metadata_subclass}]")
    if smoke:
        print(f"  target: smoke mode (≤{SMOKE_SOURCE_MAX_ROWS} HF row(s) per source)")
    else:
        print(f"  target: {target_min:.2f} min audio")
    print(f"{'─' * 60}")

    try:
        ds = load_source(entry, max_rows)
    except Exception as e:
        print(f"  [FAIL] load_source: {e}", file=sys.stderr)
        return None

    labeler = get_labeler(entry.detector)

    split_iter = _bresenham_split_iter(split_fractions)
    accumulated_min = 0.0
    split_minutes: dict[str, float] = {"train": 0.0, "validation": 0.0, "test": 0.0}
    split_rows: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    local_idx = 0
    test_split_idx = 0

    for row in tqdm(ds, desc=f"  {entry.display_name}", total=max_rows if smoke else None):
        audio = row[col]
        if hasattr(audio, "get_all_samples"):
            data = audio.get_all_samples().data
            if hasattr(data, "cpu"):
                data = data.cpu()
            audio = data.numpy().squeeze().astype("float32")
        labels = labeler.label(audio)
        if labels is None:
            print(f"  [SKIP] row {local_idx} — invalid audio", file=sys.stderr)
            local_idx += 1
            continue

        clip_min = labels[-1]["end"] / 1000.0 / 60.0

        if smoke:
            split_name = HF_SPLIT_NAMES[test_split_idx]
            test_split_idx += 1
        else:
            split_name = next(split_iter)

        writers[(entry.cls, split_name)].write(
            audio, labels, entry.display_name, local_idx, entry.cls, entry.metadata_subclass
        )
        accumulated_min += clip_min
        split_minutes[split_name] += clip_min
        split_rows[split_name] += 1
        local_idx += 1

        if smoke and test_split_idx >= len(HF_SPLIT_NAMES):
            break
        if not smoke and accumulated_min >= target_min:
            break

    if not any(split_rows.values()):
        return None

    if smoke:
        print(f"  smoke mode: wrote {test_split_idx} clip(s) across splits")
    else:
        pct = accumulated_min / target_min * 100 if target_min else 0
        print(f"  {accumulated_min:.2f} min written  (target {target_min:.2f} min, {pct:.0f}%)")
        if pct < 90:
            print("  [WARN] source exhausted before reaching target", file=sys.stderr)

    parts = []
    for s in ("train", "validation", "test"):
        parts.append(f"{s}: {split_minutes[s]:.2f}min ({split_rows[s]})")
    print(f"    {' | '.join(parts)}")

    return {s: (split_minutes[s], split_rows[s]) for s in ("train", "validation", "test")}
