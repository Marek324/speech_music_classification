"""Source processing: stream HF rows, label, and write to parquet shards."""

import sys

from tqdm import tqdm
from source_config import TEST_SOURCE_MAX_ROWS, SourceEntry

from dataloader import load_source
from labeling import get_labeler
from split_writer import SplitWriter


def process_source(
    entry: SourceEntry,
    writers: dict[tuple[str, str], SplitWriter],
    *,
    split_fractions: dict[str, float],
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

    train_cutoff_min = target_min * split_fractions["train"]
    val_cutoff_min = target_min * (split_fractions["train"] + split_fractions["validation"])
    accumulated_min = 0.0
    local_idx = 0
    any_written = False

    for row in tqdm(ds, desc=f"  {entry.display_name}", total=max_rows if test else None):
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
            audio, labels, entry.display_name, local_idx, entry.cls, entry.metadata_subclass
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
