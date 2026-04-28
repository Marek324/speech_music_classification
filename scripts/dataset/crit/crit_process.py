"""Process one critical-set recording → one parquet row in the test split.

Mirrors process.process_source's responsibility (load → label → write) but
operates on a single hand-curated local file instead of a streaming HF source,
and writes to ``test`` unconditionally.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

from labeling import SR, get_labeler
from split_writer import SplitWriter

from .crit_config import CritEntry
from .crit_loader import load_recording


def process_crit_entry(
    entry: CritEntry,
    writers: dict[tuple[str, str], SplitWriter],
    *,
    smoke: bool = False,
) -> Optional[dict[str, tuple[float, int]]]:
    """Load one recording, label it, write one parquet row.

    Returns ``{"test": (clip_minutes, 1)}`` on success, ``None`` on failure.
    The shape mirrors ``process_source``'s return so the existing summary code
    in build.py reads it via ``.get("test", (0.0, 0))`` without modification.
    """
    print(f"\n{'─' * 60}")
    print(f"  {entry.display_name}  [{entry.cls} → {entry.metadata_subclass}]")
    print(f"  source: {entry.file}")
    if smoke:
        print("  target: smoke mode (single recording)")
    print(f"{'─' * 60}")

    try:
        audio = load_recording(entry)
    except Exception as e:
        print(f"  [FAIL] load_recording: {e}", file=sys.stderr)
        return None

    duration_ms = int(len(audio) * 1000 / SR)

    if entry.labels is not None:
        # Hand-curated labels — clamp to the actual clip duration so an
        # off-by-one in the manifest doesn't poison the parquet schema.
        labels = []
        for lab in entry.labels:
            start = max(0, int(lab["start"]))
            end = min(duration_ms, int(lab["end"]))
            if end <= start:
                continue
            labels.append({"label": str(lab["label"]), "start": start, "end": end})
        if not labels:
            print(f"  [SKIP] {entry.name}: all hand-supplied labels were empty after clamping",
                  file=sys.stderr)
            return None
    else:
        labeler = get_labeler(entry.detector)
        labels = labeler.label(audio)
        if labels is None or len(labels) == 0:
            print(f"  [SKIP] {entry.name}: detector={entry.detector!r} returned no labels",
                  file=sys.stderr)
            return None

    clip_min = labels[-1]["end"] / 1000.0 / 60.0

    try:
        writer = writers[(entry.cls, "test")]
    except KeyError:
        print(f"  [FAIL] no writer registered for ({entry.cls}, 'test')", file=sys.stderr)
        return None

    writer.write(audio, labels, entry.name, 0, entry.cls, entry.metadata_subclass)
    print(f"  {clip_min:.2f} min written  (labels: {len(labels)} segment(s))")

    return {"test": (clip_min, 1)}
