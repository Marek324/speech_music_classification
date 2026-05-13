# scripts/dataset/dataloader.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

"""HuggingFace source loading for the dataset build pipeline."""

from datasets import Audio, IterableDataset, load_dataset
from source_config import RAND_SEED, SourceEntry

from labeling import SR


def load_source(entry: SourceEntry, max_rows: int, skip: int = 0) -> IterableDataset:
    """Stream one HF source, applying optional filter, skip, and audio decode settings."""
    col = entry.audio_col
    ds = load_dataset(entry.hf_id, split=entry.split, streaming=True, **entry.hf_load_kwargs())
    assert isinstance(ds, IterableDataset)
    ds = ds.shuffle(seed=RAND_SEED, buffer_size=max_rows)
    if entry.filter_col and entry.filter_val is not None:
        if not entry.audio_decode:
            ds = ds.cast_column(col, Audio(sampling_rate=SR, num_channels=1, decode=False))
        fcol, val, include = entry.filter_col, entry.filter_val, entry.filter_include
        if isinstance(val, (list, tuple)):
            ds = ds.filter(lambda row, _col=fcol, _val=val, _inc=include:
                any(tag in (row[_col] or "") for tag in _val) == _inc)
        else:
            ds = ds.filter(lambda row, _col=fcol, _val=val, _inc=include:
                (row[_col] == _val) == _inc)
    if skip > 0:
        ds = ds.skip(skip)
    ds = ds.take(max_rows)
    ds = ds.select_columns(col)
    ds = ds.cast_column(col, Audio(sampling_rate=SR, num_channels=1, decode=entry.audio_decode))
    return ds
