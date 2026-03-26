"""HuggingFace source loading for the dataset build pipeline."""

from datasets import Audio, IterableDataset, load_dataset
from source_config import RAND_SEED, SourceEntry

from labeling import SR


def load_source(entry: SourceEntry, max_rows: int, skip: int = 0) -> IterableDataset:
    col = entry.audio_col
    ds = load_dataset(entry.hf_id, split=entry.split, streaming=True, **entry.hf_load_kwargs())
    assert isinstance(ds, IterableDataset)
    # buffer_size=max_rows: default 1000 would buffer up to 9.6 GB for DEMAND
    # (which stores ~9.6 MB embedded bytes per row) before yielding anything.
    # Capping at max_rows avoids downloading far more than we'll ever consume.
    ds = ds.shuffle(seed=RAND_SEED, buffer_size=max_rows)
    if entry.filter_col and entry.filter_val is not None:
        # datasets 4.8+: cast_column(decode=False) applied *after* filter()
        # returns AudioDecoder instead of the raw {"bytes","path"} dict.
        # Pre-cast only for decode=False sources that have a filter.
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
