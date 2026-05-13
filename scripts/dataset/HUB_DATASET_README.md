---
pretty_name: Speech / Music classification
license: unknown
language: en
tags:
  - audio
  - parquet

configs:
  - config_name: full
    data_files:
      - split: train
        path: full/train/**/*.parquet
      - split: validation
        path: full/validation/**/*.parquet
      - split: test
        path: full/test/**/*.parquet
  - config_name: mid
    data_files:
      - split: train
        path: mid/train/**/*.parquet
      - split: validation
        path: mid/validation/**/*.parquet
      - split: test
        path: mid/test/**/*.parquet
  - config_name: crit
    data_files:
      - split: test
        path: crit/test/**/*.parquet
---

## Tiers

- **mid** — 1200 min target (40% speech / 40% music / 20% background), including synthetic augmentations.
- **full** — 6000 min target, same mix.
- **crit** — small hand-curated stress-test set (test split only) sourced from `scripts/dataset/crit/manifest.toml`.

## Structure

After `uv run python build.py {mid|full}` for each tier (same `--out-dir` staging parent):

```text
<staging>/
  full/   # or mid/
    train/
      speech/       part_*.parquet
      music/
      background/
    validation/
      speech/
      ...
    test/
      ...
```

Parquet columns: `audio` (HF Audio struct), `sampling_rate`, `class`, `subclass`, `source`, `row_idx`, `labels` (list of `{label, start, end}` in ms).

## Upload

From the **staging parent** (folder that contains `mid/`, `full/`):

```bash
hf upload-large-folder --repo-type dataset YOUR_ORG/YOUR_REPO .
```

## Load

```python
from datasets import load_dataset

ds = load_dataset("YOUR_ORG/YOUR_REPO", "full", split="train")
# Rows include columns from parquet; decode WAV from bytes if needed.
```

Adjust `YOUR_ORG/YOUR_REPO` and verify YAML against current [Hub dataset data files](https://huggingface.co/docs/hub/datasets-data-files-configuration) if `load_dataset` errors.
