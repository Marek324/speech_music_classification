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
      train:
        - full/train/**/*.parquet
      validation:
        - full/validation/**/*.parquet
      test:
        - full/test/**/*.parquet
  - config_name: mid
    data_files:
      train:
        - mid/train/**/*.parquet
      validation:
        - mid/validation/**/*.parquet
      test:
        - mid/test/**/*.parquet
  - config_name: mini
    data_files:
      train:
        - mini/train/**/*.parquet
      validation:
        - mini/validation/**/*.parquet
      test:
        - mini/test/**/*.parquet
---

## Structure

After `uv run python build.py {mini|mid|full}` for each tier (same `--out-dir` staging parent):

```text
<staging>/
  full/   # or mid/, mini/
    train/
      speech/       part_*.parquet
      music/
      inactive/
    validation/
      speech/
      ...
    test/
      ...
```

Parquet columns: `audio_wav`, `sampling_rate`, `class`, `subclass`, `source`, `row_idx`, `labels_json`.

## Upload

From the **staging parent** (folder that contains `mini/`, `mid/`, `full/`):

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
