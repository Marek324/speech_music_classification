"""Upload dataset tiers to HuggingFace Hub.

Normal:  uv run python upload.py                # → Marek324/speech-music-classification
Smoke:   uv run python upload.py --smoke        # → Marek324/speech-music-classification-test
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "Marek324/speech-music-classification"
STAGING_ROOT = Path(__file__).resolve().parent.parent.parent / "speech_music_dataset"
README_PATH = Path(__file__).resolve().parent / "HUB_DATASET_README.md"
TIERS = ("mini", "mid", "full")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Upload smoke builds (*_smoke dirs) to a separate -test repo",
    )
    args = parser.parse_args()

    repo_id = REPO_ID + ("-test" if args.smoke else "")
    suffix = "_smoke" if args.smoke else ""

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print(f"Repo: https://huggingface.co/datasets/{repo_id}")

    for tier in TIERS:
        local = STAGING_ROOT / f"{tier}{suffix}"
        if not local.exists():
            print(f"  [{tier}] {local} not found — skipping")
            continue
        print(f"  Uploading {local.name}/ → {tier}/ ...")
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=tier,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  [{tier}] done")

    print("  Uploading README.md ...")
    api.upload_file(
        path_or_fileobj=str(README_PATH),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("Upload complete.")


if __name__ == "__main__":
    sys.exit(main())
