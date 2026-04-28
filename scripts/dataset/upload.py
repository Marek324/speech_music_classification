"""Upload dataset tiers to HuggingFace Hub.

Normal:  uv run python upload.py                # → Marek324/speech-music-classification
Smoke:   uv run python upload.py --smoke        # → Marek324/speech-music-classification-test
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "Marek324/speech-music-classification"
STAGING_ROOT = Path(__file__).resolve().parent / "speech_music_dataset"
README_PATH = Path(__file__).resolve().parent / "HUB_DATASET_README.md"
TIERS = ("mid", "full", "crit")


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

    # Resolve each tier's local dir up-front. README references all configs in TIERS,
    # so uploading with any tier missing would leave a config that points at no files
    # (HF dataset viewer then errors out with SplitsNotFoundError). For --smoke runs,
    # fall back to the non-smoke dir if the smoke variant doesn't exist — this is the
    # natural case for crit, which has no separate _smoke build.
    locals_to_push: list[tuple[str, Path]] = []
    missing: list[str] = []
    for tier in TIERS:
        primary = STAGING_ROOT / f"{tier}{suffix}"
        fallback = STAGING_ROOT / tier
        if primary.exists():
            locals_to_push.append((tier, primary))
        elif suffix and fallback.exists():
            print(
                f"  [{tier}] {primary.name}/ not found — falling back to {fallback.name}/"
            )
            locals_to_push.append((tier, fallback))
        else:
            missing.append(f"{tier} (looked at {primary})")
    if missing:
        raise SystemExit(
            "  refusing to upload — the following tiers have no local data:\n    "
            + "\n    ".join(missing)
            + "\n  build them first or drop them from TIERS in upload.py."
        )

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print(f"Repo: https://huggingface.co/datasets/{repo_id}")

    for tier, local in locals_to_push:
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
