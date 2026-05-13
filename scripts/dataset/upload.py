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
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=TIERS,
        default=None,
        help=(
            "Restrict upload to a subset of tiers (default: all of "
            f"{list(TIERS)}). README is uploaded only when every TIER is included, "
            "since it advertises configs that may otherwise point at missing files."
        ),
    )
    args = parser.parse_args()

    repo_id = REPO_ID + ("-test" if args.smoke else "")
    suffix = "_smoke" if args.smoke else ""
    tiers_to_check = tuple(args.tiers) if args.tiers else TIERS

    # Resolve each tier's local dir up-front. README references all configs in TIERS,
    # so uploading with any tier missing would leave a config that points at no files
    # (HF dataset viewer then errors out with SplitsNotFoundError). For --smoke runs,
    # fall back to the non-smoke dir if the smoke variant doesn't exist — this is the
    # natural case for crit, which has no separate _smoke build.
    locals_to_push: list[tuple[str, Path]] = []
    missing: list[str] = []
    for tier in tiers_to_check:
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
        upload_kwargs = dict(
            folder_path=str(local),
            path_in_repo=tier,
            repo_id=repo_id,
            repo_type="dataset",
        )
        if tier == "crit":
            # Crit is small and always rebuilt from manifest — clear stale
            # remote shards before upload to avoid duplicate accumulation.
            upload_kwargs["delete_patterns"] = "*"
        api.upload_folder(**upload_kwargs)
        print(f"  [{tier}] done")

    if set(tiers_to_check) == set(TIERS):
        print("  Uploading README.md ...")
        api.upload_file(
            path_or_fileobj=str(README_PATH),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    else:
        print("  Skipping README.md (partial-tier upload — README left as-is on Hub).")

    print("Upload complete.")


if __name__ == "__main__":
    sys.exit(main())
