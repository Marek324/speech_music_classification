"""Upload weights/ and cache/ to HuggingFace."""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "Marek324/butfit-bp-artifacts"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    api = HfApi()

    create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    print(f"Repo: https://huggingface.co/{REPO_ID}")

    for folder in ("weights", "cache"):
        local = ROOT / folder
        if not local.exists() or not any(local.iterdir()):
            print(f"  {folder}/  — empty or missing, skipping")
            continue

        print(f"  Uploading {folder}/ …")
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=folder,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"  Done: {folder}/")

    print("Upload complete.")


if __name__ == "__main__":
    sys.exit(main())
