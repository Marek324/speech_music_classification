"""Download weights/ and cache/ from HuggingFace. Run after cloning."""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Marek324/butfit-bp-artifacts"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    print(f"Downloading artifacts from {REPO_ID} …")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(ROOT),
        ignore_patterns=["*.gitattributes", "README.md", ".gitignore"],
    )
    print("Done. weights/ and cache/ are ready.")


if __name__ == "__main__":
    sys.exit(main())
