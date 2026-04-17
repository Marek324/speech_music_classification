"""Download weights/, cache/, and results/*.npz (top-level + per-experiment) from HuggingFace.

Run after cloning. `snapshot_download` mirrors the repo tree under ROOT, so any
`src/exp/*/results/*.npz` uploaded by `upload_artifacts.py` lands at the same local path.
Pass `--no-cache` to skip the large `cache/` tree.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Marek324/butfit-bp-artifacts"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-cache", action="store_true", help="Skip downloading cache/")
    args = parser.parse_args()

    ignore_patterns = ["*.gitattributes", "README.md", ".gitignore"]
    if args.no_cache:
        ignore_patterns.append("cache/*")

    suffix = " (skipping cache/)" if args.no_cache else ""
    print(f"Downloading artifacts from {REPO_ID}{suffix} …")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(ROOT),
        ignore_patterns=ignore_patterns,
    )
    ready = "weights/ and results/" if args.no_cache else "weights/, cache/, and results/"
    print(f"Done. {ready} are ready.")


if __name__ == "__main__":
    sys.exit(main())
