"""Upload weights/, cache/, and results/*.npz (top-level + per-experiment) to HuggingFace.

Default mode is additive — files on HF that no longer exist locally are kept.
Pass ``--clean weights`` (and/or ``cache``, ``results``) to atomically delete
remote files in those folders that aren't in the local copy. Use after the
weights/ subdir migration to wipe the stale root-level ``tcn_*.safetensors``
duplicates before they sit alongside the new ``weights/<exp>/`` layout.

Examples:
  python scripts/upload_artifacts.py                           # additive upload
  python scripts/upload_artifacts.py --clean weights           # mirror weights/
  python scripts/upload_artifacts.py --clean weights --clean cache
  python scripts/upload_artifacts.py --clean weights --dry-run # preview only
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "Marek324/butfit-bp-artifacts"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent.parent

# Folders where ``--clean`` is supported. Per-experiment results dirs are
# scoped by sub-path; only the top-level keys above can be cleaned.
CLEAN_TARGETS = {"weights", "cache", "results"}


def _folders() -> list[tuple[Path, str, list[str] | None]]:
    entries: list[tuple[Path, str, list[str] | None]] = [
        (ROOT / "weights", "weights", None),
        (ROOT / "cache", "cache", None),
        (ROOT / "results", "results", ["*.npz"]),
    ]
    for exp_results in sorted((ROOT / "src" / "exp").glob("*/results")):
        rel = exp_results.relative_to(ROOT).as_posix()
        entries.append((exp_results, rel, ["*.npz"]))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--clean",
        action="append",
        default=[],
        choices=sorted(CLEAN_TARGETS),
        help="Folder to mirror — remote files not present locally are deleted "
             "atomically with the upload. Repeat to clean multiple folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without uploading or deleting.",
    )
    args = parser.parse_args()

    clean_set = set(args.clean)

    api = HfApi()
    if not args.dry_run:
        create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    print(f"Repo: https://huggingface.co/{REPO_ID}")
    if clean_set:
        print(f"Clean mode active for: {sorted(clean_set)} (remote-only files in those folders will be deleted)")

    for local, path_in_repo, allow_patterns in _folders():
        if not local.exists() or not any(local.iterdir()):
            print(f"  {path_in_repo}/  — empty or missing, skipping")
            continue

        # Only clean if the *top-level* segment matches the user's --clean targets.
        # Per-experiment results dirs (src/exp/*/results) are not eligible.
        top = path_in_repo.split("/", 1)[0]
        clean_this = top in clean_set and "/" not in path_in_repo
        delete_patterns = ["**"] if clean_this else None

        action = "Mirror-uploading" if clean_this else "Uploading"
        print(f"  {action} {path_in_repo}/ …")
        if args.dry_run:
            print(f"    (dry-run) delete_patterns={delete_patterns}, allow_patterns={allow_patterns}")
            continue
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=allow_patterns,
            delete_patterns=delete_patterns,
        )
        print(f"  Done: {path_in_repo}/")

    print("Upload complete." if not args.dry_run else "Dry run complete.")


if __name__ == "__main__":
    sys.exit(main())
