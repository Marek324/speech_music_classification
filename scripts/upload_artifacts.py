"""Upload weights/, cache/, and results/*.npz (top-level + per-experiment) to HuggingFace."""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "Marek324/butfit-bp-artifacts"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent.parent


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
    api = HfApi()

    create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    print(f"Repo: https://huggingface.co/{REPO_ID}")

    for local, path_in_repo, allow_patterns in _folders():
        if not local.exists() or not any(local.iterdir()):
            print(f"  {path_in_repo}/  — empty or missing, skipping")
            continue

        print(f"  Uploading {path_in_repo}/ …")
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=allow_patterns,
        )
        print(f"  Done: {path_in_repo}/")

    print("Upload complete.")


if __name__ == "__main__":
    sys.exit(main())
