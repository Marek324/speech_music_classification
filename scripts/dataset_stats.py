"""Compute per-subclass audio minutes for the HF dataset and dump a formatted report.

Usage:
    uv run python scripts/dataset_stats.py                  # full tier, all splits
    uv run python scripts/dataset_stats.py --name mid
    uv run python scripts/dataset_stats.py --splits train validation

Derives clip duration from the last label end timestamp (no audio decode).
Writes to results/dataset_stats.txt.
"""

import argparse
import tomllib
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.toml"
OUT_PATH = REPO_ROOT / "results" / "dataset_stats.txt"


def _load_dataset_cfg() -> tuple[str, str]:
    with CONFIG_PATH.open("rb") as f:
        cfg = tomllib.load(f)
    ds = cfg.get("dataset", {})
    return ds["url"], ds.get("name", "full")


def _clip_seconds(labels) -> float:
    """Return clip duration in seconds from the last label end timestamp (ms)."""
    if labels is None:
        return 0.0
    if isinstance(labels, dict):
        ends = labels.get("end") or []
    else:
        ends = [entry.get("end", 0) for entry in labels]
    if not ends:
        return 0.0
    return max(ends) / 1000.0


def collect(repo: str, name: str, splits: list[str]):
    per_split: dict[str, dict] = {}
    for split in splits:
        ds = load_dataset(repo, name=name, split=split, streaming=True)
        if "audio" in (ds.column_names or []):
            ds = ds.remove_columns(["audio"])

        sub_sec: dict[str, float] = defaultdict(float)
        sub_cnt: dict[str, int] = defaultdict(int)
        cls_sec: dict[str, float] = defaultdict(float)
        total_sec = 0.0
        total_cnt = 0

        for row in tqdm(ds, desc=split):
            secs = _clip_seconds(row.get("labels"))
            if secs <= 0:
                continue
            cls = row.get("class") or "?"
            sub = row.get("subclass") or cls
            sub_sec[sub] += secs
            sub_cnt[sub] += 1
            cls_sec[cls] += secs
            total_sec += secs
            total_cnt += 1

        per_split[split] = {
            "sub_sec": dict(sub_sec),
            "sub_cnt": dict(sub_cnt),
            "cls_sec": dict(cls_sec),
            "total_sec": total_sec,
            "total_cnt": total_cnt,
        }
    return per_split


def _fmt_minutes(sec: float) -> str:
    return f"{sec / 60:8.2f}"


def render(repo: str, name: str, per_split: dict[str, dict]) -> str:
    lines: list[str] = []
    title = f"Dataset stats — {repo} (config: {name})"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    grand_total = sum(s["total_sec"] for s in per_split.values())
    grand_clips = sum(s["total_cnt"] for s in per_split.values())
    lines.append(f"Total clips:    {grand_clips:>10,}")
    lines.append(f"Total duration: {grand_total / 60:>10.2f} min ({grand_total / 3600:.2f} h)")
    lines.append("")

    lines.append("Per split")
    lines.append("-" * 60)
    lines.append(f"{'SPLIT':<14}{'CLIPS':>10}{'MINUTES':>14}{'HOURS':>12}")
    for split, s in per_split.items():
        lines.append(
            f"{split:<14}{s['total_cnt']:>10,}{s['total_sec'] / 60:>14.2f}{s['total_sec'] / 3600:>12.2f}"
        )
    lines.append("")

    all_subs = sorted({sub for s in per_split.values() for sub in s["sub_sec"]})
    all_cls = sorted({c for s in per_split.values() for c in s["cls_sec"]})
    split_names = list(per_split.keys())

    lines.append("Per class (minutes)")
    lines.append("-" * 60)
    header = f"{'CLASS':<16}" + "".join(f"{sp.upper():>12}" for sp in split_names) + f"{'TOTAL':>12}"
    lines.append(header)
    for cls in all_cls:
        row = f"{cls:<16}"
        cls_total = 0.0
        for sp in split_names:
            v = per_split[sp]["cls_sec"].get(cls, 0.0)
            cls_total += v
            row += f"{v / 60:>12.2f}"
        row += f"{cls_total / 60:>12.2f}"
        lines.append(row)
    lines.append("")

    lines.append("Per subclass (minutes)")
    lines.append("-" * 72)
    header = (
        f"{'SUBCLASS':<28}"
        + "".join(f"{sp.upper():>12}" for sp in split_names)
        + f"{'TOTAL':>12}"
    )
    lines.append(header)
    for sub in all_subs:
        row = f"{sub:<28}"
        total = 0.0
        for sp in split_names:
            v = per_split[sp]["sub_sec"].get(sub, 0.0)
            total += v
            row += f"{v / 60:>12.2f}"
        row += f"{total / 60:>12.2f}"
        lines.append(row)
    lines.append("")

    lines.append("Per subclass (clip counts)")
    lines.append("-" * 72)
    lines.append(
        f"{'SUBCLASS':<28}"
        + "".join(f"{sp.upper():>12}" for sp in split_names)
        + f"{'TOTAL':>12}"
    )
    for sub in all_subs:
        row = f"{sub:<28}"
        total = 0
        for sp in split_names:
            v = per_split[sp]["sub_cnt"].get(sub, 0)
            total += v
            row += f"{v:>12,}"
        row += f"{total:>12,}"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    default_repo, default_name = _load_dataset_cfg()

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=default_repo, help=f"HF repo (default from config.toml: {default_repo})")
    parser.add_argument("--name", default=default_name, help=f"HF dataset config (default from config.toml: {default_name})")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to scan (default: train validation test)",
    )
    args = parser.parse_args()

    per_split = collect(args.repo, args.name, args.splits)
    report = render(args.repo, args.name, per_split)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report)
    print(report)
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
