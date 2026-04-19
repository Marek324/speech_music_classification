"""One-shot: move existing weights/tcn_<name>.* into per-experiment subdirs.

Why: until now every experiment wrote to ``weights/tcn_<variant>.safetensors``
without an experiment scope, so a "baseline" variant in tcn_hybrid would
overwrite (or get overwritten by) the same name in tcn_combined. The latter
bug actually broke a tcn_hybrid run when it inherited 80-channel preprocess
stats from a tcn_combined log_mel baseline, while its own delta2 frontend
expected 240 channels.

Mapping rules:
  - For each variant name owned by exactly one experiment → move the weight
    + stats file into ``weights/<exp>/`` (preserving filename).
  - "baseline" is shared across 5 experiments → move to a single owner
    (tcn_combined; it's the most recent baseline trainer per file mtime, so
    its eval reproduces). Other experiments' baselines need to be retrained.
  - Files at root with no experiment owner (tcn.safetensors, tcn.smoke.*,
    tcn_preprocess_stats.pt) stay at root — those belong to the production
    ``nn tcn train/eval`` flow, not an experiment.
  - Files for variants that no longer appear in any experiment config (e.g.
    historical ablations like ``filters_8``) get routed to tcn_ablation/
    since that's where they originated.

Run: ``uv run python scripts/migrate_weights_to_subdirs.py [--dry-run]``
"""

import argparse
import shutil
import sys
import tomli
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
EXP_DIR = ROOT / "src" / "exp"

# Files that stay at root (not owned by any experiment).
KEEP_AT_ROOT = {
    "tcn.safetensors",
    "tcn.smoke.safetensors",
    "tcn_preprocess_stats.pt",
}

# Experiment that owns "baseline" when collision (per user decision (b)).
BASELINE_OWNER = "tcn_combined"

# Fallback experiment for orphaned variants (historical ablations no longer in config).
ORPHAN_FALLBACK = "tcn_ablation"


def _load_variants_for_experiment(exp_dir: Path) -> set[str]:
    """Return the set of variant names declared in an experiment's config.toml.

    Handles both flat ``[tcn.variants.<name>]`` (combined/frontend/architecture/
    preprocessor/hybrid) and nested ``[tcn.ablations.<subgroup>.<name>]`` (ablation).
    """
    cfg = exp_dir / "config.toml"
    if not cfg.exists():
        return set()
    raw = tomli.load(cfg.open("rb"))
    tcn = raw.get("tcn", {})
    variants: set[str] = set()
    for k, v in tcn.get("variants", {}).items():
        variants.add(k)
    for sg, group in tcn.get("ablations", {}).items():
        if isinstance(group, dict):
            for k in group:
                if isinstance(group[k], dict):
                    variants.add(k)
    return variants


def _build_variant_to_experiment() -> dict[str, str]:
    """Map each variant name → the experiment that owns it.

    Collisions are resolved by:
      - "baseline" → BASELINE_OWNER (per user decision)
      - other collisions → first experiment encountered (sorted, deterministic)
    """
    mapping: dict[str, list[str]] = {}
    for exp in sorted(EXP_DIR.iterdir()):
        if not exp.is_dir():
            continue
        for variant in _load_variants_for_experiment(exp):
            mapping.setdefault(variant, []).append(exp.name)

    resolved: dict[str, str] = {}
    for variant, owners in mapping.items():
        if variant == "baseline":
            resolved[variant] = BASELINE_OWNER
        elif len(owners) == 1:
            resolved[variant] = owners[0]
        else:
            # Other collisions: take the first deterministically. None observed today.
            resolved[variant] = sorted(owners)[0]
    return resolved


def _strip_tcn_prefix_and_suffix(filename: str) -> str | None:
    """Extract the variant name from a weights/stats filename, or None if not a variant file.

    Examples:
      tcn_filters_8.safetensors             -> "filters_8"
      tcn_filters_8_preprocess_stats.pt     -> "filters_8"
      tcn_baseline.safetensors              -> "baseline"
      tcn.safetensors                       -> None  (production, no variant)
      tcn_preprocess_stats.pt               -> None  (production)
    """
    if not filename.startswith("tcn_"):
        return None
    stem = filename[len("tcn_"):]
    if stem.endswith(".safetensors"):
        return stem[: -len(".safetensors")]
    if stem.endswith("_preprocess_stats.pt"):
        return stem[: -len("_preprocess_stats.pt")]
    return None


def _plan_moves() -> list[tuple[Path, Path, str]]:
    """Return list of (source, destination, reason) tuples for files to move."""
    variant_to_exp = _build_variant_to_experiment()
    moves: list[tuple[Path, Path, str]] = []

    for entry in sorted(WEIGHTS.iterdir()):
        if entry.is_dir():
            continue  # Already under a subdir from a prior run, skip.
        if entry.name in KEEP_AT_ROOT:
            continue

        variant = _strip_tcn_prefix_and_suffix(entry.name)
        if variant is None:
            continue  # Not a variant weights/stats file; leave alone.

        owner = variant_to_exp.get(variant)
        reason = "config-mapped"
        if owner is None:
            owner = ORPHAN_FALLBACK
            reason = f"orphaned variant — fallback to {ORPHAN_FALLBACK}"
        if variant == "baseline":
            reason = f"shared baseline → {BASELINE_OWNER} (user decision: option b)"

        dest = WEIGHTS / owner / entry.name
        moves.append((entry, dest, reason))

    return moves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Show planned moves without doing them")
    args = parser.parse_args()

    if not WEIGHTS.exists():
        print(f"No weights/ at {WEIGHTS} — nothing to migrate.")
        return

    moves = _plan_moves()
    if not moves:
        print("No files to migrate (all weights already under per-experiment subdirs).")
        return

    print(f"Planned moves: {len(moves)}")
    for src, dst, reason in moves:
        print(f"  {src.name:60s} -> {dst.relative_to(ROOT)}  ({reason})")

    if args.dry_run:
        print("\nDry run — nothing moved. Drop --dry-run to execute.")
        return

    print()
    moved = 0
    for src, dst, _ in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"  SKIP  {src.name}: destination {dst.relative_to(ROOT)} already exists")
            continue
        shutil.move(str(src), str(dst))
        moved += 1
    print(f"\nMoved {moved} of {len(moves)} files.")


if __name__ == "__main__":
    sys.exit(main())
