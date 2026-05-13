# exp/tcn_ablation/cli.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

import logging
from pathlib import Path

import click
import torch

from ...nn.tcn.config import (
    extract_overrides,
    get_ablation_config,
    get_config,
    get_preprocess_stats_path,
    get_weights_path,
    list_ablations,
)
from ...nn.tcn.evaluation import eval_tcn
from ...nn.tcn.model import SpeechMusicDetector
from ...nn.tcn.training import train_tcn

log = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).parent / "config.toml"
_results_dir = Path(__file__).resolve().parent / "results"
_EXP_SUBDIR = "tcn_ablation"


def _load(name: str | None = None, subgroup: str | None = None):
    """Return (cfg, variant_name, weights_path, stats_path) for an ablation variant."""
    if name:
        cfg = get_ablation_config(name, config_path=_CFG_PATH, subgroup=subgroup)
    else:
        cfg = get_config(_CFG_PATH)
    variant_name = cfg.get("name", "ablation")
    return (
        cfg,
        variant_name,
        get_weights_path(name=variant_name, subdir=_EXP_SUBDIR),
        get_preprocess_stats_path(name=variant_name, subdir=_EXP_SUBDIR),
    )


@click.group("tcn-ablation")
def tcn_ablation_group():
    """TCN ablation experiments — subgroups and variants defined in src/exp/tcn_ablation/config.toml."""
    pass


@tcn_ablation_group.command("train")
@click.option("--name", "-n", default=None, help="Variant name (e.g. adam, filters_8)")
@click.option("--subgroup", "-s", default=None, help="Subgroup (optimizer, capacity, depth, regularization, training)")
def train_cmd(name, subgroup):
    """Train a single experiment variant."""
    cfg, _, weights_path, stats_path = _load(name, subgroup)
    train_tcn(cfg=cfg, weights_path=weights_path, stats_path=stats_path)


@tcn_ablation_group.command("eval")
@click.option("--name", "-n", default=None, help="Variant name")
@click.option("--subgroup", "-s", default=None, help="Subgroup name")
def eval_cmd(name, subgroup):
    """Evaluate a single experiment variant on the test split."""
    cfg, variant_name, weights_path, stats_path = _load(name, subgroup)
    eval_tcn(
        cfg=cfg,
        weights_path=weights_path,
        stats_path=stats_path,
        output_name=f"tcn_{variant_name}",
    )


@tcn_ablation_group.command("smoke-test")
@click.option("--name", "-n", default=None, help="Variant name (default: base config)")
@click.option("--subgroup", "-s", default=None)
def smoke_test_cmd(name, subgroup):
    """Smoke test: forward pass with synthetic audio (no weights or dataset needed)."""
    cfg, variant_name, _, stats_path = _load(name, subgroup)

    model = SpeechMusicDetector(cfg=cfg, stats_path=stats_path)
    nf = model.fe.n_features
    model.fe.norm_mean = torch.zeros(1, nf, 1)
    model.fe.norm_std = torch.ones(1, nf, 1)

    dummy = torch.randn(2, cfg["sample_rate"] * 3)
    probs = model(dummy)

    assert probs.shape[0] == 2 and probs.shape[1] == cfg["model"]["n_classes"], (
        f"Unexpected output shape: {probs.shape}"
    )
    assert 0 <= probs.min().item() <= 1 and 0 <= probs.max().item() <= 1
    log.info("Smoke-test passed. Variant: %s  Output shape: %s", variant_name, tuple(probs.shape))


@tcn_ablation_group.command("ablation")
@click.option("--subgroup", "-s", default=None, help="Run only this subgroup (default: all)")
@click.option("--skip-existing/--no-skip-existing", default=False,
              help="Skip variants whose weights already exist (default: off — always retrain "
                   "from scratch to avoid stale cached baselines)")
def ablation_cmd(subgroup, skip_existing):
    """Train + eval all variants across all subgroups (or a single subgroup) sequentially.

    Shared variants (e.g. 'baseline') are trained and evaluated only once.
    """
    groups = list_ablations(_CFG_PATH)
    if subgroup:
        if subgroup not in groups:
            raise click.BadParameter(f"Unknown subgroup '{subgroup}'. Available: {list(groups)}")
        groups = {subgroup: groups[subgroup]}

    done: set[str] = set()
    results: dict[str, tuple[int, float]] = {}
    diverged: set[str] = set()
    total = sum(len(names) for names in groups.values())
    log.info("Running ablation: %d variant(s) across subgroup(s): %s", total, list(groups))

    import time

    for grp, names in groups.items():
        log.info("── Subgroup: %s ──", grp)
        for name in names:
            if name in done:
                log.info("[%s] already processed — skipping", name)
                continue
            weights = get_weights_path(name=name, subdir=_EXP_SUBDIR)
            stats = get_preprocess_stats_path(name=name, subdir=_EXP_SUBDIR)
            diverged_path = _results_dir / f"tcn_{name}.diverged"
            if skip_existing and weights.exists():
                log.info("[%s] weights exist — skipping train", name)
            else:
                log.info("[%s] training...", name)
                t0 = time.monotonic()
                cfg = get_ablation_config(name, config_path=_CFG_PATH, subgroup=grp or None)
                try:
                    train_tcn(cfg=cfg, weights_path=weights, stats_path=stats)
                    log.info("[%s] training done in %.1fs", name, time.monotonic() - t0)
                    if diverged_path.exists():
                        diverged_path.unlink()
                except Exception as exc:
                    log.error("[%s] training DIVERGED: %s", name, exc)
                    _results_dir.mkdir(parents=True, exist_ok=True)
                    diverged_path.write_text(f"DIVERGED\n{type(exc).__name__}: {exc}\n")
                    diverged.add(name)
                    done.add(name)
                    continue
            log.info("[%s] evaluating...", name)
            cfg = get_ablation_config(name, config_path=_CFG_PATH, subgroup=grp or None)
            res = eval_tcn(cfg=cfg, weights_path=weights, stats_path=stats,
                           output_name=f"tcn_{name}", output_dir=_results_dir)
            results[name] = (res.n_classes, res.f1)
            log.info("[%s] eval done — %d-class macro F1: %.4f", name, res.n_classes, res.f1)
            done.add(name)

    log.info("Ablation complete. %d variant(s) processed.", len(done))
    if results or diverged:
        log.info("── Results summary ──")
        log.info("  %-32s  %s", "variant", "macro F1")
        for n, (n_cls, f1) in results.items():
            log.info("  %-32s  %.4f  (%d-class)", n, f1, n_cls)
        for n in diverged:
            log.info("  %-32s  DIVERGED", n)


@tcn_ablation_group.command("coord-ascent")
@click.option("--skip-existing/--no-skip-existing", default=True,
              help="Skip variants whose weights already exist (default: on)")
def coord_ascent_cmd(skip_existing):
    """Greedy coordinate-ascent search: winner of each subgroup becomes the baseline for the next.

    Subgroup order is defined in [tcn.coord_ascent] in src/exp/tcn_ablation/config.toml.
    All file artifacts use a '_ca' suffix to avoid colliding with OFAT results.
    """
    import json
    import time
    import tomli

    with open(_CFG_PATH, "rb") as f:
        raw = tomli.load(f)
    subgroup_order = raw.get("tcn", {}).get("coord_ascent", {}).get("subgroup_order", [])
    if not subgroup_order:
        raise click.UsageError("No subgroup_order defined in [tcn.coord_ascent] in config.toml")

    all_groups = list_ablations(_CFG_PATH)
    missing = [g for g in subgroup_order if g not in all_groups]
    if missing:
        raise click.UsageError(f"Subgroups in coord_ascent.subgroup_order not found in ablations: {missing}")

    done: set[str] = set()
    results: dict[str, tuple[int, float]] = {}
    diverged: set[str] = set()
    carried: dict = {}
    trajectory: dict = {}

    log.info("Coordinate-ascent: %d subgroup(s) in order: %s", len(subgroup_order), subgroup_order)

    for grp in subgroup_order:
        names = all_groups[grp]
        log.info("── Subgroup: %s ──", grp)
        grp_results: dict[str, tuple[int, float]] = {}

        for name in names:
            if name in done:
                log.info("[%s] already processed — skipping", name)
                continue
            full_name = f"{name}_ca"
            weights = get_weights_path(name=full_name, subdir=_EXP_SUBDIR)
            stats = get_preprocess_stats_path(name=full_name, subdir=_EXP_SUBDIR)
            diverged_path = _results_dir / f"tcn_{full_name}.diverged"

            if skip_existing and weights.exists():
                log.info("[%s] weights exist — skipping train", full_name)
            else:
                log.info("[%s] training...", full_name)
                t0 = time.monotonic()
                cfg = get_ablation_config(name, config_path=_CFG_PATH, subgroup=grp,
                                          carried_overrides=carried or None)
                cfg["name"] = full_name
                try:
                    train_tcn(cfg=cfg, weights_path=weights, stats_path=stats)
                    log.info("[%s] training done in %.1fs", full_name, time.monotonic() - t0)
                    if diverged_path.exists():
                        diverged_path.unlink()
                except Exception as exc:
                    log.error("[%s] training DIVERGED: %s", full_name, exc)
                    _results_dir.mkdir(parents=True, exist_ok=True)
                    diverged_path.write_text(f"DIVERGED\n{type(exc).__name__}: {exc}\n")
                    diverged.add(name)
                    done.add(name)
                    continue

            log.info("[%s] evaluating...", full_name)
            cfg = get_ablation_config(name, config_path=_CFG_PATH, subgroup=grp,
                                      carried_overrides=carried or None)
            cfg["name"] = full_name
            res = eval_tcn(cfg=cfg, weights_path=weights, stats_path=stats,
                           output_name=f"tcn_{full_name}", output_dir=_results_dir)
            results[name] = (res.n_classes, res.f1)
            grp_results[name] = (res.n_classes, res.f1)
            log.info("[%s] eval done — %d-class macro F1: %.4f", full_name, res.n_classes, res.f1)
            done.add(name)

        converged = {n: grp_results[n] for n in names if n in grp_results}
        if converged:
            winner = max(converged, key=lambda n: converged[n][1])
            winner_cfg = get_ablation_config(winner, config_path=_CFG_PATH, subgroup=grp,
                                             carried_overrides=carried or None)
            carried = extract_overrides(winner_cfg)
            trajectory[grp] = {
                "winner": winner,
                "f1": converged[winner][1],
                "carried": carried,
            }
            log.info("[coord-ascent] %s winner: %s  F1=%.4f", grp, winner, converged[winner][1])
        else:
            log.warning("[coord-ascent] %s — no converged variants, carried config unchanged", grp)

    log.info("Coord-ascent complete. %d variant(s) processed.", len(done))
    if results or diverged:
        log.info("── Results summary ──")
        log.info("  %-36s  %s", "variant", "macro F1")
        for n, (n_cls, f1) in results.items():
            log.info("  %-36s  %.4f  (%d-class)", f"{n}_ca", f1, n_cls)
        for n in diverged:
            log.info("  %-36s  DIVERGED", f"{n}_ca")

    if trajectory:
        traj_path = _results_dir / "coord_ascent_trajectory.json"
        traj_path.write_text(json.dumps(trajectory, indent=2))
        log.info("Trajectory saved → %s", traj_path)


