# exp/tcn_ablation/cli.py
# Thin CLI wrapper — all logic lives in src/nn/tcn/.
# The experiment is fully defined by config.toml in this directory.

import logging
from pathlib import Path

import click
import torch

from ...nn.tcn.config import (
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


def _load(name: str | None = None, subgroup: str | None = None):
    if name:
        cfg = get_ablation_config(name, config_path=_CFG_PATH, subgroup=subgroup)
    else:
        cfg = get_config(_CFG_PATH)
    variant_name = cfg.get("name", "ablation")
    return cfg, variant_name, get_weights_path(name=variant_name), get_preprocess_stats_path(name=variant_name)


@click.group("tcn-ablation")
def tcn_ablation_group():
    """TCN ablation experiments — subgroups and variants defined in src/exp/tcn_ablation/config.toml."""
    pass


@tcn_ablation_group.command("train")
@click.option("--name", "-n", default=None, help="Variant name (e.g. adam, filters_8)")
@click.option("--subgroup", "-s", default=None, help="Subgroup (optimizer, capacity, depth, regularization, training)")
@click.option("--no-wandb", is_flag=True, default=False, help="Disable W&B logging")
def train_cmd(name, subgroup, no_wandb):
    """Train a single experiment variant."""
    cfg, _, weights_path, stats_path = _load(name, subgroup)
    train_tcn(use_wandb=not no_wandb, cfg=cfg, weights_path=weights_path, stats_path=stats_path)


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
    n_mels = cfg["n_mels"]

    model = SpeechMusicDetector(cfg=cfg, stats_path=stats_path)
    model.fe.norm_mean = torch.zeros(1, n_mels, 1)
    model.fe.norm_std = torch.ones(1, n_mels, 1)

    dummy = torch.randn(2, cfg["sample_rate"] * 3)
    probs = model(dummy)

    assert probs.shape[0] == 2 and probs.shape[1] == cfg["model"]["n_classes"], (
        f"Unexpected output shape: {probs.shape}"
    )
    assert 0 <= probs.min().item() <= 1 and 0 <= probs.max().item() <= 1
    log.info("Smoke-test passed. Variant: %s  Output shape: %s", variant_name, tuple(probs.shape))


@tcn_ablation_group.command("ablation")
@click.option("--subgroup", "-s", default=None, help="Run only this subgroup (default: all)")
@click.option("--skip-existing/--no-skip-existing", default=True,
              help="Skip variants whose weights already exist (default: on)")
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
    results: dict[str, tuple[int, float]] = {}  # name -> (n_classes, macro_f1)
    total = sum(len(names) for names in groups.values())
    log.info("Running ablation: %d variant(s) across subgroup(s): %s", total, list(groups))

    import time

    for grp, names in groups.items():
        log.info("── Subgroup: %s ──", grp)
        for name in names:
            if name in done:
                log.info("[%s] already processed — skipping", name)
                continue
            weights = get_weights_path(name=name)
            if skip_existing and weights.exists():
                log.info("[%s] weights exist — skipping train", name)
            else:
                log.info("[%s] training...", name)
                t0 = time.monotonic()
                cfg = get_ablation_config(name, config_path=_CFG_PATH, subgroup=grp or None)
                train_tcn(cfg=cfg)
                log.info("[%s] training done in %.1fs", name, time.monotonic() - t0)
            log.info("[%s] evaluating...", name)
            cfg = get_ablation_config(name, config_path=_CFG_PATH, subgroup=grp or None)
            stats = get_preprocess_stats_path(name=name)
            res = eval_tcn(cfg=cfg, weights_path=weights, stats_path=stats, output_name=f"tcn_{name}")
            results[name] = (res.n_classes, res.f1)
            log.info("[%s] eval done — %d-class macro F1: %.4f", name, res.n_classes, res.f1)
            done.add(name)

    log.info("Ablation complete. %d variant(s) processed.", len(done))
    if results:
        log.info("── Results summary ──")
        log.info("  %-32s  %s", "variant", "macro F1")
        for n, (n_cls, f1) in results.items():
            log.info("  %-32s  %.4f  (%d-class)", n, f1, n_cls)


@tcn_ablation_group.command("visualize")
@click.option("--subgroup", "-s", default=None, help="Plot only this subgroup (default: all + summary)")
def visualize_cmd(subgroup):
    """Plot ablation results grouped by subgroup. Reads results/tcn_<name>.eval files."""
    import importlib.util
    import sys

    viz_path = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "visualize_ablation.py"
    spec = importlib.util.spec_from_file_location("visualize_ablation", viz_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(subgroup_filter=subgroup)
