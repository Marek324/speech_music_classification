# exp/small_tcn_pareto/cli.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

import logging
from pathlib import Path

import click
import torch

from ...nn.tcn.config import (
    get_config,
    get_preprocess_stats_path,
    get_weights_path,
    _TOP_KEYS,
    _MODEL_KEYS,
)
from ...nn.tcn.evaluation import eval_tcn
from ...nn.tcn.model import SpeechMusicDetector
from ...nn.tcn.training import train_tcn

log = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).parent / "config.toml"
_results_dir = Path(__file__).resolve().parent / "results"
_EXP_SUBDIR = "small_tcn_pareto"



def _list_variants() -> list[str]:
    """Return variant names from [tcn.variants] in the experiment config."""
    import tomli
    with open(_CFG_PATH, "rb") as f:
        raw = tomli.load(f)
    return list(raw.get("tcn", {}).get("variants", {}).keys())


def _get_variant_config(name: str) -> dict:
    """Load base config from experiment config.toml, then apply variant overrides."""
    import tomli
    cfg = get_config(_CFG_PATH)
    with open(_CFG_PATH, "rb") as f:
        raw = tomli.load(f)
    overrides = raw.get("tcn", {}).get("variants", {}).get(name, {})
    for k in _TOP_KEYS:
        if k in overrides:
            cfg[k] = overrides[k]
    cfg["model"] = {**cfg["model"], **{k: overrides[k] for k in _MODEL_KEYS if k in overrides}}
    cfg["name"] = name
    return cfg


def _load(name: str | None = None):
    """Return (cfg, variant_name, weights_path, stats_path) for a variant."""
    if name:
        cfg = _get_variant_config(name)
    else:
        cfg = get_config(_CFG_PATH)
    variant_name = cfg.get("name", "baseline")
    return (
        cfg,
        variant_name,
        get_weights_path(name=variant_name, subdir=_EXP_SUBDIR),
        get_preprocess_stats_path(name=variant_name, subdir=_EXP_SUBDIR),
    )



@click.group("small-tcn-pareto")
def small_tcn_pareto_group():
    """SmallTCN Pareto sweep — n_filters curve on delta2+no-preproc recipe."""
    pass


@small_tcn_pareto_group.command("train")
@click.option("--name", "-n", default=None, help="Variant name (e.g. filters_4)")
def train_cmd(name):
    """Train a single SmallTCN-pareto variant."""
    cfg, _, weights_path, stats_path = _load(name)
    train_tcn(cfg=cfg, weights_path=weights_path, stats_path=stats_path)


@small_tcn_pareto_group.command("eval")
@click.option("--name", "-n", default=None, help="Variant name")
def eval_cmd(name):
    """Evaluate a single SmallTCN-pareto variant on the test split."""
    cfg, variant_name, weights_path, stats_path = _load(name)
    eval_tcn(
        cfg=cfg,
        weights_path=weights_path,
        stats_path=stats_path,
        output_name=f"tcn_{variant_name}",
        output_dir=_results_dir,
    )


@small_tcn_pareto_group.command("smoke-test")
@click.option("--name", "-n", default=None, help="Variant name (default: base config)")
def smoke_test_cmd(name):
    """Smoke test: forward pass with synthetic audio (no weights or dataset needed)."""
    cfg, variant_name, _, stats_path = _load(name)

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
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Smoke-test passed. Variant: %s  Frontend: %s  n_filters: %d  n_features: %d  Params: %d  Output shape: %s",
        variant_name, cfg.get("frontend", "log_mel"), cfg["model"]["n_filters"], nf, n_params, tuple(probs.shape),
    )


@small_tcn_pareto_group.command("run-all")
@click.option("--skip-existing/--no-skip-existing", default=False,
              help="Skip variants whose weights already exist (default: off)")
def run_all_cmd(skip_existing):
    """Train + eval all SmallTCN-pareto variants sequentially."""
    import time

    names = _list_variants()
    done: set[str] = set()
    results: dict[str, tuple[int, float]] = {}
    diverged: set[str] = set()
    log.info("Running SmallTCN-pareto experiment: %d variant(s): %s", len(names), names)

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
            cfg = _get_variant_config(name)
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
        cfg = _get_variant_config(name)
        res = eval_tcn(cfg=cfg, weights_path=weights, stats_path=stats,
                       output_name=f"tcn_{name}", output_dir=_results_dir)
        results[name] = (res.n_classes, res.f1)
        log.info("[%s] eval done — %d-class macro F1: %.4f", name, res.n_classes, res.f1)
        done.add(name)

    log.info("SmallTCN-pareto experiment complete. %d variant(s) processed.", len(done))
    if results or diverged:
        log.info("── Results summary ──")
        log.info("  %-24s  %s", "variant", "macro F1")
        for n, (n_cls, f1) in results.items():
            log.info("  %-24s  %.4f  (%d-class)", n, f1, n_cls)
        for n in diverged:
            log.info("  %-24s  DIVERGED", n)
