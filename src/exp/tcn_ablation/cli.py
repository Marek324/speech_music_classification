# exp/tcn_ablation/cli.py
# Thin CLI wrapper — all logic lives in src/nn/tcn/.
# The experiment is fully defined by config.toml in this directory.

import logging
from pathlib import Path

import click
import torch

from ...nn.tcn.config import get_config, get_preprocess_stats_path, get_weights_path
from ...nn.tcn.evaluation import eval_tcn
from ...nn.tcn.model import SpeechMusicDetector
from ...nn.tcn.training import train_tcn

log = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).parent / "config.toml"


def _load():
    cfg = get_config(_CFG_PATH)
    name = cfg.get("name", "ablation")
    return cfg, name, get_weights_path(name=name), get_preprocess_stats_path(name=name)


@click.group("tcn-ablation")
def tcn_ablation_group():
    """TCN ablation experiments — edit src/exp/tcn_ablation/config.toml to change knobs."""
    pass


@tcn_ablation_group.command("train")
@click.option("--epochs", "-e", default=50, help="Max training epochs")
@click.option("--patience", "-p", default=5, help="Early stopping patience")
@click.option("--no-wandb", is_flag=True, default=False, help="Disable W&B logging")
def train_cmd(epochs, patience, no_wandb):
    """Train the experiment model defined in config.toml."""
    cfg, name, weights_path, stats_path = _load()
    train_tcn(
        epochs=epochs,
        patience=patience,
        use_wandb=not no_wandb,
        cfg=cfg,
        weights_path=weights_path,
        stats_path=stats_path,
    )


@tcn_ablation_group.command("eval")
def eval_cmd():
    """Evaluate the experiment model on the test split."""
    cfg, name, weights_path, stats_path = _load()
    eval_tcn(
        cfg=cfg,
        weights_path=weights_path,
        stats_path=stats_path,
        output_name=f"tcn_{name}",
    )


@tcn_ablation_group.command("smoke-test")
def smoke_test_cmd():
    """Smoke test: forward pass with synthetic audio (no weights or dataset needed)."""
    cfg, name, _, stats_path = _load()
    n_mels = cfg["n_mels"]

    model = SpeechMusicDetector(cfg=cfg, stats_path=stats_path)
    # Inject identity norm stats so the forward pass works without a real stats file
    model.fe.norm_mean = torch.zeros(1, n_mels, 1)
    model.fe.norm_std = torch.ones(1, n_mels, 1)

    dummy = torch.randn(2, cfg["sample_rate"] * 3)  # 2-item batch, 3 seconds
    probs = model(dummy)

    assert probs.shape[0] == 2 and probs.shape[1] == cfg["model"]["n_classes"], (
        f"Unexpected output shape: {probs.shape}"
    )
    assert 0 <= probs.min().item() <= 1 and 0 <= probs.max().item() <= 1
    log.info(
        "Smoke-test passed. Experiment: %s  Output shape: %s",
        name,
        tuple(probs.shape),
    )
