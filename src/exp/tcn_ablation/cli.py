# exp/tcn_ablation/cli.py

import logging

import click
import torch

from .config import get_config
from .evaluation import eval_exp as _eval_exp
from .model import ExpSpeechMusicDetector
from .training import train_exp as _train_exp

log = logging.getLogger(__name__)


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
    _train_exp(epochs=epochs, patience=patience, use_wandb=not no_wandb)


@tcn_ablation_group.command("eval")
def eval_cmd():
    """Evaluate the experiment model on the test split."""
    _eval_exp()


@tcn_ablation_group.command("smoke-test")
def smoke_test_cmd():
    """Smoke test: forward pass with synthetic audio (no weights or dataset needed)."""
    cfg = get_config()
    sr = cfg["sample_rate"]
    n_mels = cfg["n_mels"]

    model = ExpSpeechMusicDetector(cfg)
    # Inject identity norm stats so the forward pass works without a real stats file
    model.fe.norm_mean = torch.zeros(1, n_mels, 1)
    model.fe.norm_std = torch.ones(1, n_mels, 1)

    dummy = torch.randn(2, sr * 3)  # 2-item batch, 3 seconds
    probs = model(dummy)

    assert probs.shape[0] == 2 and probs.shape[1] == cfg["model"]["n_classes"], (
        f"Unexpected output shape: {probs.shape}"
    )
    assert 0 <= probs.min().item() <= 1 and 0 <= probs.max().item() <= 1
    log.info(
        "Smoke-test passed. Experiment: %s  Output shape: %s",
        cfg["name"],
        tuple(probs.shape),
    )
