# tcn/cli.py
# CLI commands for TCN: train, eval, smoke-test.

import logging

import click
import torch

from .config import get_config
from .evaluation import eval_tcn as _eval_tcn
from .model import SpeechMusicDetector
from .training import train_tcn as _train_tcn

log = logging.getLogger(__name__)


@click.group("tcn")
def tcn_group():
    """Causal TCN for online speech/music classification (Lemaire & Holzapfel, ISMIR 2019)."""
    pass


@tcn_group.command("train")
@click.option("--epochs", "-e", default=50, help="Max training epochs")
@click.option("--patience", "-p", default=3, help="Early stopping patience (validation checks without improvement, every 5th epoch)")
def train_cmd(epochs, patience):
    """Train the TCN model with early stopping."""
    _train_tcn(epochs=epochs, patience=patience)


@tcn_group.command("eval")
def eval_cmd():
    """Evaluate the TCN model on full test dataset."""
    _eval_tcn()


@tcn_group.command("smoke-test")
def smoke_test_cmd():
    """Run TCN smoke test (forward pass with synthetic audio)."""
    _smoke_test_tcn()


@tcn_group.command("smoke-test-online")
def smoke_test_online_cmd():
    """Run TCN smoke test on real data: 10 train rows + 3 test rows."""
    _smoke_test_online()


def _smoke_test_tcn():
    cfg = get_config()
    sr = cfg["sample_rate"]

    model = SpeechMusicDetector(sample_rate=sr)
    dummy = torch.randn(2, sr * 3)  # 2 batch, 3 seconds
    probs = model(dummy)

    assert probs.shape[0] == 2 and probs.shape[1] == 3
    assert 0 <= probs.min().item() <= 1 and 0 <= probs.max().item() <= 1
    log.info("TCN smoke-test passed. Output shape: %s", tuple(probs.shape))


def _smoke_test_online():
    """Run training (10 train rows) and evaluation (3 test rows). Verifies full pipeline.
    Uses weights/tcn.smoke and does not overwrite weights/tcn or results/tcn.eval."""
    from pathlib import Path

    from .config import get_weights_path
    from .training import train_tcn

    smoke_weights = get_weights_path().parent / "tcn.smoke.safetensors"
    log.info("Smoke-test-online: training on 10 rows (saving to %s)...", smoke_weights)
    train_tcn(epochs=2, max_train_rows=10, weights_path=smoke_weights, use_wandb=False)

    log.info("Smoke-test-online: evaluating on 3 test rows...")
    _eval_tcn_on_n_rows(3, weights_path=smoke_weights)
    log.info("TCN smoke-test-online passed.")


def _eval_tcn_on_n_rows(max_rows: int, weights_path=None):
    """Evaluate TCN on limited test rows. Used by smoke-test-online. Does not save to results/."""
    from pathlib import Path

    from .config import get_config, get_weights_path
    from .preprocess import validate_preprocess_stats
    from ...evaluator import run_evaluation
    from ..evaluation import run_nn_inference

    path = Path(weights_path) if weights_path is not None else get_weights_path()
    cfg = get_config()
    if not path.exists():
        raise FileNotFoundError(f"TCN weights not found at {path}")

    validate_preprocess_stats()

    model = SpeechMusicDetector(sample_rate=cfg["sample_rate"])
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    model.load_state_dict(state)

    y_true, y_pred, y_sub, time_per_frame_ns, _device, y_scores, clip_ids = run_nn_inference(
        model, cfg["dataset"], cfg, max_rows=max_rows
    )

    run_evaluation(
        y_true, y_pred, y_sub, time_per_frame_ns,
        output_name="tcn", save_to_file=False,
        y_scores=y_scores, clip_ids=clip_ids,
    )
