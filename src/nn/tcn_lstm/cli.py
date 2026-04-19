# nn/tcn_lstm/cli.py
# CLI for TCNLSTM — thin delegates to the TCN pipeline with tcn_lstm/ config injected.

import logging
from pathlib import Path

import click
import torch

from ..tcn.evaluation import eval_tcn
from ..tcn.training import train_tcn
from .config import get_tcn_lstm_config, get_tcn_lstm_stats_path, get_tcn_lstm_weights_path
from .model import TCNLSTM

log = logging.getLogger(__name__)

_results_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"


@click.group("tcn-lstm")
def tcn_lstm_group():
    """TCNLSTM — delta2 frontend + conv1d preprocessor + TCN + LSTM tail (combined-experiment winner)."""
    pass


@tcn_lstm_group.command("train")
@click.option("--epochs", "-e", default=50, help="Max training epochs")
@click.option("--patience", "-p", default=5, help="Early-stopping patience (val checks without improvement)")
@click.option("--no-wandb", is_flag=True, default=False, help="Disable W&B logging")
def train_cmd(epochs, patience, no_wandb):
    """Train TCNLSTM."""
    train_tcn(
        epochs=epochs,
        patience=patience,
        cfg=get_tcn_lstm_config(),
        weights_path=get_tcn_lstm_weights_path(),
        stats_path=get_tcn_lstm_stats_path(),
        use_wandb=not no_wandb,
    )


@tcn_lstm_group.command("eval")
def eval_cmd():
    """Evaluate TCNLSTM on the test split. Writes results/tcn_lstm.eval."""
    eval_tcn(
        cfg=get_tcn_lstm_config(),
        weights_path=get_tcn_lstm_weights_path(),
        stats_path=get_tcn_lstm_stats_path(),
        output_name="tcn_lstm",
        output_dir=_results_dir,
    )


@tcn_lstm_group.command("smoke-test")
def smoke_test_cmd():
    """Smoke test: forward pass with synthetic audio (no weights or dataset needed)."""
    cfg = get_tcn_lstm_config()
    model = TCNLSTM(cfg=cfg, stats_path=Path("/nonexistent"))
    nf = model.fe.n_features
    model.fe.norm_mean = torch.zeros(1, nf, 1)
    model.fe.norm_std = torch.ones(1, nf, 1)
    model.fe._stats_loaded = True

    dummy = torch.randn(2, cfg["sample_rate"] * 3)
    probs = model(dummy)

    assert probs.shape[0] == 2 and probs.shape[1] == cfg["model"]["n_classes"], (
        f"Unexpected output shape: {probs.shape}"
    )
    assert 0 <= probs.min().item() <= 1 and 0 <= probs.max().item() <= 1
    log.info(
        "TCNLSTM smoke-test passed. Frontend: %s  Preprocessor: %s  n_features: %d  Output shape: %s",
        cfg["frontend"], cfg["model"]["preprocessor"], nf, tuple(probs.shape),
    )


@tcn_lstm_group.command("smoke-test-online")
def smoke_test_online_cmd():
    """Train on 10 rows, eval on 3 — verifies full TCNLSTM pipeline. Does not overwrite real weights."""
    smoke_weights = get_tcn_lstm_weights_path().parent / "tcn_lstm.smoke.safetensors"
    log.info("TCNLSTM smoke-test-online: training on 10 rows (saving to %s)...", smoke_weights)
    train_tcn(
        epochs=2,
        max_train_rows=10,
        weights_path=smoke_weights,
        use_wandb=False,
        cfg=get_tcn_lstm_config(),
        stats_path=get_tcn_lstm_stats_path(),
    )

    log.info("TCNLSTM smoke-test-online: evaluating on 3 test rows...")
    _eval_on_n_rows(3, weights_path=smoke_weights)
    log.info("TCNLSTM smoke-test-online passed.")


def _eval_on_n_rows(max_rows: int, weights_path: Path):
    """Eval TCNLSTM on limited test rows. Does not save to results/."""
    from safetensors.torch import load_file

    from ..evaluation import run_nn_inference
    from ...evaluator import run_evaluation

    cfg = get_tcn_lstm_config()
    if not weights_path.exists():
        raise FileNotFoundError(f"TCNLSTM weights not found at {weights_path}")
    stats = get_tcn_lstm_stats_path()
    if not stats.exists():
        raise RuntimeError(f"TCNLSTM preprocess stats not found at {stats}. Run `nn tcn-lstm train` first.")

    model = TCNLSTM(cfg=cfg, stats_path=get_tcn_lstm_stats_path())
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state)

    y_true, y_pred, y_sub, time_per_frame_ns, _dev, y_scores, clip_ids = run_nn_inference(
        model, cfg["dataset"], cfg, max_rows=max_rows
    )
    run_evaluation(
        y_true, y_pred, y_sub, time_per_frame_ns,
        output_name="tcn_lstm", save_to_file=False,
        y_scores=y_scores, clip_ids=clip_ids,
    )
