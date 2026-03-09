# tcn/training.py
# Training utilities.

import logging

import torch
import torch.nn as nn

from ..wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log

from .config import get_config, get_weights_path
from .dataset import get_tcn_dataset, iter_tcn_rows
from .model import SpeechMusicDetector

log = logging.getLogger(__name__)


def build_loss() -> nn.Module:
    """BCE loss for multi-label frame-level classification."""
    return nn.BCELoss()


def train_step(model, optimizer, loss_fn, waveform, targets):
    """
    Single training step.

    Args:
        waveform: (B, samples)
        targets:  (B, 2, T) float 0/1 labels for [speech, music] per frame
    Returns:
        scalar loss
    """
    model.train()
    optimizer.zero_grad()
    probs = model(waveform)
    T_probs, T_target = probs.shape[-1], targets.shape[-1]
    if T_probs != T_target:
        T = min(T_probs, T_target)
        probs = probs[..., :T]
        targets = targets[..., :T]
    loss = loss_fn(probs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def _train_epoch(model, optimizer, loss_fn, ds, cfg, max_rows: int | None, epoch: int):
    """Train one epoch over pre-loaded dataset ds."""
    total_loss = 0.0
    n_batches = 0
    for wav, targets in iter_tcn_rows(ds, max_rows, f"Epoch {epoch}"):
        wav = wav.cuda() if torch.cuda.is_available() else wav
        targets = targets.unsqueeze(0).cuda() if torch.cuda.is_available() else targets.unsqueeze(0)
        loss = train_step(model, optimizer, loss_fn, wav, targets)
        total_loss += loss
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _validation_loss(model, loss_fn, ds_link: str) -> float:
    """Compute mean BCE loss on validation split (no gradient)."""
    model.eval()
    ds = get_tcn_dataset(ds_link, "validation")
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for wav, targets in iter_tcn_rows(ds, None, "Validation"):
            wav = wav.cuda() if torch.cuda.is_available() else wav
            targets = targets.unsqueeze(0).cuda() if torch.cuda.is_available() else targets.unsqueeze(0)
            probs = model(wav)
            T_probs, T_target = probs.shape[-1], targets.shape[-1]
            if T_probs != T_target:
                T = min(T_probs, T_target)
                probs = probs[..., :T]
                targets = targets[..., :T]
            loss = loss_fn(probs, targets)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train_tcn(
    epochs: int = 30,
    max_train_rows: int | None = None,
    weights_path=None,
    use_wandb: bool = True,
    patience: int = 5,
):
    """Train TCN with early stopping. Stops when validation loss does not improve for `patience` epochs."""
    from pathlib import Path

    cfg = get_config()
    save_path = Path(weights_path) if weights_path is not None else get_weights_path()

    model = SpeechMusicDetector(sample_rate=cfg["sample_rate"])
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = build_loss()

    if use_wandb:
        wandb_init(config={"model": "tcn", "epochs": epochs, "patience": patience, **cfg})

    ds = get_tcn_dataset(cfg["dataset"]["train"], "train")
    ds_link = cfg["dataset"]["eval"]

    best_val_loss = float("inf")
    best_state = None
    val_checks_without_improvement = 0

    for ep in range(epochs):
        loss = _train_epoch(model, optimizer, loss_fn, ds, cfg, max_train_rows, ep + 1)
        metrics = {"train/loss": loss, "epoch": ep + 1}

        if (ep + 1) % 5 == 0:
            val_loss = _validation_loss(model, loss_fn, ds_link)
            log.info("Epoch %d train loss: %.4f validation loss: %.4f", ep + 1, loss, val_loss)
            metrics["val/loss"] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                val_checks_without_improvement = 0
            else:
                val_checks_without_improvement += 1
                if val_checks_without_improvement >= patience:
                    log.info(
                        "Early stopping at epoch %d (no validation improvement for %d checks)",
                        ep + 1, patience,
                    )
                    break
        else:
            log.info("Epoch %d train loss: %.4f", ep + 1, loss)

    if use_wandb:
        wandb_finish()

    if best_state is not None:
        model.load_state_dict(best_state)
        if torch.cuda.is_available():
            model = model.cuda()

    from safetensors.torch import save_file

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), save_path)
    log.info("Saved weights to %s", save_path)
