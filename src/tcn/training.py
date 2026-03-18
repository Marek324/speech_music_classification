# tcn/training.py
# Training utilities.

import logging

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log

from .augmentation import augment
from .config import get_config, get_preprocess_stats_path, get_weights_path
from .dataset import get_tcn_dataset, iter_tcn_rows
from .model import SpeechMusicDetector
from .preprocess import compute_and_save_preprocess_stats

log = logging.getLogger(__name__)

# Paper §3.5: mini-batch sequence length (frames) and batch size
SEQ_LEN = 128   # 128 frames × 512 hop / 22050 sr ≈ 3 s; fits median speech clip (165 frames)
BATCH_SIZE = 32


def build_loss() -> nn.Module:
    """BCE loss for multi-label frame-level classification."""
    return nn.BCELoss()


def train_step(model, optimizer, loss_fn, waveform, targets):
    """
    Single training step.

    Args:
        waveform: (B, samples)
        targets:  (B, 3, T) float 0/1 labels for [speech, music, inactive] per frame
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


def _iter_batched_chunks(ds, cfg, max_rows, desc, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, training=False):
    """Slice clips into fixed-length chunks and yield mini-batches (paper §3.5).

    chunk_samples = (seq_len - 1) * hop + n_fft  →  exactly seq_len target frames.
    Stride = seq_len * hop  →  non-overlapping target windows, contiguous coverage.
    Short clips are zero-padded to yield at least one chunk (prevents silent data loss).
    When training, all chunks are collected and shuffled to interleave classes across batches.
    """
    hop = cfg["hop_length"]
    n_fft = cfg["n_fft"]
    chunk_samples = (seq_len - 1) * hop + n_fft
    stride_samples = seq_len * hop

    all_wav_chunks, all_tgt_chunks = [], []

    for wav, targets in iter_tcn_rows(ds, max_rows, desc):
        N = wav.shape[-1]
        M = targets.shape[-1]

        # Pad short clips so every clip yields at least one chunk
        if N < chunk_samples:
            wav = F.pad(wav, (0, chunk_samples - N))
            N = wav.shape[-1]
        if M < seq_len:
            targets = F.pad(targets, (0, seq_len - M))  # pad with 0 = inactive
            M = targets.shape[-1]

        s = 0
        while s + chunk_samples <= N:
            f_start = s // hop
            f_end = f_start + seq_len
            if f_end > M:
                break
            all_wav_chunks.append(wav[:, s:s + chunk_samples])   # (1, chunk_samples)
            all_tgt_chunks.append(targets[:, f_start:f_end])     # (2, seq_len)
            s += stride_samples

    # Shuffle chunks to interleave classes across batches (critical for convergence)
    if training:
        indices = list(range(len(all_wav_chunks)))
        random.shuffle(indices)
        all_wav_chunks = [all_wav_chunks[i] for i in indices]
        all_tgt_chunks = [all_tgt_chunks[i] for i in indices]

    # Yield mini-batches
    for i in range(0, len(all_wav_chunks), batch_size):
        wav_chunks = all_wav_chunks[i:i + batch_size]
        tgt_chunks = all_tgt_chunks[i:i + batch_size]
        wav_batch = torch.cat(wav_chunks, dim=0)
        if training:
            wav_batch = augment(wav_batch)
        yield wav_batch, torch.stack(tgt_chunks, dim=0)


def _train_epoch(model, optimizer, loss_fn, ds, cfg, max_rows: int | None, epoch: int):
    """Train one epoch using mini-batches of fixed-length chunks (paper §3.5)."""
    device = next(model.parameters()).device
    total_loss = 0.0
    n_batches = 0
    for wav_batch, tgt_batch in _iter_batched_chunks(ds, cfg, max_rows, f"Epoch {epoch}", training=True):
        wav_batch = wav_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        loss = train_step(model, optimizer, loss_fn, wav_batch, tgt_batch)
        total_loss += loss
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _validation_loss(model, loss_fn, ds_link: str, revision: str | None = None) -> float:
    """Compute mean BCE loss on validation split (no gradient)."""
    device = next(model.parameters()).device
    model.eval()
    ds = get_tcn_dataset(ds_link, "validation", revision=revision)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for wav, targets in iter_tcn_rows(ds, None, "Validation"):
            wav = wav.to(device)
            targets = targets.unsqueeze(0).to(device)
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
    """Train TCN with SGD+momentum, ReduceLROnPlateau, and early stopping (paper §3.5).

    Optimizer : SGD, momentum=0.9
    LR schedule: divide by 10 when val loss doesn't improve for 3 consecutive epochs
    Early stop : stop when val loss doesn't improve for `patience` consecutive epochs
    """
    from pathlib import Path

    cfg = get_config()
    save_path = Path(weights_path) if weights_path is not None else get_weights_path()
    stats_path = get_preprocess_stats_path()

    ds_url = cfg["dataset"]["url"]
    ds_rev = cfg["dataset"].get("revision")

    if not stats_path.exists():
        compute_and_save_preprocess_stats(
            ds_link=ds_url,
            max_rows=max_train_rows,
            stats_path=stats_path,
            revision=ds_rev,
        )

    model = SpeechMusicDetector(sample_rate=cfg["sample_rate"])
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    loss_fn = build_loss()

    if use_wandb:
        wandb_init(config={"model": "tcn", "epochs": epochs, "patience": patience, **cfg})

    ds = get_tcn_dataset(ds_url, "train", revision=ds_rev)
    ds_link = ds_url

    best_val_loss = float("inf")
    best_state = None
    val_checks_without_improvement = 0

    for ep in range(epochs):
        loss = _train_epoch(model, optimizer, loss_fn, ds, cfg, max_train_rows, ep + 1)
        metrics = {"train/loss": loss, "epoch": ep + 1}

        val_loss = _validation_loss(model, loss_fn, ds_link, revision=ds_rev)
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %d train loss: %.4f validation loss: %.4f lr: %.2e",
            ep + 1, loss, val_loss, current_lr,
        )
        metrics["val/loss"] = val_loss
        metrics["lr"] = current_lr

        if use_wandb:
            wandb_log(metrics, step=ep + 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            val_checks_without_improvement = 0
        else:
            val_checks_without_improvement += 1
            if val_checks_without_improvement >= patience:
                log.info(
                    "Early stopping at epoch %d (no validation improvement for %d patience epochs)",
                    ep + 1, patience,
                )
                break

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
