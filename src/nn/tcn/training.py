# tcn/training.py
# Training utilities.

import logging

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log

from .augmentation import augment
from .config import get_config, get_preprocess_stats_path, get_weights_path
from ..dataset import get_nn_dataset, iter_nn_rows
from .model import SpeechMusicDetector
from .preprocess import compute_and_save_preprocess_stats

log = logging.getLogger(__name__)

# Paper §3.5: mini-batch sequence length (frames) — default, overridden by cfg["seq_len"]
SEQ_LEN = 128   # 128 frames × 512 hop / 22050 sr ≈ 3 s; fits median speech clip (165 frames)


class _FocalBCE(nn.Module):
    """Focal BCE computed from logits. Loss = -(1-p_t)^gamma * log p_t.

    p_t = sigmoid(x) when target=1, 1-sigmoid(x) when target=0. Computed via
    logsigmoid for numerical stability.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p  = F.logsigmoid(logits)           # log sigmoid(x)
        log_1p = F.logsigmoid(-logits)          # log (1-sigmoid(x))
        p  = torch.exp(log_p)
        one_minus_p = torch.exp(log_1p)
        loss_pos = -((one_minus_p) ** self.gamma) * log_p   # when y=1
        loss_neg = -((p)          ** self.gamma) * log_1p   # when y=0
        loss = target * loss_pos + (1.0 - target) * loss_neg
        return loss.mean()


class _LabelSmoothingBCE(nn.Module):
    """BCEWithLogits with target smoothing: y' = y*(1-eps) + 0.5*eps."""
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smoothed = target * (1.0 - self.eps) + 0.5 * self.eps
        return self.bce(logits, smoothed)


def build_loss(cfg: dict | None = None) -> tuple[nn.Module, str]:
    """Build the loss module and its input mode.

    Returns:
        (loss_fn, mode) where mode is "logits" (training uses
        SpeechMusicDetector.forward_logits) or "probs" (training uses the
        sigmoid-applied SpeechMusicDetector.forward).

    Supported cfg["loss"] values:
        bce_with_logits  — nn.BCEWithLogitsLoss on raw logits                 (default, stable)
        bce              — nn.BCELoss on sigmoid(logits)                      (paper-literal, unstable)
        focal            — focal BCE on logits, gamma from cfg["focal_gamma"] (default 2.0)
        weighted_bce     — BCEWithLogitsLoss with per-class pos_weight
        mse              — nn.MSELoss on sigmoid(logits)
        label_smoothing_bce — BCEWithLogits with target smoothing cfg["label_smoothing"] (default 0.1)
    """
    cfg = cfg or {}
    name = str(cfg.get("loss", "bce_with_logits")).lower()
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(), "logits"
    if name == "bce":
        return nn.BCELoss(), "probs"
    if name == "mse":
        return nn.MSELoss(), "probs"
    if name == "focal":
        gamma = float(cfg.get("focal_gamma", 2.0))
        return _FocalBCE(gamma), "logits"
    if name == "label_smoothing_bce":
        eps = float(cfg.get("label_smoothing", 0.1))
        return _LabelSmoothingBCE(eps), "logits"
    if name == "weighted_bce":
        # Rough mid-tier class balance: inactive is ~1/3 the size of speech/music.
        # pos_weight has shape (C, 1) so it broadcasts over (B, C, T).
        pw = torch.tensor([1.0, 1.0, 3.0]).view(-1, 1)
        return nn.BCEWithLogitsLoss(pos_weight=pw), "logits"
    raise ValueError(f"Unknown loss '{name}'")


def train_step(model, optimizer, loss_fn, waveform, targets, loss_mode: str = "logits"):
    """
    Single training step.

    Args:
        waveform:  (B, samples)
        targets:   (B, 3, T) float 0/1 labels for [speech, music, inactive] per frame
        loss_mode: "logits" (use forward_logits, stable losses) or
                   "probs"  (apply sigmoid, required for nn.BCELoss / nn.MSELoss)

    Returns:
        scalar loss (float) or None if the batch was skipped due to a
        non-finite loss (guard against occasional WeightNorm blowups).
    """
    model.train()
    optimizer.zero_grad()
    if loss_mode == "probs":
        output = model(waveform)                # probs
    else:
        output = model.forward_logits(waveform) # logits
    T_out, T_target = output.shape[-1], targets.shape[-1]
    if T_out != T_target:
        T = min(T_out, T_target)
        output = output[..., :T]
        targets = targets[..., :T]
    loss = loss_fn(output, targets)
    if not torch.isfinite(loss):
        log.warning("non-finite loss (%s); skipping batch", loss.item())
        optimizer.zero_grad()
        return None
    loss.backward()
    # Clip gradients to prevent explosion (weight norm can shrink ||v|| → 0 over
    # many epochs, causing g/||v|| → ∞ and NaN activations in subsequent batches).
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def _iter_batched_chunks(ds, cfg, max_rows, desc, batch_size=None, seq_len=None, training=False):
    if batch_size is None:
        batch_size = cfg.get("batch_size", 32)
    if seq_len is None:
        seq_len = cfg.get("seq_len", SEQ_LEN)
    """Slice clips into fixed-length chunks and yield mini-batches (paper §3.5).

    chunk_samples = (seq_len - 1) * hop + n_fft  →  exactly seq_len target frames.
    Stride = seq_len * hop  →  non-overlapping target windows, contiguous coverage.
    Short clips are zero-padded to yield at least one chunk (prevents silent data loss).
    When training, all chunks are collected and shuffled to interleave classes across batches.
    """
    hop = cfg["hop_length"]
    n_fft = cfg["n_fft"]
    sr = cfg["sample_rate"]
    chunk_samples = (seq_len - 1) * hop + n_fft
    stride_samples = seq_len * hop

    all_wav_chunks, all_tgt_chunks = [], []

    for wav, targets in iter_nn_rows(ds, max_rows, desc, sr, hop, n_fft):
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
        if training and cfg.get("augment", True):
            wav_batch = augment(wav_batch)
        yield wav_batch, torch.stack(tgt_chunks, dim=0)


def _train_epoch(model, optimizer, loss_fn, ds, cfg, max_rows: int | None, epoch: int,
                 loss_mode: str = "logits"):
    """Train one epoch using mini-batches of fixed-length chunks (paper §3.5)."""
    device = next(model.parameters()).device
    total_loss = 0.0
    n_batches = 0
    for wav_batch, tgt_batch in _iter_batched_chunks(ds, cfg, max_rows, f"Epoch {epoch}", training=True):
        wav_batch = wav_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        loss = train_step(model, optimizer, loss_fn, wav_batch, tgt_batch, loss_mode=loss_mode)
        if loss is None:
            continue
        total_loss += loss
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _validation_loss(model, loss_fn, ds_link: str, cfg: dict, name: str = "full",
                     loss_mode: str = "logits") -> float:
    """Compute mean loss on validation split (no gradient)."""
    device = next(model.parameters()).device
    model.eval()
    sr = cfg["sample_rate"]
    hop = cfg["hop_length"]
    n_fft = cfg["n_fft"]
    ds = get_nn_dataset(ds_link, "validation", sr, name=name)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for wav, targets in iter_nn_rows(ds, None, "Validation", sr, hop, n_fft):
            wav = wav.to(device)
            targets = targets.unsqueeze(0).to(device)
            if loss_mode == "probs":
                output = model(wav)
            else:
                output = model.forward_logits(wav)
            T_out, T_target = output.shape[-1], targets.shape[-1]
            if T_out != T_target:
                T = min(T_out, T_target)
                output = output[..., :T]
                targets = targets[..., :T]
            loss = loss_fn(output, targets)
            if torch.isfinite(loss):
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
    cfg: dict | None = None,
    stats_path=None,
):
    """Train TCN with ReduceLROnPlateau and early stopping (paper §3.5).

    Optimizer: read from cfg["optimizer"] — "adam" (default) or "sgd" (momentum=0.9).
    LR schedule: divide by 10 when val loss doesn't improve for 3 consecutive epochs.
    Early stop: stop when val loss doesn't improve for `patience` consecutive epochs.
    """
    from pathlib import Path

    if cfg is None:
        cfg = get_config()
    exp_name = cfg.get("name")
    save_path = Path(weights_path) if weights_path is not None else get_weights_path(exp_name)
    ds_cfg = cfg["dataset"]
    ds_rev = ds_cfg.get("revision")
    if stats_path is None:
        stats_path = get_preprocess_stats_path(name=exp_name, revision=ds_rev if not exp_name else None)
    else:
        stats_path = Path(stats_path)

    if not stats_path.exists():
        compute_and_save_preprocess_stats(
            ds_link=ds_cfg["url"],
            name=ds_cfg["name"],
            max_rows=max_train_rows,
            stats_path=stats_path,
            revision=ds_rev,
        )

    model = SpeechMusicDetector(cfg=cfg, stats_path=stats_path)
    if torch.cuda.is_available():
        model = model.cuda()

    opt_name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("lr", 1e-3)
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    loss_fn, loss_mode = build_loss(cfg)
    if torch.cuda.is_available() and hasattr(loss_fn, "cuda"):
        loss_fn = loss_fn.cuda()
    log.info("Loss: %s  (mode=%s)", cfg.get("loss", "bce_with_logits"), loss_mode)

    run_name = cfg.get("name", "tcn")
    if use_wandb:
        wandb_init(config={"model": run_name, "epochs": epochs, "patience": patience, **cfg})

    ds = get_nn_dataset(ds_cfg["url"], "train", cfg["sample_rate"], name=ds_cfg["name"])
    ds_link = ds_cfg["url"]
    eval_name = ds_cfg["name"]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Starting training: variant=%s  epochs=%d  optimizer=%s  lr=%.2e  device=%s  params=%d",
        run_name, epochs, opt_name, lr, device_str, n_params,
    )

    best_val_loss = float("inf")
    best_state = None
    val_checks_without_improvement = 0

    for ep in range(epochs):
        log.info("Epoch %d/%d — loading + chunking data...", ep + 1, epochs)
        loss = _train_epoch(model, optimizer, loss_fn, ds, cfg, max_train_rows, ep + 1,
                            loss_mode=loss_mode)
        metrics = {"train/loss": loss, "epoch": ep + 1}

        val_loss = _validation_loss(model, loss_fn, ds_link, cfg, name=eval_name, loss_mode=loss_mode)
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
