# nn/tcn/training.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

import hashlib
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentation import augment_mel
from .config import get_config, get_preprocess_stats_path, get_weights_path
from ..dataset import get_nn_dataset, iter_nn_rows
from .model import SpeechMusicDetector
from .preprocess import compute_and_save_preprocess_stats
from ...seed import seed_all

log = logging.getLogger(__name__)

SEQ_LEN = 128


class _FocalBCE(nn.Module):
    """Focal BCE computed from logits. Loss = -(1-p_t)^gamma * log p_t.

    p_t = sigmoid(x) when target=1, 1-sigmoid(x) when target=0. Computed via
    logsigmoid for numerical stability.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p  = F.logsigmoid(logits)
        log_1p = F.logsigmoid(-logits)
        p  = torch.exp(log_p)
        one_minus_p = torch.exp(log_1p)
        loss_pos = -((one_minus_p) ** self.gamma) * log_p
        loss_neg = -((p)          ** self.gamma) * log_1p
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
        (loss_fn, mode) where mode is "logits" (use forward_logits_from_mel) or
        "probs" (use forward_from_mel, required for nn.BCELoss / nn.MSELoss).

    Supported cfg["loss"] values:
        bce_with_logits  — nn.BCEWithLogitsLoss on raw logits                 (default, stable)
        bce              — nn.BCELoss on sigmoid(logits)                      (matches the paper, unstable)
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
        pw = torch.tensor([1.0, 1.0, 3.0]).view(-1, 1)
        return nn.BCEWithLogitsLoss(pos_weight=pw), "logits"
    raise ValueError(f"Unknown loss '{name}'")


def _mel_cache_path(cfg: dict, split: str, fe: nn.Module | None = None) -> Path:
    """Cache path keyed by (dataset, frontend params, seq_len). Any ablation
    that changes one of these keys picks up a fresh tensor automatically.

    When a frontend is provided, ``fe.n_features`` is baked into the key so
    variants that share a ``frontend`` name but emit different channel counts
    (e.g. ``mfcc`` with ``n_mfcc=20`` vs ``n_mfcc=40``) don't collide.
    """
    ds = cfg["dataset"]
    key = f"{ds.get('url','')}|{ds.get('name','')}|{ds.get('revision','')}"
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    feat = f"_nf{fe.n_features}" if fe is not None else ""
    fname = (
        f"tcn_mel_{split}_{h}"
        f"_fe{cfg.get('frontend', 'log_mel')}{feat}"
        f"_sr{cfg['sample_rate']}_nfft{cfg['n_fft']}_hop{cfg['hop_length']}"
        f"_mels{cfg['n_mels']}_seq{cfg.get('seq_len', SEQ_LEN)}.pt"
    )
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "cache" / "nn" / fname


def _load_mel_chunks(
    ds_link: str,
    split: str,
    name: str,
    fe: nn.Module,
    fe_device: torch.device,
    cfg: dict,
    max_rows: int | None,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stacked mel + target chunk tensors for the given split.

    Fast path: load cached `.pt` if the key-derived path exists.
    Slow path: stream `iter_nn_rows` once, compute normalized log-mel per clip
    on `fe_device`, slice into non-overlapping `seq_len`-frame chunks, stack.

    Returned tensors live on CPU so the caller decides where to keep them.
    """
    cache_path = _mel_cache_path(cfg, split, fe)
    if cache_path.exists():
        log.info("[%s] loading cached mel chunks from %s", split, cache_path)
        data = torch.load(cache_path, map_location="cpu")
        return data["mel"], data["tgt"]

    sr = cfg["sample_rate"]
    hop = cfg["hop_length"]
    n_fft = cfg["n_fft"]
    seq_len = cfg.get("seq_len", SEQ_LEN)
    chunk_samples = (seq_len - 1) * hop + n_fft

    ds = get_nn_dataset(ds_link, split, sr, name=name)

    fe_was_training = fe.training
    fe.eval()

    mel_chunks: list[torch.Tensor] = []
    tgt_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for wav, targets in iter_nn_rows(ds, max_rows, desc, sr, hop, n_fft):
            if wav.shape[-1] < chunk_samples:
                wav = F.pad(wav, (0, chunk_samples - wav.shape[-1]))
            wav = wav.to(fe_device, non_blocking=True)
            mel = fe(wav).to("cpu")

            T_mel = mel.shape[-1]
            T_tgt = targets.shape[-1]
            T = min(T_mel, T_tgt)
            if T < seq_len:
                pad_amt = seq_len - T
                mel = F.pad(mel[..., :T], (0, pad_amt))
                targets = F.pad(targets[..., :T], (0, pad_amt))
                T = seq_len
            else:
                mel = mel[..., :T]
                targets = targets[..., :T]

            f = 0
            while f + seq_len <= T:
                mel_chunks.append(mel[0, :, f:f + seq_len].contiguous())
                tgt_chunks.append(targets[:, f:f + seq_len].contiguous())
                f += seq_len

    if fe_was_training:
        fe.train()

    if not mel_chunks:
        raise RuntimeError(f"No mel chunks produced for split={split!r}; dataset empty?")

    mel_t = torch.stack(mel_chunks, dim=0).contiguous()
    tgt_t = torch.stack(tgt_chunks, dim=0).contiguous()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mel": mel_t, "tgt": tgt_t}, cache_path)
    size_mb = cache_path.stat().st_size / 1e6
    log.info("[%s] saved %d mel chunks to %s (%.1f MB)", split, mel_t.shape[0], cache_path, size_mb)
    return mel_t, tgt_t


def _iter_mel_batches(
    mel: torch.Tensor,
    tgt: torch.Tensor,
    batch_size: int,
    training: bool,
    cfg: dict,
    fe: nn.Module,
):
    """Shuffle (training) or walk (validation) the precomputed mel tensor,
    yielding (mel_batch, tgt_batch) pairs. Augmentation is applied in-batch
    when training. Tensors stay on whatever device they were loaded on.
    """
    n = mel.shape[0]
    if training:
        indices = torch.randperm(n, device=mel.device)
    else:
        indices = torch.arange(n, device=mel.device)
    do_aug = training and cfg.get("augment", True)
    for i in range(0, n, batch_size):
        idx = indices[i:i + batch_size]
        mel_batch = mel.index_select(0, idx)
        tgt_batch = tgt.index_select(0, idx)
        if do_aug:
            mel_batch = augment_mel(mel_batch, fe)
        yield mel_batch, tgt_batch


def _train_epoch(
    model,
    optimizer,
    loss_fn,
    mel: torch.Tensor,
    tgt: torch.Tensor,
    cfg: dict,
    loss_mode: str,
):
    """Train one epoch over the precomputed mel tensor."""
    model.train()
    batch_size = cfg.get("batch_size", 32)
    total_loss = 0.0
    n_batches = 0
    for mel_batch, tgt_batch in _iter_mel_batches(mel, tgt, batch_size, True, cfg, model.fe):
        optimizer.zero_grad()
        if loss_mode == "probs":
            output = model.forward_from_mel(mel_batch)
        else:
            output = model.forward_logits_from_mel(mel_batch)
        T_out, T_target = output.shape[-1], tgt_batch.shape[-1]
        if T_out != T_target:
            T = min(T_out, T_target)
            output = output[..., :T]
            tgt_batch = tgt_batch[..., :T]
        loss = loss_fn(output, tgt_batch)
        if not torch.isfinite(loss):
            log.warning("non-finite loss (%s); skipping batch", loss.item())
            optimizer.zero_grad()
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _validation_loss(
    model,
    loss_fn,
    mel: torch.Tensor,
    tgt: torch.Tensor,
    cfg: dict,
    loss_mode: str,
) -> float:
    """Compute mean chunk-level loss on precomputed validation mel tensor."""
    model.eval()
    batch_size = cfg.get("batch_size", 32)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for mel_batch, tgt_batch in _iter_mel_batches(mel, tgt, batch_size, False, cfg, model.fe):
            if loss_mode == "probs":
                output = model.forward_from_mel(mel_batch)
            else:
                output = model.forward_logits_from_mel(mel_batch)
            T_out, T_target = output.shape[-1], tgt_batch.shape[-1]
            if T_out != T_target:
                T = min(T_out, T_target)
                output = output[..., :T]
                tgt_batch = tgt_batch[..., :T]
            loss = loss_fn(output, tgt_batch)
            if torch.isfinite(loss):
                total_loss += loss.item()
                n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train_tcn(
    epochs: int = 50,
    max_train_rows: int | None = None,
    weights_path=None,
    patience: int = 5,
    cfg: dict | None = None,
    stats_path=None,
):
    """Train TCN with ReduceLROnPlateau and early stopping (paper §3.5).

    Optimizer: read from cfg["optimizer"] — "adam" (default) or "sgd" (momentum=0.9).
    LR schedule: divide by 10 when val loss doesn't improve for 3 consecutive epochs.
    Early stop: stop when val loss doesn't improve for `patience` consecutive epochs.

    Data path: mel chunks are precomputed once per run (or loaded from the
    disk cache at cache/nn/tcn_mel_*.pt — shared across ablation variants with
    matching frontend params) and kept on the training device for the rest of
    training. This is the refactor that mirrors §3.4 of the paper, where the
    power spectrum is cached and augmentation is applied to the cached
    spectrograms.
    """
    seed_all()
    if cfg is None:
        cfg = get_config()

    torch.backends.cudnn.benchmark = True

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
            cfg=cfg,
        )

    model = SpeechMusicDetector(cfg=cfg, stats_path=stats_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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
    loss_fn = loss_fn.to(device)
    log.info("Loss: %s  (mode=%s)", cfg.get("loss", "bce_with_logits"), loss_mode)

    run_name = cfg.get("name", "tcn")

    ds_link = ds_cfg["url"]
    ds_name = ds_cfg["name"]

    train_mel, train_tgt = _load_mel_chunks(
        ds_link, "train", ds_name, model.fe, device, cfg, max_train_rows,
        "Computing train mel chunks",
    )
    val_mel, val_tgt = _load_mel_chunks(
        ds_link, "validation", ds_name, model.fe, device, cfg, None,
        "Computing val mel chunks",
    )
    train_mel = train_mel.to(device, non_blocking=True)
    train_tgt = train_tgt.to(device, non_blocking=True)
    val_mel = val_mel.to(device, non_blocking=True)
    val_tgt = val_tgt.to(device, non_blocking=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Starting training: variant=%s  epochs=%d  optimizer=%s  lr=%.2e  device=%s  "
        "params=%d  train_chunks=%d  val_chunks=%d",
        run_name, epochs, opt_name, lr, device, n_params,
        train_mel.shape[0], val_mel.shape[0],
    )

    best_val_loss = float("inf")
    best_state = None
    val_checks_without_improvement = 0

    for ep in range(epochs):
        loss = _train_epoch(model, optimizer, loss_fn, train_mel, train_tgt, cfg, loss_mode)
        metrics = {"train/loss": loss, "epoch": ep + 1}

        val_loss = _validation_loss(model, loss_fn, val_mel, val_tgt, cfg, loss_mode)
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %d/%d train loss: %.4f validation loss: %.4f lr: %.2e",
            ep + 1, epochs, loss, val_loss, current_lr,
        )
        metrics["val/loss"] = val_loss
        metrics["lr"] = current_lr

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            val_checks_without_improvement = 0
        else:
            val_checks_without_improvement += 1
            if val_checks_without_improvement >= patience:
                log.info(
                    "Early stopping at epoch %d (no validation improvement for %d patience epochs)",
                    ep + 1, patience,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    from safetensors.torch import save_file

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        save_path,
    )
    log.info("Saved weights to %s", save_path)
