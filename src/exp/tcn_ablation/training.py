# exp/tcn_ablation/training.py
# Training loop for TCN ablation experiments.

import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.dataset import get_nn_dataset, iter_nn_rows
from ...nn.tcn.augmentation import augment
from ...nn.tcn.preprocess import LogMelSpectrogram
from ...wandb_logger import finish as wandb_finish, init as wandb_init, log_metrics as wandb_log
from .config import get_config, get_preprocess_stats_path, get_weights_path
from .model import ExpSpeechMusicDetector

log = logging.getLogger(__name__)


def _compute_and_save_preprocess_stats(cfg: dict, max_rows: int | None, stats_path: Path) -> None:
    """Compute log-mel normalization stats from training set using this experiment's params."""
    train_cfg = cfg["dataset"]["train"]
    ds_link = train_cfg["url"]
    ds_name = train_cfg["name"]
    if ds_link is None:
        raise ValueError("No dataset URL — set [dataset.train] url in root config.toml.")

    fe = LogMelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        f_min=cfg["f_min"],
        f_max=cfg["f_max"],
    )
    fe.eval()

    sr, hop, n_fft = cfg["sample_rate"], cfg["hop_length"], cfg["n_fft"]
    ds = get_nn_dataset(ds_link, "train", sr, name=ds_name)

    sums = None
    sumsq = None
    count = 0

    for wav, _ in iter_nn_rows(ds, max_rows, "Computing preprocess stats", sr, hop, n_fft):
        with torch.no_grad():
            wav_mono = wav if wav.ndim == 2 and wav.shape[0] == 1 else wav.mean(dim=0, keepdim=True)
            if fe.resampler is not None:
                wav_mono = fe.resampler(wav_mono)
            mel = fe.mel(wav_mono)
            log_mel = torch.log(mel + 1e-7)

        b, m, t = log_mel.shape
        flat = log_mel.permute(0, 2, 1).reshape(-1, m)
        n = flat.shape[0]
        s = flat.sum(dim=0)
        s2 = (flat**2).sum(dim=0)
        sums = s if sums is None else sums + s
        sumsq = s2 if sumsq is None else sumsq + s2
        count += n

    if count == 0:
        raise RuntimeError("No frames in training set; cannot compute preprocess stats.")

    mean = sums / count
    var = (sumsq / count) - (mean**2)
    std = torch.sqrt(torch.clamp(var, min=1e-10))

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, stats_path)
    log.info("Saved preprocess stats to %s", stats_path)


def _iter_batched_chunks(ds, cfg, max_rows, desc, batch_size=32, training=False):
    """Slice clips into fixed-length chunks and yield mini-batches (paper §3.5)."""
    seq_len = cfg["seq_len"]
    hop = cfg["hop_length"]
    n_fft = cfg["n_fft"]
    sr = cfg["sample_rate"]
    chunk_samples = (seq_len - 1) * hop + n_fft
    stride_samples = seq_len * hop

    all_wav_chunks, all_tgt_chunks = [], []

    for wav, targets in iter_nn_rows(ds, max_rows, desc, sr, hop, n_fft):
        N = wav.shape[-1]
        M = targets.shape[-1]

        if N < chunk_samples:
            wav = F.pad(wav, (0, chunk_samples - N))
            N = wav.shape[-1]
        if M < seq_len:
            targets = F.pad(targets, (0, seq_len - M))
            M = targets.shape[-1]

        s = 0
        while s + chunk_samples <= N:
            f_start = s // hop
            f_end = f_start + seq_len
            if f_end > M:
                break
            all_wav_chunks.append(wav[:, s : s + chunk_samples])
            all_tgt_chunks.append(targets[:, f_start:f_end])
            s += stride_samples

    if training:
        indices = list(range(len(all_wav_chunks)))
        random.shuffle(indices)
        all_wav_chunks = [all_wav_chunks[i] for i in indices]
        all_tgt_chunks = [all_tgt_chunks[i] for i in indices]

    for i in range(0, len(all_wav_chunks), batch_size):
        wav_batch = torch.cat(all_wav_chunks[i : i + batch_size], dim=0)
        if training:
            wav_batch = augment(wav_batch)
        yield wav_batch, torch.stack(all_tgt_chunks[i : i + batch_size], dim=0)


def _train_epoch(model, optimizer, loss_fn, ds, cfg, max_rows, epoch):
    device = next(model.parameters()).device
    total_loss, n_batches = 0.0, 0
    for wav_batch, tgt_batch in _iter_batched_chunks(ds, cfg, max_rows, f"Epoch {epoch}", training=True):
        wav_batch = wav_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        model.train()
        optimizer.zero_grad()
        probs = model(wav_batch)
        T_p, T_t = probs.shape[-1], tgt_batch.shape[-1]
        if T_p != T_t:
            T = min(T_p, T_t)
            probs = probs[..., :T]
            tgt_batch = tgt_batch[..., :T]
        loss = loss_fn(probs, tgt_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _validation_loss(model, loss_fn, cfg):
    device = next(model.parameters()).device
    model.eval()
    eval_cfg = cfg["dataset"]["eval"]
    ds = get_nn_dataset(eval_cfg["url"], "validation", cfg["sample_rate"], name=eval_cfg["name"])
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for wav, targets in iter_nn_rows(ds, None, "Validation", cfg["sample_rate"], cfg["hop_length"], cfg["n_fft"]):
            wav = wav.to(device)
            targets = targets.unsqueeze(0).to(device)
            probs = model(wav)
            T_p, T_t = probs.shape[-1], targets.shape[-1]
            if T_p != T_t:
                T = min(T_p, T_t)
                probs = probs[..., :T]
                targets = targets[..., :T]
            total_loss += loss_fn(probs, targets).item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train_exp(
    epochs: int = 30,
    max_train_rows: int | None = None,
    weights_path=None,
    use_wandb: bool = True,
    patience: int = 5,
):
    """Train experiment model with config from src/exp/tcn_ablation/config.toml."""
    cfg = get_config()
    name = cfg["name"]
    save_path = Path(weights_path) if weights_path is not None else get_weights_path(name)
    stats_path = get_preprocess_stats_path(name)

    if not stats_path.exists():
        _compute_and_save_preprocess_stats(cfg, max_train_rows, stats_path)

    model = ExpSpeechMusicDetector(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    opt_name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("lr", 1e-3)
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    loss_fn = nn.BCELoss()

    if use_wandb:
        wandb_init(config={"model": f"exp_tcn_ablation_{name}", "epochs": epochs, "patience": patience, **cfg})

    train_cfg = cfg["dataset"]["train"]
    ds = get_nn_dataset(train_cfg["url"], "train", cfg["sample_rate"], name=train_cfg["name"])

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        train_loss = _train_epoch(model, optimizer, loss_fn, ds, cfg, max_train_rows, ep + 1)
        val_loss = _validation_loss(model, loss_fn, cfg)
        current_lr = optimizer.param_groups[0]["lr"]
        log.info("Epoch %d  train=%.4f  val=%.4f  lr=%.2e", ep + 1, train_loss, val_loss, current_lr)

        if use_wandb:
            wandb_log({"train/loss": train_loss, "val/loss": val_loss, "lr": current_lr, "epoch": ep + 1}, step=ep + 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d", ep + 1)
                break

    if use_wandb:
        wandb_finish()

    if best_state is not None:
        model.load_state_dict(best_state)

    from safetensors.torch import save_file

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), save_path)
    log.info("Saved weights to %s", save_path)
