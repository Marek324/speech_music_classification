# tcn/cli.py
# CLI commands for TCN: train, eval, smoke-test.

import logging
from pathlib import Path

import numpy as np
import torch

from ..wandb_logger import init as wandb_init, log_metrics as wandb_log, finish as wandb_finish
from datasets import Audio, load_dataset
from tqdm import tqdm

from .config import get_config
from .model import SpeechMusicDetector
from .training import build_loss, train_step

log = logging.getLogger(__name__)

LABEL_MAP = {"speech": 0, "music": 1, "noise": -1, "inactive": -1}  # -1 = skip


def _get_weights_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "weights" / "tcn"


def _class_to_target(cls: str) -> torch.Tensor | None:
    """Convert class string to (2,) target tensor [speech, music]. Returns None for skip."""
    idx = LABEL_MAP.get(cls, -1)
    if idx < 0:
        return None
    t = torch.zeros(2, dtype=torch.float32)
    t[idx] = 1.0
    return t


def _load_tcn_dataset(ds_link: str, split: str, max_rows: int | None = None):
    """Load HF dataset, resample to TCN sample_rate, yield (waveform, targets) per row."""
    cfg = get_config()
    sr = cfg["sample_rate"]
    hop = cfg["hop_length"]

    ds = load_dataset(ds_link, split=split).cast_column(
        "audio",
        Audio(sampling_rate=sr, num_channels=1),
    )

    for i, row in enumerate(tqdm(ds, desc=f"Loading {split}")):
        if max_rows and i >= max_rows:
            break
        audio = row["audio"]["array"] if isinstance(row["audio"], dict) else row["audio"]
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        cls_name = row["class"]
        target = _class_to_target(cls_name)
        if target is None:
            continue

        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, samples)
        n_frames = (wav.shape[-1] - cfg["n_fft"]) // hop + 1
        if n_frames < 1:
            continue
        targets = target.unsqueeze(1).expand(2, n_frames)  # (2, T)
        yield wav, targets


def _train_epoch(model, optimizer, loss_fn, ds_link: str, split: str, max_rows: int | None = None):
    total_loss = 0.0
    n_batches = 0
    for wav, targets in _load_tcn_dataset(ds_link, split, max_rows=max_rows):
        wav = wav.cuda() if torch.cuda.is_available() else wav
        targets = targets.unsqueeze(0).cuda() if torch.cuda.is_available() else targets.unsqueeze(0)
        loss = train_step(model, optimizer, loss_fn, wav, targets)
        total_loss += loss
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_tcn(epochs: int = 3, max_train_rows: int | None = None):
    cfg = get_config()
    sr = cfg["sample_rate"]
    ds_link = cfg["dataset"]["train"]

    model = SpeechMusicDetector(sample_rate=sr)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = build_loss()

    wandb_init(config={"model": "tcn", "dataset": ds_link, "epochs": epochs, **cfg})

    for ep in range(epochs):
        loss = _train_epoch(model, optimizer, loss_fn, ds_link, "train", max_rows=max_train_rows)
        log.info("Epoch %d train loss: %.4f", ep + 1, loss)
        wandb_log({"train/loss": loss, "epoch": ep + 1})

    wandb_finish()

    path = _get_weights_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info("Saved weights to %s", path)


def eval_tcn(max_rows: int = 100):
    cfg = get_config()
    sr = cfg["sample_rate"]
    path = _get_weights_path()

    if not path.exists():
        raise FileNotFoundError(f"TCN weights not found at {path}. Run train first.")

    ds_link = cfg["dataset"]["eval"]

    model = SpeechMusicDetector(sample_rate=sr)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    correct, total = 0, 0
    for wav, targets in _load_tcn_dataset(ds_link, "test", max_rows=max_rows):
        with torch.no_grad():
            probs = model(wav)  # (1, 2, T)
        pred = (probs[0, 1, :] > 0.5).float()  # music prob
        target_class = targets[1, 0].item()  # 0 or 1
        pred_class = 1 if pred.mean().item() > 0.5 else 0
        correct += int(pred_class == target_class)
        total += 1

    acc = correct / total if total else 0.0
    log.info("TCN eval: %d/%d correct, accuracy %.4f", correct, total, acc)
    wandb_init(config={})
    wandb_log({"eval/accuracy": acc, "eval/correct": correct, "eval/total": total})
    wandb_finish()
    return acc


def smoke_test_tcn():
    cfg = get_config()
    sr = cfg["sample_rate"]

    model = SpeechMusicDetector(sample_rate=sr)
    dummy = torch.randn(2, sr * 3)  # 2 batch, 3 seconds
    probs = model(dummy)

    assert probs.shape[0] == 2 and probs.shape[1] == 2
    assert 0 <= probs.min().item() <= 1 and 0 <= probs.max().item() <= 1
    log.info("TCN smoke-test passed. Output shape: %s", tuple(probs.shape))
