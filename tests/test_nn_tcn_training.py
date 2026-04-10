# tests/test_nn_tcn_training.py

import numpy as np
import torch
import pytest
from pathlib import Path

from src.nn.tcn.training import _iter_batched_chunks, train_step, build_loss, SEQ_LEN, BATCH_SIZE


SR = 22050
HOP = 512
N_FFT = 1024

TCN_CFG = {
    "sample_rate": SR,
    "hop_length": HOP,
    "n_fft": N_FFT,
    "n_mels": 80,
    "f_min": 27.5,
    "f_max": 8000.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "seq_len": SEQ_LEN,
    "dataset": {"url": None, "name": "full"},
    "model": {
        "n_filters": 8,
        "kernel_size": 3,
        "n_layers": 2,
        "n_stacks": 1,
        "dropout": 0.0,
        "n_classes": 3,
        "use_weight_norm": False,
    },
}

CHUNK_SAMPLES = (SEQ_LEN - 1) * HOP + N_FFT   # 66048
STRIDE_SAMPLES = SEQ_LEN * HOP                 # 65536


def _make_rows(n_clips, duration_s=10):
    """Synthetic dataset rows (list of dicts)."""
    n_samples = SR * duration_s
    return [
        {
            "audio": {"array": np.zeros(n_samples, dtype=np.float32)},
            "class": "speech",
            "labels": [],
        }
        for _ in range(n_clips)
    ]


def _make_detector():
    from src.nn.tcn.model import SpeechMusicDetector
    model = SpeechMusicDetector(cfg=TCN_CFG, stats_path=Path("/nonexistent"))
    model.fe.norm_mean = torch.zeros(1, 80, 1)
    model.fe.norm_std = torch.ones(1, 80, 1)
    model.fe._stats_loaded = True
    return model


# ── _iter_batched_chunks ─────────────────────────────────────────────────

def test_iter_batched_chunks_yields_batches():
    rows = _make_rows(2)
    batches = list(_iter_batched_chunks(rows, TCN_CFG, None, "test", batch_size=4))
    assert len(batches) > 0


def test_iter_batched_chunks_wav_shape():
    rows = _make_rows(2)
    for wav_batch, tgt_batch in _iter_batched_chunks(rows, TCN_CFG, None, "test", batch_size=4):
        B = wav_batch.shape[0]
        assert wav_batch.shape == (B, CHUNK_SAMPLES)
        break


def test_iter_batched_chunks_target_shape():
    rows = _make_rows(2)
    for wav_batch, tgt_batch in _iter_batched_chunks(rows, TCN_CFG, None, "test", batch_size=4):
        B = wav_batch.shape[0]
        assert tgt_batch.shape == (B, 3, SEQ_LEN)
        break


def test_iter_batched_chunks_short_clip_yields_one_chunk():
    # Clip shorter than CHUNK_SAMPLES gets padded; must yield exactly 1 chunk
    short_n = CHUNK_SAMPLES // 2
    rows = [{"audio": {"array": np.zeros(short_n, dtype=np.float32)}, "class": "speech", "labels": []}]
    batches = list(_iter_batched_chunks(rows, TCN_CFG, None, "test", batch_size=32))
    total_chunks = sum(wb.shape[0] for wb, _ in batches)
    assert total_chunks == 1


def test_iter_batched_chunks_count_for_10s():
    # 10s clip → expected 3 non-overlapping chunks (see plan for derivation)
    rows = _make_rows(1, duration_s=10)
    batches = list(_iter_batched_chunks(rows, TCN_CFG, None, "test", batch_size=32))
    total_chunks = sum(wb.shape[0] for wb, _ in batches)
    assert total_chunks == 3


def test_iter_batched_chunks_no_overlap():
    # Verify consecutive chunks from the same clip use non-overlapping sample ranges.
    # We check this by inspecting the source waveform with unique values.
    n = SR * 10
    unique_wav = torch.arange(n, dtype=torch.float32).unsqueeze(0)  # (1, N)

    rows = [{"audio": {"array": unique_wav.numpy().squeeze()}, "class": "speech", "labels": []}]
    batches = list(_iter_batched_chunks(rows, TCN_CFG, None, "test", batch_size=32, training=False))
    all_chunks = torch.cat([wb for wb, _ in batches], dim=0)  # (total_chunks, CHUNK_SAMPLES)

    # The start of each chunk should differ by exactly STRIDE_SAMPLES
    starts = [int(c[0].item()) for c in all_chunks]
    for i in range(1, len(starts)):
        assert starts[i] - starts[i - 1] == STRIDE_SAMPLES


def test_iter_batched_chunks_shuffle_changes_order():
    rows = _make_rows(3, duration_s=10)
    batches_no_shuffle = list(_iter_batched_chunks(rows, TCN_CFG, None, "test", batch_size=1, training=False))
    import random; random.seed(42)
    batches_shuffle = list(_iter_batched_chunks(rows, TCN_CFG, None, "test", batch_size=1, training=True))
    # At least one batch should differ in position
    for_check = min(len(batches_no_shuffle), len(batches_shuffle), 5)
    diffs = sum(
        0 if torch.equal(batches_no_shuffle[i][0], batches_shuffle[i][0]) else 1
        for i in range(for_check)
    )
    assert diffs > 0


# ── train_step ────────────────────────────────────────────────────────────

def test_train_step_returns_float():
    model = _make_detector()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_loss()
    waveform = torch.randn(2, CHUNK_SAMPLES)
    targets = torch.rand(2, 3, SEQ_LEN)
    # Normalize targets to be valid BCE targets (rows sum to 1)
    targets = targets / targets.sum(dim=1, keepdim=True)
    loss = train_step(model, optimizer, loss_fn, waveform, targets)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_train_step_shape_mismatch_handled():
    """T_probs ≠ T_targets should not crash; loss should be finite."""
    model = _make_detector()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_loss()
    waveform = torch.randn(1, CHUNK_SAMPLES)
    # mismatched T: targets have 100 frames, model will produce SEQ_LEN frames
    targets = torch.rand(1, 3, 100)
    targets = targets / targets.sum(dim=1, keepdim=True)
    loss = train_step(model, optimizer, loss_fn, waveform, targets)
    assert np.isfinite(loss)


def test_build_loss_returns_bce():
    import torch.nn as nn
    loss_fn = build_loss()
    assert isinstance(loss_fn, nn.BCELoss)
