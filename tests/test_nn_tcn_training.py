# tests/test_nn_tcn_training.py
# Tests for the current mel-level training API. Pre-refactor waveform chunker
# (_iter_batched_chunks / train_step / BATCH_SIZE) was removed when training
# switched to precomputed mel caches — see CLAUDE.md "Per-epoch HF re-streaming".

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.nn.tcn.training import (
    SEQ_LEN,
    _iter_mel_batches,
    build_loss,
)


# ---------------------------------------------------------------------------
# build_loss — contract: (loss_fn, mode_str)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,cls,mode", [
    ("bce_with_logits", nn.BCEWithLogitsLoss, "logits"),
    ("bce", nn.BCELoss, "probs"),
    ("mse", nn.MSELoss, "probs"),
    ("weighted_bce", nn.BCEWithLogitsLoss, "logits"),
])
def test_build_loss_dispatch(name, cls, mode):
    loss_fn, returned_mode = build_loss({"loss": name})
    assert isinstance(loss_fn, cls)
    assert returned_mode == mode


def test_build_loss_default_is_bce_with_logits():
    loss_fn, mode = build_loss()
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
    assert mode == "logits"


def test_build_loss_focal_uses_logits_mode():
    loss_fn, mode = build_loss({"loss": "focal", "focal_gamma": 1.5})
    assert mode == "logits"
    # Focal is a custom module; compute a sample loss to confirm it runs.
    logits = torch.randn(2, 3, 10)
    tgt = torch.rand(2, 3, 10).clamp(0, 1)
    value = loss_fn(logits, tgt)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_build_loss_label_smoothing_uses_logits_mode():
    loss_fn, mode = build_loss({"loss": "label_smoothing_bce", "label_smoothing": 0.1})
    assert mode == "logits"
    logits = torch.randn(2, 3, 5)
    tgt = torch.zeros(2, 3, 5)
    tgt[:, 0] = 1.0  # speech one-hot
    value = loss_fn(logits, tgt)
    assert torch.isfinite(value)


def test_build_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss({"loss": "garbage"})


# ---------------------------------------------------------------------------
# SEQ_LEN sanity
# ---------------------------------------------------------------------------


def test_seq_len_is_positive_int():
    assert isinstance(SEQ_LEN, int)
    assert SEQ_LEN > 0


# ---------------------------------------------------------------------------
# _iter_mel_batches — shape / shuffling / augmentation semantics
# ---------------------------------------------------------------------------


class _FakeFe:
    """Frontend stub exposing only what _iter_mel_batches touches for augment."""

    def __init__(self, n_mels: int):
        self.norm_std = torch.ones(1, n_mels, 1)


def _synth(n_chunks=20, n_mels=80, seq_len=128):
    torch.manual_seed(0)
    mel = torch.randn(n_chunks, n_mels, seq_len)
    tgt = torch.zeros(n_chunks, 3, seq_len)
    tgt[:, 0] = 1.0  # all-speech targets — uniqueness via first channel index
    return mel, tgt


def test_iter_mel_batches_yields_batches():
    mel, tgt = _synth(n_chunks=20)
    fe = _FakeFe(n_mels=80)
    batches = list(_iter_mel_batches(mel, tgt, batch_size=4, training=False, cfg={"augment": False}, fe=fe))
    assert len(batches) == 5  # 20 / 4


def test_iter_mel_batches_batch_shape():
    mel, tgt = _synth()
    fe = _FakeFe(n_mels=80)
    for mb, tb in _iter_mel_batches(mel, tgt, batch_size=4, training=False, cfg={"augment": False}, fe=fe):
        assert mb.shape == (4, 80, 128)
        assert tb.shape == (4, 3, 128)
        break


def test_iter_mel_batches_last_batch_may_be_smaller():
    mel, tgt = _synth(n_chunks=10)
    fe = _FakeFe(n_mels=80)
    shapes = [mb.shape[0] for mb, _ in _iter_mel_batches(mel, tgt, batch_size=4, training=False, cfg={"augment": False}, fe=fe)]
    assert shapes == [4, 4, 2]


def test_iter_mel_batches_no_shuffle_walks_in_order():
    mel, tgt = _synth(n_chunks=6)
    fe = _FakeFe(n_mels=80)
    # Replace mel entries with unique markers so we can verify order.
    mel = torch.arange(6, dtype=torch.float32).view(6, 1, 1).expand(6, 80, 128).contiguous()
    out = torch.cat([mb[:, 0, 0] for mb, _ in _iter_mel_batches(
        mel, tgt, batch_size=2, training=False, cfg={"augment": False}, fe=fe
    )])
    assert torch.equal(out, torch.arange(6, dtype=torch.float32))


def test_iter_mel_batches_shuffle_changes_order():
    mel, tgt = _synth(n_chunks=32)
    fe = _FakeFe(n_mels=80)
    torch.manual_seed(1)
    batches = list(_iter_mel_batches(mel, tgt, batch_size=4, training=True, cfg={"augment": False}, fe=fe))
    # At least some rows must appear in a different position than in-order walk would put them.
    first_batch_shuffled = batches[0][0]
    diff = (first_batch_shuffled - mel[:4]).abs().max().item()
    assert diff > 0, "training=True should shuffle; first batch equals unshuffled head"


def test_iter_mel_batches_augment_applied_when_training_and_augment_true():
    mel, tgt = _synth(n_chunks=4)
    fe = _FakeFe(n_mels=80)
    torch.manual_seed(2)
    # With augment=True + training=True, mel batch should differ from its pre-augment values.
    for mb, _ in _iter_mel_batches(mel, tgt, batch_size=4, training=True, cfg={"augment": True}, fe=fe):
        # Compare the augmented batch to the raw indexed version (via permutation trick:
        # take each augmented row and check it is NOT equal to any raw row).
        raw_vals = {tuple(mel[i].flatten()[:4].tolist()) for i in range(4)}
        aug_first = tuple(mb[0].flatten()[:4].tolist())
        assert aug_first not in raw_vals
        break


def test_iter_mel_batches_augment_skipped_in_eval():
    mel, tgt = _synth(n_chunks=4)
    fe = _FakeFe(n_mels=80)
    torch.manual_seed(3)
    # training=False must not apply augmentation even if cfg["augment"] is True.
    for mb, _ in _iter_mel_batches(mel, tgt, batch_size=4, training=False, cfg={"augment": True}, fe=fe):
        assert torch.equal(mb, mel), "eval mode must not augment"
        break
