# tests/test_nn_hybrids.py
# Sanity tests for tail modules and build_tail() factory.

import pytest
import torch

from src.nn.hybrids import (
    TailAttn,
    TailGRU,
    TailLSTM,
    TailTwoBranch,
    _TAIL_TYPES,
    build_tail,
)


N_CLASSES = 3
N_FEATURES = 16  # TCN feature-map channel count used in tests
BATCH = 2
T = 50


# ---------------------------------------------------------------------------
# Shape + is_stateful sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls, kwargs, expect_stateful",
    [
        (TailGRU, dict(n_features=N_FEATURES, hidden=32, n_classes=N_CLASSES), True),
        (TailLSTM, dict(n_features=N_FEATURES, hidden=32, n_classes=N_CLASSES), True),
        (TailAttn, dict(n_features=N_FEATURES, hidden=32, n_classes=N_CLASSES, n_heads=4), True),
        (TailTwoBranch, dict(n_features=N_FEATURES, branch_filters=16, n_classes=N_CLASSES), False),
    ],
)
def test_tail_forward_shape_and_stateful_flag(cls, kwargs, expect_stateful):
    torch.manual_seed(0)
    tail = cls(**kwargs)
    x = torch.randn(BATCH, N_FEATURES, T)
    y = tail(x)
    assert y.shape == (BATCH, N_CLASSES, T), f"{cls.__name__}: got {tuple(y.shape)}"
    assert y.dtype == x.dtype
    assert tail.is_stateful is expect_stateful


# ---------------------------------------------------------------------------
# Streaming round-trip for stateful tails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, kwargs", [
    (TailGRU, dict(n_features=N_FEATURES, hidden=32, n_classes=N_CLASSES)),
    (TailLSTM, dict(n_features=N_FEATURES, hidden=32, n_classes=N_CLASSES)),
])
def test_rnn_tail_streaming_accumulates_state(cls, kwargs):
    """Streaming over chunks with carried state must match one-pass output."""
    torch.manual_seed(0)
    tail = cls(**kwargs)
    tail.eval()
    x = torch.randn(BATCH, N_FEATURES, T)

    with torch.no_grad():
        full = tail(x)

        state = None
        chunk = 10
        streamed = []
        for start in range(0, T, chunk):
            x_new = x[..., start : start + chunk]
            y_new, state = tail.forward_streaming(x_new, state)
            streamed.append(y_new)
        streamed = torch.cat(streamed, dim=-1)

    diff = (full - streamed).abs().max().item()
    assert diff < 1e-5, f"{cls.__name__}: streaming vs full diff {diff:.3e}"


def test_attn_tail_streaming_runs_and_preserves_cache():
    """Attn tail: streaming runs without error, produces correct shape, cache grows + trims."""
    torch.manual_seed(0)
    tail = TailAttn(n_features=N_FEATURES, hidden=32, n_classes=N_CLASSES, n_heads=4, cache_len=20)
    tail.eval()
    x = torch.randn(BATCH, N_FEATURES, T)

    state = None
    streamed = []
    with torch.no_grad():
        for start in range(0, T, 10):
            x_new = x[..., start : start + 10]
            y_new, state = tail.forward_streaming(x_new, state)
            streamed.append(y_new)
    streamed = torch.cat(streamed, dim=-1)
    assert streamed.shape == (BATCH, N_CLASSES, T)
    # Cache trimmed to cache_len on the last dim (seq).
    assert state["k"].shape[-2] == 20
    assert state["v"].shape[-2] == 20


# ---------------------------------------------------------------------------
# Two-branch causality
# ---------------------------------------------------------------------------

def test_two_branch_is_causal():
    """Modifying x[t=T-1] must not change any output before T-1 (strict causality)."""
    torch.manual_seed(0)
    tail = TailTwoBranch(n_features=N_FEATURES, branch_filters=16, n_classes=N_CLASSES)
    tail.eval()
    x = torch.randn(1, N_FEATURES, T)

    with torch.no_grad():
        y1 = tail(x)
        x2 = x.clone()
        x2[..., -1] = torch.randn(N_FEATURES)
        y2 = tail(x2)

    # Outputs before the last position must be identical.
    diff = (y1[..., :-1] - y2[..., :-1]).abs().max().item()
    assert diff < 1e-6, f"causality violation: diff={diff:.3e}"


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,kind", [
    ("gru", TailGRU),
    ("lstm", TailLSTM),
    ("attn", TailAttn),
    ("two_branch", TailTwoBranch),
])
def test_build_tail_dispatch(name, kind):
    cfg = {"model": {"tail": name, "tail_width": 32, "n_classes": N_CLASSES, "n_heads": 4}}
    tail = build_tail(cfg, n_features=N_FEATURES)
    assert isinstance(tail, kind)


def test_build_tail_returns_none_when_unset():
    cfg = {"model": {"n_classes": N_CLASSES}}
    assert build_tail(cfg, n_features=N_FEATURES) is None


def test_build_tail_empty_string_returns_none():
    cfg = {"model": {"tail": "", "n_classes": N_CLASSES}}
    assert build_tail(cfg, n_features=N_FEATURES) is None


def test_build_tail_unknown_raises():
    cfg = {"model": {"tail": "bogus", "tail_width": 32, "n_classes": N_CLASSES}}
    with pytest.raises(ValueError, match="Unknown tail"):
        build_tail(cfg, n_features=N_FEATURES)


def test_tail_types_tuple_in_sync_with_factory():
    """If someone adds a tail class without updating _TAIL_TYPES, the error message lies."""
    assert set(_TAIL_TYPES) == {"gru", "lstm", "attn", "two_branch"}
