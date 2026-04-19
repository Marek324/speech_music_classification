# nn/hybrids.py
# Tail modules stacked on top of the TCN feature map for the tcn_hybrid experiment.
# Each tail consumes backbone features (B, C_in, T) and produces logits (B, n_classes, T).
# Stateful tails implement forward_streaming(x_new, state) for online inference.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CausalConv1d


class _TailBase(nn.Module):
    """Protocol: offline forward on (B, C_in, T); streaming forward on new frames with state."""

    is_stateful: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C_in, T) -> (B, n_classes, T)
        raise NotImplementedError

    def forward_streaming(self, x_new: torch.Tensor, state):  # -> (logits_new, new_state)
        # Stateless default: state is opaque and passes through.
        return self.forward(x_new), state

    def init_state(self, batch: int, device):
        return None


# ---------------------------------------------------------------------------
# RNN tails
# ---------------------------------------------------------------------------

class _RNNTail(_TailBase):
    is_stateful = True
    _rnn_cls: type[nn.RNNBase]

    def __init__(self, n_features: int, hidden: int, n_classes: int):
        super().__init__()
        self.rnn = self._rnn_cls(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Conv1d(hidden, n_classes, kernel_size=1)

    def _run(self, x: torch.Tensor, state):
        # x: (B, C, T) -> RNN wants (B, T, C)
        y, new_state = self.rnn(x.transpose(1, 2), state)
        logits = self.classifier(y.transpose(1, 2))
        return logits, new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self._run(x, None)
        return logits

    def forward_streaming(self, x_new: torch.Tensor, state):
        rnn_state = None if state is None else state.get("rnn")
        logits, new_rnn_state = self._run(x_new, rnn_state)
        new_state = dict(state or {})
        new_state["rnn"] = new_rnn_state
        return logits, new_state


class TailGRU(_RNNTail):
    _rnn_cls = nn.GRU


class TailLSTM(_RNNTail):
    _rnn_cls = nn.LSTM


# ---------------------------------------------------------------------------
# Causal self-attention tail (with KV cache)
# ---------------------------------------------------------------------------

class TailAttn(_TailBase):
    is_stateful = True

    def __init__(self, n_features: int, hidden: int, n_classes: int, n_heads: int = 4, ffn: int = 64, cache_len: int = 128):
        super().__init__()
        assert hidden % n_heads == 0, f"hidden ({hidden}) must divide n_heads ({n_heads})"
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.cache_len = cache_len
        self.in_proj = nn.Linear(n_features, 3 * hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.GELU(),
            nn.Linear(ffn, hidden),
        )
        self.classifier = nn.Conv1d(hidden, n_classes, kernel_size=1)
        self._scale = 1.0 / math.sqrt(self.head_dim)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, T, hidden) -> (B, H, T, head_dim)
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, H, T, head_dim) -> (B, T, hidden)
        B, H, T, D = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * D)

    def _attend(self, q, k, v, causal_offset: int):
        # q: (B, H, Tq, D); k, v: (B, H, Tk, D). causal_offset = Tk - Tq (past frames in cache).
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale  # (B, H, Tq, Tk)
        Tq = q.shape[-2]
        Tk = k.shape[-2]
        # Causal mask: position i in q corresponds to absolute position (causal_offset + i).
        # Allowed keys: absolute positions <= (causal_offset + i) → key index j ≤ causal_offset + i.
        i = torch.arange(Tq, device=q.device).view(-1, 1)
        j = torch.arange(Tk, device=q.device).view(1, -1)
        mask = j > (causal_offset + i)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)  # (B, H, Tq, D)

    def _block(self, x_new: torch.Tensor, k_cache: Optional[torch.Tensor], v_cache: Optional[torch.Tensor]):
        # x_new: (B, T_new, C_in)
        qkv = self.in_proj(x_new)  # (B, T_new, 3*hidden)
        q, k_new, v_new = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k_new = self._split_heads(k_new)
        v_new = self._split_heads(v_new)

        if k_cache is not None:
            k = torch.cat([k_cache, k_new], dim=-2)
            v = torch.cat([v_cache, v_new], dim=-2)
        else:
            k, v = k_new, v_new
        causal_offset = k.shape[-2] - q.shape[-2]

        attn_out = self._attend(q, k, v, causal_offset)  # (B, H, T_new, D)
        attn_out = self._merge_heads(attn_out)          # (B, T_new, hidden)
        attn_out = self.out_proj(attn_out)

        # Residual + LN(x_new projected up is tricky — we skip a residual on the input and rely
        # on norm1 on attn output; keeps things simple for a 1-layer stack.)
        h = self.norm1(attn_out)
        h = h + self.ffn(self.norm2(h))

        # Trim caches to last cache_len frames.
        k_out = k[..., -self.cache_len :, :]
        v_out = v[..., -self.cache_len :, :]
        return h, k_out, v_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C_in, T) -> (B, n_classes, T)
        x = x.transpose(1, 2)  # (B, T, C_in)
        h, _, _ = self._block(x, None, None)
        return self.classifier(h.transpose(1, 2))

    def forward_streaming(self, x_new: torch.Tensor, state):
        x_new = x_new.transpose(1, 2)
        k_cache = None if state is None else state.get("k")
        v_cache = None if state is None else state.get("v")
        h, k_out, v_out = self._block(x_new, k_cache, v_cache)
        logits = self.classifier(h.transpose(1, 2))
        new_state = dict(state or {})
        new_state["k"] = k_out
        new_state["v"] = v_out
        return logits, new_state


# ---------------------------------------------------------------------------
# Two-branch TCN tail (stateless, larger-dilation parallel branch)
# ---------------------------------------------------------------------------

class TailTwoBranch(_TailBase):
    is_stateful = False

    def __init__(self, n_features: int, branch_filters: int, n_classes: int, kernel_size: int = 5, n_layers: int = 3):
        super().__init__()
        # Parallel dilated conv branch with larger dilation base (4x the baseline).
        blocks = []
        in_ch = n_features
        for i in range(n_layers):
            dilation = 4 * (2 ** i)  # 4, 8, 16, ...
            blocks.append(nn.Sequential(
                CausalConv1d(in_ch, branch_filters, kernel_size=kernel_size, dilation=dilation),
                nn.ReLU(),
            ))
            in_ch = branch_filters
        self.branch = nn.Sequential(*blocks)
        self.classifier = nn.Conv1d(n_features + branch_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        b = self.branch(x)
        return self.classifier(torch.cat([x, b], dim=1))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TAIL_TYPES = ("gru", "lstm", "attn", "two_branch")


def build_tail(cfg: dict, n_features: int) -> Optional[nn.Module]:
    """Construct a tail module from config. Returns None when no tail is configured."""
    m = cfg.get("model", {})
    tail = m.get("tail")
    if not tail:
        return None
    n_classes = m["n_classes"]
    width = m.get("tail_width", 32)
    if tail == "gru":
        return TailGRU(n_features=n_features, hidden=width, n_classes=n_classes)
    if tail == "lstm":
        return TailLSTM(n_features=n_features, hidden=width, n_classes=n_classes)
    if tail == "attn":
        return TailAttn(
            n_features=n_features, hidden=width, n_classes=n_classes,
            n_heads=m.get("n_heads", 4),
        )
    if tail == "two_branch":
        return TailTwoBranch(n_features=n_features, branch_filters=width, n_classes=n_classes)
    raise ValueError(f"Unknown tail {tail!r}. Supported: {_TAIL_TYPES}")
