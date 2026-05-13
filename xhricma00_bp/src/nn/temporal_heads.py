# nn/temporal_heads.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CausalConv1d


class _HeadBase(nn.Module):
    """Protocol: offline forward on (B, C_in, T); streaming forward on new frames with state."""

    is_stateful: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_streaming(self, x_new: torch.Tensor, state):
        """Stateless default: state is opaque and passes through."""
        return self.forward(x_new), state

    def init_state(self, batch: int, device):
        return None



class _RNNHead(_HeadBase):
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
        """Apply the RNN + classifier on (B, C, T) input and return (logits, new_state)."""
        y, new_state = self.rnn(x.transpose(1, 2), state)
        logits = self.classifier(y.transpose(1, 2))
        return logits, new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self._run(x, None)
        return logits

    def forward_streaming(self, x_new: torch.Tensor, state):
        """Stream new frames through the RNN, threading hidden state across calls."""
        rnn_state = None if state is None else state.get("rnn")
        logits, new_rnn_state = self._run(x_new, rnn_state)
        new_state = dict(state or {})
        new_state["rnn"] = new_rnn_state
        return logits, new_state


class GRUHead(_RNNHead):
    _rnn_cls = nn.GRU


class LSTMHead(_RNNHead):
    _rnn_cls = nn.LSTM



class AttnHead(_HeadBase):
    """Single-block causal multi-head self-attention head with KV cache for streaming."""

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
        """(B, T, hidden) -> (B, H, T, head_dim)."""
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) -> (B, T, hidden)."""
        B, H, T, D = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * D)

    def _attend(self, q, k, v, causal_offset: int):
        """Compute masked scaled-dot-product attention with a cache offset on key indices."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        Tq = q.shape[-2]
        Tk = k.shape[-2]
        i = torch.arange(Tq, device=q.device).view(-1, 1)
        j = torch.arange(Tk, device=q.device).view(1, -1)
        mask = j > (causal_offset + i)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def _block(self, x_new: torch.Tensor, k_cache: Optional[torch.Tensor], v_cache: Optional[torch.Tensor]):
        """Run one attention + FFN block on new frames, returning hidden state and trimmed KV caches."""
        qkv = self.in_proj(x_new)
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

        attn_out = self._attend(q, k, v, causal_offset)
        attn_out = self._merge_heads(attn_out)
        attn_out = self.out_proj(attn_out)

        h = self.norm1(attn_out)
        h = h + self.ffn(self.norm2(h))

        k_out = k[..., -self.cache_len :, :]
        v_out = v[..., -self.cache_len :, :]
        return h, k_out, v_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the offline attention block on (B, C_in, T) input and return logits."""
        x = x.transpose(1, 2)
        h, _, _ = self._block(x, None, None)
        return self.classifier(h.transpose(1, 2))

    def forward_streaming(self, x_new: torch.Tensor, state):
        """Stream new frames through attention with cached keys/values, returning logits and updated state."""
        x_new = x_new.transpose(1, 2)
        k_cache = None if state is None else state.get("k")
        v_cache = None if state is None else state.get("v")
        h, k_out, v_out = self._block(x_new, k_cache, v_cache)
        logits = self.classifier(h.transpose(1, 2))
        new_state = dict(state or {})
        new_state["k"] = k_out
        new_state["v"] = v_out
        return logits, new_state



_TEMPORAL_HEAD_TYPES = ("gru", "lstm", "attn")


def build_temporal_head(cfg: dict, n_features: int) -> Optional[nn.Module]:
    """Construct a temporal-head module from config. Returns None when no head is configured."""
    m = cfg.get("model", {})
    head = m.get("temporal_head")
    if not head:
        return None
    n_classes = m["n_classes"]
    width = m.get("temporal_head_width", 32)
    if head == "gru":
        return GRUHead(n_features=n_features, hidden=width, n_classes=n_classes)
    if head == "lstm":
        return LSTMHead(n_features=n_features, hidden=width, n_classes=n_classes)
    if head == "attn":
        return AttnHead(
            n_features=n_features, hidden=width, n_classes=n_classes,
            n_heads=m.get("n_heads", 4),
        )
    raise ValueError(f"Unknown temporal head {head!r}. Supported: {_TEMPORAL_HEAD_TYPES}")
