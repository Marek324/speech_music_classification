# nn/backbones.py
# Alternative causal backbone architectures for speech/music classification.
# All backbones: forward(x: (B, n_features, T)) -> logits: (B, n_classes, T).

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class CausalGRU(nn.Module):
    """Unidirectional GRU backbone. Strictly causal: output_t depends only on input_{0..t}."""

    def __init__(self, n_features: int, n_filters: int, n_layers: int,
                 dropout: float, n_classes: int, **_kwargs):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, n_filters, kernel_size=1)
        self.rnn = nn.GRU(
            n_filters, n_filters, num_layers=n_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Conv1d(n_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)        # (B, n_filters, T)
        x = x.permute(0, 2, 1)        # (B, T, n_filters)
        x, _ = self.rnn(x)            # (B, T, n_filters)
        x = x.permute(0, 2, 1)        # (B, n_filters, T)
        return self.classifier(x)     # (B, n_classes, T)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class CausalLSTM(nn.Module):
    """Unidirectional LSTM backbone. Cell state provides richer memory than GRU."""

    def __init__(self, n_features: int, n_filters: int, n_layers: int,
                 dropout: float, n_classes: int, **_kwargs):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, n_filters, kernel_size=1)
        self.rnn = nn.LSTM(
            n_filters, n_filters, num_layers=n_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Conv1d(n_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x.permute(0, 2, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017).

    Non-learnable, added per-frame — no cross-frame dependence.
    """

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CausalTransformer(nn.Module):
    """Causal Transformer backbone.

    Uses ``is_causal=True`` on ``nn.TransformerEncoder`` (PyTorch ≥ 2.0) to
    prevent attention to future positions.
    """

    def __init__(self, n_features: int, n_filters: int, n_layers: int,
                 n_heads: int, dropout: float, n_classes: int, **_kwargs):
        super().__init__()
        self.input_proj = nn.Linear(n_features, n_filters)
        self.pe = _SinusoidalPE(n_filters)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_filters, nhead=n_heads,
            dim_feedforward=n_filters * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Conv1d(n_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)            # (B, T, n_features)
        x = self.input_proj(x)            # (B, T, n_filters)
        x = self.pe(x)
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.encoder(x, mask=mask, is_causal=True)  # (B, T, n_filters)
        x = x.permute(0, 2, 1)            # (B, n_filters, T)
        return self.classifier(x)         # (B, n_classes, T)
