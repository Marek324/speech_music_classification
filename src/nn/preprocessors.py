# nn/preprocessors.py
# Preprocessor layers between frontend and backbone.
# All preprocessors: forward(x: (B, n_features, T)) -> (B, n_features, T).
# All preserve both n_features and T.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CausalConv1d


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class IdentityPreprocessor(nn.Module):
    """No-op preprocessor — features pass directly to backbone."""

    def __init__(self, n_features: int):
        super().__init__()
        self._n_features = n_features

    @property
    def n_features(self) -> int:
        return self._n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# Causal Conv1D
# ---------------------------------------------------------------------------

class Conv1dPreprocessor(nn.Module):
    """Two causal Conv1D layers with BN + ReLU. Reuses CausalConv1d from blocks.py."""

    def __init__(self, n_features: int):
        super().__init__()
        self._n_features = n_features
        self.conv1 = CausalConv1d(n_features, n_features, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm1d(n_features)
        self.conv2 = CausalConv1d(n_features, n_features, kernel_size=3, dilation=1)
        self.bn2 = nn.BatchNorm1d(n_features)

    @property
    def n_features(self) -> int:
        return self._n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


# ---------------------------------------------------------------------------
# Causal Conv2D
# ---------------------------------------------------------------------------

class _CausalConv2d(nn.Module):
    """2D convolution that is causal on the time axis and symmetric on frequency.

    Time axis: left-padded only (causal, no lookahead).
    Frequency axis: symmetric padding (no causality needed).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple[int, int] = (3, 3)):
        super().__init__()
        kf, kt = kernel_size
        self._pad_freq = kf // 2        # symmetric on frequency
        self._pad_time = kt - 1         # causal on time (left only)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        # F.pad order: (left_T, right_T, top_F, bottom_F)
        x = F.pad(x, (self._pad_time, 0, self._pad_freq, self._pad_freq))
        return self.conv(x)


class Conv2dPreprocessor(nn.Module):
    """Causal 2D convolution preprocessor — captures local spectro-temporal patterns.

    Reshapes (B, F, T) → (B, 1, F, T), applies two causal Conv2D layers,
    squeezes back to (B, F, T). Preserves both n_features and T.
    """

    def __init__(self, n_features: int, mid_channels: int = 32):
        super().__init__()
        self._n_features = n_features
        self.conv1 = _CausalConv2d(1, mid_channels, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = _CausalConv2d(mid_channels, 1, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(1)

    @property
    def n_features(self) -> int:
        return self._n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        x = x.unsqueeze(1)                    # (B, 1, F, T)
        x = F.relu(self.bn1(self.conv1(x)))   # (B, mid, F, T)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 1, F, T)
        return x.squeeze(1)                    # (B, F, T)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PREPROCESSOR_TYPES = ("none", "conv1d", "conv2d")


def build_preprocessor(cfg: dict, n_features: int) -> nn.Module:
    """Construct a preprocessor module from config."""
    m = cfg.get("model", {})
    name = m.get("preprocessor", "none")
    if name == "none":
        return IdentityPreprocessor(n_features)
    if name == "conv1d":
        return Conv1dPreprocessor(n_features)
    if name == "conv2d":
        return Conv2dPreprocessor(n_features)
    raise ValueError(f"Unknown preprocessor {name!r}. Supported: {_PREPROCESSOR_TYPES}")
