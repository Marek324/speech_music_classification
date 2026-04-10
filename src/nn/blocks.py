# nn/blocks.py
# Causal dilated convolution building blocks — shared across all NN models.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1-D convolution that is strictly causal.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        use_weight_norm: bool = False,
        skip_connections: bool = True,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        if use_weight_norm:
            nn.utils.weight_norm(self.conv1.conv)
            nn.utils.weight_norm(self.conv2.conv)
            self.bn1 = None
            self.bn2 = None
        else:
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip_connections = skip_connections
        self.drop = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.drop(F.relu(out))
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.drop(F.relu(out))
        return F.relu(out + residual) if self.skip_connections else F.relu(out)
