# nn/blocks.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {
    "relu":       F.relu,
    "leaky_relu": F.leaky_relu,
    "elu":        F.elu,
    "gelu":       F.gelu,
}


def _legacy_weight_norm_load_hook(state_dict, prefix, *_):
    """Rename legacy weight_norm keys to the parametrizations layout in-place.

    Pre-hooks run before ``load_state_dict`` validates keys, so checkpoints
    saved under the deprecated ``nn.utils.weight_norm`` (``weight_g`` /
    ``weight_v``) load cleanly into modules that now use
    ``nn.utils.parametrizations.weight_norm`` (``parametrizations.weight.
    original0`` / ``original1``).
    """
    g_key = prefix + "weight_g"
    v_key = prefix + "weight_v"
    if g_key in state_dict:
        state_dict[prefix + "parametrizations.weight.original0"] = state_dict.pop(g_key)
    if v_key in state_dict:
        state_dict[prefix + "parametrizations.weight.original1"] = state_dict.pop(v_key)


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
    """Two stacked causal dilated convolutions with residual + dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        use_weight_norm: bool = False,
        skip_connections: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        if use_weight_norm:
            for c in (self.conv1.conv, self.conv2.conv):
                nn.utils.parametrizations.weight_norm(c)
                c._register_load_state_dict_pre_hook(_legacy_weight_norm_load_hook)
            self.bn1 = None
            self.bn2 = None
        else:
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip_connections = skip_connections
        self.act = _ACTIVATIONS.get(activation, F.relu)
        self.drop = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two causal convs with optional skip connection and return activations."""
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.drop(self.act(out))
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.drop(self.act(out))
        return self.act(out + residual) if self.skip_connections else self.act(out)
