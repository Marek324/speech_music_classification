# exp/tcn_ablation/model.py
# Ablation-ready variant of the causal TCN pipeline.
# Supports BatchNorm (default) or WeightNorm normalization.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.blocks import CausalConv1d
from ...nn.tcn.preprocess import LogMelSpectrogram
from .config import get_config, get_preprocess_stats_path


class ExpTCNResidualBlock(nn.Module):
    """TCN residual block with configurable normalization (BatchNorm or WeightNorm)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        if use_weight_norm:
            nn.utils.weight_norm(self.conv1.conv)
            nn.utils.weight_norm(self.conv2.conv)
            self.norm1 = None
            self.norm2 = None
        else:
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)

        self.drop = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        out = F.relu(out)
        out = self.drop(out)
        return F.relu(out + residual)


class ExpCausalTCN(nn.Module):
    """Causal TCN built entirely from explicit config — no hidden defaults."""

    def __init__(
        self,
        n_mels: int,
        n_filters: int,
        kernel_size: int,
        n_layers: int,
        n_stacks: int,
        dropout: float,
        n_classes: int,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(n_mels, n_filters, kernel_size=1)

        blocks = []
        in_ch = n_filters
        for _ in range(n_stacks):
            for layer_idx in range(n_layers):
                dilation = 2**layer_idx
                blocks.append(
                    ExpTCNResidualBlock(
                        in_ch, n_filters, kernel_size, dilation, dropout, use_weight_norm
                    )
                )
                in_ch = n_filters
        self.tcn = nn.Sequential(*blocks)
        self.classifier = nn.Conv1d(n_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.tcn(x)
        x = self.classifier(x)
        return torch.sigmoid(x)


class ExpSpeechMusicDetector(nn.Module):
    """End-to-end waveform -> per-frame probabilities for ablation experiments."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        if cfg is None:
            cfg = get_config()
        m = cfg["model"]
        stats_path = get_preprocess_stats_path(cfg["name"])
        self.fe = LogMelSpectrogram(
            sample_rate=cfg["sample_rate"],
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
            f_min=cfg["f_min"],
            f_max=cfg["f_max"],
            stats_path=stats_path,
        )
        self.model = ExpCausalTCN(
            n_mels=cfg["n_mels"],
            n_filters=m["n_filters"],
            kernel_size=m["kernel_size"],
            n_layers=m["n_layers"],
            n_stacks=m["n_stacks"],
            dropout=m["dropout"],
            n_classes=m["n_classes"],
            use_weight_norm=m.get("use_weight_norm", False),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.fe(waveform)
        return self.model(spec)
