# tcn/model.py
# Causal TCN and full SpeechMusicDetector pipeline.

import torch
import torch.nn as nn

from ..blocks import TCNResidualBlock
from .config import get_config, get_preprocess_stats_path
from .preprocess import LogMelSpectrogram


class CausalTCN(nn.Module):
    """
    Causal TCN for frame-level speech / music detection.
    Receptive field = n_stacks * sum(2^i) * (kernel-1) + 1 frames.
    """

    def __init__(
        self,
        n_mels: int | None = None,
        n_filters: int | None = None,
        kernel_size: int | None = None,
        n_layers: int | None = None,
        n_stacks: int | None = None,
        dropout: float | None = None,
        n_classes: int | None = None,
        cfg: dict | None = None,
    ):
        super().__init__()
        if cfg is None:
            cfg = get_config()
        m = cfg["model"]
        n_mels = n_mels or cfg["n_mels"]
        n_filters = n_filters or m["n_filters"]
        kernel_size = kernel_size or m["kernel_size"]
        n_layers = n_layers or m["n_layers"]
        n_stacks = n_stacks or m["n_stacks"]
        dropout = dropout if dropout is not None else m["dropout"]
        n_classes = n_classes or m["n_classes"]
        use_weight_norm = m.get("use_weight_norm", False)
        skip_connections = m.get("skip_connections", True)
        activation = m.get("activation", "relu")

        self.input_proj = nn.Conv1d(n_mels, n_filters, kernel_size=1)

        blocks = []
        in_ch = n_filters
        for _ in range(n_stacks):
            for layer_idx in range(n_layers):
                dilation = 2**layer_idx
                blocks.append(
                    TCNResidualBlock(in_ch, n_filters, kernel_size, dilation, dropout, use_weight_norm, skip_connections, activation)
                )
                in_ch = n_filters
        self.tcn = nn.Sequential(*blocks)

        self.classifier = nn.Conv1d(n_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: log-mel spectrogram (batch, n_mels, time_frames)
        Returns:
            probs: (batch, n_classes, time_frames)
        """
        x = self.input_proj(x)
        x = self.tcn(x)
        x = self.classifier(x)
        return torch.sigmoid(x)


class SpeechMusicDetector(nn.Module):
    """
    End-to-end: raw waveform -> per-frame speech/music probabilities.
    Fully causal, suitable for online/streaming inference.
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        cfg: dict | None = None,
        stats_path=None,
        **tcn_kwargs,
    ):
        super().__init__()
        if cfg is None:
            cfg = get_config()
        sr = sample_rate or cfg["sample_rate"]
        if stats_path is None:
            rev = cfg["dataset"].get("revision")
            stats_path = get_preprocess_stats_path(revision=rev)
        self.fe = LogMelSpectrogram(
            sample_rate=sr,
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
            f_min=cfg["f_min"],
            f_max=cfg["f_max"],
            stats_path=stats_path,
        )
        self.model = CausalTCN(cfg=cfg, **tcn_kwargs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) mono at sample_rate
        Returns:
            probs: (batch, 3, time_frames)
        """
        spec = self.fe(waveform)
        probs = self.model(spec)
        return probs
