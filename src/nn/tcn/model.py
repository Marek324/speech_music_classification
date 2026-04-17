# tcn/model.py
# Causal TCN and full SpeechMusicDetector pipeline.

import torch
import torch.nn as nn

from ..backbones import CausalGRU, CausalLSTM, CausalTransformer
from ..blocks import TCNResidualBlock
from ..preprocessors import build_preprocessor
from .config import get_config, get_preprocess_stats_path
from .preprocess import build_frontend


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
            logits: (batch, n_classes, time_frames) — raw (pre-sigmoid) logits.
            Apply torch.sigmoid externally to obtain probabilities. Training
            uses BCEWithLogitsLoss directly on these logits; inference paths
            apply sigmoid in SpeechMusicDetector.forward.
        """
        x = self.input_proj(x)
        x = self.tcn(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

_BACKBONE_TYPES = ("tcn", "gru", "lstm", "transformer")


def build_backbone(cfg: dict, n_features: int) -> nn.Module:
    """Construct a backbone module from config."""
    m = cfg.get("model", {})
    name = m.get("backbone", "tcn")
    common = dict(
        n_features=n_features,
        n_filters=m["n_filters"],
        n_layers=m["n_layers"],
        dropout=m["dropout"],
        n_classes=m["n_classes"],
    )
    if name == "tcn":
        return CausalTCN(cfg=cfg, n_mels=n_features)
    if name == "gru":
        return CausalGRU(**common)
    if name == "lstm":
        return CausalLSTM(**common)
    if name == "transformer":
        return CausalTransformer(n_heads=m.get("n_heads", 4), **common)
    raise ValueError(f"Unknown backbone {name!r}. Supported: {_BACKBONE_TYPES}")


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
    ):
        super().__init__()
        if cfg is None:
            cfg = get_config()
        if stats_path is None:
            rev = cfg["dataset"].get("revision")
            stats_path = get_preprocess_stats_path(revision=rev)
        if sample_rate is not None:
            cfg = {**cfg, "sample_rate": sample_rate}
        self.fe = build_frontend(cfg, stats_path=stats_path)
        self.preproc = build_preprocessor(cfg, n_features=self.fe.n_features)
        self.model = build_backbone(cfg, n_features=self.preproc.n_features)

    def forward_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        """Raw logits — used by training with BCEWithLogitsLoss."""
        spec = self.fe(waveform)
        spec = self.preproc(spec)
        return self.model(spec)

    def forward_logits_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Raw logits from pre-computed normalized log-mel spectrograms."""
        mel = self.preproc(mel)
        return self.model(mel)

    def forward_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Sigmoid probs from pre-computed normalized log-mel spectrograms."""
        mel = self.preproc(mel)
        return torch.sigmoid(self.model(mel))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) mono at sample_rate
        Returns:
            probs: (batch, 3, time_frames) — sigmoid applied for inference.
        """
        return torch.sigmoid(self.forward_logits(waveform))
