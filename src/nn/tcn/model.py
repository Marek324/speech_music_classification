# src/nn/tcn/model.py
# Marek Hric
# Causal TCN and full SpeechMusicDetector pipeline.

import torch
import torch.nn as nn

from ..backbones import CausalGRU, CausalLSTM, CausalTransformer
from ..blocks import TCNResidualBlock
from ..temporal_heads import build_temporal_head
from ..preprocessors import build_preprocessor
from .config import get_config, get_preprocess_stats_path
from .preprocess import build_frontend


class CausalTCN(nn.Module):
    """
    Causal TCN for frame-level speech / music detection.
    Receptive field = n_stacks * sum(2^i) * (kernel-1) + 1 frames.

    When ``return_features=True``, the internal classifier head is skipped and
    the raw feature map ``(B, n_filters, T)`` is returned — used when a temporal-head
    module provides its own classification head.
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
        return_features: bool = False,
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

        self.return_features = return_features
        self.n_filters = n_filters

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
            When ``return_features=True``, returns features (batch, n_filters, T) instead.
        """
        x = self.input_proj(x)
        x = self.tcn(x)
        if self.return_features:
            return x
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

_BACKBONE_TYPES = ("tcn", "gru", "lstm", "transformer")


def build_backbone(cfg: dict, n_features: int) -> nn.Module:
    """Construct a backbone module from config.

    When ``cfg["model"]["temporal_head"]`` is set, the TCN backbone is built with
    ``return_features=True`` so the head module can provide the classification head.
    Only the ``tcn`` backbone supports temporal heads — other backbones ignore the config.
    """
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
        return CausalTCN(cfg=cfg, n_mels=n_features, return_features=bool(m.get("temporal_head")))
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
        # Temporal head sits on the TCN feature map (only when backbone == "tcn" + cfg.model.temporal_head set).
        self.temporal_head = build_temporal_head(cfg, n_features=cfg["model"]["n_filters"])

    def _apply_backbone(self, mel: torch.Tensor) -> torch.Tensor:
        """Run backbone + optional temporal head on a normalized mel tensor. Returns logits."""
        x = self.model(mel)
        if self.temporal_head is not None:
            x = self.temporal_head(x)
        return x

    def forward_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        """Raw logits — used by training with BCEWithLogitsLoss."""
        spec = self.fe(waveform)
        spec = self.preproc(spec)
        return self._apply_backbone(spec)

    def forward_logits_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Raw logits from pre-computed normalized log-mel spectrograms."""
        mel = self.preproc(mel)
        return self._apply_backbone(mel)

    def forward_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Sigmoid probs from pre-computed normalized log-mel spectrograms."""
        return torch.sigmoid(self.forward_logits_from_mel(mel))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) mono at sample_rate
        Returns:
            probs: (batch, 3, time_frames) — sigmoid applied for inference.
        """
        return torch.sigmoid(self.forward_logits(waveform))

    def forward_streaming(self, waveform: torch.Tensor, state=None):
        """Streaming forward.

        For stateless models (no head): equivalent to forward(); state passes through.
        For stateful temporal heads: the TCN recomputes on the full audio buffer (it is stateless),
        then only the last ``state["n_new_frames"]`` feature frames are fed through the
        head with carried state. Returns (probs, new_state).
        """
        if self.temporal_head is None or not getattr(self.temporal_head, "is_stateful", False):
            return self.forward(waveform), state

        spec = self.fe(waveform)
        spec = self.preproc(spec)
        feats = self.model(spec)  # (B, n_filters, T_buf)

        n_new = (state or {}).get("n_new_frames")
        if n_new is None or n_new <= 0 or n_new > feats.shape[-1]:
            # No frame counter yet (e.g. first offline call) — process the whole buffer.
            logits, new_state = self.temporal_head.forward_streaming(feats, state)
        else:
            feats_new = feats[..., -n_new:]
            logits_new, new_state = self.temporal_head.forward_streaming(feats_new, state)
            # Pad prefix with zeros so caller sees a tensor aligned to buffer length;
            # last frame (the one StreamingInference reads) is always valid.
            prefix = feats.shape[-1] - n_new
            if prefix > 0:
                pad = torch.zeros(
                    logits_new.shape[0], logits_new.shape[1], prefix,
                    device=logits_new.device, dtype=logits_new.dtype,
                )
                logits = torch.cat([pad, logits_new], dim=-1)
            else:
                logits = logits_new
        return torch.sigmoid(logits), new_state
