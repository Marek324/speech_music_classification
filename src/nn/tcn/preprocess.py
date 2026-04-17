# tcn/preprocess.py
# Audio frontend modules for TCN.
# Base pipeline: Resample 22050 mono -> feature extraction -> normalize (precomputed mean/var).
# Frontends: log-mel spectrogram (paper), log-mel + deltas, MFCC, PCEN.

import logging
import math
from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio.functional as F_audio
import torchaudio.transforms as T

from .config import get_config, get_preprocess_stats_path
from ..dataset import get_nn_dataset, iter_nn_rows

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 80
F_MIN = 27.5
F_MAX = 8000.0


# ---------------------------------------------------------------------------
# Base frontend
# ---------------------------------------------------------------------------

class _BaseFrontend(nn.Module):
    """Abstract base for audio frontends.

    Subclasses implement ``_n_features`` and ``forward_raw``.  The base class
    handles resampling, mono conversion, stats loading, and z-normalization.
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        norm_mean: torch.Tensor | None = None,
        norm_std: torch.Tensor | None = None,
        stats_path: Path | str | None = None,
    ):
        super().__init__()
        cfg = get_config()
        sr = sample_rate if sample_rate is not None else cfg["sample_rate"]
        self.sample_rate = sr
        self.target_sr = TARGET_SAMPLE_RATE
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.resampler = None
        if sr != TARGET_SAMPLE_RATE:
            self.resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)

        self.register_buffer("norm_mean", None)
        self.register_buffer("norm_std", None)
        self._stats_loaded = False

        if norm_mean is not None and norm_std is not None:
            self._set_norm_stats(norm_mean, norm_std)
            self._stats_loaded = True
        elif stats_path is not None:
            path = Path(stats_path)
            if path.exists():
                self._load_stats(path)
                self._stats_loaded = True
            else:
                log.warning("Preprocess stats not found at %s; normalization will fail at inference.", path)
        else:
            rev = get_config()["dataset"].get("revision")
            stats_path = get_preprocess_stats_path(revision=rev)
            if stats_path.exists():
                self._load_stats(stats_path)
                self._stats_loaded = True

    # -- properties ----------------------------------------------------------

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of output channels (frequency bins / coefficients)."""

    # -- stats ---------------------------------------------------------------

    def _set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.view(1, -1, 1)
        std = std.view(1, -1, 1)
        std = torch.clamp(std, min=1e-7)
        self.norm_mean = mean
        self.norm_std = std

    def _load_stats(self, path: Path):
        data = torch.load(path, map_location="cpu")
        if "mean" in data and "std" in data:
            self._set_norm_stats(data["mean"], data["std"])
        else:
            raise ValueError(
                f"Invalid preprocess stats file {path}: expected 'mean' and 'std' keys"
            )

    def _validate_stats(self):
        if self.norm_mean is None or self.norm_std is None:
            rev = get_config()["dataset"].get("revision")
            path = get_preprocess_stats_path(revision=rev)
            raise RuntimeError(
                "TCN preprocessing requires precomputed normalization stats (mean, std). "
                f"Run training first to compute and save stats to {path}, "
                "or ensure the stats file exists before inference."
            )

    # -- waveform helpers ----------------------------------------------------

    def _ensure_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 3 and waveform.shape[-2] > 1:
            return waveform.mean(dim=-2, keepdim=False)
        return waveform

    def _resample_if_needed(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.resampler is not None:
            return self.resampler(waveform)
        return waveform

    # -- forward -------------------------------------------------------------

    @abstractmethod
    def forward_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return features *before* z-normalization.

        Input: mono waveform at ``self.sample_rate`` (already resampled).
        Output: ``(batch, n_features, time_frames)``.
        """

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Waveform → z-normalized features ``(batch, n_features, T)``."""
        x = self._ensure_mono(waveform)
        x = self._resample_if_needed(x)
        raw = self.forward_raw(x)
        if self.norm_mean is None or self.norm_std is None:
            self._validate_stats()
        return (raw - self.norm_mean) / self.norm_std


# ---------------------------------------------------------------------------
# Log-mel spectrogram (paper frontend)
# ---------------------------------------------------------------------------

class LogMelSpectrogram(_BaseFrontend):
    """Log-mel spectrogram — Lemaire & Holzapfel ISMIR 2019 §3.4.

    Pipeline: Hann STFT → power spectrum → mel filterbank → log → z-norm.
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, **kwargs)
        self._n_mels = n_mels
        self.mel = T.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            window_fn=torch.hann_window,
        )

    @property
    def n_features(self) -> int:
        return self._n_mels

    def forward_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel(waveform)
        return torch.log(mel + 1e-7)


# ---------------------------------------------------------------------------
# Log-mel + temporal deltas
# ---------------------------------------------------------------------------

class LogMelDeltaFrontend(_BaseFrontend):
    """Log-mel spectrogram concatenated with temporal derivatives.

    ``order=1``: mel + Δ  (2× n_mels features).
    ``order=2``: mel + Δ + ΔΔ  (3× n_mels features).
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        order: int = 1,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, **kwargs)
        self._n_mels = n_mels
        self._order = order
        self.mel = T.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            window_fn=torch.hann_window,
        )

    @property
    def n_features(self) -> int:
        return self._n_mels * (1 + self._order)

    def forward_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel(waveform)
        log_mel = torch.log(mel + 1e-7)
        parts = [log_mel]
        prev = log_mel
        for _ in range(self._order):
            delta = F_audio.compute_deltas(prev)
            parts.append(delta)
            prev = delta
        return torch.cat(parts, dim=-2)  # (B, n_mels*(1+order), T)


# ---------------------------------------------------------------------------
# MFCC
# ---------------------------------------------------------------------------

class MFCCFrontend(_BaseFrontend):
    """Mel-frequency cepstral coefficients + z-normalization.

    Uses DCT-II of the log-mel spectrogram (standard MFCC pipeline).
    The underlying mel filterbank uses the same parameters as LogMelSpectrogram.
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        n_mfcc: int = 20,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, **kwargs)
        self._n_mfcc = n_mfcc
        self.mfcc = T.MFCC(
            sample_rate=TARGET_SAMPLE_RATE,
            n_mfcc=n_mfcc,
            log_mels=False,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": n_fft,
                "f_min": f_min,
                "f_max": f_max,
                "n_mels": n_mels,
                "power": 2.0,
                "window_fn": torch.hann_window,
            },
        )

    @property
    def n_features(self) -> int:
        return self._n_mfcc

    def forward_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mfcc(waveform)


# ---------------------------------------------------------------------------
# PCEN (Per-Channel Energy Normalization)
# ---------------------------------------------------------------------------

class PCENFrontend(_BaseFrontend):
    """Mel spectrogram with PCEN compression (Wang et al. 2017) + z-normalization.

    PCEN replaces the log compression with an adaptive gain control:
        ``PCEN(S) = (S / (eps + M)^alpha + delta)^r - delta^r``
    where M is an IIR-smoothed version of S along the time axis.
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        gain: float = 0.98,
        bias: float = 2.0,
        power: float = 0.5,
        time_constant: float = 0.4,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, **kwargs)
        self._n_mels = n_mels
        self.mel = T.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            window_fn=torch.hann_window,
        )
        self._gain = gain
        self._bias = bias
        self._power = power
        self._eps = eps
        # IIR smoothing coefficient: s = 1 - exp(-hop / (tc * sr))
        self._smooth = 1.0 - math.exp(-hop_length / (time_constant * TARGET_SAMPLE_RATE))

    @property
    def n_features(self) -> int:
        return self._n_mels

    def _pcen(self, S: torch.Tensor) -> torch.Tensor:
        """Apply PCEN to a mel spectrogram ``(B, n_mels, T)``."""
        s = self._smooth
        # IIR smoother along time axis
        M = torch.empty_like(S)
        M[..., 0] = S[..., 0]
        for t in range(1, S.shape[-1]):
            M[..., t] = (1.0 - s) * M[..., t - 1] + s * S[..., t]
        # AGC
        return (S / (self._eps + M) ** self._gain + self._bias) ** self._power - self._bias ** self._power

    def forward_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel(waveform)
        return self._pcen(mel)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_FRONTEND_TYPES = ("log_mel", "log_mel_delta", "log_mel_delta2", "mfcc", "pcen")


def build_frontend(cfg: dict, stats_path: Path | str | None = None) -> _BaseFrontend:
    """Construct a frontend module from config."""
    frontend = cfg.get("frontend", "log_mel")
    common = dict(
        sample_rate=cfg.get("sample_rate", TARGET_SAMPLE_RATE),
        n_fft=cfg.get("n_fft", N_FFT),
        hop_length=cfg.get("hop_length", HOP_LENGTH),
        f_min=cfg.get("f_min", F_MIN),
        f_max=cfg.get("f_max", F_MAX),
        stats_path=stats_path,
    )
    n_mels = cfg.get("n_mels", N_MELS)

    if frontend == "log_mel":
        return LogMelSpectrogram(n_mels=n_mels, **common)
    if frontend == "log_mel_delta":
        return LogMelDeltaFrontend(n_mels=n_mels, order=1, **common)
    if frontend == "log_mel_delta2":
        return LogMelDeltaFrontend(n_mels=n_mels, order=2, **common)
    if frontend == "mfcc":
        return MFCCFrontend(n_mels=n_mels, n_mfcc=cfg.get("n_mfcc", 20), **common)
    if frontend == "pcen":
        return PCENFrontend(n_mels=n_mels, **common)
    raise ValueError(
        f"Unknown frontend {frontend!r}. Supported: {_FRONTEND_TYPES}"
    )


# ---------------------------------------------------------------------------
# Stats validation / computation
# ---------------------------------------------------------------------------

def validate_preprocess_stats() -> None:
    """Ensure precomputed normalization stats exist. Raise RuntimeError if missing."""
    rev = get_config()["dataset"].get("revision")
    path = get_preprocess_stats_path(revision=rev)
    if not path.exists():
        raise RuntimeError(
            "TCN preprocessing requires precomputed normalization stats. "
            f"Stats file not found at {path}. Run training first to compute and save stats."
        )


def compute_and_save_preprocess_stats(
    ds_link: str | None = None,
    name: str = "full",
    max_rows: int | None = None,
    stats_path: Path | None = None,
    revision: str | None = None,
    cfg: dict | None = None,
) -> Path:
    """Compute preprocess stats from training set and save. Returns path to saved file.

    Works with any frontend — uses ``frontend.forward_raw()`` to get the
    pre-normalization features, then computes per-channel mean and std.
    """
    if cfg is None:
        cfg = get_config()
    ds_link = ds_link or cfg["dataset"]["url"]
    if ds_link is None:
        raise ValueError("No dataset link; set [dataset] url in config or pass ds_link.")
    if revision is None:
        revision = cfg["dataset"].get("revision")
    stats_path = stats_path or get_preprocess_stats_path(revision=revision)

    fe = build_frontend(cfg, stats_path=None)
    fe.eval()

    sr = cfg.get("sample_rate", TARGET_SAMPLE_RATE)
    hop = cfg.get("hop_length", HOP_LENGTH)
    n_fft = cfg.get("n_fft", N_FFT)

    ds = get_nn_dataset(ds_link, "train", sr, name=name)
    sums = None
    sumsq = None
    count = 0

    for wav, _ in iter_nn_rows(ds, max_rows, "Computing preprocess stats", sr, hop, n_fft):
        with torch.no_grad():
            wav_mono = fe._ensure_mono(wav)
            wav_mono = fe._resample_if_needed(wav_mono)
            features = fe.forward_raw(wav_mono)

        b, m, t = features.shape
        flat = features.permute(0, 2, 1).reshape(-1, m)
        n = flat.shape[0]
        s = flat.sum(dim=0)
        s2 = (flat ** 2).sum(dim=0)
        if sums is None:
            sums = s
            sumsq = s2
        else:
            sums = sums + s
            sumsq = sumsq + s2
        count += n

    if count == 0:
        raise RuntimeError("No frames in training set; cannot compute preprocess stats.")
    mean = sums / count
    var = (sumsq / count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-10))

    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, stats_path)
    log.info("Saved preprocess stats to %s", stats_path)
    return stats_path
