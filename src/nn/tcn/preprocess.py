# tcn/preprocess.py
# Log-mel spectrogram frontend for TCN.
# Pipeline: Resample 22050 mono -> Hann STFT (1024, 512) -> power spectrum ->
# 80-band Mel (27.5-8000 Hz) -> log -> normalize (precomputed mean/var).

import logging
from pathlib import Path

import torch
import torch.nn as nn
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


class LogMelSpectrogram(nn.Module):
    """
    Converts raw waveform to log-mel spectrogram with exact preprocessing pipeline:
    1. Resample to 22050 Hz mono (if input sr differs)
    2. Hann-windowed STFT — frame length 1024, hop size 512
    3. Power spectrum (magnitude squared)
    4. 80-band Mel filterbank, 27.5 Hz to 8000 Hz
    5. Log scale
    6. Normalize using precomputed training set mean and variance
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        norm_mean: torch.Tensor | None = None,
        norm_std: torch.Tensor | None = None,
        stats_path: Path | str | None = None,
    ):
        super().__init__()
        cfg = get_config()
        sr = sample_rate if sample_rate is not None else cfg["sample_rate"]
        self.sample_rate = sr
        self.target_sr = TARGET_SAMPLE_RATE

        self.resampler = None
        if sr != TARGET_SAMPLE_RATE:
            self.resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)

        self.mel = T.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,  # power spectrum (magnitude squared)
            window_fn=torch.hann_window,
        )

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

    def _ensure_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure (batch, samples) or (batch, channels, samples) -> (batch, samples) mono."""
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 3 and waveform.shape[-2] > 1:
            return waveform.mean(dim=-2, keepdim=False)
        return waveform

    def _resample_if_needed(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.resampler is not None:
            return self.resampler(waveform)
        return waveform

    def _validate_stats(self):
        if self.norm_mean is None or self.norm_std is None:
            rev = get_config()["dataset"].get("revision")
            path = get_preprocess_stats_path(revision=rev)
            raise RuntimeError(
                "TCN preprocessing requires precomputed normalization stats (mean, std). "
                f"Run training first to compute and save stats to {path}, "
                "or ensure the stats file exists before inference."
            )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) or (batch, channels, samples), mono expected at sample_rate
        Returns:
            (batch, n_mels, time_frames) normalized log-mel spectrogram
        """
        x = self._ensure_mono(waveform)
        x = self._resample_if_needed(x)

        mel = self.mel(x)
        log_mel = torch.log(mel + 1e-7)

        if self.norm_mean is None or self.norm_std is None:
            self._validate_stats()
        return (log_mel - self.norm_mean) / self.norm_std


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
) -> Path:
    """Compute preprocess stats from training set and save. Returns path to saved file."""
    cfg = get_config()
    ds_link = ds_link or cfg["dataset"]["train"]["url"]
    if ds_link is None:
        raise ValueError("No dataset link; set [dataset] url in config or pass ds_link.")
    if revision is None:
        revision = cfg["dataset"].get("revision")
    stats_path = stats_path or get_preprocess_stats_path(revision=revision)

    fe = LogMelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        f_min=cfg["f_min"],
        f_max=cfg["f_max"],
    )
    fe.eval()

    sr = cfg["sample_rate"]
    hop = cfg["hop_length"]
    n_fft = cfg["n_fft"]

    ds = get_nn_dataset(ds_link, "train", sr, name=name)
    sums = None
    sumsq = None
    count = 0

    for wav, _ in iter_nn_rows(ds, max_rows, "Computing preprocess stats", sr, hop, n_fft):
        with torch.no_grad():
            wav_mono = wav
            if wav.ndim == 1:
                wav_mono = wav.unsqueeze(0)
            elif wav.shape[0] > 1:
                wav_mono = wav.mean(dim=0, keepdim=True)
            if fe.resampler is not None:
                wav_mono = fe.resampler(wav_mono)
            mel = fe.mel(wav_mono)
            log_mel = torch.log(mel + 1e-7)

        b, m, t = log_mel.shape
        flat = log_mel.permute(0, 2, 1).reshape(-1, m)
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
