# feat_extractor.py
# Marek Hric

from collections import deque
from copy import deepcopy
from typing import Optional

import numpy as np
from librosa import feature as libfeat
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, lfilter

import config


class FeatExtractor:
    def __init__(self):
        cfg = config.get_config()
        assert isinstance(cfg, config.Config)
        self.model = cfg.model.name
        self.cfg = cfg.fext
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        self.defaults = cfg.defaults
        assert isinstance(self.defaults, config.Defaults)

        # buffers
        # signal buffer for raw signal, non-overlapping samples
        self.signal_buffer: deque[float] = deque(maxlen=self.defaults.max_buffer)

        # feat buffer for previous features signal, overlapping feature vectors
        self.feat_buf_size = int(
            (self.defaults.max_buffer / self.defaults.hop_length) - 1
        )
        self.feat_buffer: deque[np.ndarray] = deque(maxlen=self.feat_buf_size)

        if self.cfg.mfcc.enable:
            self.last_mfcc: Optional[np.ndarray]

    def extract(self, frame: np.ndarray) -> np.ndarray:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        feats = []
        mfccs = None

        match self.model:
            case "decision_tree":
                mfccs = self._mfcc(frame).flatten()
                mfccs = np.nan_to_num(mfccs, nan=0.0, posinf=0.0, neginf=0.0)
                feats.append(mfccs)
                feats.append(self._mfcc_diff_norm(mfccs))
                self.last_mfcc = deepcopy(mfccs)

                feats.append(self._short_time_energy(frame))
                feats.append(self._zero_crossing_rate(frame))
                feats.append(self._band_energy_ratio(frame))
                feats.append(self._autocorrelation_coefficient(frame))
                feats.append(self._spectral_rolloff_point(frame))
                feats.append(self._spectrum_centroid(frame))
                feats.append(self._spectrum_spread(frame))
                feats.append(self._spectral_flux(frame))

            case "gmm" | "svm":
                pass

        feat = np.hstack(feats)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        # feat /= np.linalg.norm(feat) + 1e-10

        return feat

    # =============================
    #           HELPERS
    # =============================

    def _get_sig_buffer(self) -> np.ndarray:
        buffer: deque[float] = self.signal_buffer
        max_size = self.defaults.max_buffer

        out = np.zeros(max_size, dtype=float)
        out[-len(buffer) :] = np.fromiter(buffer, dtype=float)

        return out

    def _get_feat_buffer(self) -> np.ndarray:
        buffer = self.feat_buffer

        out = np.zeros(self.feat_buf_size, dtype=float)
        out[-len(buffer) :] = np.fromiter(buffer, dtype=float)

        return out

    def _get_last_frame(self) -> Optional[np.ndarray]:
        return self.feat_buffer[-1] if self.feat_buffer else None

    # =============================
    #           FEATURES
    # =============================

    def _short_time_energy(self, frame: np.ndarray) -> float:
        return 10 * np.log10(1 / frame.shape[0] * np.sum(frame**2) + 1e-10)

    def _zero_crossing_rate(self, frame: np.ndarray) -> float:
        count = 0
        for i in range(1, frame.shape[0]):
            if frame[i] * frame[i - 1] < 0:
                count += 1
        return count / 2

    def _band_energy_ratio(self, frame: np.ndarray) -> float:
        sr = self.defaults.sample_rate
        berconf = self.cfg.band_energy_ratio
        assert isinstance(berconf, config.BandEnergyRatioConfig)

        lowlow = berconf.lower_band.lower_bound
        lowup = berconf.lower_band.upper_bound
        uplow = berconf.upper_band.lower_bound
        upup = berconf.upper_band.upper_bound

        def bin_num(f: float, K: int) -> int:
            return int(np.floor((K * f) / sr))

        def band_energy(dft: np.ndarray, lb: float, ub: float) -> float:
            K = dft.shape[0]
            b1 = bin_num(lb, K)
            b2 = bin_num(ub, K)
            return np.sum(np.abs(dft[b1:b2]) ** 2)

        dft = np.fft.fft(frame, n=self.defaults.n_fft)
        E1 = band_energy(dft, lowlow, lowup)
        E2 = band_energy(dft, uplow, upup)
        ratio = (E1 + 1e-10) / (E2 + 1e-10)
        ratio = np.clip(ratio, 1e-10, 1e10)
        result = 10 * np.log10(ratio)

        return 0.0 if np.isinf(result) or np.isnan(result) else float(result)

    def _autocorrelation_coefficient(self, frame: np.ndarray) -> float:
        ac = correlate(frame, frame, mode="full")
        ac = ac[ac.shape[0] // 2 :]
        min_lag = self.cfg.autocorrelation_coefficient.min_lag_ms * int(
            self.defaults.sample_rate / 1000
        )
        max_lag = self.cfg.autocorrelation_coefficient.max_lag_ms * int(
            self.defaults.sample_rate / 1000
        )
        return np.max(ac[min_lag:max_lag])

    def _spectral_rolloff_point(self, frame: np.ndarray) -> float:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        return float(
            libfeat.spectral_rolloff(
                y=frame,
                sr=self.defaults.sample_rate,
                roll_percent=self.cfg.spectral_rolloff_point.thr,
                n_fft=self.defaults.n_fft,
            )[0, 0]
        )

    def _spectrum_centroid(self, frame: np.ndarray) -> float:
        value = float(
            libfeat.spectral_centroid(
                y=frame, sr=self.defaults.sample_rate, n_fft=self.defaults.n_fft
            )[0, 0]
        )
        return 0.0 if np.isnan(value) or np.isinf(value) else value

    def _spectrum_spread(self, frame: np.ndarray) -> float:
        return float(
            libfeat.spectral_bandwidth(
                y=frame, sr=self.defaults.sample_rate, n_fft=self.defaults.n_fft
            )[0, 0]
        )

    def _spectral_flux(self, frame: np.ndarray) -> float:
        last_frame = self._get_last_frame()
        if last_frame is None:
            return 0

        return np.sum(np.abs(np.fft.fft(frame) - np.fft.fft(last_frame)) ** 2)

    def _mfcc(self, frame: np.ndarray) -> np.ndarray:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        return libfeat.mfcc(
            y=frame,
            sr=self.defaults.sample_rate,
            n_mfcc=10,
            hop_length=self.defaults.hop_length,
            n_fft=self.defaults.n_fft,
        )
        return np.array([])

    def _mfcc_diff_norm(self, mfccs: np.ndarray) -> float:
        last_mfccs = getattr(self, "last_mfcc", None)
        if last_mfccs is None:
            return 0

        return np.sqrt(np.sum(np.abs(mfccs - last_mfccs) ** 2))

    def _naps_of_zffs(self, frame: np.ndarray) -> float:
        # Add frame to buffer
        # TODO: Move to separate function
        self.signal_buffer.extend(frame)

        sig = self._get_sig_buffer()

        # ZFFS
        diff = np.diff(sig, prepend=sig[0])

        y = lfilter([1], [1, -4, 6, -4, 1], diff)

        N = int(self.defaults.sample_rate * 0.01)  # 10ms
        y_1 = y - uniform_filter1d(y, size=N, mode="nearest")
        zffs = y_1 - uniform_filter1d(y_1, size=N, mode="nearest")

        # NAPS
        zffs = zffs[-self.defaults.frame_length]

        return 0.0
