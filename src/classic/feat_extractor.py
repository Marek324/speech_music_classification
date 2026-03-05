# feat_extractor.py
# Marek Hric

import warnings
from collections import deque
from copy import deepcopy
from typing import Literal, Optional

import numpy as np
from librosa import feature as libfeat
from librosa.util import fix_length
from librosa import lpc
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, lfilter, find_peaks, hilbert
from scipy.stats import skew
from scipy.fft import fft

from .. import config

FeatureSet = Literal["decision_tree", "gmm_svm"]

N_FFT = "n_fft"
BUFFERS = "buffers"
FEATURES = "features"


def _safe(arr: np.ndarray) -> np.ndarray:
    """Replace NaN and Inf with 0."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


class FeatExtractor:
    def __init__(self, sec_buffer_scale: int = 30) -> None:
        self.cfg = config.get_config()
        self.sr = self.cfg["sample_rate"]
        self.n_fft = self.cfg[N_FFT]
        buf = self.cfg[BUFFERS]
        self.fl = buf["frame_length_ms"] * self.sr // 1000
        self.fh = buf["hop_length_ms"] * self.sr // 1000
        self.lt_len = int(buf["lt_len_ms"] * self.sr // 1000)

        self.signal_buffer: deque[float] = deque(maxlen=self.lt_len)
        self.feat_buf_size = int(buf["lt_len_ms"] / buf["hop_length_ms"]) - 1
        self.feat_buffer: deque[np.ndarray] = deque(maxlen=self.feat_buf_size)
        self.sec_buf_size = self.feat_buf_size * sec_buffer_scale
        self.sec_feat_buffer: deque[np.ndarray] = deque(maxlen=self.sec_buf_size)

        self.last_fft: Optional[np.ndarray] = None
        self.last_mfcc: Optional[np.ndarray] = None

    def _get_feature_set(self) -> FeatureSet:
        features = self.cfg.get(FEATURES, {})
        name = features.get("set") or self.cfg["model"]["name"]
        return "gmm_svm" if name in ("gmm", "svm") else "decision_tree"

    def _frame_for_fft(self, frame: np.ndarray) -> np.ndarray:
        return fix_length(frame, size=self.n_fft)

    def _get_sig_buffer(self) -> np.ndarray:
        buffer = self.signal_buffer
        out = np.zeros(self.lt_len, dtype=float)
        out[-len(buffer) :] = np.fromiter(buffer, dtype=float)
        return out

    def _get_feat_buffer(self, secondary: bool = False) -> np.ndarray:
        buf = self.sec_feat_buffer if secondary else self.feat_buffer
        if not buf:
            return np.empty((0, 0), dtype=float)
        return np.asarray(buf, dtype=float)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        self.signal_buffer.extend(frame[-self.fh :])
        feature_set = self._get_feature_set()
        if feature_set == "decision_tree":
            return self._extract_decision_tree(frame)
        return self._extract_gmm_svm(frame)

    def _extract_decision_tree(self, frame: np.ndarray) -> np.ndarray:
        feats = [
            self._short_time_energy(frame),
            self._zero_crossing_rate(frame),
            self._autocorrelation_coefficient(frame),
        ]
        mfccs = _safe(self._mfcc(frame).flatten())
        feats.extend(mfccs)
        feats.append(self._mfcc_diff_norm(mfccs))
        self.last_mfcc = deepcopy(mfccs)

        feats.extend([
            self._band_energy_ratio(frame),
            self._spectral_rolloff_point(frame),
            self._spectrum_centroid(frame),
            self._spectrum_spread(frame),
            self._spectral_flux(frame),
        ])

        feats = np.array(feats)
        self.feat_buffer.append(feats)

        buf = self._get_feat_buffer()
        energies, zcrs = buf[:, 0], buf[:, 1]
        diffs = np.abs(np.diff(buf, axis=0))

        feats = np.hstack([
            feats,
            self._stats_mean(buf),
            self._stats_std(buf),
            self._stats_mean(diffs),
            self._stats_std(diffs),
            self._stats_skew(zcrs),
            self._stats_skew(diffs[:, 1]),
            self._low_short_time_energy_ratio(energies),
        ])
        return _safe(feats)

    def _extract_gmm_svm(self, frame: np.ndarray) -> np.ndarray:
        base_feats = np.array([
            self._short_time_energy(frame),
            self._zero_crossing_rate(frame),
            self._spectrum_centroid(frame),
            self._spectral_flux(frame),
            self._spectral_rolloff_point(frame),
        ])
        feats = _safe(np.hstack(base_feats))
        self.feat_buffer.append(feats)

        lster = self._low_short_time_energy_ratio(self._get_feat_buffer()[:, 0])
        self._fill_speech_specific_buffer()

        existing = self._get_feat_buffer()
        speech = self._get_feat_buffer(secondary=True)
        feat = np.hstack([
            self._stats_var(existing[:, 1]),
            self._stats_var(existing[:, 2]),
            self._stats_var(existing[:, 3]),
            self._stats_var(existing[:, 4]),
            lster,
            self._stats_mean(speech[:, 0]),
            self._stats_mean(speech[:, 1]),
            self._stats_var(speech[:, 2]),
            self._stats_mean(speech[:, 3]),
        ])
        return feat

    def _fill_speech_specific_buffer(self) -> None:
        """Fill sec_feat_buffer with 1ms-shifted speech-specific features."""
        step = int(self.sr * 0.001)
        sigbuf = self._get_sig_buffer()
        start_min = len(sigbuf) - 2 * self.fl
        start_max = len(sigbuf) - self.fl
        for start in range(start_min, start_max, step):
            subframe = sigbuf[start : start + self.fl]
            row = [
                self._naps_of_zffs(subframe),
                self._psr_he_lp_residual(subframe),
                self._log_mel_spectrum_energy(subframe),
                self._modulation_spectrum_energy(subframe),
            ]
            self.sec_feat_buffer.append(_safe(np.hstack(row)))

    # --- Low-level feature extractors ---

    def _short_time_energy(self, frame: np.ndarray) -> float:
        return float(10 * np.log10(np.mean(frame**2) + 1e-10))

    def _zero_crossing_rate(self, frame: np.ndarray) -> float:
        return float(np.sum(np.abs(np.diff(np.sign(frame)))) / 2)

    def _band_energy_ratio(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        berconf = self.cfg[FEATURES]["band_energy_ratio"]
        low, high = berconf["low"], berconf["high"]

        def band_energy(dft: np.ndarray, lb: float, ub: float) -> float:
            K = dft.shape[0]
            b1 = int(np.floor(K * lb / self.sr))
            b2 = int(np.floor(K * ub / self.sr))
            return float(np.sum(np.abs(dft[b1:b2]) ** 2))

        dft = np.fft.fft(frame, n=self.n_fft)
        nyquist = self.sr / 2
        E1 = band_energy(dft, 0.0, low)
        E2 = band_energy(dft, high, nyquist)
        ratio = np.clip((E1 + 1e-10) / (E2 + 1e-10), 1e-10, 1e10)
        result = 10 * np.log10(ratio)
        return 0.0 if np.isnan(result) or np.isinf(result) else float(result)

    def _autocorrelation_coefficient(self, frame: np.ndarray) -> float:
        acconf = self.cfg[FEATURES]["autocorrelation_coefficient"]
        ac = correlate(frame, frame, mode="full")[frame.shape[0] // 2 :]
        min_lag = acconf["min_lag_ms"] * self.sr // 1000
        max_lag = acconf["max_lag_ms"] * self.sr // 1000
        return float(np.max(ac[min_lag:max_lag]))

    def _spectral_rolloff_point(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        srp = self.cfg[FEATURES]["spectral_rolloff_point"]
        return float(
            libfeat.spectral_rolloff(
                y=frame, sr=self.sr, roll_percent=srp["threshold"], n_fft=self.n_fft
            )[0, 0]
        )

    def _spectrum_centroid(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        val = libfeat.spectral_centroid(y=frame, sr=self.sr, n_fft=self.n_fft)[0, 0]
        return 0.0 if np.isnan(val) or np.isinf(val) else float(val)

    def _spectrum_spread(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        val = libfeat.spectral_bandwidth(y=frame, sr=self.sr, n_fft=self.n_fft)[0, 0]
        return 0.0 if np.isnan(val) or np.isinf(val) else float(val)

    def _spectral_flux(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        current_fft = np.fft.fft(frame, n=self.n_fft)
        if self.last_fft is None:
            self.last_fft = current_fft
            return 0.0
        flux = float(np.sum(np.abs(current_fft - self.last_fft) ** 2))
        self.last_fft = current_fft
        return flux

    def _mfcc(self, frame: np.ndarray) -> np.ndarray:
        frame = self._frame_for_fft(frame)
        return libfeat.mfcc(
            y=frame, sr=self.sr, n_mfcc=10, hop_length=self.fh, n_fft=self.n_fft
        )

    def _mfcc_diff_norm(self, mfccs: np.ndarray) -> float:
        if self.last_mfcc is None:
            return 0.0
        return float(np.sqrt(np.sum(np.abs(mfccs - self.last_mfcc) ** 2)))

    def _low_short_time_energy_ratio(self, energies: np.ndarray) -> float:
        if len(energies) == 0:
            return 0.0
        thr = float(np.mean(energies) / 3)
        return float(np.sum(energies < thr) / len(energies))

    def _naps_of_zffs(self, frame: np.ndarray) -> float:
        diff = np.diff(frame, prepend=frame[0])
        y = lfilter([1], [1, -4, 6, -4, 1], diff)
        N = int(self.sr * 0.01)
        y_1 = y - uniform_filter1d(y, size=N, mode="nearest")
        zffs = y_1 - uniform_filter1d(y_1, size=N, mode="nearest")
        zffs = zffs[-self.fl :]

        r = correlate(zffs, zffs, mode="full")[zffs.shape[0] // 2 :]
        if r[0] == 0:
            return 0.0
        r = r / r[0]
        peaks, _ = find_peaks(r, distance=int(self.sr * 0.002))
        return 0.0 if len(peaks) == 0 else float(r[peaks[0]])

    def _psr_he_lp_residual(self, frame: np.ndarray) -> float:
        a = lpc(frame, order=10)
        residual = lfilter([1], a, frame)
        he = np.abs(hilbert(residual))
        min_dist = int(self.sr * 0.005)
        peaks, _ = find_peaks(he, distance=min_dist)
        if len(peaks) == 0:
            return 0.0

        peak_idx = int(peaks[np.argmax(he[peaks])])
        peak_val = he[peak_idx]
        half = int(self.sr * 0.01) // 2
        left = he[max(0, peak_idx - half) : max(0, peak_idx - 4)]
        right = he[min(len(he), peak_idx + 4) : min(len(he), peak_idx + half)]
        sidelobes = np.concatenate((left, right))
        if sidelobes.size == 0:
            return 0.0
        return float(peak_val / (np.var(sidelobes) + 1e-8))

    def _log_mel_spectrum_energy(self, frame: np.ndarray) -> float:
        frame = self._frame_for_fft(frame)
        S = libfeat.melspectrogram(
            y=frame, sr=self.sr, n_fft=self.n_fft,
            n_mels=22, fmin=0, fmax=4000, power=2.0,
        )
        return float(np.sum(np.log(S + 1e-10)[:18, :]))

    def _modulation_spectrum_energy(self, frame: np.ndarray) -> float:
        envelope = np.abs(frame) - np.mean(np.abs(frame))
        if len(envelope) == 0:
            return 0.0
        fft_val = np.abs(fft(envelope))
        df = self.sr / len(envelope)
        lower_idx = int(2.0 / df)
        upper_idx = int(10.0 / df)
        if lower_idx >= len(fft_val):
            return 0.0
        mod_syllabic = np.sum(fft_val[lower_idx:upper_idx] ** 2)
        limit_idx = int(50.0 / df)
        total = np.sum(fft_val[1:limit_idx] ** 2)
        return 0.0 if total == 0 else float(mod_syllabic / total)

    def _stats_mean(self, feats: np.ndarray) -> np.ndarray:
        return self._stats_agg(feats, np.mean)

    def _stats_std(self, feats: np.ndarray) -> np.ndarray:
        return self._stats_agg(feats, np.std)

    def _stats_var(self, feats: np.ndarray) -> np.ndarray:
        return self._stats_agg(feats, np.var)

    def _stats_skew(self, feats: np.ndarray) -> np.ndarray:
        if feats.size == 0 or feats.shape[0] < 3:
            return self._empty_like(feats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nan_to_num(skew(feats, axis=0), nan=0.0)

    def _stats_agg(self, feats: np.ndarray, agg) -> np.ndarray:
        if feats.size == 0 or feats.shape[0] < 1:
            return self._empty_like(feats)
        return agg(feats, axis=0)

    def _empty_like(self, feats: np.ndarray) -> np.ndarray:
        dim = 1 if feats.ndim == 1 else feats.shape[1]
        return np.zeros(dim)
