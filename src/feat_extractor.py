# feat_extractor.py
# Marek Hric

from collections import deque
from copy import deepcopy
from typing import Optional

import numpy as np
from librosa import feature as libfeat
from librosa.util import fix_length
from librosa import lpc
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, lfilter, find_peaks, hilbert
from scipy.stats import skew
from scipy.fft import fft

import config


class FeatExtractor:
    def __init__(self, sec_buffer_scale: int = 30) -> None:
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
        # secondary buffer for speech-specific features for GMM/SVM
        # needed because they are calculated 30 times more frequently
        self.sec_buf_size = self.feat_buf_size * sec_buffer_scale
        self.sec_feat_buffer: deque[np.ndarray] = deque(maxlen=self.sec_buf_size)

        self.last_fft: Optional[np.ndarray] = None

        if self.cfg.mfcc.enable:
            self.last_mfcc: Optional[np.ndarray]

    def extract(self, frame: np.ndarray) -> np.ndarray:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        feats = []
        sec_feats = []
        mfccs = None

        # add frame to buffer
        self.signal_buffer.extend(frame[-self.defaults.hop_length :])

        match self.model:
            case "decision_tree":
                # accumulate feats
                # time domain features
                feats.append(self._short_time_energy(frame))
                feats.append(self._zero_crossing_rate(frame))
                feats.append(self._autocorrelation_coefficient(frame))

                # frequency domain features
                padd_frame = self._get_frame(padd=True)
                mfccs = self._mfcc(padd_frame).flatten()
                mfccs = np.nan_to_num(mfccs, nan=0.0, posinf=0.0, neginf=0.0)
                feats.extend(mfccs)
                feats.append(self._mfcc_diff_norm(mfccs))
                self.last_mfcc = deepcopy(mfccs)

                feats.append(self._band_energy_ratio(padd_frame))
                feats.append(self._spectral_rolloff_point(padd_frame))
                feats.append(self._spectrum_centroid(padd_frame))
                feats.append(self._spectrum_spread(padd_frame))
                feats.append(self._spectral_flux(padd_frame))

                feats = np.array(feats)

                # update buffer
                self.feat_buffer.append(feats)

                # calc stats
                buf = self._get_feat_buffer()
                energies = buf[:, 0]
                zcrs = buf[:, 1]

                diffs = np.abs(np.diff(buf, axis=0))
                zcr_diffs = diffs[1]

                feats = np.hstack(
                    [
                        feats,
                        self._stats_mean(buf),
                        self._stats_std(buf),
                        self._stats_mean(diffs),
                        self._stats_std(diffs),
                        np.nan_to_num(skew(zcrs), nan=0.0),
                        np.nan_to_num(skew(zcr_diffs), nan=0.0),
                        self._low_short_time_energy_ratio(energies),
                    ]
                )

                feat = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                return feat

            case "gmm" | "svm":
                # current features calculation

                padd_frame = self._get_frame(padd=True)
                # energy for lster
                feats.append(self._short_time_energy(frame))  # 0

                # existing
                feats.append(self._zero_crossing_rate(frame))  # 1
                feats.append(self._spectrum_centroid(padd_frame))  # 2
                feats.append(self._spectral_flux(padd_frame))  # 3
                feats.append(self._spectral_rolloff_point(padd_frame))  # 4

                feats = np.nan_to_num(np.hstack(feats), nan=0.0, posinf=0.0, neginf=0.0)
                self.feat_buffer.append(feats)

                lster = self._low_short_time_energy_ratio(self._get_feat_buffer()[:, 0])

                # speech-specific, 1ms shift
                step = int(self.defaults.sample_rate * 0.001)
                for offset in range(0, self.defaults.frame_length, step):
                    sec_buf_row = []
                    pf = self._get_frame(padd=True)[
                        -self.defaults.frame_length - offset - 1 : -offset
                        if offset != 0
                        else None
                    ]
                    sec_buf_row.append(self._naps_of_zffs(offset))
                    sec_buf_row.append(self._psr_he_lp_residual(pf))
                    sec_buf_row.append(self._log_mel_spectrum_energy(pf))
                    sec_buf_row.append(self._modulation_spectrum_energy(offset))

                    self.sec_feat_buffer.append(
                        np.nan_to_num(
                            np.hstack(sec_buf_row), nan=0.0, posinf=0.0, neginf=0.0
                        )
                    )

                # stats calculation

                existing = self._get_feat_buffer()
                speech = self._get_feat_buffer(secondary=True)

                feat = np.hstack(
                    (
                        self._stats_var(existing[:, 1]),  # zcr
                        self._stats_var(existing[:, 2]),  # centroid
                        self._stats_var(existing[:, 3]),  # flux
                        self._stats_var(existing[:, 4]),  # rolloff
                        lster,  # lster
                        self._stats_mean(speech[:, 0]),  # naps
                        self._stats_mean(speech[:, 1]),  # psr
                        self._stats_var(speech[:, 2]),  # logmel
                        self._stats_mean(speech[:, 3]),  # modul
                    )
                )

                return feat

    # =============================
    #           HELPERS
    # =============================

    def _get_sig_buffer(self) -> np.ndarray:
        buffer: deque[float] = self.signal_buffer

        out = np.zeros(self.defaults.max_buffer, dtype=float)
        out[-len(buffer) :] = np.fromiter(buffer, dtype=float)

        return out

    def _get_frame(self, padd: bool = False) -> np.ndarray:
        if not self.signal_buffer:
            return np.zeros(self.defaults.frame_length, dtype=float)

        frame = np.fromiter(self.signal_buffer, dtype=float)[
            -self.defaults.frame_length :
        ]

        if padd:
            frame = fix_length(
                frame,
                size=self.defaults.n_fft,
            )

        return frame

    def _get_feat_buffer(self, secondary: bool = False) -> np.ndarray:
        buf = self.feat_buffer if not secondary else self.sec_feat_buffer

        if not buf:
            return np.empty((0, 0), dtype=float)

        return np.asarray(buf, dtype=float)

    # =============================
    #           FEATURES
    # =============================

    def _short_time_energy(self, frame: np.ndarray) -> float:
        return 10 * np.log10(1 / frame.shape[0] * np.sum(frame**2) + 1e-10)

    def _zero_crossing_rate(self, frame: np.ndarray) -> float:
        return float(np.sum(np.abs(np.diff(np.sign(frame)))) / 2)

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
        last_fft = self.last_fft
        current_fft = np.fft.fft(frame)
        if last_fft is None:
            self.last_fft = current_fft
            return 0

        flux = np.sum(np.abs(current_fft - last_fft) ** 2)
        self.last_fft = current_fft
        return flux

    def _mfcc(self, frame: np.ndarray) -> np.ndarray:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        return libfeat.mfcc(
            y=frame,
            sr=self.defaults.sample_rate,
            n_mfcc=10,
            hop_length=self.defaults.hop_length,
            n_fft=self.defaults.n_fft,
        )

    def _mfcc_diff_norm(self, mfccs: np.ndarray) -> float:
        last_mfccs = getattr(self, "last_mfcc", None)
        if last_mfccs is None:
            return 0

        return np.sqrt(np.sum(np.abs(mfccs - last_mfccs) ** 2))

    def _low_short_time_energy_ratio(self, energies: np.ndarray) -> float:
        thr = float(np.mean(energies) / 3)
        return np.sum(energies < thr) / len(energies)

    def _naps_of_zffs(self, offset: int) -> float:
        sig = self._get_sig_buffer()[
            -self.defaults.frame_length - offset - 1 : -offset if offset != 0 else None
        ]

        # ZFFS
        diff = np.diff(sig, prepend=sig[0])

        y = lfilter([1], [1, -4, 6, -4, 1], diff)

        N = int(self.defaults.sample_rate * 0.01)  # 10ms
        y_1 = y - uniform_filter1d(y, size=N, mode="nearest")
        zffs = y_1 - uniform_filter1d(y_1, size=N, mode="nearest")

        # NAPS
        zffs = zffs[-self.defaults.frame_length :]

        # autocorrelation
        r = correlate(zffs, zffs, mode="full")
        r = r[r.shape[0] // 2 :]

        if r[0] == 0:
            return 0.0

        r /= r[0]

        peaks, _ = find_peaks(r, distance=int(self.defaults.sample_rate * 0.002))
        if len(peaks) == 0:
            return 0.0

        return r[peaks[0]]

    def _psr_he_lp_residual(self, frame: np.ndarray) -> float:
        # LPC analysis
        a = lpc(frame, order=10)
        residual = lfilter(a, [1], frame)

        # Hilbert envelope
        he = np.abs(hilbert(residual))

        # Peak detection
        min_dist = int(self.defaults.sample_rate * 0.005)
        peaks, _ = find_peaks(he, distance=min_dist)

        if len(peaks) == 0:
            return 0.0

        peak_idx = peaks[np.argmax(he[peaks])]
        peak_val = he[peak_idx]

        # Sidelobe window (one pitch period)
        pitch_period = int(self.defaults.sample_rate * 0.01)
        half = pitch_period // 2

        left = he[max(0, peak_idx - half) : max(0, peak_idx - 4)]
        right = he[min(len(he), peak_idx + 4) : min(len(he), peak_idx + half)]

        sidelobes = np.concatenate((left, right))
        if sidelobes.size == 0:
            return 0.0

        sidelobe_var = np.var(sidelobes) + 1e-8
        return float(peak_val / sidelobe_var)

    def _log_mel_spectrum_energy(self, frame: np.ndarray) -> float:
        S = libfeat.melspectrogram(
            y=frame,
            sr=self.defaults.sample_rate,
            n_fft=self.defaults.n_fft,
            n_mels=22,
            fmin=0,
            fmax=4000,
            power=2.0,
        )

        log_mel = np.log(S + 1e-10)

        return float(np.sum(log_mel[:18, :], axis=0))

    def _modulation_spectrum_energy(self, offset: int) -> float:
        sig = self._get_sig_buffer()[
            -self.defaults.frame_length - offset - 1 : -offset if offset != 0 else None
        ]

        # Rectification
        envelope = np.abs(sig)
        envelope -= np.mean(envelope)

        N = len(envelope)
        if N == 0:
            return 0.0

        fft_val = np.abs(fft(envelope))

        df = self.defaults.sample_rate / N

        # Sum energy in syllabic band (approx 2-6 Hz around 4 Hz center)
        lower_idx = int(2.0 / df)
        upper_idx = int(10.0 / df)

        # Ensure indices are within bounds
        if lower_idx >= len(fft_val):
            return 0.0

        mod_energy_syllabic = np.sum(fft_val[lower_idx:upper_idx] ** 2)

        # Total modulation energy (up to 50 Hz)
        limit_idx = int(50.0 / df)
        total_mod_energy = np.sum(fft_val[1:limit_idx] ** 2)

        if total_mod_energy == 0:
            return 0.0

        return float(mod_energy_syllabic / total_mod_energy)

    # =============================
    #           STATS
    # =============================

    def _stats_mean(self, feats: np.ndarray) -> np.ndarray:
        return np.mean(feats, axis=0)

    def _stats_std(self, feats: np.ndarray) -> np.ndarray:
        return np.std(feats, axis=0)

    def _stats_var(self, feats: np.ndarray) -> np.ndarray:
        return np.var(feats, axis=0)

    def _stats_skew(self, feats: np.ndarray) -> np.ndarray:
        return np.nan_to_num(skew(feats, axis=0), nan=0.0)
