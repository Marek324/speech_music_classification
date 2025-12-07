# feat_extractor.py
# Marek Hric

from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.signal import correlate
from librosa import feature as libfeat

import config


class FeatExtractor:
    def __init__(self):
        cfg = config.get_config()
        assert isinstance(self.cfg, config.Config)
        self.cfg = cfg.fext
        self.defaults = cfg.defaults
        self.last_frame: Optional[np.ndarray]
        if self.cfg.mfcc.enable:
            self.last_mfcc: Optional[np.ndarray]
        pass

    def extract(self, frame: np.ndarray) -> np.ndarray:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        feats = []
        if self.cfg.mfcc.enable:
            mfccs = self._mfcc(frame).flatten()
            if self.cfg.mfcc_diff_norm.enable:
                feats.append(self._mfcc_diff_norm(mfccs))

        if self.cfg.st_energy.enable:
            feats.append(self._ste(frame))
        if self.cfg.zcr.enable:
            feats.append(self._zcr(frame))
        if self.cfg.band_energy_ratio.enable:
            feats.append(self._ber(frame))
        if self.cfg.ac_coeff.enable:
            feats.append(self._acc(frame))
        if self.cfg.s_rolloff_point.enable:
            feats.append(self._srp(frame))
        if self.cfg.s_centroid.enable:
            feats.append(self._sc(frame))
        if self.cfg.s_flux.enable:
            feats.append(self._sf(frame))

        feat = np.hstack((mfccs, np.array(feats)))

        self.last_frame = deepcopy(frame)
        if self.cfg.mfcc.enable:
            self.last_mfcc = deepcopy(mfccs)
        return feat

    def _ste(self, frame: np.ndarray) -> float:
        return 10 * np.log10(1 / frame.shape[0] * np.sum(frame**2))

    def _zcr(self, frame: np.ndarray) -> float:
        count = 0
        for i in range(1, frame.shape[0]):
            if frame[i] * frame[i - 1] < 0:
                count += 1
        return count / 2

    def _ber(self, frame: np.ndarray) -> float:
        sr = self.defaults.sample_rate
        berconf = self.cfg.band_energy_ratio
        assert isinstance(berconf, config.BandEnergyRatioConfig)
        llb = berconf.lband.lbound
        lub = berconf.lband.ubound
        ulb = berconf.uband.lbound
        uub = berconf.uband.ubound

        def bin_num(f: float, K: int) -> int:
            return int(np.floor((K * f) / sr))

        def band_energy(dft: np.ndarray, lb: float, ub: float) -> float:
            K = dft.shape[0]
            b1 = bin_num(lb, K)
            b2 = bin_num(ub, K)
            return np.sum(np.abs(dft[b1:b2]) ** 2)

        dft = np.fft.fft(frame, n=self.defaults.n_fft)
        E1 = band_energy(dft, llb, lub)
        E2 = band_energy(dft, ulb, uub)

        return 10 * np.log10(E1 / E2) if E2 > 0 else np.inf

    def _acc(self, frame: np.ndarray) -> float:
        ac = correlate(frame, frame, mode="full")
        ac = ac[ac.shape[0] // 2 :]
        min_lag = self.cfg.ac_coeff.min_lag_ms * int(self.defaults.sample_rate / 1000)
        max_lag = self.cfg.ac_coeff.max_lag_ms * int(self.defaults.sample_rate / 1000)
        return np.max(ac[min_lag:max_lag])

    def _srp(self, frame: np.ndarray) -> float:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        return float(
            libfeat.spectral_rolloff(
                y=frame,
                sr=self.defaults.sample_rate,
                roll_percent=self.cfg.s_rolloff_point.thr,
                n_fft=self.defaults.n_fft,
            )[0, 0]
        )

    def _sc(self, frame: np.ndarray) -> float:
        return float(
            libfeat.spectral_centroid(
                y=frame, sr=self.defaults.sample_rate, n_fft=self.defaults.n_fft
            )[0, 0]
        )

    def _ss(self, frame: np.ndarray) -> float:
        return float(
            libfeat.spectral_bandwidth(
                y=frame, sr=self.defaults.sample_rate, n_fft=self.defaults.n_fft
            )[0, 0]
        )

    def _sf(self, frame: np.ndarray) -> float:
        last_frame = self.last_frame
        if last_frame is None:
            return 0

        return np.sum(np.abs(np.fft.fft(frame) - np.fft.fft(last_frame)) ** 2)

    def _mfcc(self, frame: np.ndarray) -> np.ndarray:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        return libfeat.mfcc(
            y=frame,
            sr=self.defaults.sample_rate,
            n_mfcc=10,
            hop_length=len(frame) + 1,
            n_fft=self.defaults.n_fft,
        )
        return np.array([])

    def _mfcc_diff_norm(self, mfccs: np.ndarray) -> float:
        last_mfccs = self.last_mfcc
        if last_mfccs is None:
            return 0

        return np.sqrt(np.sum(np.abs(mfccs - last_mfccs) ** 2))
