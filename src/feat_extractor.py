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
        assert isinstance(cfg, config.Config)
        self.cfg = cfg.fext
        self.defaults = cfg.defaults
        self.last_frame: Optional[np.ndarray]
        if self.cfg.mfcc.enable:
            self.last_mfcc: Optional[np.ndarray]

    def extract(self, frame: np.ndarray) -> np.ndarray:
        assert isinstance(self.cfg, config.FeatExtractorConfig)
        feats = []
        mfccs = None
        # DEBUG
        feat_names = []

        if self.cfg.mfcc.enable:
            mfccs = self._mfcc(frame).flatten()
            mfccs = np.nan_to_num(mfccs, nan=0.0, posinf=0.0, neginf=0.0)
            feats.append(mfccs)
            feat_names.append("mfcc")
            if self.cfg.mfcc_diff_norm.enable:
                feats.append(self._mfcc_diff_norm(mfccs))
                feat_names.append("mfcc_diff_norm")

            self.last_mfcc = deepcopy(mfccs)

        if self.cfg.st_energy.enable:
            feats.append(self._ste(frame))
            feat_names.append("ste")
        if self.cfg.zcr.enable:
            feats.append(self._zcr(frame))
            feat_names.append("zcr")
        if self.cfg.band_energy_ratio.enable:
            feats.append(self._ber(frame))
            feat_names.append("ber")
        if self.cfg.ac_coeff.enable:
            feats.append(self._acc(frame))
            feat_names.append("acc")
        if self.cfg.s_rolloff_point.enable:
            feats.append(self._srp(frame))
            feat_names.append("srp")
        if self.cfg.s_centroid.enable:
            feats.append(self._sc(frame))
            feat_names.append("sc")
        if self.cfg.s_flux.enable:
            feats.append(self._sf(frame))
            feat_names.append("sf")

        # DEBUG: Check each feature before concatenation
        # for i, (f, name) in enumerate(zip(feats, feat_names)):
        #     if np.any(np.isinf(f)) or np.any(np.isnan(f)):
        #         print(f"WARNING: Feature '{name}' contains inf/nan: {f}")
        #     feats[i] = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

        feat = np.hstack(feats)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feat /= np.linalg.norm(feat) + 1e-10

        self.last_frame = deepcopy(frame)
        return feat

    def _ste(self, frame: np.ndarray) -> float:
        return 10 * np.log10(1 / frame.shape[0] * np.sum(frame**2) + 1e-10)

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
        ratio = (E1 + 1e-10) / (E2 + 1e-10)
        ratio = np.clip(ratio, 1e-10, 1e10)
        result = 10 * np.log10(ratio)

        return 0.0 if np.isinf(result) or np.isnan(result) else float(result)

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
        value = float(
            libfeat.spectral_centroid(
                y=frame, sr=self.defaults.sample_rate, n_fft=self.defaults.n_fft
            )[0, 0]
        )
        return 0.0 if np.isnan(value) or np.isinf(value) else value

    def _ss(self, frame: np.ndarray) -> float:
        return float(
            libfeat.spectral_bandwidth(
                y=frame, sr=self.defaults.sample_rate, n_fft=self.defaults.n_fft
            )[0, 0]
        )

    def _sf(self, frame: np.ndarray) -> float:
        last_frame = getattr(self, "last_frame", None)
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
