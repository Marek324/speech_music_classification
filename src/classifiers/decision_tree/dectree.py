# decision_tree.py
# Marek Hric


import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import brentq

from ..sm_classifier import SMClassifier
from sm_lib import SMDataset


class SMDecisionTree(SMClassifier):
    def __init__(self):
        self.speech_pdfs: dict[int, gaussian_kde] = {} # {feat_i: pdf}
        self.music_pdfs: dict[int, gaussian_kde] = {}
        # {feat_i: {'ex_speech' : {'thr': float, 'metrics': {'I': float, 'Er': float}}}}
        self.thresholds: dict[int, dict[str, dict[str, float|dict[str, float]]]] = {} 
        self.top_features: dict[str, list[int]] = {
            'ex_speech': [], 'ex_music':  [],
            'high_prob_speech': [], 'high_prob_music': [],
            'separation': []
        }
        self.n_top_feat: int = 5


    def fit(self, X: SMDataset):
        print(X.speech.shape)
        print(X.music.shape)
        assert X.n_feat >= self.n_top_feat

        for i in range(X.n_feat):
            S = X.speech[:, i]
            M = X.music[:, i]

            self.speech_pdfs[i] = gaussian_kde(S)
            self.music_pdfs[i] = gaussian_kde(M)

            self._comp_thresholds(i, S, M)

            if False:
                debug_plot(self, i, S, M)
                #break

        self._select_features(X)
        #self._select_features_separation(X_speech, X_music)


    def predict(self, X: np.ndarray) -> int:
        pass
        return 1


    def _comp_thresholds(self, i: int, S: np.ndarray, M: np.ndarray):
        if np.mean(S) < np.mean(M):
            s_ex = np.max(S[S < np.min(M)], initial=np.min(M))
            m_ex = np.min(M[M > np.max(S)], initial=np.max(S))
        else:
            s_ex = np.min(S[S > np.max(M)], initial=np.max(M))
            m_ex = np.max(M[M < np.min(S)], initial=np.min(S))

        x_range: np.ndarray = np.linspace(
            min(S.min(), M.min()),
            max(S.max(), M.max()),
            1000
        )

        # d - discrete
        d_pdf_s = self.speech_pdfs[i](x_range)
        d_pdf_m = self.music_pdfs[i](x_range)
        d_pdf_diff = d_pdf_s - d_pdf_m

        s_hprob = x_range[np.argmax(d_pdf_diff)]
        m_hprob = x_range[np.argmin(d_pdf_diff)]

        def pdf_diff(x): return self.speech_pdfs[i](x) - self.music_pdfs[i](x)
        sep = brentq(pdf_diff, min(s_hprob, m_hprob), max(s_hprob, m_hprob))

        self.thresholds[i] = {
            'ex_speech':        { 'thr': s_ex, 'metrics': {} },
            'ex_music':         { 'thr': m_ex, 'metrics': {} },
            'high_prob_speech': { 'thr': s_hprob, 'metrics': {} },
            'high_prob_music':  { 'thr': m_hprob, 'metrics': {} },
            'separation':       { 'thr': sep }
        }

        self._comp_thr_metrics(i, S, M)


    def _comp_thr_metrics(self, i: int, S: np.ndarray, M: np.ndarray):
        for key, data in self.thresholds[i].items():
            if key == 'separation':
                continue

            if 'speech' in key:
                target, opp = S, M
            else:
                target, opp = M, S

            thr = data['thr']

            if np.mean(target) > np.mean(opp):
                inc = np.sum(target > thr)
                err = np.sum(opp > thr)
            else:
                inc = np.sum(target < thr)
                err = np.sum(opp < thr)

            inc = inc * 100 / len(target)
            err = err * 100 / len(opp)

            data['metrics'] = { 'I': inc, 'Er': err }


    def _select_features(self, X: SMDataset):
        def correlation(Xi: np.ndarray, Xj: np.ndarray) -> float:
            return np.dot(Xi, Xj) / np.sqrt(np.sum(Xi)**2 * np.sum(Xj)**2)

        def sep_score(j: int, C: list[float], X: SMDataset, K: list[int]) -> float:
            if len(K) == 0:
                return C[j]

            ALPHA = BETA = 0.5
            sep_power = ALPHA * C[j]
            other_correlation = np.sum([np.sum(np.abs(correlation(X.xs[:, j], X.xs[:, k]))) for k in K])
            return sep_power - (BETA / len(K)) * other_correlation

        for thr_n in ['ex_speech', 'ex_music', 'high_prob_speech', 'high_prob_music']:
            feat_i = list(range(X.n_feat))
            C = [
                m['I'] if 'ex' in thr_n
                else m['I']**2 / (m['Er'] + 1e-15)

                for _, data in self.thresholds.items()
                if (m := data[thr_n]['metrics']) 
            ]

            top_f = []

            while len(top_f) < self.n_top_feat:
                top_f.append(int(ik := np.argmax([sep_score(j, C, X, top_f) for j in feat_i])))
                feat_i.remove(feat_i[ik])

            self.top_features[thr_n] = top_f

        # selection of separation calculated here for now with FDR

        C = [
            (np.mean(X.speech[:, i]) - np.mean(X.music[:, i]))**2 /\
                (np.var(X.speech[:, i]) + np.var(X.music[:, i]))
            for i in range(X.n_feat)
        ]

        feat_i = list(range(X.n_feat))
        top_f = []

        while len(top_f) < 5:
            top_f.append(int(ik := np.argmax([sep_score(j, C, X, top_f) for j in feat_i])))
            feat_i.remove(feat_i[ik])

        self.top_features['separation'] = top_f


    # TODO: finish SFFS
    # not used now
    def _select_features_separation(self, X_speech: np.ndarray, X_music: np.ndarray):
        # cov expects data to be in transposed layout with rowvar=True (default)
        def sel_crit(X_speech: np.ndarray, X_music: np.ndarray) -> float:
            X = np.vstack((X_speech, X_music))
            Sspeech = np.cov(X_speech, rowvar=False)
            Smusic = np.cov(X_music, rowvar=True)
            Sw = (Sspeech + Smusic) / 2
            Sm = np.cov(X, rowvar=False)
            return np.abs(Sm) / np.abs(Sw)

        pass
