# decision_tree.py
# Marek Hric

import sys

DEBUG = True

try:
    import numpy as np
    from scipy.stats import gaussian_kde
    if DEBUG: import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)


class DecisionTree:
    def __init__(self):
        self.n_features = None
        self.speech_pdfs = {} # {feat_i: pdf}
        self.music_pdfs = {}
        self.thresholds = {} # {feat_i: {'ex_speech' : {'thr': float, 'metrics': {'I': float, 'Er': float}}}}
        self.n_top_feat = 5
        self.top_features = {
            'ex_speech': [], 'ex_music':  [],
            'high_prob_speech': [], 'high_prob_music': [],
            'separation': []
        }


    def train(self, X_speech: np.ndarray, X_music: np.ndarray):
        print(X_speech.shape)
        print(X_music.shape)
        assert X_speech.shape[1] == X_music.shape[1]
        assert X_speech.shape[1] >= self.n_top_feat

        self.n_features = X_speech.shape[1]
        for i in range(self.n_features):
            S = X_speech[:, i]
            M = X_music[:, i]

            self.speech_pdfs[i] = gaussian_kde(S)
            self.music_pdfs[i] = gaussian_kde(M)

            self._comp_thresholds(i, S, M)

            if DEBUG:
                debug_plot(self, i, S, M)
                #break

        self._select_features(X_speech, X_music)
        #self._select_features_separation(X_speech, X_music)

        # remove not selected thresholds


    def _comp_thresholds(self, i: int, S: np.ndarray, M: np.ndarray):
        if np.mean(S) < np.mean(M):
            s_ex = np.max(S[S < np.min(M)], initial=np.min(M))
            m_ex = np.min(M[M > np.max(S)], initial=np.max(S))
        else:
            s_ex = np.min(S[S > np.max(M)], initial=np.max(M))
            m_ex = np.max(M[M < np.min(S)], initial=np.min(S))

        x_range = np.linspace(
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

        from scipy.optimize import brentq
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
            if key == 'separation': continue

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


    def _select_features(self, X_speech: np.ndarray, X_music: np.ndarray):
        def correlation(Xi: np.ndarray, Xj: np.ndarray) -> float:
            return np.dot(Xi, Xj) / np.sqrt(np.sum(Xi)**2 * np.sum(Xj)**2)

        def sep_score(j: int, C: np.ndarray, X: np.ndarray, K: list[int]) -> float:
            if len(K) == 0:
                return C[j]

            ALPHA = BETA = 0.5
            sep_power = ALPHA * C[j]
            other_correlation = np.sum([np.sum(np.abs(correlation(X[:, j], X[:, k]))) for k in K])
            return sep_power - (BETA / len(K)) * other_correlation

        X = np.vstack((X_speech, X_music))

        feat_i = list(range(X.shape[1]))
        for thr_n in ['ex_speech', 'ex_music', 'high_prob_speech', 'high_prob_music']:
            C = [
                (m := data[thr_n]['metrics'])['I'] if 'ex' in thr_t
                else m['I']**2 / m['Er']
                for _, data in self.thresholds.item()
            ]

            top_f = [(first_i := np.argmax(C))] #TODO: test if while works with empty array
            feat_i.remove(first_i)

            while len(top_f) < self.n_top_feat:
                top_f.append(ik := np.argmax([sep_score(j, C, X, top_f) for j in feat_i]))
                feat_i.pop(ik)

            self.top_features[thr_n] = top_f

        # selection of separation calculated here for now with FDR

        C = [
            (np.mean(X_speech[:, i]) - np.mean(X_music[:, i]))**2 /\
                (np.var(X_speech[:, i]) + np.var(X_music[:, i]))
            for i in range(self.n_features)
        ]

        top_f = [(first_i := np.argmax(C))] #TODO: test if while works with empty array
        feat_i.remove(first_i)

        while len(top_f) < self.n_top_feat:
            top_f.append(ik := np.argmax([sep_score(j, C, X, top_f) for j in feat_i]))
            feat_i.pop(ik)

        self.top_features[thr_n] = top_f





    def _select_features_separation(self, X_speech: np.ndarray, X_music: np.ndarray):
        # cov expects data to be in transposed layout with rowvar=True (default)
        def sel_crit(
            X: np.ndarray,
            X_speech: np.ndarray,
            X_music: np.ndarray
        ) -> float:
            Sspeech = np.cov(X_speech, rowvar=False)
            Smusic = np.cov(X_music, rowvar=True)
            Sw = (Sspeech + Smusic) / 2
            Sm = np.cov(X, rowvar=False)
            return np.abs(Sm) / np.abs(Sw)

        X = np.vstack((X_speech, X_music))


def debug_plot(dectree: DecisionTree, i: int, S: np.ndarray, M: np.ndarray):
    x_min = min(S.min(), M.min())
    x_max = max(S.max(), M.max())

    x_range = np.linspace(x_min - 0.5, x_max + 0.5, 1000)

    plt.figure(figsize=(12,6))

    plt.hist(S, bins=75, density=True, alpha=0.5, label='Speech hist', color='blue')
    plt.hist(M, bins=75, density=True, alpha=0.5, label='Music hist', color='orange')

    s_y = dectree.speech_pdfs[i](x_range)
    m_y = dectree.music_pdfs[i](x_range)

    plt.plot(x_range, s_y, color='darkblue', linewidth=2, label='Speech distrib')
    plt.plot(x_range, m_y, color='darkorange', linewidth=2, label='Music distrib')

    thr = dectree.thresholds[i]
    plt.axvline(thr['ex_speech']['thr'], color='blue', linestyle='--', linewidth=2, label='Extreme Speech')
    plt.axvline(thr['ex_music']['thr'], color='orange', linestyle='--', linewidth=2, label='Extreme Music')
    plt.axvline(thr['high_prob_speech']['thr'], color='blue', linestyle=':', linewidth=2, label='High Prob Speech')
    plt.axvline(thr['high_prob_music']['thr'], color='orange', linestyle=':', linewidth=2, label='High Prob Music')
    plt.axvline(thr['separation']['thr'], color='red', linestyle='-', linewidth=2, label='Separation')

    plt.title(f'Feature {i}: Estimated PDFs and thresholds')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'debug/debug{i}.png')

    #for key, data in dectree.thresholds[i].items():
    #    print(key)
    #    if key == 'separation':
    #        print(f'\tthr: {data['thr']}')
    #    else:
    #        print(f'\tthr: {data['thr']}\tI: {data['metrics']['I']}\tEr: {data['metrics']['Er']}')

    for key, data in dectree.top_features.items():
        print(f'Top features for {key}:')
        print(f'\t{data}')

