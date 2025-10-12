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


    def train(self, X_speech: np.ndarray, X_music: np.ndarray):
        print(X_speech.shape)
        print(X_music.shape)

        self.n_features = X_speech.shape[1]
        for i in range(self.n_features):
            S = X_speech[:, i]
            M = X_music[:, i]

            self.speech_pdfs[i] = gaussian_kde(S)
            self.music_pdfs[i] = gaussian_kde(M)

            self._comp_thresholds(i, S, M)

            if DEBUG:
                debug_plot(self, i, S, M)
                break


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

            target, opp = (S, M) if 'speech' in key else (M, S)
            thr = data['thr']
            inc = np.sum(target > thr) / len(target)
            err = np.sum(opp > thr) / len(opp) if 'ex' in key else 0.0

            data['metrics'] = { 'I': inc, 'Er': err }


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
    plt.xlabel(f'Value')
    plt.ylabel(f'Density')
    plt.legend()
    plt.savefig('debug.png')

    for key, data in dectree.thresholds[i].items():
        print(key)
        if key == 'separation':
            print(f'\tthr: {data['thr']}')
        else:
            print(f'\tthr: {data['thr']}\tI: {data['metrics']['I']}\tEr: {data['metrics']['Er']}')


