# feature_selection.py
# Marek Hric


import numpy as np

from sm_lib import SMDataset


def select_features(
    X: SMDataset,
    thrs: dict[int, dict[str, dict[str, float|dict[str, float]]]],
    n_top: int
) -> dict[str, list[int]]:
    top_features: dict[str, list[int]] = {
            'sx': [], 'mx':  [],
            'hs': [], 'mh': [],
            's': []
        }

    for thr in top_features:
        C: list[float] = []
        if 's':
            C = [ (np.mean(X.speech[:, i]) - np.mean(X.music[:, i]))**2 /\
                    (np.var(X.speech[:, i]) + np.var(X.music[:, i]))
                for i in range(X.n_feat) ]
        else:
            if 'ex' in thr:
                C = [ data[thr]['metrics']['I'] for _, data in thrs.items() ]
            else:
                C = [ m['I']**2 / (m['Er'] + 1e-15) 
                        for _, data in thrs.items() if (m := data[thr]['metrics']) ]

        top_features[thr] = select(X, C, n_top)

    return top_features


def select(X: SMDataset, C: list[float], n_top: int) -> list[int]:
    feat_i = list(range(X.n_feat))
    top_f = []

    while len(top_f) < n_top:
        top_f.append(int(ik := np.argmax([sep_score(j, C, X, top_f) for j in feat_i])))
        feat_i.remove(feat_i[ik])

    return top_f


def sep_score(j: int, C: list[float], X: SMDataset, K: list[int]) -> float:
    def correlation(Xi: np.ndarray, Xj: np.ndarray) -> float:
        return np.dot(Xi, Xj) / np.sqrt(np.sum(Xi)**2 * np.sum(Xj)**2)

    if len(K) == 0:
        return C[j]

    ALPHA = BETA = 0.5
    sep_power = ALPHA * C[j]
    other_correlation = np.sum([np.sum(np.abs(correlation(X.xs[:, j], X.xs[:, k]))) for k in K])
    return sep_power - (BETA / len(K)) * other_correlation




# TODO: finish SFFS
# not used now
# remove s from thr list and first if
def select_features_s(dectree: None, X_speech: np.ndarray, X_music: np.ndarray):
    def sel_crit(X_speech: np.ndarray, X_music: np.ndarray) -> float:
        X = np.vstack((X_speech, X_music))
        # cov expects data to be in transposed layout with rowvar=True (default)
        Sspeech = np.cov(X_speech, rowvar=False)
        Smusic = np.cov(X_music, rowvar=True)
        Sw = (Sspeech + Smusic) / 2
        Sm = np.cov(X, rowvar=False)
        return np.abs(Sm) / np.abs(Sw)

    pass
