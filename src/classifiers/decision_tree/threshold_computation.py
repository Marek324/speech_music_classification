# threshold_computation.py
# Marek Hric


import numpy as np
from scipy.optimize import brentq
from scipy.stats import gaussian_kde


def feature_thresholds(
    i: int, 
    S: np.ndarray, M: np.ndarray,
    pdf_s: gaussian_kde, pdf_m: gaussian_kde
) -> dict[str, dict[str, float|dict[str, float]]]:

    x_range: np.ndarray = np.linspace(
        min(S.min(), M.min()),
        max(S.max(), M.max()),
        1000
    )

    # d - discrete
    d_pdf_s = pdf_s(x_range)
    d_pdf_m = pdf_m(x_range)
    d_pdf_diff = d_pdf_s - d_pdf_m

    s_hprob = x_range[np.argmax(d_pdf_diff)]
    m_hprob = x_range[np.argmin(d_pdf_diff)]

    if s_hprob < m_hprob:
        s_ex = np.max(S[S < np.min(M)], initial=np.min(M))
        m_ex = np.min(M[M > np.max(S)], initial=np.max(S))
    else:
        s_ex = np.min(S[S > np.max(M)], initial=np.max(M))
        m_ex = np.max(M[M < np.min(S)], initial=np.min(S))

    def pdf_diff(x): return pdf_s(x) - pdf_m(x)

    sep = brentq(pdf_diff, min(s_hprob, m_hprob), max(s_hprob, m_hprob))

    ret = {
        'sx':        { 'thr': s_ex, 'metrics': {} },
        'mx':         { 'thr': m_ex, 'metrics': {} },
        'hs': { 'thr': s_hprob, 'metrics': {} },
        'mh':  { 'thr': m_hprob, 'metrics': {} },
        's':       { 'thr': sep }
    }

    metrics(ret, i, S, M)


    # debug
    if False:
        xs = float(ret['sx']['thr'])
        xm = float(ret['mx']['thr'])
        hs = float(ret['hs']['thr'])
        hm = float(ret['mh']['thr'])
        s = float(ret['s']['thr'])

        print(f"""
{i}: 
    s : {s},
    hs: {xs}, hm: {hm}
    xs: {xs}, xm: {xm}
    OK: {'YES' if ( xs >= hs >= s >= hm >= xm or xs <= hs <= s <= hm <= xm ) else 'NO'}
""")

    return ret


def metrics(
    thrs: dict[str, dict[str, float|dict[str, float]]],
    i: int, 
    S: np.ndarray, M: np.ndarray
):
    for key, data in thrs.items():
        if key == 's':
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


