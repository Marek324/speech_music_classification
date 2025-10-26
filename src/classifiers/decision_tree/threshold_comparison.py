# threshold_comparison.py
# Marek Hric
#TODO: WTF refactor :D


import numpy as np


def comp_thresholds(
    x: np.ndarray,
    top: dict[str, list[int]],
    thrs: dict[int, dict[str, dict[str, float|dict[str, float]]]]
) -> dict[str, int]:
    res = {
        "sx": 0, "sh": 0, "sp": 0,
        "mx": 0, "mh": 0, "mp": 0
    }

    for t in top:
        sel_thrs = [thrs[s][t]['thr'] for s in top[t]]
        sel_feats = [x[s] for s in top[t]]

        if t == "p":
            res['sp'] = comp(sel_feats, sel_thrs, thrs, top[t], t)
            res['mp'] = len(top[t]) - res['sp']
            continue

        res[t] = comp(sel_feats, sel_thrs, thrs, top[t], t)

    return res


def comp(x: list[float], sel_thrs: list[float], thrs: dict[int, dict[str, dict[str, float|dict[str, float]]]], ti: list[int], tn: str) -> int:
    count = 0
    for f, t, i in zip(x, sel_thrs, ti):
        if max_direction(tn, thrs[i]):
            count += 1 if f >= t else 0
        else:
            count += 1 if f < t else 0

    return count


def max_direction(t:str, thrs: dict[str, dict[str, float|dict[str, float]]]) -> bool:
    """
    res = true if extreme speech threshold is bigger than extreme music threshold
    return flips res based on t (threshold name, e.g. no flip for speech thresholds)
    """
    res = float(thrs['sx']['thr']) >= float(thrs['mx']['thr'])
    return res if 's' in t else not res
