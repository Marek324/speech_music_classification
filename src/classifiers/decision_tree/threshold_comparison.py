# threshold_comparison.py
# Marek Hric


import numpy as np


def comp_thresholds(
    x: np.ndarray,  # frame features
    top: dict[str, list[int]],  # selected features indices
    thrs: dict[
        int, dict[str, dict[str, float | bool | dict[str, float]]]
    ],  # all feature thresholds
) -> dict[str, int]:
    res = {"sx": 0, "sh": 0, "ss": 0, "mx": 0, "mh": 0, "ms": 0}

    for t_name, feat_ids in top.items():
        count = sum(
            (x[i] > thrs[i][t_name]["thr"])
            if thrs[i][t_name]["dir_max"]
            else (x[i] < thrs[i][t_name]["thr"])
            for i in feat_ids
        )
        if t_name == "s":
            res["ss"], res["ms"] = count, len(feat_ids) - count
        else:
            res[t_name] = count

    return res
