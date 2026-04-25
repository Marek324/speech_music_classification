"""Cheap existence probe for model weights used by the demo UI.

Called from the page-load handler to badge the picker: no torch import here,
no model construction — just filesystem checks.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WEIGHTS = _REPO_ROOT / "weights"

CLASSIC_MODELS = ("decision_tree", "gmm", "svm")
NN_MODELS = ("tcn", "tcn_lstm", "small_tcn")
ALL_MODELS = CLASSIC_MODELS + NN_MODELS

_NN_PATHS = {
    "tcn": (
        _WEIGHTS / "tcn.safetensors",
        _WEIGHTS / "tcn_preprocess_stats.pt",
    ),
    "tcn_lstm": (
        _WEIGHTS / "tcn_lstm" / "tcn_lstm.safetensors",
        _WEIGHTS / "tcn_lstm" / "tcn_lstm_preprocess_stats.pt",
    ),
    "small_tcn": (
        _WEIGHTS / "small_tcn" / "tcn_small.safetensors",
        _WEIGHTS / "small_tcn" / "tcn_small_preprocess_stats.pt",
    ),
}


def has_weights(model_name: str) -> bool:
    if model_name in CLASSIC_MODELS:
        return (_WEIGHTS / model_name).exists()
    if model_name in NN_MODELS:
        return all(p.exists() for p in _NN_PATHS[model_name])
    return False


def all_status() -> dict[str, bool]:
    return {name: has_weights(name) for name in ALL_MODELS}
