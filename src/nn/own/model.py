# nn/own/model.py
# OwnModel — thin wrapper over SpeechMusicDetector baked to the delta2_conv1d config.
# See src/nn/own/config.py for the default config and the programmatic override hook.

from ..tcn.model import SpeechMusicDetector
from .config import get_own_config, get_own_stats_path


class OwnModel(SpeechMusicDetector):
    """TCN + log_mel_delta2 frontend + conv1d preprocessor.

    Inherits everything from SpeechMusicDetector — only the default config differs.
    Pass a custom ``cfg`` to override for experiment sweeps.
    """

    def __init__(self, cfg: dict | None = None, stats_path=None):
        cfg = cfg or get_own_config()
        super().__init__(cfg=cfg, stats_path=stats_path or get_own_stats_path())
