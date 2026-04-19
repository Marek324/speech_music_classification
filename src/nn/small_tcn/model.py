# nn/small_tcn/model.py
# SmallTCN — small-footprint TCN: log_mel_delta2 frontend + n_filters=8.
# See src/nn/small_tcn/config.py for the default config and the programmatic override hook.

from ..tcn.model import SpeechMusicDetector
from .config import get_small_tcn_config, get_small_tcn_stats_path


class SmallTCN(SpeechMusicDetector):
    """log_mel_delta2 frontend + n_filters=8 TCN backbone, no preprocessor, no tail.

    Inherits everything from SpeechMusicDetector — only the default config differs.
    Pass a custom ``cfg`` to override for experiment sweeps.
    """

    def __init__(self, cfg: dict | None = None, stats_path=None):
        cfg = cfg or get_small_tcn_config()
        super().__init__(cfg=cfg, stats_path=stats_path or get_small_tcn_stats_path())
