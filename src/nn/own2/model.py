# nn/own2/model.py
# Own2Model — small-footprint TCN: log_mel_delta2 frontend + n_filters=8.
# See src/nn/own2/config.py for the default config and the programmatic override hook.

from ..tcn.model import SpeechMusicDetector
from .config import get_own2_config, get_own2_stats_path


class Own2Model(SpeechMusicDetector):
    """log_mel_delta2 frontend + n_filters=8 TCN backbone, no preprocessor, no tail.

    Inherits everything from SpeechMusicDetector — only the default config differs.
    Pass a custom ``cfg`` to override for experiment sweeps.
    """

    def __init__(self, cfg: dict | None = None, stats_path=None):
        cfg = cfg or get_own2_config()
        super().__init__(cfg=cfg, stats_path=stats_path or get_own2_stats_path())
