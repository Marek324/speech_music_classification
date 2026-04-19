# nn/tcn_lstm/model.py
# TCNLSTM — thin wrapper over SpeechMusicDetector baked to the delta2+conv1d+LSTM config.
# See src/nn/tcn_lstm/config.py for the default config and the programmatic override hook.

from ..tcn.model import SpeechMusicDetector
from .config import get_tcn_lstm_config, get_tcn_lstm_stats_path


class TCNLSTM(SpeechMusicDetector):
    """TCN + log_mel_delta2 frontend + conv1d preprocessor + LSTM tail.

    Inherits everything from SpeechMusicDetector — only the default config differs.
    Pass a custom ``cfg`` to override for experiment sweeps.
    """

    def __init__(self, cfg: dict | None = None, stats_path=None):
        cfg = cfg or get_tcn_lstm_config()
        super().__init__(cfg=cfg, stats_path=stats_path or get_tcn_lstm_stats_path())
