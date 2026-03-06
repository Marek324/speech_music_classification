# tcn/__init__.py
# Causal TCN for online speech/music classification.
# Based on: Lemaire & Holzapfel, ISMIR 2019.

from .blocks import CausalConv1d, TCNResidualBlock
from .model import CausalTCN, SpeechMusicDetector
from .preprocess import LogMelSpectrogram
from .streaming import StreamingInference
from .training import build_loss, train_step

__all__ = [
    "CausalConv1d",
    "CausalTCN",
    "LogMelSpectrogram",
    "SpeechMusicDetector",
    "StreamingInference",
    "TCNResidualBlock",
    "build_loss",
    "train_step",
]
