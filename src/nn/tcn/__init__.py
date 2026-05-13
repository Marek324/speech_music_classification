# src/nn/tcn/__init__.py
# Marek Hric
# Causal TCN for online speech/music classification.
# Based on: Lemaire & Holzapfel, ISMIR 2019.

from .model import SpeechMusicDetector
from .streaming import StreamingInference

__all__ = ["SpeechMusicDetector", "StreamingInference"]
