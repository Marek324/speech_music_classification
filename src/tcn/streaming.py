# tcn/streaming.py
# Chunk-by-chunk online inference.

import torch

from .config import get_config


class StreamingInference:
    """Wraps the model for streaming inference."""

    def __init__(self, model, hop_length=None, sample_rate=None):
        self.model = model
        cfg = get_config()
        self.hop_length = hop_length or cfg["hop_length"]
        self.sample_rate = sample_rate or cfg["sample_rate"]
        self.buffer = torch.zeros(1, 0)

    @torch.no_grad()
    def process_chunk(self, chunk):
        self.model.eval()
        chunk = chunk.unsqueeze(0)
        self.buffer = torch.cat([self.buffer, chunk], dim=-1)

        min_samples = self.hop_length * 4
        if self.buffer.shape[-1] < min_samples:
            return 0.0, 0.0

        probs = self.model(self.buffer)
        speech_prob = probs[0, 0, -1].item()
        music_prob = probs[0, 1, -1].item()

        max_samples = self.sample_rate * 10
        self.buffer = self.buffer[:, -max_samples:]

        return speech_prob, music_prob
