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
        device = next(model.parameters()).device
        self.buffer = torch.zeros(1, 0, device=device)
        self._state = None
        self._frames_emitted = 0

        m = cfg["model"]
        kernel_size = m["kernel_size"]
        n_layers = m["n_layers"]
        n_stacks = m["n_stacks"]
        left_rf_frames = 0
        for _ in range(n_stacks):
            for i in range(n_layers):
                left_rf_frames += 2 * (kernel_size - 1) * (2 ** i)  # 2 convs per block
        self.left_rf_samples = left_rf_frames * self.hop_length

    @torch.no_grad()
    def process_chunk(self, chunk):
        self.model.eval()
        chunk = chunk.unsqueeze(0).to(self.buffer.device)
        chunk_frames = chunk.shape[-1] // self.hop_length
        self.buffer = torch.cat([self.buffer, chunk], dim=-1)

        min_samples = self.hop_length * 4
        if self.buffer.shape[-1] < min_samples:
            return 0.0, 0.0, 0.0

        # Stateful-streaming protocol: model reads/advances state each call.
        # Stateless models ignore n_new_frames and return state unchanged.
        if self._state is None:
            self._state = {}
        self._state["n_new_frames"] = chunk_frames
        probs, self._state = self.model.forward_streaming(self.buffer, self._state)
        self._frames_emitted += chunk_frames

        speech_prob   = probs[0, 0, -1].item()
        music_prob    = probs[0, 1, -1].item()
        inactive_prob = probs[0, 2, -1].item()

        self.buffer = self.buffer[:, -self.left_rf_samples:]

        return speech_prob, music_prob, inactive_prob
