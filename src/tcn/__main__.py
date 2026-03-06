# tcn/__main__.py
# Run with: uv run python -m src.tcn

import logging

import torch

from . import SpeechMusicDetector, StreamingInference
from .config import get_config

log = logging.getLogger(__name__)


def main():
    cfg = get_config()
    sr = cfg["sample_rate"]
    hop = cfg["hop_length"]
    m = cfg["model"]

    log.info("Building model...")
    detector = SpeechMusicDetector(sample_rate=sr)

    n_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    log.info("Trainable parameters: %s", f"{n_params:,}")

    n_layers, n_stacks, k = m["n_layers"], m["n_stacks"], m["kernel_size"]
    rf_frames = n_stacks * sum(2**i for i in range(n_layers)) * (k - 1) + 1
    rf_seconds = rf_frames * hop / sr
    log.info("Receptive field: %s frames (~%.1fs)", rf_frames, rf_seconds)

    dummy_audio = torch.randn(2, sr * 5)
    probs = detector(dummy_audio)
    log.info("Input shape:  %s", dummy_audio.shape)
    log.info("Output shape: %s  (batch, [speech, music], frames)", probs.shape)

    log.info("Streaming inference (10 chunks)...")
    streamer = StreamingInference(detector, hop_length=hop, sample_rate=sr)
    for i in range(10):
        chunk = torch.randn(sr // 2)
        sp, mu = streamer.process_chunk(chunk)
        log.info("  chunk %02d: speech=%.3f  music=%.3f", i + 1, sp, mu)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    main()
