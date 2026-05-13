# scripts/parity_tcn_streaming.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.
"""Parity check: streaming TCN forward is bit-identical to offline forward.

Usage:
    uv run python scripts/parity_tcn_streaming.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nn.tcn.model import SpeechMusicDetector


CFG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 80,
    "f_min": 27.5,
    "f_max": 8000.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "seq_len": 128,
    "dataset": {"url": None, "name": "full"},
    "model": {
        "n_filters": 8,
        "kernel_size": 3,
        "n_layers": 2,
        "n_stacks": 1,
        "dropout": 0.0,
        "n_classes": 3,
        "use_weight_norm": False,
    },
}


def _make_detector() -> SpeechMusicDetector:
    model = SpeechMusicDetector(cfg=CFG, stats_path=Path("/nonexistent"))
    model.fe.norm_mean = torch.zeros(1, CFG["n_mels"], 1)
    model.fe.norm_std = torch.ones(1, CFG["n_mels"], 1)
    model.fe._stats_loaded = True
    model.eval()
    return model


def main() -> int:
    torch.manual_seed(0)
    model = _make_detector()

    torch.manual_seed(0)
    audio = torch.randn(1, CFG["sample_rate"])

    with torch.no_grad():
        offline = model(audio)
        streaming, state = model.forward_streaming(audio, state=None)

    if torch.equal(offline, streaming):
        print("PASS: streaming output is bit-identical to offline output")
        print(f"  output shape: {tuple(offline.shape)} (batch, classes, frames)")
        print(f"  state for stateless TCN: {state}")
        return 0

    diff = (offline - streaming).abs().max().item()
    print(f"FAIL: streaming and offline outputs differ — max abs diff = {diff:.6e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
