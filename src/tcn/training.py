# tcn/training.py
# Training utilities.

import torch.nn as nn


def build_loss() -> nn.Module:
    """BCE loss for multi-label frame-level classification."""
    return nn.BCELoss()


def train_step(model, optimizer, loss_fn, waveform, targets):
    """
    Single training step.

    Args:
        waveform: (B, samples)
        targets:  (B, 2, T) float 0/1 labels for [speech, music] per frame
    Returns:
        scalar loss
    """
    model.train()
    optimizer.zero_grad()
    probs = model(waveform)
    loss = loss_fn(probs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()
