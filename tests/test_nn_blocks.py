# tests/test_nn_blocks.py

import torch
import pytest
from src.nn.blocks import CausalConv1d, TCNResidualBlock


# ── CausalConv1d ──────────────────────────────────────────────────────────

def test_causal_conv_output_shape():
    layer = CausalConv1d(4, 8, kernel_size=3, dilation=1)
    x = torch.randn(2, 4, 50)
    out = layer(x)
    assert out.shape == (2, 8, 50)


def test_causal_conv_output_shape_dilation():
    layer = CausalConv1d(4, 4, kernel_size=5, dilation=4)
    x = torch.randn(1, 4, 100)
    out = layer(x)
    assert out.shape == (1, 4, 100)


def test_causal_conv_causality():
    """Output at time t must not depend on input at time t+1 or later."""
    torch.manual_seed(0)
    layer = CausalConv1d(4, 4, kernel_size=5, dilation=2)
    layer.eval()

    x = torch.randn(1, 4, 40)
    with torch.no_grad():
        out_full = layer(x)

    # Zero out all future frames beyond t=20 and recompute
    t = 20
    x_masked = x.clone()
    x_masked[:, :, t:] = 0.0
    with torch.no_grad():
        out_masked = layer(x_masked)

    # Outputs up to t should be identical
    assert torch.allclose(out_full[:, :, :t], out_masked[:, :, :t], atol=1e-5)


# ── TCNResidualBlock ──────────────────────────────────────────────────────

def test_residual_block_same_channels_output_shape():
    block = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0)
    x = torch.randn(2, 8, 50)
    out = block(x)
    assert out.shape == (2, 8, 50)


def test_residual_block_diff_channels_output_shape():
    block = TCNResidualBlock(4, 8, kernel_size=3, dilation=1, dropout=0.0)
    x = torch.randn(2, 4, 50)
    out = block(x)
    assert out.shape == (2, 8, 50)


def test_residual_block_same_channels_no_downsample():
    block = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0)
    assert block.downsample is None


def test_residual_block_diff_channels_has_downsample():
    block = TCNResidualBlock(4, 8, kernel_size=3, dilation=1, dropout=0.0)
    assert block.downsample is not None


def test_residual_block_weight_norm():
    block = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0, use_weight_norm=True)
    x = torch.randn(1, 8, 20)
    out = block(x)
    assert out.shape == (1, 8, 20)
    assert block.bn1 is None
    assert block.bn2 is None


def test_residual_block_batch_norm():
    block = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0, use_weight_norm=False)
    assert block.bn1 is not None
    assert block.bn2 is not None


def test_residual_block_causality():
    """Block output at time t must not depend on input at time t+1 or later."""
    torch.manual_seed(1)
    block = TCNResidualBlock(8, 8, kernel_size=5, dilation=2, dropout=0.0)
    block.eval()

    x = torch.randn(1, 8, 60)
    with torch.no_grad():
        out_full = block(x)

    t = 30
    x_masked = x.clone()
    x_masked[:, :, t:] = 0.0
    with torch.no_grad():
        out_masked = block(x_masked)

    assert torch.allclose(out_full[:, :, :t], out_masked[:, :, :t], atol=1e-5)


# ── Activation variants ───────────────────────────────────────────────────────

@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "elu", "gelu"])
def test_residual_block_activation_output_shape(activation):
    block = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0,
                             activation=activation)
    x = torch.randn(2, 8, 50)
    out = block(x)
    assert out.shape == (2, 8, 50)


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "elu", "gelu"])
def test_residual_block_activation_output_finite(activation):
    block = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0,
                             activation=activation)
    block.eval()
    x = torch.randn(2, 8, 50)
    with torch.no_grad():
        out = block(x)
    assert torch.isfinite(out).all(), f"non-finite output with activation={activation}"


def test_residual_block_default_activation_is_relu():
    """Default activation must behave identically to explicit relu."""
    torch.manual_seed(42)
    block_default = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0)
    torch.manual_seed(42)
    block_relu = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0,
                                  activation="relu")
    block_default.eval()
    block_relu.eval()
    x = torch.randn(1, 8, 30)
    with torch.no_grad():
        assert torch.allclose(block_default(x), block_relu(x))


def test_residual_block_unknown_activation_falls_back_to_relu():
    """Unknown activation string should fall back to relu without raising."""
    block = TCNResidualBlock(8, 8, kernel_size=3, dilation=1, dropout=0.0,
                             activation="unknown_act")
    x = torch.randn(1, 8, 20)
    out = block(x)
    assert out.shape == (1, 8, 20)
