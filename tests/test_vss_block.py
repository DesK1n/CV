"""Tests for 2D Visual State Space (VSS) and SS2D PyTorch modules."""
import pytest
import torch


def test_vss_block_forward_shape():
    """Verify that VSSBlock preserves spatial tensor dimensions (B, C, H, W)."""
    from modules.vss_block import VSSBlock

    batch, channels, height, width = 2, 64, 16, 16
    x = torch.randn(batch, channels, height, width)

    block = VSSBlock(d_model=channels, d_state=16)
    out = block(x)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_vss_block_backward_pass():
    """Verify that gradients flow properly through the 4-way selective scan."""
    from modules.vss_block import VSSBlock

    batch, channels, height, width = 1, 32, 8, 8
    x = torch.randn(batch, channels, height, width, requires_grad=True)

    block = VSSBlock(d_model=channels, d_state=8)
    out = block(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients must propagate back to input tensor"
    assert not torch.isnan(x.grad).any(), "Gradients should not contain NaNs"
