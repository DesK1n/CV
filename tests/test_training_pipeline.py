"""Tests for training pipelines (YOLO26 and Mamba-YOLO)."""
from pathlib import Path
import pytest
import torch


def test_train_mamba_loss_computation():
    """Verify combined detection loss computation for Mamba-YOLO."""
    from train_mamba import compute_detection_loss

    # Batch of 2, 5 classes, spatial grid 20x20
    pred_cls = torch.randn(2, 5, 20, 20)
    pred_reg = torch.randn(2, 4, 20, 20)

    target_cls = torch.zeros(2, 20, 20, dtype=torch.long)
    target_boxes = torch.rand(2, 4, 20, 20)  # [cx, cy, w, h] normalized

    loss, loss_dict = compute_detection_loss(
        pred_cls, pred_reg, target_cls, target_boxes
    )

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert "cls_loss" in loss_dict and "box_loss" in loss_dict
