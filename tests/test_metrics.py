"""Tests for tracking metrics (IoU, MOTA, IDF1, HOTA)."""
import numpy as np
import pytest


def test_iou_calculation():
    """Verify standard bounding box intersection over union calculation."""
    from modules.metrics import box_iou

    box1 = np.array([0.0, 0.0, 10.0, 10.0])
    box2 = np.array([0.0, 0.0, 10.0, 10.0])
    assert box_iou(box1, box2) == pytest.approx(1.0, 1e-4)

    box3 = np.array([5.0, 0.0, 15.0, 10.0])
    # Inter: 5 * 10 = 50. Union: 100 + 100 - 50 = 150. IoU: 50 / 150 = 1/3
    assert box_iou(box1, box3) == pytest.approx(1.0 / 3.0, 1e-4)

    box4 = np.array([20.0, 20.0, 30.0, 30.0])
    assert box_iou(box1, box4) == pytest.approx(0.0, 1e-4)


def test_mota_perfect_tracking():
    """Verify MOTA is 1.0 for perfect ground-truth tracking."""
    from modules.metrics import compute_mota

    # 3 frames, 2 targets per frame, perfectly matched
    gt_tracks = {
        0: [(1, np.array([10, 10, 20, 20])), (2, np.array([30, 30, 40, 40]))],
        1: [(1, np.array([12, 12, 22, 22])), (2, np.array([32, 32, 42, 42]))],
    }
    pred_tracks = {
        0: [(1, np.array([10, 10, 20, 20])), (2, np.array([30, 30, 40, 40]))],
        1: [(1, np.array([12, 12, 22, 22])), (2, np.array([32, 32, 42, 42]))],
    }

    metrics = compute_mota(gt_tracks, pred_tracks, iou_thresh=0.5)
    assert metrics["mota"] == pytest.approx(1.0, 1e-4)
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["idsw"] == 0
