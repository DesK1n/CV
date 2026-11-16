"""Tests for detection model wrappers (BaseDetector, StandardYOLO26, MambaYOLO)."""
import numpy as np
import pytest
import torch


def test_detector_interface_and_inference():
    """Verify that both detectors adhere to the unified predict() signature."""
    from models.base_detector import BaseDetector
    from models.baseline_yolo26 import StandardYOLO26
    from models.mamba_yolo import MambaYOLO

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    yolo_model = StandardYOLO26()
    assert isinstance(yolo_model, BaseDetector)
    yolo_preds = yolo_model.predict(dummy_frame, conf=0.1)
    assert isinstance(yolo_preds, np.ndarray)
    assert yolo_preds.ndim == 2
    assert yolo_preds.shape[1] == 6  # [x1, y1, x2, y2, score, class_id]

    mamba_model = MambaYOLO()
    assert isinstance(mamba_model, BaseDetector)
    mamba_preds = mamba_model.predict(dummy_frame, conf=0.1)
    assert isinstance(mamba_preds, np.ndarray)
    assert mamba_preds.ndim == 2
    assert mamba_preds.shape[1] == 6
