"""Tests for the video tracking pipeline and CLI execution."""
from pathlib import Path
import numpy as np
import pytest
import cv2


def test_run_tracking_on_synthetic_sequence(tmp_path: Path):
    """Verify run_tracking processes frame sequence and outputs valid JSON records."""
    from track import run_tracking
    from models.mamba_yolo import MambaYOLO
    from modules.ssm_tracker import SSMTracker

    # Generate small sequence of 5 frames
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Moving vehicle
        cv2.rectangle(img, (50 + i * 5, 50), (100 + i * 5, 90), (255, 255, 255), -1)
        cv2.imwrite(str(seq_dir / f"frame_{i:03d}.jpg"), img)

    out_json = tmp_path / "output.json"
    detector = MambaYOLO()
    tracker = SSMTracker()

    records = run_tracking(
        source=seq_dir,
        detector=detector,
        tracker=tracker,
        conf_thresh=0.1,
        out_json=out_json,
    )

    assert len(records) == 5
    assert out_json.exists()
    assert "frame" in records[0] and "latency_ms" in records[0]
