"""Unit tests for SORT (Simple Online and Realtime Tracking) module.

Tests KalmanBoxTracker state evolution (7-dim state), 2-stage association
(IoU + normalized Euclidean center distance), and Track NMS deduplication.
"""
from typing import List

import numpy as np
import pytest

from modules.sort_tracker import KalmanBoxTracker, SORTTracker, Sort, compute_iou_batch


def test_compute_iou_batch() -> None:
    """Verify pairwise IoU matrix computation between bounding box sets.

    Returns:
        None
    """
    bb_test = np.array(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32
    )
    bb_gt = np.array(
        [[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]], dtype=np.float32
    )
    iou_matrix = compute_iou_batch(bb_test, bb_gt)
    assert iou_matrix.shape == (2, 2)
    assert iou_matrix[0, 0] == pytest.approx(1.0, 1e-4)
    assert iou_matrix[0, 1] == pytest.approx(1.0 / 3.0, 1e-4)
    assert iou_matrix[1, 0] == pytest.approx(0.0, 1e-4)


def test_kalman_box_tracker_7dim_state() -> None:
    """Verify KalmanBoxTracker initializes 7-dim state [u, v, s, r, u_dot, v_dot, s_dot].

    Returns:
        None
    """
    bbox = np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float32)
    tracker = KalmanBoxTracker(bbox=bbox, score=0.9, class_id=0.0)

    # 7-dimensional state check
    assert tracker.x.shape == (7,)
    assert isinstance(tracker.x, np.ndarray)

    # u = 150, v = 200, w = 100, h = 200 => s = 20000, r = 0.5
    assert tracker.x[0] == pytest.approx(150.0, 1e-3)
    assert tracker.x[1] == pytest.approx(200.0, 1e-3)
    assert tracker.x[2] == pytest.approx(20000.0, 1e-1)
    assert tracker.x[3] == pytest.approx(0.5, 1e-3)
    # Velocities initialized to zero
    assert tracker.x[4] == 0.0
    assert tracker.x[5] == 0.0
    assert tracker.x[6] == 0.0

    # Test predict advances age
    pred_box = tracker.predict()
    assert pred_box.shape == (4,)
    assert tracker.age == 1
    assert tracker.time_since_update == 1

    # Test update with shifted box estimates velocity
    new_box = np.array([110.0, 105.0, 210.0, 305.0], dtype=np.float32)
    tracker.update(new_box, score=0.95, class_id=0.0)
    assert tracker.time_since_update == 0
    assert tracker.hits == 2
    assert tracker.x[4] != 0.0  # u_dot should be non-zero after movement
    assert len(tracker.history) >= 2


def test_sort_tracker_identity_maintenance() -> None:
    """Verify SORTTracker assigns and maintains consistent track_id across frames.

    Returns:
        None
    """
    tracker = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.25)

    # Frame 1: Detection at (100, 100, 150, 150)
    det1 = np.array([[100.0, 100.0, 150.0, 150.0, 0.9, 0.0]], dtype=np.float32)
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    track_id_1 = int(tracks1[0, 0])

    # Frame 2: Slight movement to (104, 102, 154, 152)
    det2 = np.array([[104.0, 102.0, 154.0, 152.0, 0.88, 0.0]], dtype=np.float32)
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    track_id_2 = int(tracks2[0, 0])

    assert track_id_1 == track_id_2, "Track ID must persist across continuous frames"


def test_sort_tracker_center_distance_fallback() -> None:
    """Verify stage 2 association via normalized Euclidean center distance.

    Returns:
        None
    """
    tracker = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.3)

    # Frame 1: Vehicle at (100, 100, 160, 160)
    det1 = np.array([[100.0, 100.0, 160.0, 160.0, 0.9, 0.0]], dtype=np.float32)
    tracks1 = tracker.update(det1)
    track_id_1 = int(tracks1[0, 0])

    # Frame 2: Fast movement where IoU drops below 0.3 but center distance < 0.90
    # Shift by 30 pixels horizontally and vertically:
    # Overlap = 30x30 = 900, Union = 6300 => IoU ≈ 0.143 < 0.30 (fails Stage 1)
    # Center distance: sqrt((30/60)^2 + (30/60)^2) ≈ 0.707 < 0.90 (succeeds Stage 2)
    det2 = np.array([[130.0, 130.0, 190.0, 190.0, 0.85, 0.0]], dtype=np.float32)
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    track_id_2 = int(tracks2[0, 0])

    assert track_id_1 == track_id_2, "Fast-moving target must be matched via stage 2 center distance fallback"


def test_sort_tracker_track_nms_deduplication() -> None:
    """Verify Track NMS removes duplicate overlapping bounding boxes on the same vehicle.

    Returns:
        None
    """
    tracker = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.25, track_nms=0.45)

    # Two heavily overlapping detections (IoU > 0.80) for the same vehicle
    det = np.array(
        [
            [100.0, 100.0, 180.0, 180.0, 0.92, 0.0],
            [102.0, 101.0, 181.0, 182.0, 0.85, 0.0],
        ],
        dtype=np.float32,
    )
    tracks = tracker.update(det)
    # Track NMS must deduplicate to a single vehicle track
    assert len(tracks) == 1


def test_sort_alias() -> None:
    """Verify Sort is an alias of SORTTracker.

    Returns:
        None
    """
    assert Sort is SORTTracker
    tracker = Sort()
    assert isinstance(tracker, SORTTracker)
