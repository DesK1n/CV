"""Tests for Multi-Object Tracking (MOT) modules: SORTTracker, SSMTracker, and TrafficTracker."""
from pathlib import Path
import numpy as np
import pytest


def test_sort_tracker_maintains_identity():
    """Verify SORTTracker assigns and maintains track_id across consecutive frames."""
    from modules.sort_tracker import SORTTracker

    tracker = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.3)

    # Frame 1: Detection at (100, 100, 150, 150)
    det1 = np.array([[100.0, 100.0, 150.0, 150.0, 0.9, 0.0]], dtype=np.float32)
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    track_id_1 = int(tracks1[0, 0])

    # Frame 2: Slight movement to (105, 102, 155, 152)
    det2 = np.array([[105.0, 102.0, 155.0, 152.0, 0.88, 0.0]], dtype=np.float32)
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    track_id_2 = int(tracks2[0, 0])

    # The track identity must be preserved across frames
    assert track_id_1 == track_id_2, "Track ID must persist for continuous motion"


def test_ssm_tracker_state_modeling():
    """Verify SSMTracker handles track association via state space predictive memory."""
    from modules.ssm_tracker import SSMTracker

    tracker = SSMTracker(max_age=5, min_hits=1)

    det1 = np.array([[200.0, 200.0, 260.0, 260.0, 0.95, 0.0]], dtype=np.float32)
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    track_id_1 = int(tracks1[0, 0])

    det2 = np.array([[210.0, 205.0, 270.0, 265.0, 0.92, 0.0]], dtype=np.float32)
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    track_id_2 = int(tracks2[0, 0])

    assert track_id_1 == track_id_2, "SSMTracker must maintain consistent track_id"


def test_traffic_tracker_multi_algorithm():
    """Verify TrafficTracker instantiates and switches between enhanced_sort, bytetrack, botsort."""
    from tracker import TrafficTracker

    tracker = TrafficTracker(tracker_type="enhanced_sort")
    assert tracker.tracker_type == "enhanced_sort"

    tracker.set_tracker("bytetrack")
    assert tracker.tracker_type == "bytetrack"

    tracker.set_tracker("botsort")
    assert tracker.tracker_type == "botsort"

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    ann, stats = tracker.process_frame(frame)
    assert "fps" in stats
    assert "latency_ms" in stats
    assert "active_tracks" in stats
    assert "total_count" in stats
