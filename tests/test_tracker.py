"""Unit tests for the TrafficTracker engine in tracker.py.

Tests multi-tracker algorithm selection ('enhanced_sort', 'bytetrack', 'botsort'),
model detection backend switching, trail history, and frame streaming telemetry.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from data_manager import create_mock_video
from tracker import TrafficTracker, get_track_color


def test_get_track_color() -> None:
    """Verify distinct color tuple generation for tracking IDs.

    Returns:
        None
    """
    c0 = get_track_color(0)
    c1 = get_track_color(1)
    assert isinstance(c0, tuple) and len(c0) == 3
    assert isinstance(c1, tuple) and len(c1) == 3
    assert c0 != c1


def test_traffic_tracker_initialization_and_selection() -> None:
    """Verify TrafficTracker initialization and supported tracking algorithm selection.

    Returns:
        None
    """
    tracker = TrafficTracker(tracker_type="enhanced_sort", draw_trails=True)
    assert tracker.tracker_type == "enhanced_sort"

    # Switching to supported algorithms
    tracker.set_tracker("bytetrack")
    assert tracker.tracker_type == "bytetrack"

    tracker.set_tracker("botsort")
    assert tracker.tracker_type == "botsort"

    # Error on unsupported algorithm
    with pytest.raises(ValueError) as exc_info:
        tracker.set_tracker("invalid_tracker_algorithm")
    assert "Unsupported tracker" in str(exc_info.value)


def test_traffic_tracker_available_metadata() -> None:
    """Verify class methods for retrieving available trackers and models.

    Returns:
        None
    """
    trackers = TrafficTracker.get_available_trackers()
    assert isinstance(trackers, list)
    tracker_ids = [t["id"] for t in trackers]
    assert "enhanced_sort" in tracker_ids
    assert "bytetrack" in tracker_ids
    assert "botsort" in tracker_ids

    models = TrafficTracker.get_available_models()
    assert isinstance(models, list)
    assert len(models) >= 1
    for m in models:
        assert "id" in m and "name" in m and "path" in m


def test_traffic_tracker_reset() -> None:
    """Verify tracker reset clears history, counts, and active tracks.

    Returns:
        None
    """
    tracker = TrafficTracker(tracker_type="enhanced_sort")
    tracker.trail_history[1].append((100, 100))
    tracker.unique_track_ids.add(1)
    tracker.frame_index = 42

    tracker.reset()
    assert len(tracker.trail_history) == 0
    assert len(tracker.unique_track_ids) == 0
    assert tracker.frame_index == 0


def test_traffic_tracker_stream_frames_on_mock_video(tmp_path: Path) -> None:
    """Verify frame streaming generator executes and produces valid telemetry.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None
    """
    mock_vid = tmp_path / "stream_test.mp4"
    create_mock_video(mock_vid, num_frames=5, width=320, height=240, fps=10)

    tracker = TrafficTracker(tracker_type="enhanced_sort", draw_trails=True)

    records = []
    frame_count = 0
    for frame, stats in tracker.stream_frames(
        source=mock_vid, max_frames=3, draw=True, yield_jpeg=False
    ):
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)
        assert "frame_idx" in stats
        assert "active_tracks" in stats
        assert "total_count" in stats
        assert "fps" in stats
        assert "tracks" in stats
        records.append(stats)
        frame_count += 1

    assert frame_count == 3
    assert len(records) == 3
