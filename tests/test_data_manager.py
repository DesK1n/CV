"""Tests for UA-DETRAC dataset manager and synthetic data generation."""
from pathlib import Path

import pytest

from config import UA_DETRAC_DIR
from data_manager import (
    UADetracDatasetManager,
    create_mock_video,
    generate_mock_sequence,
)


def test_no_synthetic_data_rule() -> None:
    """Verify that failure to locate UA-DETRAC raises RuntimeError when download disabled.

    Returns:
        None

    Raises:
        AssertionError: If RuntimeError is not raised with appropriate message.
    """
    manager = UADetracDatasetManager(destination_dir=Path("non_existent_test_dir"))
    with pytest.raises(RuntimeError) as exc_info:
        manager.prepare_dataset(download_if_missing=False)

    assert "UA-DETRAC" in str(exc_info.value) or "dataset" in str(exc_info.value)


def test_uadetrac_manager_validation() -> None:
    """Verify UA-DETRAC dataset manager validation and structure inspection methods.

    Returns:
        None
    """
    manager = UADetracDatasetManager(destination_dir=UA_DETRAC_DIR)
    assert manager.is_valid() is True
    stats = manager.verify_dataset_structure()
    assert "train_images" in stats
    assert "val_images" in stats
    assert stats["train_images"] > 0
    assert stats["val_images"] > 0


def test_mock_video_generation(tmp_path: Path) -> None:
    """Verify creation of mock synthetic test video.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None
    """
    mock_video_path = tmp_path / "test_mock.mp4"
    created = create_mock_video(
        mock_video_path, num_frames=10, width=320, height=240, fps=10
    )

    assert created.exists()
    assert created.stat().st_size > 0


def test_mock_sequence_generation(tmp_path: Path) -> None:
    """Verify generation of synthetic image sequences for unit tests.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None
    """
    seq_dir = tmp_path / "seq"
    created_dir = generate_mock_sequence(
        seq_dir, num_frames=5, width=320, height=240
    )

    assert created_dir.exists()
    frames = list(created_dir.glob("*.jpg"))
    assert len(frames) == 5
