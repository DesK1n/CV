"""Tests for project configuration constants, paths, and hyperparameters."""
from pathlib import Path

import pytest

from config import (
    BASE_DIR,
    CUSTOM_BOTSORT_YAML,
    CUSTOM_BYTETRACK_YAML,
    DATA_DIR,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_YOLO_WEIGHTS,
    DEVICE,
    IOU_THRESH,
    MAMBA_YOLO_WEIGHTS,
    MAX_AGE,
    MIN_HITS,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    TRACK_NMS,
    TrainingHyperparameters,
    UA_DETRAC_DIR,
    UAV_YOLO_WEIGHTS,
    VEHICLE_CLASSES,
    WEB_DIR,
    get_device,
)


def test_config_paths_and_constants() -> None:
    """Verify that configuration defines necessary paths, device, and tracking parameters.

    Returns:
        None
    """
    assert isinstance(BASE_DIR, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(MODELS_DIR, Path)
    assert isinstance(REPORTS_DIR, Path)
    assert isinstance(RUNS_DIR, Path)
    assert isinstance(OUTPUTS_DIR, Path)
    assert isinstance(WEB_DIR, Path)
    assert isinstance(UA_DETRAC_DIR, Path)

    # Tracking parameters
    assert MAX_AGE == 15
    assert MIN_HITS == 2
    assert IOU_THRESH == 0.25
    assert TRACK_NMS == 0.45

    # Device
    assert isinstance(DEVICE, str)
    assert DEVICE in ("cpu", "cuda:0")
    assert isinstance(get_device(), str)

    # Thresholds
    assert 0.0 < DEFAULT_CONF_THRESHOLD < 1.0
    assert 0.0 < DEFAULT_IOU_THRESHOLD < 1.0


def test_config_classes_and_weights() -> None:
    """Verify vehicle class mappings and model weights file configuration.

    Returns:
        None
    """
    assert isinstance(VEHICLE_CLASSES, dict)
    assert VEHICLE_CLASSES[0] == "car"
    assert VEHICLE_CLASSES[1] == "bus"
    assert VEHICLE_CLASSES[2] == "van"
    assert VEHICLE_CLASSES[3] == "truck"

    # Model and tracker configuration files
    assert isinstance(DEFAULT_YOLO_WEIGHTS, Path)
    assert isinstance(UAV_YOLO_WEIGHTS, Path)
    assert isinstance(MAMBA_YOLO_WEIGHTS, Path)
    assert isinstance(CUSTOM_BYTETRACK_YAML, Path)
    assert isinstance(CUSTOM_BOTSORT_YAML, Path)

    assert CUSTOM_BYTETRACK_YAML.exists()
    assert CUSTOM_BOTSORT_YAML.exists()


def test_training_hyperparameters() -> None:
    """Verify default values of TrainingHyperparameters dataclass.

    Returns:
        None
    """
    params = TrainingHyperparameters()
    assert params.epochs == 100
    assert params.batch_size == 32
    assert params.learning_rate == 1e-3
    assert params.use_fp16 is True
