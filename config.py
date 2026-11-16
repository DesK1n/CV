"""Configuration parameters and paths for CCTV Vehicle Detection and Tracking.

This module centralizes filesystem paths, hardware acceleration device setup,
target vehicle classification mappings, training hyperparameters, and tracking
constants (MAX_AGE, MIN_HITS, IOU_THRESH, TRACK_NMS).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

# Base filesystem directories
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "dataset"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
RUNS_DIR: Path = BASE_DIR / "runs"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
WEB_DIR: Path = BASE_DIR / "web"

# Ensure all primary working directories exist
for directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR, RUNS_DIR, OUTPUTS_DIR, WEB_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Determines optimal computational device with preference for CUDA.

    Returns:
        str: Device string identifier ('cuda:0' or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


DEVICE: str = get_device()

# UA-DETRAC & CCTV dataset paths and classes
UA_DETRAC_DIR: Path = DATA_DIR / "ua_detrac"
VEHICLE_CLASSES: Dict[int, str] = {
    0: "car",
    1: "bus",
    2: "van",
    3: "truck",
}

# Detection defaults
DEFAULT_CONF_THRESHOLD: float = 0.25
DEFAULT_IOU_THRESHOLD: float = 0.45
DEFAULT_INPUT_SIZE: Tuple[int, int] = (640, 640)

# Tracking constants
MAX_AGE: int = 15
MIN_HITS: int = 2
IOU_THRESH: float = 0.25
TRACK_NMS: float = 0.45

# Backward compatibility aliases
DEFAULT_TRACK_MAX_AGE: int = MAX_AGE
DEFAULT_TRACK_MIN_HITS: int = MIN_HITS
DEFAULT_IOU_MATCH_THRESH: float = IOU_THRESH
DEFAULT_TRACK_NMS: float = TRACK_NMS

# Model weights paths
DEFAULT_YOLO_WEIGHTS: Path = MODELS_DIR / "yolo26_uadetrac_best.pt"
UAV_YOLO_WEIGHTS: Path = MODELS_DIR / "uav_yolo26s_best.pt"
MAMBA_YOLO_WEIGHTS: Path = MODELS_DIR / "mamba_yolo_uadetrac_best.pt"

# Tracker configuration paths
CUSTOM_BYTETRACK_YAML: Path = BASE_DIR / "custom_bytetrack.yaml"
CUSTOM_BOTSORT_YAML: Path = BASE_DIR / "custom_botsort.yaml"


@dataclass(frozen=True)
class TrainingHyperparameters:
    """Hyperparameters for deep training sessions on RTX 5060 / CUDA.

    Attributes:
        epochs: Number of complete training epochs (100 for YOLO, 30 for Mamba).
        batch_size: Mini-batch size optimized for 8 GB VRAM.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization factor.
        warmup_epochs: Epochs for linear warmup.
        patience: Early stopping patience epochs.
        use_fp16: Whether to use mixed precision.
    """

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    patience: int = 20
    use_fp16: bool = True
