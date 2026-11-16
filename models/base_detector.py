"""Abstract base detector interface for CCTV vehicle detection."""
from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """Abstract interface that all vehicle detection models must implement."""

    @abstractmethod
    def predict(self, frame: np.ndarray, conf: float = 0.25) -> np.ndarray:
        """Runs object detection on a single BGR image.

        Args:
            frame: Input image array in BGR format with shape (H, W, 3).
            conf: Confidence threshold for filtering bounding boxes.

        Returns:
            np.ndarray: Detections array of shape (N, 6) where each row contains
                [x1, y1, x2, y2, confidence, class_id] in pixel coordinates.
                Returns empty array of shape (0, 6) if no objects detected.
        """
        pass
