"""Standard YOLO26 vehicle detector implementation."""
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from config import BASE_DIR, DEVICE, MODELS_DIR
from models.base_detector import BaseDetector

DEFAULT_YOLO26_WEIGHTS = BASE_DIR.parent.parent.parent / "K1 S2" / "Розпізн" / "yolo26n.pt"


class StandardYOLO26(BaseDetector):
    """Wrapper for standard YOLO26 / YOLOv8 baseline detector."""

    def __init__(self, weights_path: Optional[str | Path] = None) -> None:
        """Initializes the Standard YOLO26 detector.

        Args:
            weights_path: Path to .pt weights file. If None, checks models/ and K1 S2.
        """
        if weights_path is not None and Path(weights_path).exists():
            self.model_path = str(weights_path)
        elif (MODELS_DIR / "yolo26_uadetrac_best.pt").exists():
            self.model_path = str(MODELS_DIR / "yolo26_uadetrac_best.pt")
        elif DEFAULT_YOLO26_WEIGHTS.exists():
            self.model_path = str(DEFAULT_YOLO26_WEIGHTS)
        else:
            self.model_path = "yolov8n.pt"

        self.device = DEVICE
        self.model = YOLO(self.model_path)

    def predict(self, frame: np.ndarray, conf: float = 0.25) -> np.ndarray:
        """Performs object detection on frame.

        Args:
            frame: Input BGR image (H, W, 3).
            conf: Confidence threshold.

        Returns:
            np.ndarray: Array of shape (N, 6) with [x1, y1, x2, y2, score, cls].
        """
        # COCO vehicle classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        # If COCO model (80 classes), filter to vehicle classes only
        classes_filter = [2, 3, 5, 7] if len(self.model.names) > 10 else None
        results = self.model(
            frame,
            conf=conf,
            iou=0.45,
            device=self.device,
            classes=classes_filter,
            verbose=False,
        )

        boxes_list = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()[:, None]
                cls_ids = r.boxes.cls.cpu().numpy()[:, None]
                # Map COCO classes to standard vehicle IDs
                if len(self.model.names) > 10:
                    coco_to_std = {2: 0, 5: 1, 7: 2, 3: 4}
                    cls_ids = np.vectorize(lambda c: coco_to_std.get(int(c), 0))(cls_ids)
                row = np.hstack([xyxy, confs, cls_ids])
                boxes_list.append(row)

        if not boxes_list:
            return np.empty((0, 6), dtype=np.float32)

        return np.vstack(boxes_list).astype(np.float32)
