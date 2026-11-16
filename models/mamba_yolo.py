"""Mamba-YOLO hybrid detector implementation using Visual State Space (VSS) blocks."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import BASE_DIR, DEVICE, MAMBA_YOLO_WEIGHTS
from models.base_detector import BaseDetector
from modules.vss_block import VSSBlock


class MambaYOLONet(nn.Module):
    """Deep neural network combining CNN backbone with 2D Visual State Space neck."""

    def __init__(self, num_classes: int = 5) -> None:
        """Initializes the MambaYOLONet.

        Args:
            num_classes: Number of vehicle classification categories.
        """
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        self.vss_neck = VSSBlock(d_model=128, d_state=16)

        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass generating classification logits and bounding box offsets.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Tuple of (pred_cls, pred_reg):
                pred_cls: Class logits (B, num_classes, H/32, W/32).
                pred_reg: Box offsets (B, 4, H/32, W/32).
        """
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.vss_neck(feat)

        cls_out = self.cls_head(feat)
        reg_out = self.reg_head(feat)
        return cls_out, reg_out


class MambaYOLO(BaseDetector):
    """Wrapper for Mamba-YOLO detector conforming to BaseDetector interface."""

    def __init__(self, weights_path: Optional[str | Path] = None) -> None:
        """Initializes MambaYOLO detector.

        Args:
            weights_path: Path to checkpoint weights (.pt).
        """
        self.device = DEVICE
        self.net = MambaYOLONet(num_classes=5)

        target_file = Path(weights_path) if weights_path else MAMBA_YOLO_WEIGHTS
        if target_file.exists():
            try:
                state_dict = torch.load(target_file, map_location=self.device)
                self.net.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

        self.net.to(self.device)
        self.net.eval()

    def predict(self, frame: np.ndarray, conf: float = 0.25) -> np.ndarray:
        """Runs object detection on single BGR frame.

        Args:
            frame: Input BGR image (H, W, 3).
            conf: Confidence threshold.

        Returns:
            np.ndarray: Array of shape (N, 6) with [x1, y1, x2, y2, score, cls].
        """
        orig_h, orig_w = frame.shape[:2]
        img_resized = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255.0)
            .to(self.device)
        )

        with torch.no_grad():
            cls_logits, reg_offsets = self.net(tensor)

        # Softmax probabilities and decode grid
        probs = torch.softmax(cls_logits, dim=1)[0]  # (5, 20, 20)
        max_prob, pred_cls = torch.max(probs, dim=0)  # (20, 20)
        reg = reg_offsets[0]  # (4, 20, 20)

        mask = max_prob >= conf
        if not mask.any():
            return np.empty((0, 6), dtype=np.float32)

        indices = mask.nonzero(as_tuple=False)
        boxes_list = []
        for idx in indices:
            y_idx, x_idx = idx[0].item(), idx[1].item()
            score = max_prob[y_idx, x_idx].item()
            cls_id = float(pred_cls[y_idx, x_idx].item())

            # Decode center and dimensions relative to 640x640
            cx = (x_idx + 0.5 + reg[0, y_idx, x_idx].item()) / 20.0 * orig_w
            cy = (y_idx + 0.5 + reg[1, y_idx, x_idx].item()) / 20.0 * orig_h
            bw = torch.exp(reg[2, y_idx, x_idx]).item() * 50.0 / 640.0 * orig_w
            bh = torch.exp(reg[3, y_idx, x_idx]).item() * 50.0 / 640.0 * orig_h

            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(float(orig_w), cx + bw / 2.0)
            y2 = min(float(orig_h), cy + bh / 2.0)
            boxes_list.append([x1, y1, x2, y2, score, cls_id])

        if not boxes_list:
            return np.empty((0, 6), dtype=np.float32)

        return np.array(boxes_list, dtype=np.float32)
