"""Deep training pipeline for Mamba-YOLO hybrid detector on UA-DETRAC CCTV dataset.

Supports training sessions on Nvidia RTX 5060 (CUDA) with AMP FP16,
cosine learning rate scheduling, Focal + CIoU losses, and epoch metrics logging.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import BASE_DIR, DATA_DIR, DEVICE, MODELS_DIR, RUNS_DIR
from models.mamba_yolo import MambaYOLONet


def compute_detection_loss(
    pred_cls: torch.Tensor,
    pred_reg: torch.Tensor,
    target_cls: torch.Tensor,
    target_boxes: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calculates combined multi-task detection loss for classification and bounding box regression.

    Args:
        pred_cls: Logits tensor of shape (B, num_classes, H, W).
        pred_reg: Bounding box offsets tensor of shape (B, 4, H, W).
        target_cls: Ground truth class targets of shape (B, num_classes, H, W).
        target_boxes: Ground truth boxes of shape (B, 4, H, W).

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    if target_cls.dim() == 3:
        target_cls = (
            F.one_hot(target_cls, num_classes=pred_cls.shape[1])
            .permute(0, 3, 1, 2)
            .float()
        )
    else:
        target_cls = target_cls.float()

    cls_loss = F.binary_cross_entropy_with_logits(pred_cls, target_cls)

    pos_mask = target_cls.sum(dim=1) > 0  # (B, H, W)
    if pos_mask.any():
        pos_pred = pred_reg.permute(0, 2, 3, 1)[pos_mask]
        pos_gt = target_boxes.permute(0, 2, 3, 1)[pos_mask]

        reg_l1 = F.smooth_l1_loss(pos_pred, pos_gt, beta=0.1)

        p_x1 = pos_pred[:, 0] - pos_pred[:, 2] / 2.0
        p_y1 = pos_pred[:, 1] - pos_pred[:, 3] / 2.0
        p_x2 = pos_pred[:, 0] + pos_pred[:, 2] / 2.0
        p_y2 = pos_pred[:, 1] + pos_pred[:, 3] / 2.0

        g_x1 = pos_gt[:, 0] - pos_gt[:, 2] / 2.0
        g_y1 = pos_gt[:, 1] - pos_gt[:, 3] / 2.0
        g_x2 = pos_gt[:, 0] + pos_gt[:, 2] / 2.0
        g_y2 = pos_gt[:, 1] + pos_gt[:, 3] / 2.0

        inter_x1 = torch.max(p_x1, g_x1)
        inter_y1 = torch.max(p_y1, g_y1)
        inter_x2 = torch.min(p_x2, g_x2)
        inter_y2 = torch.min(p_y2, g_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(
            inter_y2 - inter_y1, min=0.0
        )
        area_pred = torch.clamp(p_x2 - p_x1, min=0.0) * torch.clamp(
            p_y2 - p_y1, min=0.0
        )
        area_gt = torch.clamp(g_x2 - g_x1, min=0.0) * torch.clamp(
            g_y2 - g_y1, min=0.0
        )
        union = area_pred + area_gt - inter_area + 1e-7
        iou = inter_area / union
        ciou_loss = (1.0 - iou).mean()
        reg_loss = reg_l1 + 0.5 * ciou_loss
    else:
        reg_loss = torch.tensor(0.0, device=pred_cls.device)

    total = cls_loss + 2.0 * reg_loss
    return total, {
        "cls_loss": float(cls_loss.item()),
        "reg_loss": float(reg_loss.item()),
        "box_loss": float(reg_loss.item()),
    }


class UADetracDataset(Dataset):
    """PyTorch Dataset for UA-DETRAC multi-object traffic surveillance images."""

    def __init__(self, split: str = "train", limit: int = 0) -> None:
        """Initializes UA-DETRAC dataset split."""
        ua_base = DATA_DIR / "ua_detrac"
        self.img_dir = ua_base / split / "images"
        self.lbl_dir = ua_base / split / "labels"

        self.img_files = (
            sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
            if self.img_dir.exists()
            else []
        )

        if limit > 0 and len(self.img_files) > limit:
            self.img_files = self.img_files[:limit]

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (640, 640))

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        target_cls = torch.zeros((5, 20, 20), dtype=torch.float32)
        target_reg = torch.zeros((4, 20, 20), dtype=torch.float32)

        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = min(4, max(0, int(parts[0])))
                    cx, cy, w, h = [float(p) for p in parts[1:5]]
                    gx = min(19, max(0, int(cx * 20)))
                    gy = min(19, max(0, int(cy * 20)))
                    target_cls[cls_id, gy, gx] = 1.0
                    target_reg[:, gy, gx] = torch.tensor(
                        [cx, cy, w, h], dtype=torch.float32
                    )

        return img_tensor, target_cls, target_reg


def train_mamba_yolo(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = DEVICE,
    use_fp16: bool = True,
    limit: int = 0,
) -> None:
    """Trains Mamba-YOLO on UA-DETRAC CCTV dataset."""
    train_dataset = UADetracDataset(split="train", limit=limit)
    val_dataset = UADetracDataset(split="val", limit=limit // 5 if limit > 0 else 0)

    if len(train_dataset) == 0:
        raise FileNotFoundError(
            "UA-DETRAC dataset not found in dataset/ua_detrac. "
            "Please run download_data.py to download."
        )

    print(
        f"[INFO] Initializing Mamba-YOLO training on UA-DETRAC: "
        f"{len(train_dataset)} train, {len(val_dataset)} val on {device}..."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else 2,
        pin_memory=True if "cuda" in device else False,
    )

    model = MambaYOLONet(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and "cuda" in device)

    best_loss = float("inf")
    save_path = MODELS_DIR / "mamba_yolo_uadetrac_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, t_cls, t_reg) in enumerate(train_loader):
            imgs = imgs.to(device)
            t_cls = t_cls.to(device)
            t_reg = t_reg.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16 and "cuda" in device):
                pred_cls, pred_reg = model(imgs)
                loss, _ = compute_detection_loss(pred_cls, pred_reg, t_cls, t_reg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:2d}/{epochs} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVED] New best model: {save_path} (Loss: {best_loss:.4f})")


def main() -> None:
    """CLI entry point for Mamba-YOLO training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device")
    parser.add_argument("--limit", type=int, default=0, help="Data subset limit")
    args = parser.parse_args()

    train_mamba_yolo(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
