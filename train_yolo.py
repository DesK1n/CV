"""Deep fine-tuning pipeline for Standard YOLO26 on UA-DETRAC CCTV traffic dataset.

Executes multi-epoch training on Nvidia RTX 5060 / CUDA with AMP FP16,
cosine learning rate schedule, data augmentations, and full evaluation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

from ultralytics import YOLO

from config import BASE_DIR, DATA_DIR, DEVICE, MODELS_DIR, RUNS_DIR
from models.baseline_yolo26 import DEFAULT_YOLO26_WEIGHTS


def train_yolo26(
    epochs: int = 25,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = DEVICE,
    patience: int = 15,
) -> None:
    """Trains/fine-tunes Standard YOLO26 on UA-DETRAC.

    Args:
        epochs: Number of complete training epochs.
        batch_size: Size of mini-batches.
        imgsz: Target square image dimension.
        device: Device string (e.g. '0' or 'cpu').
        patience: Epochs to wait for early stopping.
    """
    data_yaml = BASE_DIR / "ua_detrac.yaml"
    if not data_yaml.exists():
        raise RuntimeError(
            f"UA-DETRAC configuration not found at {data_yaml}. "
            f"Please run download_data.py first."
        )

    base_model_path = (
        str(DEFAULT_YOLO26_WEIGHTS) if DEFAULT_YOLO26_WEIGHTS.exists() else "yolov8n.pt"
    )

    print(
        f"[INFO] Initializing YOLO26 fine-tuning on UA-DETRAC from {base_model_path} on device={device}..."
    )
    model = YOLO(base_model_path)
    cuda_device = "0" if "cuda" in device else device

    start_time = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=cuda_device,
        project=str(RUNS_DIR / "yolo"),
        name="uadetrac_train",
        exist_ok=True,
        patience=patience,
        cos_lr=True,
        workers=0 if os.name == "nt" else 4,
    )

    elapsed = time.time() - start_time
    print(
        f"[SUCCESS] YOLO26 UA-DETRAC training completed in {elapsed:.2f}s (~{elapsed/3600:.2f}h)."
    )

    trained_best = RUNS_DIR / "yolo" / "uadetrac_train" / "weights" / "best.pt"
    target_best = MODELS_DIR / "yolo26_uadetrac_best.pt"
    if trained_best.exists():
        shutil.copy2(trained_best, target_best)
        print(f"[INFO] Exported best model weights to {target_best}")


def main() -> None:
    """CLI entry point for YOLO26 training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=DEVICE, help="Training device")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    args = parser.parse_args()

    train_yolo26(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
