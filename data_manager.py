"""Dataset manager for UA-DETRAC dataset verification and mock video generation.

Provides tools for verifying dataset integrity, downloading/linking from Kaggle,
generating synthetic mock test videos and image sequences for rapid pipeline verification.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import DATA_DIR, UA_DETRAC_DIR, VEHICLE_CLASSES

KAGGLE_UADETRAC_REF: str = "nguynttthanh/ua-detrac-yolo"


class UADetracDatasetManager:
    """Manages downloading, validation, and preparation of the UA-DETRAC CCTV dataset."""

    def __init__(self, destination_dir: Path = UA_DETRAC_DIR) -> None:
        """Initializes UA-DETRAC dataset manager.

        Args:
            destination_dir: Directory where UA-DETRAC dataset is stored or linked.
        """
        self.dest_dir: Path = Path(destination_dir)

    def is_valid(self) -> bool:
        """Checks if dataset exists and contains required train and val images.

        Returns:
            bool: True if images exist and pass count threshold, False otherwise.
        """
        train_img = self.dest_dir / "train" / "images"
        val_img = self.dest_dir / "val" / "images"
        if train_img.exists() and val_img.exists():
            train_count = len(list(train_img.glob("*.jpg")))
            val_count = len(list(val_img.glob("*.jpg")))
            return train_count > 1000 and val_count > 500
        return False

    def verify_dataset_structure(self) -> Dict[str, int]:
        """Inspects and returns image and label counts across dataset splits.

        Returns:
            Dict[str, int]: Mapping of split keys to file counts.
        """
        stats: Dict[str, int] = {}
        for split in ("train", "val"):
            img_dir = self.dest_dir / split / "images"
            lbl_dir = self.dest_dir / split / "labels"
            stats[f"{split}_images"] = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
            stats[f"{split}_labels"] = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        return stats

    def download_and_link(self) -> Path:
        """Downloads UA-DETRAC from Kaggle via kagglehub and creates directory junction.

        Returns:
            Path: Path to destination directory.

        Raises:
            RuntimeError: If download fails or verification fails.
        """
        if self.is_valid():
            train_count = len(list((self.dest_dir / "train" / "images").glob("*.jpg")))
            print(
                f"[INFO] UA-DETRAC dataset already active at {self.dest_dir} "
                f"({train_count} train images)."
            )
            return self.dest_dir

        print(f"[INFO] Downloading/verifying UA-DETRAC from Kaggle: '{KAGGLE_UADETRAC_REF}'...")
        try:
            import kagglehub

            cache_path = Path(kagglehub.dataset_download(KAGGLE_UADETRAC_REF))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download UA-DETRAC via kagglehub: {exc}. Check connection/credentials."
            ) from exc

        print(f"[SUCCESS] Kagglehub cache path: {cache_path}")
        self.dest_dir.parent.mkdir(parents=True, exist_ok=True)

        if not self.dest_dir.exists():
            if os.name == "nt":
                import subprocess

                cmd = f'cmd /c mklink /J "{self.dest_dir}" "{cache_path}"'
                print(f"[INFO] Linking junction: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
            else:
                self.dest_dir.symlink_to(cache_path, target_is_directory=True)

        if not self.is_valid():
            raise RuntimeError(f"UA-DETRAC validation failed at {self.dest_dir}.")

        # Generate ua_detrac.yaml
        yaml_path = self.dest_dir.parent.parent / "ua_detrac.yaml"
        train_abs = (self.dest_dir / "train" / "images").resolve().as_posix()
        val_abs = (self.dest_dir / "val" / "images").resolve().as_posix()
        yaml_content = f"""path: {self.dest_dir.resolve().as_posix()}
train: {train_abs}
val: {val_abs}
names:
  0: vehicle
"""
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"[SUCCESS] UA-DETRAC dataset ready. Config: {yaml_path}")
        return self.dest_dir

    def prepare_dataset(self, download_if_missing: bool = True) -> Path:
        """Ensures the dataset is ready and verified.

        Args:
            download_if_missing: Whether to trigger download if not present.

        Returns:
            Path: Path to verified dataset directory.

        Raises:
            RuntimeError: If dataset is missing and download is disabled.
        """
        if self.is_valid():
            return self.dest_dir
        if not download_if_missing:
            raise RuntimeError(
                f"UA-DETRAC dataset not found at {self.dest_dir} and download is disabled."
            )
        return self.download_and_link()


def create_mock_video(
    output_path: str | Path,
    num_frames: int = 60,
    width: int = 640,
    height: int = 480,
    fps: int = 20,
) -> Path:
    """Generates a synthetic MP4 video with moving vehicles for testing and streaming.

    Args:
        output_path: Target video file path (.mp4).
        num_frames: Number of video frames to generate.
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        fps: Frames per second.

    Returns:
        Path: Path to generated mock video file.

    Raises:
        RuntimeError: If video writer fails to open.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_file}")

    # Simulated vehicle trajectories: (start_x, start_y, w, h, dx, dy, color, label)
    vehicles = [
        (30, 100, 80, 50, 4, 1, (46, 204, 113), "Car 1"),
        (width - 120, 200, 90, 55, -3, 0, (52, 152, 219), "Car 2"),
        (60, 320, 120, 70, 3, -1, (241, 196, 15), "Bus 1"),
    ]

    for frame_idx in range(num_frames):
        # Road asphalt background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Draw lane dividing dashed markings
        for y_lane in (height // 3, 2 * height // 3):
            for x_dash in range(0, width, 40):
                cv2.line(frame, (x_dash, y_lane), (x_dash + 20, y_lane), (200, 200, 200), 2)

        for sx, sy, vw, vh, dx, dy, color, label in vehicles:
            vx = int(sx + dx * frame_idx) % (width + vw) - vw // 2
            vy = int(sy + dy * frame_idx)

            # Clamp coordinates
            x1 = max(0, min(width - 1, vx))
            y1 = max(0, min(height - 1, vy))
            x2 = max(0, min(width - 1, vx + vw))
            y2 = max(0, min(height - 1, vy + vh))

            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Frame stamp
        cv2.putText(
            frame,
            f"Mock CCTV Frame: {frame_idx:03d}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    return out_file


def generate_mock_sequence(
    output_dir: str | Path,
    num_frames: int = 10,
    width: int = 320,
    height: int = 240,
) -> Path:
    """Generates an image sequence of JPEG frames for quick unit tests.

    Args:
        output_dir: Target directory for sequential frames.
        num_frames: Number of frames to create.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Path: Path to image directory.
    """
    seq_path = Path(output_dir)
    seq_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        img = np.full((height, width, 3), 30, dtype=np.uint8)
        # Moving box simulating vehicle
        x1 = 50 + i * 6
        y1 = 60 + i * 2
        cv2.rectangle(img, (x1, y1), (x1 + 60, y1 + 40), (255, 255, 255), -1)
        cv2.imwrite(str(seq_path / f"frame_{i:04d}.jpg"), img)

    return seq_path


def main() -> None:
    """CLI entry point for dataset management and mock data generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest", type=Path, default=UA_DETRAC_DIR, help="Destination directory for dataset"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing dataset"
    )
    parser.add_argument(
        "--create-mock-video", type=Path, default=None, help="Generate synthetic test video"
    )
    args = parser.parse_args()

    if args.create_mock_video:
        path = create_mock_video(args.create_mock_video)
        print(f"[SUCCESS] Mock video created at: {path}")
        return

    manager = UADetracDatasetManager(destination_dir=args.dest)
    if args.verify_only:
        valid = manager.is_valid()
        stats = manager.verify_dataset_structure()
        print(f"[INFO] UA-DETRAC valid: {valid}, stats: {stats}")
    else:
        manager.prepare_dataset(download_if_missing=True)


if __name__ == "__main__":
    main()
