"""Batch video tracking pipeline for CCTV Vehicle Analytics.

Processes video files or image sequence directories using either Standard YOLO26
or Mamba-YOLO paired with SORT or SSM-Tracker. Draws vehicle bounding boxes,
motion trajectory trails, IDs, and exports JSON logs and annotated MP4 video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from config import DEFAULT_CONF_THRESHOLD, OUTPUTS_DIR, VEHICLE_CLASSES
from models.base_detector import BaseDetector
from models.baseline_yolo26 import StandardYOLO26
from models.mamba_yolo import MambaYOLO
from modules.sort_tracker import SORTTracker
from modules.ssm_tracker import SSMTracker


def get_color(idx: int) -> Tuple[int, int, int]:
    """Generates distinct BGR color tuple based on track index."""
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (241, 196, 15),
        (230, 126, 34),
        (231, 76, 60),
        (26, 188, 156),
        (0, 168, 255),
    ]
    return palette[idx % len(palette)]


def run_tracking(
    source: str | Path,
    detector: BaseDetector,
    tracker: SORTTracker | SSMTracker,
    conf_thresh: float = DEFAULT_CONF_THRESHOLD,
    save_video: bool = False,
    out_json: Path | None = None,
    output_video_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """Runs end-to-end tracking pipeline on a video source.

    Args:
        source: Video file path or sequence folder.
        detector: Instantiated vehicle detector.
        tracker: Instantiated tracker.
        conf_thresh: Detection confidence threshold.
        save_video: Whether to write annotated MP4 video.
        out_json: Optional path to save JSON tracking log.
        output_video_path: Optional output video path.

    Returns:
        List of per-frame tracking records.
    """
    source_path = Path(source)
    is_image_folder = source_path.is_dir()

    if is_image_folder:
        image_files = sorted(
            list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        )
        if not image_files:
            raise FileNotFoundError(f"No images found in sequence directory: {source_path}")
        first_frame = cv2.imread(str(image_files[0]))
        h, w = first_frame.shape[:2]
        fps = 25.0
        total_frames = len(image_files)
    else:
        if not source_path.exists():
            raise FileNotFoundError(f"Video file not found: {source_path}")
        cap = cv2.VideoCapture(str(source_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if save_video and output_video_path is None:
        output_video_path = OUTPUTS_DIR / f"{source_path.stem}_tracked.mp4"

    video_writer = None
    if save_video and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    records: List[Dict[str, Any]] = []
    trail_history: Dict[int, List[Tuple[int, int]]] = {}

    print(f"[INFO] Tracking source: {source_path} ({total_frames} frames, {w}x{h} @ {fps:.1f} FPS)...")
    frame_idx = 0

    while True:
        if is_image_folder:
            if frame_idx >= total_frames:
                break
            frame = cv2.imread(str(image_files[frame_idx]))
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        t0 = time.perf_counter()
        detections = detector.predict(frame, conf=conf_thresh)
        tracks = tracker.update(detections)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        current_frame_tracks = []

        # Annotate frame
        for trk in tracks:
            track_id = int(trk[0])
            x1, y1, x2, y2 = [int(v) for v in trk[1:5]]
            score = float(trk[5])
            cls_id = int(trk[6])
            cls_name = VEHICLE_CLASSES.get(cls_id, "car")

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if track_id not in trail_history:
                trail_history[track_id] = []
            trail_history[track_id].append((center_x, center_y))
            if len(trail_history[track_id]) > 25:
                trail_history[track_id].pop(0)

            current_frame_tracks.append(
                {
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(score, 3),
                    "cls": cls_name,
                }
            )

            # Draw motion trails
            color = get_color(track_id)
            pts = trail_history[track_id]
            for i in range(1, len(pts)):
                thickness = max(1, int(np.sqrt(i)))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {cls_name} {score:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 4), (x1 + lw, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw HUD banner
        hud_text = (
            f"FPS: {1000.0/max(1.0, latency_ms):.1f} | Latency: {latency_ms:.1f}ms | "
            f"Active Vehicles: {len(tracks)}"
        )
        cv2.putText(
            frame,
            hud_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if video_writer:
            video_writer.write(frame)

        records.append(
            {
                "frame": frame_idx,
                "latency_ms": round(latency_ms, 2),
                "tracks": current_frame_tracks,
            }
        )

        frame_idx += 1

    if not is_image_folder:
        cap.release()
    if video_writer:
        video_writer.release()
        print(f"[SUCCESS] Annotated video saved to: {output_video_path}")

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"[SUCCESS] Tracking log saved to: {out_json}")

    return records


def main() -> None:
    """CLI entry point for video tracking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, required=True, help="Input video or folder")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo26", "mamba_yolo"],
        default="mamba_yolo",
        help="Detection model",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["sort", "ssm"],
        default="ssm",
        help="Tracking algorithm",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save annotated MP4 video"
    )
    parser.add_argument(
        "--out-json", type=Path, default=None, help="Save JSON tracking log"
    )
    args = parser.parse_args()

    detector: BaseDetector = (
        MambaYOLO() if args.model == "mamba_yolo" else StandardYOLO26()
    )
    tracker: SORTTracker | SSMTracker = (
        SSMTracker() if args.tracker == "ssm" else SORTTracker()
    )

    run_tracking(
        source=args.source,
        detector=detector,
        tracker=tracker,
        conf_thresh=args.conf,
        save_video=args.save_video,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
