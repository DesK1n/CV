"""Unified Multi-Object Traffic Tracking engine.

Provides the TrafficTracker class which integrates:
1. Enhanced SORT (KalmanBoxTracker + 2-stage association + Track NMS deduplication)
2. ByteTrack (low-confidence recovery via custom_bytetrack.yaml)
3. BoT-SORT (motion compensation + score fusion via custom_botsort.yaml)

Supports frame-by-frame processing, trajectory trails, and streaming frame generators.
"""
from __future__ import annotations

import collections
from pathlib import Path
import time
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CUSTOM_BOTSORT_YAML,
    CUSTOM_BYTETRACK_YAML,
    DATA_DIR,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_YOLO_WEIGHTS,
    DEVICE,
    IOU_THRESH,
    MAMBA_YOLO_WEIGHTS,
    MAX_AGE,
    MIN_HITS,
    MODELS_DIR,
    OUTPUTS_DIR,
    TRACK_NMS,
    UAV_YOLO_WEIGHTS,
    VEHICLE_CLASSES,
)
from models.base_detector import BaseDetector
from models.baseline_yolo26 import StandardYOLO26
try:
    from models.mamba_yolo import MambaYOLO
except ImportError:
    MambaYOLO = None
from modules.sort_tracker import SORTTracker


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generates distinct BGR color tuple based on track integer identity.

    Args:
        track_id: Tracking integer identifier.

    Returns:
        Tuple[int, int, int]: BGR color code.
    """
    palette: List[Tuple[int, int, int]] = [
        (46, 204, 113),   # Emerald green
        (52, 152, 219),   # Dodger blue
        (155, 89, 182),  # Amethyst purple
        (241, 196, 15),   # Sunflower yellow
        (230, 126, 34),   # Carrot orange
        (231, 76, 60),    # Alizarin red
        (26, 188, 156),   # Turquoise
        (0, 168, 255),    # Neon blue
        (255, 107, 129),  # Coral pink
        (123, 237, 159),  # Mint
    ]
    return palette[track_id % len(palette)]


class TrafficTracker:
    """Unified vehicle tracking engine for CCTV and UAV video streams.

    Attributes:
        tracker_type: Active tracking algorithm ('enhanced_sort', 'bytetrack', 'botsort').
        model_name: Name of active detection model.
        conf_thresh: Detection confidence threshold.
        draw_trails: Whether trajectory lines are rendered on output frames.
        trail_length: Maximum points retained in trajectory trail history.
        trail_history: Mapping of track_id to list of (center_x, center_y) positions.
        unique_track_ids: Set of all unique tracking IDs observed in session.
    """

    SUPPORTED_TRACKERS: Dict[str, str] = {
        "enhanced_sort": "Enhanced SORT (Kalman + 2-Stage Matching + Track NMS)",
        "bytetrack": "ByteTrack (Custom 2-Stage Association)",
        "botsort": "BoT-SORT (Motion Compensation + Fusion)",
    }

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        tracker_type: str = "enhanced_sort",
        conf_thresh: float = DEFAULT_CONF_THRESHOLD,
        max_age: int = MAX_AGE,
        min_hits: int = MIN_HITS,
        iou_thresh: float = IOU_THRESH,
        track_nms: float = TRACK_NMS,
        draw_trails: bool = True,
        trail_length: int = 30,
    ) -> None:
        """Initializes the TrafficTracker engine.

        Args:
            model_path: Path or identifier for detection model weights.
            tracker_type: Algorithm key ('enhanced_sort', 'bytetrack', 'botsort').
            conf_thresh: Detection confidence cutoff.
            max_age: Maximum frame buffer for lost tracks.
            min_hits: Minimum hits before track is confirmed active.
            iou_thresh: IoU assignment threshold.
            track_nms: Track deduplication NMS threshold.
            draw_trails: Whether to render trajectory trails.
            trail_length: Maximum trail length in points.
        """
        self.device: str = DEVICE
        self.conf_thresh: float = float(conf_thresh)
        self.max_age: int = int(max_age)
        self.min_hits: int = int(min_hits)
        self.iou_thresh: float = float(iou_thresh)
        self.track_nms: float = float(track_nms)
        self.draw_trails: bool = draw_trails
        self.trail_length: int = int(trail_length)

        self.trail_history: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        self.unique_track_ids: Set[int] = set()
        self.frame_index: int = 0

        self.model_name: str = "yolo26_uadetrac"
        self.yolo_model: Optional[YOLO] = None
        self.custom_detector: Optional[BaseDetector] = None

        self._load_model(model_path)
        self.set_tracker(tracker_type)

    def _load_model(self, model_path: Optional[str | Path]) -> None:
        """Loads and initializes detection model backend.

        Args:
            model_path: Path or identifier for detection weights.
        """
        target_path: Path = DEFAULT_YOLO_WEIGHTS
        if model_path is not None:
            raw_str = str(model_path).lower()
            if raw_str in ("uav", "uav_yolo", "uav_yolo26s", "uav_yolo26s_best.pt"):
                target_path = UAV_YOLO_WEIGHTS
                self.model_name = "uav_yolo26s"
            elif raw_str in ("mamba", "mamba_yolo", "mamba_yolo_uadetrac_best.pt"):
                target_path = MAMBA_YOLO_WEIGHTS
                self.model_name = "mamba_yolo"
            elif raw_str in ("yolo", "yolo26", "yolo26n", "yolo26_uadetrac_best.pt"):
                target_path = DEFAULT_YOLO_WEIGHTS
                self.model_name = "yolo26_uadetrac"
            else:
                target_path = Path(model_path)
                self.model_name = target_path.stem

        if self.model_name == "mamba_yolo":
            self.custom_detector = MambaYOLO(weights_path=target_path)
            self.yolo_model = None
        else:
            actual_weights = target_path if target_path.exists() else DEFAULT_YOLO_WEIGHTS
            if not actual_weights.exists():
                actual_weights = Path("yolov8n.pt")
            self.yolo_model = YOLO(str(actual_weights))
            self.custom_detector = None

    def set_tracker(self, tracker_type: str) -> None:
        """Switches active tracking algorithm.

        Args:
            tracker_type: Target tracker key ('enhanced_sort', 'bytetrack', 'botsort').

        Raises:
            ValueError: If tracker_type is not supported.
        """
        clean_type = tracker_type.lower().strip()
        if clean_type not in self.SUPPORTED_TRACKERS:
            raise ValueError(
                f"Unsupported tracker '{tracker_type}'. Must be one of {list(self.SUPPORTED_TRACKERS.keys())}"
            )
        self.tracker_type = clean_type
        self.sort_tracker = SORTTracker(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_thresh,
            track_nms=self.track_nms,
        )

    def set_model(self, model_path: str | Path) -> None:
        """Changes detection model and resets session state.

        Args:
            model_path: New model path or identifier.
        """
        self._load_model(model_path)
        self.reset()

    def reset(self) -> None:
        """Resets tracking state, trail histories, and vehicle counts."""
        self.trail_history.clear()
        self.unique_track_ids.clear()
        self.frame_index = 0
        self.sort_tracker = SORTTracker(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_thresh,
            track_nms=self.track_nms,
        )

    @classmethod
    def get_available_trackers(cls) -> List[Dict[str, str]]:
        """Returns list of supported tracking algorithms with descriptions.

        Returns:
            List[Dict[str, str]]: Tracker metadata dictionaries.
        """
        return [{"id": k, "name": v} for k, v in cls.SUPPORTED_TRACKERS.items()]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, str]]:
        """Scans filesystem and returns list of accessible detection models.

        Returns:
            List[Dict[str, str]]: Model identifier dictionaries.
        """
        models: List[Dict[str, str]] = [
            {
                "id": "yolo26_uadetrac",
                "name": "YOLO26n UA-DETRAC (CCTV Traffic Best)",
                "path": str(DEFAULT_YOLO_WEIGHTS),
                "exists": str(DEFAULT_YOLO_WEIGHTS.exists()),
            },
            {
                "id": "uav_yolo26s",
                "name": "UAV YOLO26s (Aerial Drone Multi-Class)",
                "path": str(UAV_YOLO_WEIGHTS),
                "exists": str(UAV_YOLO_WEIGHTS.exists()),
            },
            {
                "id": "mamba_yolo",
                "name": "Mamba-YOLO (Visual State Space Hybrid)",
                "path": str(MAMBA_YOLO_WEIGHTS),
                "exists": str(MAMBA_YOLO_WEIGHTS.exists()),
            },
        ]
        return models

    @classmethod
    def get_available_videos(cls) -> List[Dict[str, str]]:
        """Scans dataset directory and returns list of test videos.

        Returns:
            List[Dict[str, str]]: Video file metadata dictionaries.
        """
        videos: List[Dict[str, str]] = []
        for ext in ("*.mp4", "*.mkv", "*.avi"):
            for v_path in sorted(DATA_DIR.glob(ext)):
                videos.append(
                    {
                        "name": v_path.name,
                        "path": str(v_path),
                        "size_mb": f"{v_path.stat().st_size / (1024 * 1024):.1f} MB",
                    }
                )
        return videos

    def process_frame(
        self, frame: np.ndarray, draw: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Processes a single BGR video frame through detection and tracking.

        Args:
            frame: Input image array in BGR format with shape (H, W, 3).
            draw: Whether to render bounding boxes and HUD annotations.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (annotated_frame, telemetry_dict).
        """
        self.frame_index += 1
        t_start = time.perf_counter()

        annotated = frame.copy() if draw else frame
        tracks_data: List[Dict[str, Any]] = []

        if self.tracker_type == "enhanced_sort" or self.yolo_model is None:
            # 1. Detection Stage
            if self.yolo_model is not None:
                # Ultralytics detection
                pred_results = self.yolo_model(
                    frame,
                    conf=self.conf_thresh,
                    iou=0.45,
                    device=self.device,
                    verbose=False,
                )
                boxes_list: List[np.ndarray] = []
                for r in pred_results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()[:, None]
                        cls_ids = r.boxes.cls.cpu().numpy()[:, None]
                        boxes_list.append(np.hstack([xyxy, confs, cls_ids]))
                dets = (
                    np.vstack(boxes_list).astype(np.float32)
                    if boxes_list
                    else np.empty((0, 6), dtype=np.float32)
                )
            else:
                # Mamba-YOLO detection
                dets = self.custom_detector.predict(frame, conf=self.conf_thresh)

            # 2. Association Stage via Enhanced SORT
            active_sort_tracks = self.sort_tracker.update(dets)

            for trk in active_sort_tracks:
                trk_id = int(trk[0])
                x1, y1, x2, y2 = [int(v) for v in trk[1:5]]
                score = float(trk[5])
                cls_id = int(trk[6])
                cls_name = self._resolve_class_name(cls_id)

                self.unique_track_ids.add(trk_id)
                tracks_data.append(
                    {
                        "track_id": trk_id,
                        "bbox": [x1, y1, x2, y2],
                        "conf": round(score, 3),
                        "cls": cls_name,
                    }
                )
        else:
            # ByteTrack or BoT-SORT via Ultralytics engine
            cfg_path = (
                CUSTOM_BYTETRACK_YAML
                if self.tracker_type == "bytetrack"
                else CUSTOM_BOTSORT_YAML
            )
            results = self.yolo_model.track(
                source=frame,
                persist=True,
                tracker=str(cfg_path),
                conf=self.conf_thresh,
                device=self.device,
                verbose=False,
            )

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    track_ids = (
                        r.boxes.id.int().cpu().numpy()
                        if r.boxes.id is not None
                        else np.zeros(len(xyxy), dtype=int)
                    )

                    for i in range(len(xyxy)):
                        trk_id = int(track_ids[i])
                        if trk_id <= 0:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                        score = float(confs[i])
                        cls_id = int(classes[i])
                        cls_name = self._resolve_class_name(cls_id)

                        self.unique_track_ids.add(trk_id)
                        tracks_data.append(
                            {
                                "track_id": trk_id,
                                "bbox": [x1, y1, x2, y2],
                                "conf": round(score, 3),
                                "cls": cls_name,
                            }
                        )

        t_elapsed = time.perf_counter() - t_start
        latency_ms = max(0.1, t_elapsed * 1000.0)
        fps = 1000.0 / latency_ms

        # Render annotations if requested
        if draw:
            self._render_annotations(annotated, tracks_data, fps, latency_ms)

        stats: Dict[str, Any] = {
            "frame_idx": self.frame_index,
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 2),
            "active_tracks": len(tracks_data),
            "total_count": len(self.unique_track_ids),
            "tracker": self.tracker_type,
            "model": self.model_name,
            "tracks": tracks_data,
        }

        return annotated, stats

    def _resolve_class_name(self, cls_id: int) -> str:
        """Translates integer class index into human-readable label.

        Args:
            cls_id: Class integer index.

        Returns:
            str: Resolved vehicle category name.
        """
        if self.yolo_model is not None and hasattr(self.yolo_model, "names"):
            if cls_id in self.yolo_model.names:
                return str(self.yolo_model.names[cls_id])
        return VEHICLE_CLASSES.get(cls_id, "car")

    def _render_annotations(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        fps: float,
        latency_ms: float,
    ) -> None:
        """Draws bounding boxes, trajectory trails, and HUD banner on frame.

        Args:
            frame: Target BGR image array to mutate.
            tracks: List of active track dictionaries.
            fps: Current frames-per-second metric.
            latency_ms: Processing latency in milliseconds.
        """
        for item in tracks:
            trk_id = int(item["track_id"])
            x1, y1, x2, y2 = item["bbox"]
            score = float(item["conf"])
            cls_name = str(item["cls"])
            color = get_track_color(trk_id)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Trajectory Trail
            if self.draw_trails:
                trail = self.trail_history[trk_id]
                trail.append((cx, cy))
                if len(trail) > self.trail_length:
                    trail.pop(0)

                for i in range(1, len(trail)):
                    thickness = max(1, int(np.sqrt(i)))
                    cv2.line(frame, trail[i - 1], trail[i], color, thickness)

            # Bounding Box & Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{trk_id} {cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # HUD Top Banner
        hud_bg_w = min(620, frame.shape[1] - 20)
        cv2.rectangle(frame, (10, 8), (10 + hud_bg_w, 42), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 8), (10 + hud_bg_w, 42), (70, 70, 70), 1)

        hud_str = (
            f"FPS: {fps:4.1f} | Latency: {latency_ms:4.1f}ms | "
            f"Active: {len(tracks):2d} | Total Count: {len(self.unique_track_ids):3d}"
        )
        cv2.putText(
            frame,
            hud_str,
            (18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

    def stream_frames(
        self,
        source: str | Path | int,
        max_frames: Optional[int] = None,
        loop: bool = False,
        draw: bool = True,
        yield_jpeg: bool = False,
        jpeg_quality: int = 80,
    ) -> Generator[Tuple[np.ndarray | bytes, Dict[str, Any]], None, None]:
        """Streams processed frames and tracking metrics from video or camera.

        Args:
            source: Video path, sequence directory, or webcam index.
            max_frames: Optional upper bound on processed frames.
            loop: Whether to loop video playback infinitely.
            draw: Whether to render bounding boxes and HUD.
            yield_jpeg: When True, frame is yielded as encoded JPEG bytes.
            jpeg_quality: JPEG compression quality (1-100).

        Yields:
            Tuple[np.ndarray | bytes, Dict[str, Any]]: (frame_data, telemetry).

        Raises:
            FileNotFoundError: If video source is not found.
        """
        source_path = Path(source) if isinstance(source, (str, Path)) else None
        is_seq_dir = source_path is not None and source_path.is_dir()

        if is_seq_dir:
            img_files = sorted(
                list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
            )
            if not img_files:
                raise FileNotFoundError(f"No images in directory: {source_path}")
            cap = None
        else:
            if source_path is not None and not source_path.exists():
                raise FileNotFoundError(f"Video file not found: {source_path}")
            cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))

        frame_count: int = 0
        img_idx: int = 0

        try:
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break

                if is_seq_dir:
                    if img_idx >= len(img_files):
                        if loop:
                            img_idx = 0
                        else:
                            break
                    frame = cv2.imread(str(img_files[img_idx]))
                    img_idx += 1
                else:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        if loop:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break

                annotated, stats = self.process_frame(frame, draw=draw)
                frame_count += 1

                if yield_jpeg:
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    success, buffer = cv2.imencode(".jpg", annotated, encode_params)
                    frame_out: np.ndarray | bytes = (
                        buffer.tobytes() if success else b""
                    )
                else:
                    frame_out = annotated

                yield frame_out, stats
        finally:
            if cap is not None:
                cap.release()
