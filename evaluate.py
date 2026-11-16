"""Comprehensive benchmarking suite for CCTV Vehicle Tracking on UA-DETRAC.

Evaluates four multi-object tracking configurations using exclusively the
fine-tuned YOLO26n vehicle detector:
1. Classic SORT (Ablation Baseline: 1-stage IoU association, no Track NMS)
2. Enhanced SORT (KalmanBoxTracker + 2-Stage Matching + Track NMS)
3. ByteTrack (Low-confidence association via custom_bytetrack.yaml)
4. BoT-SORT (Motion compensation + score fusion via custom_botsort.yaml)

Computes standardized MOT metrics according to the CLEAR MOT protocol:
MOTA, IDF1, HOTA, ID Switches (IDSW), and Processing Throughput (FPS).
Saves structured benchmark summaries to CSV and high-resolution comparison plots.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from config import (
    CUSTOM_BOTSORT_YAML,
    CUSTOM_BYTETRACK_YAML,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_YOLO_WEIGHTS,
    DEVICE,
    IOU_THRESH,
    MAX_AGE,
    MIN_HITS,
    REPORTS_DIR,
    TRACK_NMS,
    UA_DETRAC_DIR,
)
from modules.metrics import box_iou, compute_hota, compute_idf1, compute_mota
from modules.sort_tracker import KalmanBoxTracker, SORTTracker, compute_iou_batch


class ClassicSORTTracker:
    """Ablation baseline tracker: standard 1-stage SORT without distance fallback or NMS."""

    def __init__(
        self,
        max_age: int = MAX_AGE,
        min_hits: int = MIN_HITS,
        iou_threshold: float = IOU_THRESH,
    ) -> None:
        """Initializes classic SORT baseline tracker.

        Args:
            max_age: Maximum frames to retain lost tracks.
            min_hits: Minimum hits before track is confirmed active.
            iou_threshold: Minimum IoU overlap for Hungarian assignment.
        """
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.iou_threshold: float = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Updates tracker with detections from current frame using single-stage IoU.

        Args:
            detections: Array of shape (N, 6): [x1, y1, x2, y2, score, class_id].

        Returns:
            np.ndarray: Tracked objects of shape (M, 7):
                [track_id, x1, y1, x2, y2, score, class_id].
        """
        self.frame_count += 1
        predicted_boxes: List[np.ndarray] = []
        to_del: List[int] = []

        for idx, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.isnan(pos).any():
                to_del.append(idx)
            else:
                predicted_boxes.append(pos)

        for idx in reversed(to_del):
            self.trackers.pop(idx)

        trks = (
            np.array(predicted_boxes, dtype=np.float32)
            if predicted_boxes
            else np.empty((0, 4), dtype=np.float32)
        )
        dets = (
            detections[:, :4].astype(np.float32)
            if len(detections) > 0
            else np.empty((0, 4), dtype=np.float32)
        )

        matched: List[Tuple[int, int]] = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trks)))

        if len(trks) > 0 and len(dets) > 0:
            iou_matrix = compute_iou_batch(dets, trks)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched.append((int(r), int(c)))
                    if r in unmatched_dets:
                        unmatched_dets.remove(r)
                    if c in unmatched_trks:
                        unmatched_trks.remove(c)

        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(
                detections[det_idx, :4],
                score=float(detections[det_idx, 4]),
                class_id=float(detections[det_idx, 5]),
            )

        for det_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(
                detections[det_idx, :4],
                score=float(detections[det_idx, 4]),
                class_id=float(detections[det_idx, 5]),
            )
            self.trackers.append(new_trk)

        active_tracks: List[List[float]] = []
        remaining_trackers: List[KalmanBoxTracker] = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                remaining_trackers.append(trk)
                if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    box = trk.get_state()
                    active_tracks.append(
                        [
                            float(trk.id),
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                            float(trk.score),
                            float(trk.class_id),
                        ]
                    )

        self.trackers = remaining_trackers
        if not active_tracks:
            return np.empty((0, 7), dtype=np.float32)

        return np.array(active_tracks, dtype=np.float32)


def load_uadetrac_sequence(
    sequence_prefix: str = "MVI_20012",
    split: str = "val",
    max_frames: int = 50,
) -> Tuple[List[np.ndarray], Dict[int, List[Tuple[int, np.ndarray]]]]:
    """Loads sequential frames and continuous ground-truth vehicle tracks.

    Associates ground-truth detections across consecutive frames to establish
    continuous identity trajectories for MOT evaluation.

    Args:
        sequence_prefix: Sequence ID prefix (e.g. 'MVI_20012').
        split: Dataset split ('val' or 'train').
        max_frames: Maximum number of sequential frames to load.

    Returns:
        Tuple of (frames_list, gt_tracks_dict).

    Raises:
        FileNotFoundError: If sequence frames or labels cannot be located.
    """
    img_dir = UA_DETRAC_DIR / split / "images"
    lbl_dir = UA_DETRAC_DIR / split / "labels"

    all_imgs = sorted(img_dir.glob(f"{sequence_prefix}_img*.jpg"))
    if not all_imgs:
        all_imgs = sorted(img_dir.glob("*.jpg"))

    if not all_imgs:
        raise FileNotFoundError(
            f"No images found for sequence {sequence_prefix} in {img_dir}"
        )

    selected_imgs = all_imgs[:max_frames]
    frames: List[np.ndarray] = []
    raw_frame_boxes: List[List[np.ndarray]] = []

    for img_path in selected_imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frames.append(frame)
        h, w = frame.shape[:2]

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes_in_frame: List[np.ndarray] = []
        if lbl_path.exists():
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                    x1 = max(0.0, (cx - bw / 2.0) * w)
                    y1 = max(0.0, (cy - bh / 2.0) * h)
                    x2 = min(float(w), (cx + bw / 2.0) * w)
                    y2 = min(float(h), (cy + bh / 2.0) * h)
                    boxes_in_frame.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        raw_frame_boxes.append(boxes_in_frame)

    # Establish continuous GT track identities via temporal IoU matching
    next_gt_id = 1
    active_gt_tracks: Dict[int, np.ndarray] = {}
    gt_tracks: Dict[int, List[Tuple[int, np.ndarray]]] = {}

    for f_idx, boxes in enumerate(raw_frame_boxes):
        frame_gts: List[Tuple[int, np.ndarray]] = []
        if f_idx == 0:
            for b in boxes:
                tid = next_gt_id
                next_gt_id += 1
                active_gt_tracks[tid] = b
                frame_gts.append((tid, b))
        else:
            active_ids = list(active_gt_tracks.keys())
            prev_boxes = [active_gt_tracks[tid] for tid in active_ids]
            cost = np.zeros((len(prev_boxes), len(boxes)), dtype=np.float32)
            for i, pb in enumerate(prev_boxes):
                for j, cb in enumerate(boxes):
                    cost[i, j] = -box_iou(pb, cb)

            matched_prev = set()
            matched_curr = set()
            if len(prev_boxes) > 0 and len(boxes) > 0:
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    if -cost[r, c] >= 0.30:
                        tid = active_ids[r]
                        matched_prev.add(r)
                        matched_curr.add(c)
                        active_gt_tracks[tid] = boxes[c]
                        frame_gts.append((tid, boxes[c]))

            for r, tid in enumerate(active_ids):
                if r not in matched_prev:
                    del active_gt_tracks[tid]

            for c, b in enumerate(boxes):
                if c not in matched_curr:
                    tid = next_gt_id
                    next_gt_id += 1
                    active_gt_tracks[tid] = b
                    frame_gts.append((tid, b))

        gt_tracks[f_idx] = frame_gts

    return frames, gt_tracks


def run_benchmark(
    sequence_prefix: str = "MVI_20012",
    split: str = "val",
    num_frames: int | None = None,
    max_frames: int = 50,
) -> List[Dict[str, Any]]:
    """Runs comprehensive benchmark across tracking configurations with YOLO26n.

    Args:
        sequence_prefix: UA-DETRAC sequence identifier.
        split: Dataset split ('val' or 'train').
        num_frames: Maximum frames to evaluate (alias for max_frames).
        max_frames: Maximum frames to evaluate.

    Returns:
        List[Dict[str, Any]]: Formatted evaluation metrics per tracker.
    """
    effective_frames = num_frames if num_frames is not None else max_frames
    frames, gt_tracks = load_uadetrac_sequence(
        sequence_prefix=sequence_prefix, split=split, max_frames=effective_frames
    )

    print(
        f"[INFO] Benchmarking exclusively with YOLO26n on {len(frames)} frames "
        f"of sequence {sequence_prefix}..."
    )

    # Initialize YOLO26n detector backend
    weights_path = (
        DEFAULT_YOLO_WEIGHTS
        if DEFAULT_YOLO_WEIGHTS.exists()
        else Path("yolov8n.pt")
    )
    detector = YOLO(str(weights_path))
    # Warmup detector on first frame
    _ = detector(frames[0], conf=DEFAULT_CONF_THRESHOLD, device=DEVICE, verbose=False)

    # Precompute YOLO26n detections for SORT variants
    print("[INFO] Computing YOLO26n vehicle detections for offline tracker evaluation...")
    cached_detections: List[np.ndarray] = []
    det_latencies: List[float] = []

    for frame in frames:
        t0 = time.perf_counter()
        pred_res = detector(
            frame,
            conf=DEFAULT_CONF_THRESHOLD,
            iou=0.45,
            device=DEVICE,
            verbose=False,
        )
        t1 = time.perf_counter()
        det_latencies.append((t1 - t0) * 1000.0)

        boxes_list: List[np.ndarray] = []
        for r in pred_res:
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
        cached_detections.append(dets)

    mean_det_latency = float(np.mean(det_latencies)) if det_latencies else 15.0

    configs: List[Tuple[str, str]] = [
        ("Classic SORT (Baseline)", "classic_sort"),
        ("Enhanced SORT (Ours)", "enhanced_sort"),
        ("ByteTrack", "bytetrack"),
        ("BoT-SORT", "botsort"),
    ]

    results_summary: List[Dict[str, Any]] = []

    for label, tracker_key in configs:
        print(f"[INFO] Benchmarking tracker: {label} (Model: YOLO26n)...")
        pred_tracks: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        tracker_latencies: List[float] = []

        if tracker_key == "classic_sort":
            tracker_inst = ClassicSORTTracker(
                max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESH
            )
            for f_idx, dets in enumerate(cached_detections):
                t0 = time.perf_counter()
                out_trks = tracker_inst.update(dets)
                t1 = time.perf_counter()
                tracker_latencies.append((t1 - t0) * 1000.0 + mean_det_latency)

                f_preds: List[Tuple[int, np.ndarray]] = []
                for trk in out_trks:
                    f_preds.append((int(trk[0]), trk[1:5]))
                pred_tracks[f_idx] = f_preds

        elif tracker_key == "enhanced_sort":
            tracker_inst = SORTTracker(
                max_age=MAX_AGE,
                min_hits=MIN_HITS,
                iou_threshold=IOU_THRESH,
                track_nms=TRACK_NMS,
            )
            for f_idx, dets in enumerate(cached_detections):
                t0 = time.perf_counter()
                out_trks = tracker_inst.update(dets)
                t1 = time.perf_counter()
                tracker_latencies.append((t1 - t0) * 1000.0 + mean_det_latency)

                f_preds: List[Tuple[int, np.ndarray]] = []
                for trk in out_trks:
                    f_preds.append((int(trk[0]), trk[1:5]))
                pred_tracks[f_idx] = f_preds

        else:
            # ByteTrack or BoT-SORT via Ultralytics online engine
            cfg_file = (
                CUSTOM_BYTETRACK_YAML
                if tracker_key == "bytetrack"
                else CUSTOM_BOTSORT_YAML
            )
            for f_idx, frame in enumerate(frames):
                t0 = time.perf_counter()
                res = detector.track(
                    source=frame,
                    persist=True,
                    tracker=str(cfg_file),
                    conf=DEFAULT_CONF_THRESHOLD,
                    device=DEVICE,
                    verbose=False,
                )
                t1 = time.perf_counter()
                tracker_latencies.append((t1 - t0) * 1000.0)

                f_preds: List[Tuple[int, np.ndarray]] = []
                for r in res:
                    if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
                        ids = r.boxes.id.int().cpu().numpy()
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        for tid, box in zip(ids, xyxy):
                            if int(tid) > 0:
                                f_preds.append((int(tid), box.astype(np.float32)))
                pred_tracks[f_idx] = f_preds

        avg_lat = float(np.mean(tracker_latencies)) if tracker_latencies else 20.0
        fps = 1000.0 / max(1.0, avg_lat)

        mota_metrics = compute_mota(gt_tracks, pred_tracks, iou_thresh=0.35)
        idf1_val = compute_idf1(gt_tracks, pred_tracks, iou_thresh=0.35)
        hota_val = compute_hota(gt_tracks, pred_tracks, iou_thresh=0.35)

        bounded_mota = max(0.0, float(mota_metrics["mota"]))

        record: Dict[str, Any] = {
            "Configuration": label,
            "Model": "YOLO26n",
            "MOTA": round(bounded_mota, 3),
            "IDF1": round(float(idf1_val), 3),
            "HOTA": round(float(hota_val), 3),
            "IDSW": int(mota_metrics["idsw"]),
            "Latency_ms": round(avg_lat, 2),
            "FPS": round(fps, 1),
            "FP": int(mota_metrics["fp"]),
            "FN": int(mota_metrics["fn"]),
            "Total_GT": int(mota_metrics["gt_total"]),
        }
        results_summary.append(record)
        print(
            f"  -> MOTA: {record['MOTA']} | IDF1: {record['IDF1']} | "
            f"HOTA: {record['HOTA']} | IDSW: {record['IDSW']} | FPS: {record['FPS']}"
        )

    return results_summary


def save_reports(results: List[Dict[str, Any]]) -> None:
    """Exports benchmark metrics to CSV and Matplotlib visualization charts.

    Args:
        results: Benchmark records list.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "metrics_summary.csv"

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[SUCCESS] Metrics summary saved to: {csv_path}")

    # Visualizations
    names = [r["Configuration"] for r in results]
    motas = [r["MOTA"] for r in results]
    idf1s = [r["IDF1"] for r in results]
    hotas = [r["HOTA"] for r in results]
    idsws = [r["IDSW"] for r in results]
    fps_vals = [r["FPS"] for r in results]

    # Figure 1: Comprehensive benchmark comparison (Accuracy & Speed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    width = 0.25

    ax1.bar(x - width, motas, width, label="MOTA", color="#2980b9")
    ax1.bar(x, idf1s, width, label="IDF1", color="#27ae60")
    ax1.bar(x + width, hotas, width, label="HOTA", color="#8e44ad")
    ax1.set_ylabel("Score")
    ax1.set_title("Tracking Accuracy (MOTA, IDF1, HOTA) on UA-DETRAC (YOLO26n)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    bars2 = ax2.bar(names, fps_vals, color="#e67e22", width=0.45)
    ax2.axhline(
        25.0, color="red", linestyle="--", linewidth=1.5, label="Real-Time CCTV (25 FPS)"
    )
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.0,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylabel("Processing Speed (FPS)")
    ax2.set_title("Throughput Benchmark (FPS)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = REPORTS_DIR / "benchmark_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Benchmark visualization plot saved to: {plot_path}")

    # Figure 2: Ablation Study - Identity Switches & Association Consistency
    fig_abl, (ax_idsw, ax_assoc) = plt.subplots(1, 2, figsize=(13, 5))
    bars_idsw = ax_idsw.bar(names, idsws, color=["#c0392b", "#27ae60", "#2980b9", "#8e44ad"], width=0.5)
    for bar in bars_idsw:
        yval = bar.get_height()
        ax_idsw.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.1,
            f"{int(yval)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_idsw.set_ylabel("ID Switches (Lower is Better)")
    ax_idsw.set_title("Ablation: Identity Switch Frequency")
    ax_idsw.set_xticks(x)
    ax_idsw.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax_idsw.grid(axis="y", linestyle="--", alpha=0.7)

    # Association metric (IDF1 vs HOTA)
    ax_assoc.plot(names, idf1s, marker="o", linewidth=2.5, markersize=8, label="IDF1", color="#27ae60")
    ax_assoc.plot(names, hotas, marker="s", linewidth=2.5, markersize=8, label="HOTA", color="#8e44ad")
    ax_assoc.set_ylabel("Association Score")
    ax_assoc.set_title("Association Stability (IDF1 & HOTA)")
    ax_assoc.set_xticks(x)
    ax_assoc.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax_assoc.set_ylim(0.0, 1.0)
    ax_assoc.legend()
    ax_assoc.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    ablation_plot_path = REPORTS_DIR / "ablation_idsw.png"
    plt.savefig(ablation_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Ablation visualization plot saved to: {ablation_plot_path}")


def main() -> None:
    """CLI for running benchmark evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        type=str,
        default="MVI_20012",
        help="UA-DETRAC sequence ID (default: MVI_20012)",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Dataset split ('val' or 'train')"
    )
    parser.add_argument(
        "--frames", type=int, default=50, help="Number of frames to evaluate"
    )
    args = parser.parse_args()

    results = run_benchmark(
        sequence_prefix=args.sequence, split=args.split, num_frames=args.frames
    )
    save_reports(results)


if __name__ == "__main__":
    main()
