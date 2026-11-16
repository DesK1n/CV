"""SORT (Simple Online and Realtime Tracking) implementation for vehicle tracking.

Uses Kalman filtering for motion prediction, two-stage Hungarian assignment
(IoU + normalized Euclidean center distance), and track deduplication (Track NMS).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import IOU_THRESH, MAX_AGE, MIN_HITS, TRACK_NMS


def compute_iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Computes pairwise Intersection-over-Union (IoU) between bounding box sets.

    Args:
        bb_test: Array of shape (N, 4) in format [x1, y1, x2, y2].
        bb_gt: Array of shape (M, 4) in format [x1, y1, x2, y2].

    Returns:
        np.ndarray: IoU matrix of shape (N, M).
    """
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float32)

    xx1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0:1].T)
    yy1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1:2].T)
    xx2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2:3].T)
    yy2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3:4].T)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    union = area_test[:, None] + area_gt[None, :] - inter

    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class KalmanBoxTracker:
    """Tracks a single bounding box using dynamic linear Kalman motion filtering.

    Attributes:
        id: Unique tracking identity integer.
        x: 7-dimensional state vector [u, v, s, r, u_dot, v_dot, s_dot].
        score: Most recent detection confidence score.
        class_id: Detected vehicle class index.
        time_since_update: Frames elapsed since last matched observation.
        hits: Total successful match count.
        hit_streak: Consecutive frames with active detections.
        age: Total track lifetime in frames.
        history: Sequence of previous (center_x, center_y) positions.
    """

    count: int = 0

    def __init__(
        self, bbox: np.ndarray, score: float = 1.0, class_id: float = 0.0
    ) -> None:
        """Initializes Kalman tracker from bounding box [x1, y1, x2, y2].

        Args:
            bbox: Bounding box array [x1, y1, x2, y2].
            score: Initial detection confidence.
            class_id: Initial detected class ID.
        """
        self.id: int = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        u = float(bbox[0] + w / 2.0)
        v = float(bbox[1] + h / 2.0)
        s = w * h
        r = w / h

        # 7-dimensional state: [u, v, s, r, u_dot, v_dot, s_dot]
        self.x: np.ndarray = np.array([u, v, s, r, 0.0, 0.0, 0.0], dtype=np.float32)
        self.score: float = float(score)
        self.class_id: float = float(class_id)
        self.time_since_update: int = 0
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0
        self.history: List[Tuple[float, float]] = [(u, v)]

    def update(self, bbox: np.ndarray, score: float, class_id: float) -> None:
        """Updates state vector with observed bounding box and estimates velocity.

        Args:
            bbox: New observation bounding box [x1, y1, x2, y2].
            score: Observation confidence score.
            class_id: Vehicle class identifier.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)
        self.class_id = float(class_id)

        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        u = float(bbox[0] + w / 2.0)
        v = float(bbox[1] + h / 2.0)
        s = w * h
        r = w / h

        # Velocity estimation
        prev_u = float(self.x[0])
        prev_v = float(self.x[1])
        prev_s = float(self.x[2])

        meas_u_dot = u - prev_u
        meas_v_dot = v - prev_v
        meas_s_dot = s - prev_s

        # Position correction
        alpha = 0.75
        self.x[0] = alpha * u + (1.0 - alpha) * (self.x[0] + self.x[4])
        self.x[1] = alpha * v + (1.0 - alpha) * (self.x[1] + self.x[5])
        self.x[2] = alpha * s + (1.0 - alpha) * (self.x[2] + self.x[6])
        self.x[3] = alpha * r + (1.0 - alpha) * self.x[3]

        # Velocity momentum update
        beta = 0.40
        self.x[4] = beta * meas_u_dot + (1.0 - beta) * self.x[4]
        self.x[5] = beta * meas_v_dot + (1.0 - beta) * self.x[5]
        self.x[6] = beta * meas_s_dot + (1.0 - beta) * self.x[6]

        self.history.append((float(self.x[0]), float(self.x[1])))
        if len(self.history) > 30:
            self.history.pop(0)

    def predict(self) -> np.ndarray:
        """Advances state vector using estimated velocity and returns predicted bbox.

        Returns:
            np.ndarray: Predicted bounding box [x1, y1, x2, y2].
        """
        self.x[0] += self.x[4]
        self.x[1] += self.x[5]
        self.x[2] += self.x[6]
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        w = np.sqrt(max(1.0, float(self.x[2] * self.x[3])))
        h = max(1.0, float(self.x[2] / w))
        x1 = self.x[0] - w / 2.0
        y1 = self.x[1] - h / 2.0
        x2 = self.x[0] + w / 2.0
        y2 = self.x[1] + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def get_state(self) -> np.ndarray:
        """Returns current bounding box estimate [x1, y1, x2, y2].

        Returns:
            np.ndarray: Bounding box coordinates [x1, y1, x2, y2].
        """
        w = np.sqrt(max(1.0, float(self.x[2] * self.x[3])))
        h = max(1.0, float(self.x[2] / w))
        return np.array(
            [
                self.x[0] - w / 2.0,
                self.x[1] - h / 2.0,
                self.x[0] + w / 2.0,
                self.x[1] + h / 2.0,
            ],
            dtype=np.float32,
        )


class SORTTracker:
    """Multi-Object Tracker with two-stage association and track deduplication.

    Attributes:
        max_age: Maximum frames to retain track without association.
        min_hits: Minimum hits before track is confirmed.
        iou_threshold: Minimum IoU overlap for primary Hungarian matching.
        track_nms: IoU threshold for track deduplication.
    """

    def __init__(
        self,
        max_age: int = MAX_AGE,
        min_hits: int = MIN_HITS,
        iou_threshold: float = IOU_THRESH,
        track_nms: float = TRACK_NMS,
    ) -> None:
        """Initializes SORT tracker.

        Args:
            max_age: Maximum frames to retain track without association.
            min_hits: Minimum hits before track is confirmed.
            iou_threshold: Minimum IoU overlap for primary Hungarian matching.
            track_nms: IoU threshold for track deduplication.
        """
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.iou_threshold: float = iou_threshold
        self.track_nms: float = track_nms
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Updates tracker with detections from current frame.

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

        # Two-stage Hungarian assignment (IoU + Euclidean center distance fallback)
        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks)

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(
                detections[det_idx, :4],
                score=float(detections[det_idx, 4]),
                class_id=float(detections[det_idx, 5]),
            )

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(
                detections[det_idx, :4],
                score=float(detections[det_idx, 4]),
                class_id=float(detections[det_idx, 5]),
            )
            self.trackers.append(new_trk)

        # Extract active tracks
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
                            float(trk.hits),
                        ]
                    )

        self.trackers = remaining_trackers

        # Track NMS / Deduplication: prevents multiple boxes on the same vehicle
        if len(active_tracks) > 1:
            boxes = np.array([a[1:5] for a in active_tracks], dtype=np.float32)
            ious = compute_iou_batch(boxes, boxes)
            keep = set(range(len(active_tracks)))
            for i in range(len(active_tracks)):
                if i not in keep:
                    continue
                for j in range(i + 1, len(active_tracks)):
                    if j in keep and ious[i, j] > self.track_nms:
                        # Keep the track with more hits
                        if active_tracks[i][7] >= active_tracks[j][7]:
                            keep.remove(j)
                        else:
                            keep.remove(i)
                            break
            filtered_tracks = [active_tracks[k][:7] for k in sorted(keep)]
        else:
            filtered_tracks = [a[:7] for a in active_tracks]

        if not filtered_tracks:
            return np.empty((0, 7), dtype=np.float32)

        return np.array(filtered_tracks, dtype=np.float32)

    def _associate(
        self, dets: np.ndarray, trks: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associates detections with predicted tracks using IoU with distance fallback.

        Args:
            dets: Detections bounding box array (N, 4).
            trks: Predicted tracks bounding box array (M, 4).

        Returns:
            Tuple: (matched_indices, unmatched_detections, unmatched_trackers).
        """
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))

        # Stage 1: Hungarian assignment on IoU
        iou_matrix = compute_iou_batch(dets, trks)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trks)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((int(r), int(c)))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        # Stage 2: Normalized Euclidean center distance fallback for fast-moving vehicles
        new_unmatched_dets: List[int] = []
        for d in unmatched_dets:
            d_box = dets[d]
            d_cx = (d_box[0] + d_box[2]) / 2.0
            d_cy = (d_box[1] + d_box[3]) / 2.0
            best_t: int | None = None
            best_dist: float = 0.90  # within 90% of vehicle dimensions
            for t in unmatched_trks:
                t_box = trks[t]
                t_cx = (t_box[0] + t_box[2]) / 2.0
                t_cy = (t_box[1] + t_box[3]) / 2.0
                tw = max(10.0, float(t_box[2] - t_box[0]))
                th = max(10.0, float(t_box[3] - t_box[1]))
                dist = float(
                    np.sqrt(((d_cx - t_cx) / tw) ** 2 + ((d_cy - t_cy) / th) ** 2)
                )
                if dist < best_dist:
                    best_dist = dist
                    best_t = t
            if best_t is not None:
                matched.append((d, best_t))
                unmatched_trks.remove(best_t)
            else:
                new_unmatched_dets.append(d)

        return matched, new_unmatched_dets, unmatched_trks


# Backward compatibility and alternative naming
Sort = SORTTracker
