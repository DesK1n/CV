"""Temporal State Space Model (SSM) Tracker for vehicle trajectory estimation.

Implements the selective state space recurrence for temporal trajectory modeling
with two-stage association (IoU + Euclidean center distance) and track deduplication.
"""
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.sort_tracker import compute_iou_batch


class SSMTargetState:
    """Manages the recurrent state space dynamics for an individual vehicle track."""

    count = 0

    def __init__(self, bbox: np.ndarray, score: float, class_id: float, d_state: int = 16) -> None:
        """Initializes a new SSM target track.

        Args:
            bbox: [x1, y1, x2, y2]
            score: Confidence score
            class_id: Class label
            d_state: State vector dimension
        """
        self.id = SSMTargetState.count
        SSMTargetState.count += 1

        self.score = score
        self.class_id = class_id
        self.d_state = d_state
        self.time_since_update = 0
        self.hits = 1

        # Current bounding box estimate
        self.bbox = bbox.copy().astype(np.float32)

        # Recurrent state vector h in R^d_state
        self.h = np.zeros((d_state,), dtype=np.float32)

        # Discretized SSM parameters
        self.A = np.diag(np.linspace(0.85, 0.99, d_state).astype(np.float32))
        self.B = np.ones((d_state, 4), dtype=np.float32) * 0.1
        self.C = np.ones((4, d_state), dtype=np.float32) * (1.0 / float(d_state))

        # Motion velocity vector [dx1, dy1, dx2, dy2]
        self.velocity = np.zeros(4, dtype=np.float32)
        self.history: List[Tuple[float, float]] = [
            ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        ]

    def predict(self) -> np.ndarray:
        """Propagates state forward in time using SSM recurrence."""
        self.time_since_update += 1

        # Advance hidden state h_t = A * h_{t-1} + B * velocity
        self.h = self.A @ self.h + self.B @ self.velocity

        # Predict motion delta via output matrix C
        delta = self.C @ self.h
        predicted_bbox = self.bbox + delta

        # Ensure valid coordinates
        predicted_bbox[2] = max(predicted_bbox[0] + 5.0, predicted_bbox[2])
        predicted_bbox[3] = max(predicted_bbox[1] + 5.0, predicted_bbox[3])
        return predicted_bbox

    def update(self, bbox: np.ndarray, score: float, class_id: float) -> None:
        """Updates track with observed detection."""
        observed_delta = bbox - self.bbox
        self.velocity = 0.7 * self.velocity + 0.3 * observed_delta

        # Update SSM state
        self.h = self.A @ self.h + self.B @ observed_delta
        self.bbox = bbox.copy().astype(np.float32)

        self.score = score
        self.class_id = class_id
        self.time_since_update = 0
        self.hits += 1

        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.history.append((cx, cy))
        if len(self.history) > 30:
            self.history.pop(0)


class SSMTracker:
    """Multi-object vehicle tracker powered by selective state space modeling."""

    def __init__(
        self, max_age: int = 15, min_hits: int = 2, iou_threshold: float = 0.25
    ) -> None:
        """Initializes SSMTracker.

        Args:
            max_age: Frames allowed without matching before track termination.
            min_hits: Hits required for confirmation.
            iou_threshold: Association threshold.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.targets: List[SSMTargetState] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Updates tracks with current frame detections.

        Args:
            detections: Array of shape (N, 6): [x1, y1, x2, y2, score, class_id].

        Returns:
            np.ndarray: Tracked vehicles of shape (M, 7):
                [track_id, x1, y1, x2, y2, score, class_id].
        """
        self.frame_count += 1
        predicted_boxes = [t.predict() for t in self.targets]

        trks = np.array(predicted_boxes, dtype=np.float32) if predicted_boxes else np.empty((0, 4))
        dets = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))

        # Two-stage Hungarian assignment (IoU + Euclidean center distance fallback)
        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks)

        for det_idx, trk_idx in matched:
            self.targets[trk_idx].update(
                detections[det_idx, :4],
                score=float(detections[det_idx, 4]),
                class_id=float(detections[det_idx, 5]),
            )

        for det_idx in unmatched_dets:
            new_target = SSMTargetState(
                detections[det_idx, :4],
                score=float(detections[det_idx, 4]),
                class_id=float(detections[det_idx, 5]),
            )
            self.targets.append(new_target)

        active_tracks = []
        surviving_targets = []
        for t in self.targets:
            if t.time_since_update <= self.max_age:
                surviving_targets.append(t)
                if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    active_tracks.append(
                        [
                            float(t.id),
                            t.bbox[0],
                            t.bbox[1],
                            t.bbox[2],
                            t.bbox[3],
                            t.score,
                            t.class_id,
                            float(t.hits),
                        ]
                    )

        self.targets = surviving_targets

        # Track NMS / Deduplication: prevents multiple boxes on the same vehicle
        if len(active_tracks) > 1:
            boxes = np.array([a[1:5] for a in active_tracks], dtype=np.float32)
            ious = compute_iou_batch(boxes, boxes)
            keep = set(range(len(active_tracks)))
            for i in range(len(active_tracks)):
                if i not in keep:
                    continue
                for j in range(i + 1, len(active_tracks)):
                    if j in keep and ious[i, j] > 0.45:
                        if active_tracks[i][7] >= active_tracks[j][7]:
                            keep.remove(j)
                        else:
                            keep.remove(i)
                            break
            active_tracks = [active_tracks[k][:7] for k in sorted(keep)]
        else:
            active_tracks = [a[:7] for a in active_tracks]

        if not active_tracks:
            return np.empty((0, 7), dtype=np.float32)

        return np.array(active_tracks, dtype=np.float32)

    def _associate(
        self, dets: np.ndarray, trks: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian association using IoU distance matrix with distance fallback."""
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
                matched.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        # Stage 2: Normalized Euclidean center distance fallback
        new_unmatched_dets = []
        for d in unmatched_dets:
            d_box = dets[d]
            d_cx = (d_box[0] + d_box[2]) / 2.0
            d_cy = (d_box[1] + d_box[3]) / 2.0
            best_t = None
            best_dist = 0.90
            for t in unmatched_trks:
                t_box = trks[t]
                t_cx = (t_box[0] + t_box[2]) / 2.0
                t_cy = (t_box[1] + t_box[3]) / 2.0
                tw = max(10.0, float(t_box[2] - t_box[0]))
                th = max(10.0, float(t_box[3] - t_box[1]))
                dist = np.sqrt(((d_cx - t_cx) / tw) ** 2 + ((d_cy - t_cy) / th) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_t = t
            if best_t is not None:
                matched.append((d, best_t))
                unmatched_trks.remove(best_t)
            else:
                new_unmatched_dets.append(d)

        return matched, new_unmatched_dets, unmatched_trks
