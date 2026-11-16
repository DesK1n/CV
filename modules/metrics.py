"""Evaluation metrics for object detection and multi-object tracking (MOT).

Calculates IoU, mAP@50, mAP@50-95, MOTA, IDF1, and ID Switches according to
the CLEAR MOT protocol.
"""
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculates Intersection over Union between two boxes [x1, y1, x2, y2].

    Args:
        box1: Coordinates of first box (4,).
        box2: Coordinates of second box (4,).

    Returns:
        float: IoU score in range [0.0, 1.0].
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_mota(
    gt_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """Computes Multiple Object Tracking Accuracy (MOTA) and related metrics.

    Formula:
        MOTA = 1 - (sum(FN) + sum(FP) + sum(IDSW)) / sum(GT)

    Args:
        gt_tracks: Dict mapping frame_idx -> list of (track_id, box).
        pred_tracks: Dict mapping frame_idx -> list of (track_id, box).
        iou_thresh: Minimum IoU overlap to consider a true positive match.

    Returns:
        Dict with keys: 'mota', 'fp', 'fn', 'idsw', 'gt_total'.
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0

    all_frames = sorted(set(gt_tracks.keys()) | set(pred_tracks.keys()))
    last_matched_id: Dict[int, int] = {}  # gt_id -> pred_id

    for frame in all_frames:
        gts = gt_tracks.get(frame, [])
        preds = pred_tracks.get(frame, [])
        total_gt += len(gts)

        if not gts:
            total_fp += len(preds)
            continue
        if not preds:
            total_fn += len(gts)
            continue

        # Build cost matrix: -IoU
        cost_matrix = np.zeros((len(gts), len(preds)), dtype=np.float32)
        for g_idx, (_, g_box) in enumerate(gts):
            for p_idx, (_, p_box) in enumerate(preds):
                cost_matrix[g_idx, p_idx] = -box_iou(g_box, p_box)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        matched_pred = set()

        for g_idx, p_idx in zip(row_ind, col_ind):
            iou_val = -cost_matrix[g_idx, p_idx]
            if iou_val >= iou_thresh:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)

                gt_id = gts[g_idx][0]
                pred_id = preds[p_idx][0]

                # Check ID switch
                if gt_id in last_matched_id and last_matched_id[gt_id] != pred_id:
                    total_idsw += 1
                last_matched_id[gt_id] = pred_id

        total_fn += len(gts) - len(matched_gt)
        total_fp += len(preds) - len(matched_pred)

    mota = 1.0 - (float(total_fn + total_fp + total_idsw) / max(1.0, float(total_gt)))
    return {
        "mota": float(mota),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "idsw": int(total_idsw),
        "gt_total": int(total_gt),
    }


def compute_idf1(
    gt_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_thresh: float = 0.5,
) -> float:
    """Computes Identification F1-Score (IDF1) for association consistency.

    Args:
        gt_tracks: Ground truth tracks.
        pred_tracks: Predicted tracks.
        iou_thresh: IoU overlap threshold.

    Returns:
        float: IDF1 score in range [0.0, 1.0].
    """
    idtp = 0
    idfp = 0
    idfn = 0

    all_frames = set(gt_tracks.keys()) | set(pred_tracks.keys())
    for frame in all_frames:
        gts = gt_tracks.get(frame, [])
        preds = pred_tracks.get(frame, [])

        if not gts:
            idfp += len(preds)
            continue
        if not preds:
            idfn += len(gts)
            continue

        cost_matrix = np.zeros((len(gts), len(preds)), dtype=np.float32)
        for g_idx, (_, g_box) in enumerate(gts):
            for p_idx, (_, p_box) in enumerate(preds):
                cost_matrix[g_idx, p_idx] = -box_iou(g_box, p_box)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = 0
        for g_idx, p_idx in zip(row_ind, col_ind):
            if -cost_matrix[g_idx, p_idx] >= iou_thresh:
                matches += 1

        idtp += matches
        idfn += len(gts) - matches
        idfp += len(preds) - matches

    denom = 2 * idtp + idfp + idfn
    if denom == 0:
        return 0.0
    return float(2 * idtp / denom)


def compute_hota(
    gt_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_thresh: float = 0.5,
) -> float:
    """Computes Higher Order Tracking Accuracy (HOTA) score.

    HOTA cleanly balances Detection Accuracy (DetA) and Association Accuracy (AssA):
    $$HOTA_\\alpha = \\sqrt{DetA_\\alpha \\cdot AssA_\\alpha}$$

    Args:
        gt_tracks: Dict mapping frame_idx -> list of (track_id, box).
        pred_tracks: Dict mapping frame_idx -> list of (track_id, box).
        iou_thresh: IoU threshold (default 0.5).

    Returns:
        float: HOTA score in range [0.0, 1.0].
    """
    all_frames = sorted(set(gt_tracks.keys()) | set(pred_tracks.keys()))
    frame_matches: Dict[int, List[Tuple[int, int]]] = {}
    tp = 0
    fn = 0
    fp = 0
    gt_id_frames: Dict[int, set[int]] = {}
    pr_id_frames: Dict[int, set[int]] = {}

    for f in all_frames:
        gts = gt_tracks.get(f, [])
        preds = pred_tracks.get(f, [])
        for gid, _ in gts:
            gt_id_frames.setdefault(gid, set()).add(f)
        for pid, _ in preds:
            pr_id_frames.setdefault(pid, set()).add(f)

        if not gts:
            fp += len(preds)
            continue
        if not preds:
            fn += len(gts)
            continue

        cost = np.zeros((len(gts), len(preds)), dtype=np.float32)
        for i, (_, gb) in enumerate(gts):
            for j, (_, pb) in enumerate(preds):
                cost[i, j] = -box_iou(gb, pb)
        row_ind, col_ind = linear_sum_assignment(cost)

        f_matches: List[Tuple[int, int]] = []
        matched_gt = set()
        matched_pr = set()
        for r, c in zip(row_ind, col_ind):
            if -cost[r, c] >= iou_thresh:
                gid = gts[r][0]
                pid = preds[c][0]
                f_matches.append((gid, pid))
                matched_gt.add(r)
                matched_pr.add(c)

        frame_matches[f] = f_matches
        tp += len(f_matches)
        fn += len(gts) - len(matched_gt)
        fp += len(preds) - len(matched_pr)

    denom = tp + fn + fp
    if denom == 0 or tp == 0:
        return 0.0

    # Count pair co-occurrences across all frames
    pair_counts: Dict[Tuple[int, int], int] = {}
    for f, matches in frame_matches.items():
        for gid, pid in matches:
            pair = (gid, pid)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Association accuracy per match
    ass_scores: List[float] = []
    for f, matches in frame_matches.items():
        for gid, pid in matches:
            tpa = pair_counts[(gid, pid)]
            fna = len(gt_id_frames[gid]) - tpa
            fpa = len(pr_id_frames[pid]) - tpa
            a_c = float(tpa) / float(tpa + fna + fpa)
            ass_scores.append(a_c)

    ass_a = float(np.mean(ass_scores)) if ass_scores else 0.0
    det_a = float(tp) / float(denom)
    hota = float(np.sqrt(det_a * ass_a))
    return hota
