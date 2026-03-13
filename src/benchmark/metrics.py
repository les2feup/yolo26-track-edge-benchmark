from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

from benchmark.config import MIN_TRACK_LEN_FRAMES
from benchmark.mot_gt import load_gt, load_seqinfo

# Fragmentation statistics: ratio plus raw counts for denominator-collapse diagnosis
FragStats = namedtuple("FragStats", ["ratio", "total_initiated", "short_tracks_abs"])


def compute_mot_metrics(raw_csv: Path, seq_dir: Path) -> dict:
    """Tracking quality metrics for one (model, sequence, resolution) run.

    Evaluates the raw inference CSV against MOT17 ground truth using the
    py-motmetrics accumulator. The IoU distance threshold follows the
    MOTChallenge standard (max_iou = 0.5, i.e. IoU >= 0.5 is a match).

    Bounding-box conversion: the CSV stores bboxes in xyxy format; motmetrics
    expects xywh (top-left corner + width/height), so w = x2-x1, h = y2-y1.

    The fragmentation ratio is appended as a derived metric: GT-matched tracks
    shorter than MIN_TRACK_LEN_FRAMES divided by the GT track count. Spurious
    initiations (unmatched pred IDs) are excluded from the count.

    Returns a flat dict with keys:
        mota, idf1, num_switches, mostly_tracked, mostly_lost,
        frag_ratio, mean_inference_ms, fps, peak_mem_mb
    """
    gt_df   = load_gt(seq_dir)
    info    = load_seqinfo(seq_dir)
    raw_df  = pd.read_csv(raw_csv)

    pred_df          = _decode_predictions(raw_df)
    acc              = _build_accumulator(gt_df, pred_df)
    matched_pred_ids = _matched_pred_ids(acc)

    mh      = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "mostly_tracked", "mostly_lost"],
        name="seq",
    )

    # Timing summary (warm-up NaN rows are excluded by skipna default in mean())
    mean_ms = float(raw_df["inference_ms"].mean(skipna=True))
    fps     = 1000.0 / mean_ms if mean_ms > 0 else float("nan")

    # Full-process RSS: Python + framework + model weights. This is the deployment-
    # relevant figure for edge devices where the framework floor dominates RSS and
    # model-weight deltas are below measurement resolution.
    mem_col = next((c for c in ("mem_total_bytes", "mem_bytes") if c in raw_df.columns), None)
    peak_mem_mb = float(raw_df[mem_col].max()) / 1e6 if mem_col else float("nan")

    num_switches   = int(summary.loc["seq", "num_switches"])
    n_gt_tracks    = int(gt_df["track_id"].nunique())
    idsw_per_gt    = num_switches / n_gt_tracks if n_gt_tracks > 0 else float("nan")

    # Fragmentation: GT-matched short tracks only, anchored to GT track count.
    # Excludes spurious initiations (unmatched pred IDs) so the signal measures
    # real people the tracker found but then lost, not false-positive track churn.
    frag_stats = _fragmentation_ratio(raw_df, n_gt_tracks, matched_pred_ids)

    mostly_tracked     = int(summary.loc["seq", "mostly_tracked"])
    mostly_tracked_ratio = mostly_tracked / n_gt_tracks if n_gt_tracks > 0 else float("nan")

    return {
        "mota":                float(summary.loc["seq", "mota"]),
        "idf1":                float(summary.loc["seq", "idf1"]),
        "num_switches":        num_switches,
        "idsw_per_gt_track":   idsw_per_gt,
        "mostly_tracked":      mostly_tracked,
        "mostly_tracked_ratio": mostly_tracked_ratio,
        "mostly_lost":         int(summary.loc["seq", "mostly_lost"]),
        "frag_ratio":          frag_stats.ratio,
        "total_initiated":     frag_stats.total_initiated,
        "short_tracks_abs":    frag_stats.short_tracks_abs,
        "mean_inference_ms":   mean_ms,
        "fps":                 fps,
        "peak_mem_mb":         peak_mem_mb,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _decode_predictions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Expand JSON-encoded bbox and track_id columns into a flat DataFrame.

    Output columns: frame_id, track_id, x, y, w, h  (xywh, matching GT format).
    """
    rows = []
    for _, row in raw_df.iterrows():
        track_ids = json.loads(row["track_ids"])
        bboxes    = json.loads(row["bboxes_xyxy"])
        for tid, (x1, y1, x2, y2) in zip(track_ids, bboxes):
            rows.append({
                "frame_id": int(row["frame_id"]),
                "track_id": int(tid),
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
            })
    return pd.DataFrame(rows, columns=["frame_id", "track_id", "x", "y", "w", "h"])


def _build_accumulator(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> mm.MOTAccumulator:
    """Feed GT and predictions frame-by-frame into a MOTAccumulator.

    Frames present in GT but absent from predictions are treated as misses.
    The IoU distance matrix uses the MOTChallenge threshold of 0.5.
    """
    acc         = mm.MOTAccumulator(auto_id=False)
    all_frames  = sorted(gt_df["frame_id"].unique())

    for fid in all_frames:
        gt_frame   = gt_df[gt_df["frame_id"] == fid]
        pred_frame = pred_df[pred_df["frame_id"] == fid] if not pred_df.empty else pd.DataFrame()

        gt_ids   = gt_frame["track_id"].tolist()
        pred_ids = pred_frame["track_id"].tolist() if not pred_frame.empty else []

        gt_boxes   = gt_frame[["x", "y", "w", "h"]].values.astype(float)
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values.astype(float) if not pred_frame.empty else np.empty((0, 4))

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            dist = np.full((len(gt_ids), len(pred_ids)), np.nan)
        else:
            # motmetrics.distances.iou_matrix uses np.asfarray which was removed
            # in NumPy 2.0.  Compute the 1-IoU distance matrix directly.
            dist = _iou_distance(gt_boxes, pred_boxes)

        acc.update(gt_ids, pred_ids, dist, frameid=fid)

    return acc


def _matched_pred_ids(acc: mm.MOTAccumulator) -> set:
    """Set of pred track IDs that were matched to a GT track at least once.

    Extracted from the accumulator event log (Type == 'MATCH', column HId).
    Used to exclude spurious initiations from the fragmentation count — only
    tracks that correspond to a real GT person are counted as fragmented.
    """
    events = acc.mot_events
    matched = events.loc[events["Type"] == "MATCH", "HId"].dropna()
    return set(matched.astype(int).unique())


def _fragmentation_ratio(
    raw_df: pd.DataFrame,
    n_gt_tracks: int,
    matched_pred_ids: set,
) -> "FragStats":
    """Fragmentation statistics for GT-matched tracks in one run.

    Returns a FragStats namedtuple with three fields:
    - ratio: GT-matched short tracks / GT track count (resolution-invariant)
    - total_initiated: total unique track IDs emitted by ByteTrack (diagnostic only)
    - short_tracks_abs: GT-matched tracks shorter than MIN_TRACK_LEN_FRAMES

    Only pred IDs that were matched to a GT track at least once are counted.
    Spurious initiations (false-positive detections that quickly vanish) are
    excluded so the signal measures real people the tracker found but then lost.
    """
    if raw_df.empty:
        return FragStats(ratio=float("nan"), total_initiated=0, short_tracks_abs=0)

    track_lengths: dict[int, int] = {}
    for track_ids_json in raw_df["track_ids"]:
        for tid in json.loads(track_ids_json):
            track_lengths[tid] = track_lengths.get(tid, 0) + 1

    total     = len(track_lengths)
    # Restrict to GT-matched pred IDs only
    short_abs = sum(
        1 for tid, length in track_lengths.items()
        if tid in matched_pred_ids and length < MIN_TRACK_LEN_FRAMES
    )
    ratio = short_abs / n_gt_tracks if n_gt_tracks > 0 else 0.0

    return FragStats(ratio=ratio, total_initiated=total, short_tracks_abs=short_abs)


def _iou_distance(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """1 − IoU distance matrix for xywh bounding boxes, shape (N_gt, N_pred).

    Entries above the 0.5 IoU match threshold are set to NaN so motmetrics
    treats them as infeasible assignments (same semantics as iou_matrix's
    max_iou parameter).
    """
    # Convert xywh → xyxy for vectorised intersection computation
    def to_xyxy(b: np.ndarray) -> np.ndarray:
        return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)

    ga = to_xyxy(gt_boxes)    # (N_gt, 4)
    pa = to_xyxy(pred_boxes)  # (N_pred, 4)

    # Intersection
    inter_x1 = np.maximum(ga[:, None, 0], pa[None, :, 0])
    inter_y1 = np.maximum(ga[:, None, 1], pa[None, :, 1])
    inter_x2 = np.minimum(ga[:, None, 2], pa[None, :, 2])
    inter_y2 = np.minimum(ga[:, None, 3], pa[None, :, 3])
    inter_w  = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h  = np.maximum(0.0, inter_y2 - inter_y1)
    inter    = inter_w * inter_h

    area_g = gt_boxes[:, 2] * gt_boxes[:, 3]    # (N_gt,)
    area_p = pred_boxes[:, 2] * pred_boxes[:, 3] # (N_pred,)
    union  = area_g[:, None] + area_p[None, :] - inter

    iou  = np.where(union > 0, inter / union, 0.0)
    dist = 1.0 - iou
    # Infeasible assignment: IoU < 0.5 → distance > 0.5, flagged as NaN
    dist[iou < 0.5] = np.nan
    return dist
