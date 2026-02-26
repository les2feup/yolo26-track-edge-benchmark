import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.metrics import compute_mot_metrics
from benchmark.mot_gt import load_seqinfo


def detection_stability(csv_path: Path, baseline_csv: Path) -> float:
    """Mean absolute deviation of per-frame detection count vs the 640 baseline.

    Measures whether the detector continues to find approximately the same
    number of objects as resolution decreases.  The MAD is computed relative
    to the mean detection count of the baseline run — not the current run —
    so that zero degradation corresponds to a value near zero.
    """
    df_cur  = pd.read_csv(csv_path)
    df_base = pd.read_csv(baseline_csv)

    # Align on frame_id to handle any length mismatches
    merged = df_cur.set_index("frame_id")[["n_detections"]].join(
        df_base.set_index("frame_id")[["n_detections"]], rsuffix="_base", how="inner"
    )

    base_mean = merged["n_detections_base"].mean()
    if base_mean == 0:
        return 0.0

    mad = (merged["n_detections"] - merged["n_detections_base"]).abs().mean()
    return float(mad)


def track_continuity(csv_path: Path, seq_dir: Path) -> dict:
    """Two primary track-continuity signals plus fragmentation diagnostics.

    Delegates to compute_mot_metrics for IDSW computation against GT.

    Primary signals (methodology v3):
    - idsw_per_gt_track: identity confusion — switches per unique GT track. Rises when IoU
      precision degrades enough to cause mis-associations between spatially close detections.
      Note: at very low resolutions in dense scenes, detection recall collapse suppresses the
      association pool and drives IDSW toward zero despite worsening tracking — always read
      alongside mostly_tracked_ratio.
    - mostly_tracked_ratio: end-to-end continuity — fraction of GT tracks with ≥80% detection
      coverage. Monotonically degrades as resolution drops across all scene types; the consistent
      operating-envelope signal.

    Diagnostic columns (not primary signals):
    - frag_ratio: GT-matched short tracks / GT track count. Post-Fix-2 analysis showed
      ByteTrack almost never re-initiates a new track ID for a lost real person; losses are
      absorbed into mostly_lost state and reflected in MT, not in short-track counts. Retained
      for denominator-collapse diagnosis only (0–14 events per condition across MOT17 sequences).
    - total_initiated: total unique pred track IDs — denominator-collapse diagnostic.
    - short_tracks_abs: raw numerator of frag_ratio.

    Returns:
        {
            "num_switches":          int,    # absolute IDSW count
            "idsw_per_gt_track":     float,  # primary confusion signal
            "frag_ratio":            float,  # diagnostic only (not a primary claim)
            "total_initiated":       int,    # diagnostic denominator
            "short_tracks_abs":      int,    # diagnostic numerator
            "mostly_tracked_ratio":  float,  # primary continuity signal
        }
    """
    from benchmark.mot_gt import load_gt  # local import avoids circular dependency risk

    m           = compute_mot_metrics(csv_path, seq_dir)
    n_gt_tracks = load_gt(seq_dir)["track_id"].nunique()

    # Mostly Tracked: fraction of GT tracks with >= 80% detection coverage.
    # Shares the same GT track denominator as idsw_per_gt_track for consistency.
    mostly_tracked_ratio = m["mostly_tracked"] / n_gt_tracks if n_gt_tracks > 0 else 0.0

    return {
        "num_switches":          int(m["num_switches"]),
        "idsw_per_gt_track":     float(m["idsw_per_gt_track"]),
        "frag_ratio":            float(m["frag_ratio"]),
        "total_initiated":       int(m["total_initiated"]),
        "short_tracks_abs":      int(m["short_tracks_abs"]),
        "mostly_tracked_ratio":  float(mostly_tracked_ratio),
    }


def spatial_precision(csv_path: Path, baseline_csv: Path) -> float:
    """Mean footpoint displacement vs the 640 baseline, in pixels.

    Footpoints are the bottom-centre of each bounding box — the metric
    used for pedestrian ground position in crowd monitoring.

    Matching strategy: for each frame, pair detections between the current
    and baseline run by nearest footpoint (greedy assignment).  Track IDs
    are not used because ByteTrack may assign different IDs across runs
    that start from different resolutions.

    Returns the mean Euclidean displacement across all matched pairs,
    expressed in pixels at the reduced resolution.  Returns NaN if the
    current run has no detections.
    """
    df_cur  = pd.read_csv(csv_path)
    df_base = pd.read_csv(baseline_csv)

    displacements: list[float] = []

    frame_ids = df_cur["frame_id"].unique()
    for fid in frame_ids:
        cur_fps  = _get_footpoints(df_cur, fid)
        base_fps = _get_footpoints(df_base, fid)

        if len(cur_fps) == 0 or len(base_fps) == 0:
            continue

        # Greedy nearest-neighbour matching (sufficient for monotone displacement tracking)
        cur_arr  = np.array(cur_fps)   # (N, 2)
        base_arr = np.array(base_fps)  # (M, 2)

        # Scale baseline footpoints to the current resolution for a fair comparison
        imgsz_cur  = int(df_cur.loc[df_cur["frame_id"] == fid, "imgsz"].iloc[0])
        imgsz_base = int(df_base.loc[df_base["frame_id"] == fid, "imgsz"].iloc[0])
        scale = imgsz_cur / imgsz_base
        base_arr_scaled = base_arr * scale

        matched = _greedy_match(cur_arr, base_arr_scaled)
        displacements.extend(matched)

    return float(np.mean(displacements)) if displacements else float("nan")


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_footpoints(df: pd.DataFrame, frame_id: int) -> list[tuple[float, float]]:
    """Decoded footpoint list for one frame."""
    rows = df[df["frame_id"] == frame_id]
    if rows.empty:
        return []
    fps_json = rows.iloc[0]["footpoints"]
    return [(cx, y2) for cx, y2 in json.loads(fps_json)]


def _greedy_match(cur: np.ndarray, base: np.ndarray) -> list[float]:
    """Greedy nearest-neighbour pairing between two sets of 2D points.

    For each point in cur (the smaller or equal set), find the nearest
    unmatched point in base and record the Euclidean distance.  Points
    left unmatched due to set size difference are ignored.
    """
    if len(cur) == 0 or len(base) == 0:
        return []

    # Pairwise Euclidean distances: shape (N_cur, N_base)
    diff  = cur[:, None, :] - base[None, :, :]   # (N_cur, N_base, 2)
    dists = np.sqrt((diff ** 2).sum(axis=2))       # (N_cur, N_base)

    used_base = set()
    results   = []

    # Sort cur indices by their closest base distance (greedy nearest-first)
    for i in np.argsort(dists.min(axis=1)):
        row         = dists[i].copy()
        row[list(used_base)] = np.inf
        j = int(np.argmin(row))
        if row[j] < np.inf:
            results.append(float(row[j]))
            used_base.add(j)

    return results
