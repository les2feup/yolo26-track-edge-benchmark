"""
Hailo-8L benchmark loop.

Mirrors runner.run_sequence() in output schema and timing protocol so that
downstream metrics and notebook cells work unchanged.

Inference:  HailoInfer (hailo_platform async API, no ultralytics)
Tracking:   supervision.ByteTrack (pure Python, no YOLO dependency)
Detection:  hailo_postprocess.decode_detections (DFL + NMS on CPU)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil

from benchmark.config import CONF, CLASSES, WARMUP_FRAMES
from benchmark.hailo_infer import HailoInfer
from benchmark.hailo_postprocess import decode_detections
from benchmark.runner import preprocess_frame   # CLAHE shared with pt runner

try:
    import supervision as sv
except ImportError as exc:
    raise ImportError(
        "supervision is required for Hailo ByteTrack: pip install supervision"
    ) from exc


def run_sequence_hailo(
    hef_path: str | Path,
    seq_dir: Path,
    imgsz: int,
    out_csv: Path,
    clahe: bool = False,
) -> pd.DataFrame:
    """
    Per-frame Hailo-8L tracking loop for one MOT17 sequence.

    Timing protocol matches runner.run_sequence():
    - First WARMUP_FRAMES are run but their inference_ms is recorded as NaN.
    - GPU memory measurement is not applicable; CPU RSS is used instead.

    The output CSV schema is identical to runner.run_sequence() so that
    compute_mot_metrics() and all notebook aggregation cells work unchanged.

    Args:
        hef_path: Path to the compiled HEF model file.
        seq_dir:  MOT17 sequence directory (contains img1/, gt/, seqinfo.ini).
        imgsz:    Inference resolution (must match HEF input shape, usually 640).
        out_csv:  Destination path for per-frame CSV.
        clahe:    Apply CLAHE luminance normalisation before inference.

    Returns:
        DataFrame with one row per frame.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    seq_name    = seq_dir.name.rsplit("-", 1)[0]   # "MOT17-09-SDP" → "MOT17-09"
    model_name  = Path(hef_path).name

    # Original image dimensions for scaling boxes back from model input space.
    # HailoInfer resizes frames to imgsz×imgsz internally; decoded boxes are in
    # that space. GT coordinates are in original image pixels, so we must scale.
    _probe = cv2.imread(str(frame_paths[0]))
    orig_h, orig_w = _probe.shape[:2]
    scale_x = orig_w / imgsz
    scale_y = orig_h / imgsz

    process = psutil.Process()
    tracker = sv.ByteTrack()
    records = []

    with HailoInfer(hef_path) as model:
        for frame_idx, img_path in enumerate(frame_paths):
            frame_id  = frame_idx + 1
            frame_bgr = cv2.imread(str(img_path))
            if clahe:
                frame_bgr = preprocess_frame(frame_bgr)

            t0 = time.perf_counter()
            raw = model.infer(frame_bgr)
            t1 = time.perf_counter()

            boxes_xyxy, confs, cls_ids = decode_detections(
                raw,
                imgsz=imgsz,
                conf_thres=CONF,
                target_classes=CLASSES,
            )

            # Feed detections into supervision ByteTrack (still in model space)
            if boxes_xyxy:
                dets = sv.Detections(
                    xyxy=np.array(boxes_xyxy, dtype=np.float32),
                    confidence=np.array(confs, dtype=np.float32),
                    class_id=np.array(cls_ids, dtype=np.int32),
                )
            else:
                dets = sv.Detections.empty()

            tracked = tracker.update_with_detections(dets)

            track_ids   = tracked.tracker_id.tolist() if tracked.tracker_id is not None else []
            track_confs = tracked.confidence.tolist()  if tracked.confidence is not None else []

            # Scale tracked boxes from model input space → original image pixels
            if len(tracked) > 0:
                scaled = tracked.xyxy.copy()
                scaled[:, [0, 2]] *= scale_x
                scaled[:, [1, 3]] *= scale_y
                bboxes = scaled.tolist()
            else:
                bboxes = []

            footpoints = [((x1 + x2) / 2, y2) for x1, y1, x2, y2 in bboxes]

            inference_ms = (t1 - t0) * 1000 if frame_idx >= WARMUP_FRAMES else float("nan")
            mem_bytes    = process.memory_info().rss

            records.append({
                "frame_id":     frame_id,
                "inference_ms": inference_ms,
                "n_detections": len(track_ids),
                "track_ids":    json.dumps(track_ids),
                "bboxes_xyxy":  json.dumps(bboxes),
                "confs":        json.dumps(track_confs),
                "footpoints":   json.dumps(footpoints),
                "mem_bytes":    mem_bytes,
                "imgsz":        imgsz,
                "model":        model_name,
                "seq":          seq_name,
            })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return df
