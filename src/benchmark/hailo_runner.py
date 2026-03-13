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
    max_duration_s: float | None = None,
    baseline_ram: int | None = None,
) -> pd.DataFrame:
    """
    Per-frame Hailo-8L tracking loop for one MOT17 sequence.

    Timing protocol matches runner.run_sequence():
    - First WARMUP_FRAMES are run but their inference_ms is recorded as NaN.
    - Host RSS delta around HailoInfer construction is used as memory footprint
      (NPU SRAM usage is opaque; this captures driver + I/O buffer allocations).

    The output CSV schema is identical to runner.run_sequence() so that
    compute_mot_metrics() and all notebook aggregation cells work unchanged.

    Args:
        hef_path: Path to the compiled HEF model file.
        seq_dir:  MOT17 sequence directory (contains img1/, gt/, seqinfo.ini).
        imgsz:    Inference resolution (must match HEF input shape, usually 640).
        out_csv:  Destination path for per-frame CSV.

    Returns:
        DataFrame with one row per frame.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    seq_name    = seq_dir.name.rsplit("-", 1)[0]   # "MOT17-09-SDP" → "MOT17-09"
    model_name  = Path(hef_path).name

    tracker = sv.ByteTrack()
    records = []

    _process = psutil.Process()
    _rss_before = baseline_ram if baseline_ram is not None else _process.memory_info().rss
    with HailoInfer(hef_path) as model:
        mem_total_bytes = _process.memory_info().rss
        mem_delta_bytes = max(mem_total_bytes - _rss_before, 0)
        # HEF input resolution is fixed at compile time — always use this for
        # coordinate decoding and scaling, regardless of the requested imgsz.
        hef_h, hef_w, _ = model.input_shape
        hef_imgsz = hef_w   # assumed square; hef_h == hef_w for YOLO26

        # Original image dimensions for scaling boxes from HEF input space → pixels.
        _probe = cv2.imread(str(frame_paths[0]))
        orig_h, orig_w = _probe.shape[:2]
        scale_x = orig_w / hef_imgsz
        scale_y = orig_h / hef_imgsz

        t_loop_start = time.perf_counter()
        frame_idx = 0
        budget_expired = False

        # Loop over the sequence continuously until max_duration_s expires.
        # Single-pass when max_duration_s is None (standard benchmark mode).
        while not budget_expired:
            for img_path in frame_paths:
                if max_duration_s is not None and (time.perf_counter() - t_loop_start) >= max_duration_s:
                    budget_expired = True
                    break

                frame_id  = frame_idx + 1
                frame_idx += 1
                frame_bgr = cv2.imread(str(img_path))

                t0 = time.perf_counter()
                raw = model.infer(frame_bgr)
                t1 = time.perf_counter()

                boxes_xyxy, confs, cls_ids = decode_detections(
                    raw,
                    imgsz=hef_imgsz,
                    conf_thres=CONF,
                    target_classes=CLASSES,
                )

                # Feed detections into supervision ByteTrack (still in model space)
                if len(boxes_xyxy):
                    dets = sv.Detections(
                        xyxy=boxes_xyxy,
                        confidence=confs,
                        class_id=cls_ids,
                    )
                else:
                    dets = sv.Detections.empty()

                tracked = tracker.update_with_detections(dets)
                t2 = time.perf_counter()

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

                inference_ms   = (t1 - t0) * 1000 if frame_idx > WARMUP_FRAMES else float("nan")
                postprocess_ms = (t2 - t1) * 1000 if frame_idx > WARMUP_FRAMES else float("nan")

                records.append({
                    "frame_id":        frame_id,
                    "inference_ms":    inference_ms,
                    "postprocess_ms":  postprocess_ms,
                    "n_detections":    len(track_ids),
                    "track_ids":       json.dumps(track_ids),
                    "bboxes_xyxy":     json.dumps(bboxes),
                    "confs":           json.dumps(track_confs),
                    "footpoints":      json.dumps(footpoints),
                    "mem_total_bytes": mem_total_bytes,
                    "mem_delta_bytes": mem_delta_bytes,
                    "mem_bytes":       mem_total_bytes,   # backward-compat alias
                    "imgsz":           hef_imgsz,
                    "model":           model_name,
                    "seq":             seq_name,
                })

            # Single-pass mode: no time budget → exit after one pass
            if max_duration_s is None:
                break

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return df
