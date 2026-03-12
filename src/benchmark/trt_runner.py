"""
TensorRT HQ benchmark loop for Jetson Nano.

Mirrors hailo_runner.run_sequence_hailo() in output schema and timing protocol
so that downstream metrics, degradation analysis, and notebook cells work unchanged.

Inference:  TrtInfer (tensorrt + pycuda, no ultralytics)
Tracking:   supervision.ByteTrack (pure Python, no YOLO dependency)
Detection:  trt_postprocess.decode_detections (DFL decode + NMS on CPU)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import os

import cv2
import numpy as np
import pandas as pd
import psutil
import pycuda.driver as _cuda_drv

from benchmark.config import CONF, CLASSES, WARMUP_FRAMES
from benchmark.trt_infer import TrtInfer
from benchmark.trt_postprocess import decode_detections

try:
    import supervision as sv
except ImportError as exc:
    raise ImportError(
        "supervision is required for TRT HQ ByteTrack: pip install supervision"
    ) from exc


def run_sequence_trt(
    engine_path: str | Path,
    seq_dir: Path,
    imgsz: int,
    out_csv: Path,
    max_duration_s: float | None = None,
    baseline_ram: int | None = None,
) -> pd.DataFrame:
    """
    Per-frame TensorRT HQ tracking loop for one MOT17 sequence.

    Timing protocol matches runner.run_sequence():
    - First WARMUP_FRAMES are run but their inference_ms is recorded as NaN.
    - mem_total_bytes is measured after engine load (platform RSS view).
    - mem_delta_bytes = mem_total_bytes − baseline_ram (model-only footprint).

    baseline_ram should be the process RSS before any framework imports, supplied
    by worker.py. If omitted, the current RSS is used as baseline (less accurate).

    The output CSV schema is identical to runner.run_sequence() so that
    compute_mot_metrics() and all notebook aggregation cells work unchanged.

    Args:
        engine_path:  Path to the HQ .engine file (6 raw Conv outputs).
        seq_dir:      MOT17 sequence directory (contains img1/, gt/, seqinfo.ini).
        imgsz:        Inference resolution (must match engine's compiled input shape).
        out_csv:      Destination path for per-frame CSV.
        baseline_ram: Process RSS (bytes) before framework imports, from worker.py.

    Returns:
        DataFrame with one row per frame.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    seq_name    = seq_dir.name.rsplit("-", 1)[0]   # "MOT17-09-SDP" → "MOT17-09"
    model_name  = Path(engine_path).name

    tracker = sv.ByteTrack()
    records = []

    process = psutil.Process(os.getpid())
    if baseline_ram is None:
        baseline_ram = process.memory_info().rss

    with TrtInfer(engine_path) as model:
        # Platform RSS after engine load: captures driver + weights + I/O buffers.
        mem_total_bytes = process.memory_info().rss
        mem_delta_bytes = max(mem_total_bytes - baseline_ram, 0)

        # Engine input resolution is baked at compile time
        engine_imgsz = model.input_w   # assumed square

        # Original image dimensions for scaling boxes
        _probe = cv2.imread(str(frame_paths[0]))
        orig_h, orig_w = _probe.shape[:2]
        scale_x = orig_w / engine_imgsz
        scale_y = orig_h / engine_imgsz
        del _probe

        t_loop_start = time.perf_counter()
        frame_idx = 0
        budget_expired = False

        while not budget_expired:
            for img_path in frame_paths:
                if max_duration_s is not None and (time.perf_counter() - t_loop_start) >= max_duration_s:
                    budget_expired = True
                    break

                frame_id  = frame_idx + 1
                frame_idx += 1
                frame_bgr = cv2.imread(str(img_path))

                # Inference: TRT engine produces 6 raw Conv outputs (NCHW)
                t0  = time.perf_counter()
                raw = model.infer(frame_bgr)
                t1  = time.perf_counter()

                # CPU post-processing: DFL decode + NMS
                boxes_xyxy, confs, cls_ids = decode_detections(
                    raw,
                    imgsz=engine_imgsz,
                    conf_thres=CONF,
                    target_classes=CLASSES,
                )

                # Feed detections into supervision ByteTrack
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

                # Scale tracked boxes from engine input space → original image pixels
                if len(tracked) > 0:
                    scaled = tracked.xyxy.copy()
                    scaled[:, [0, 2]] *= scale_x
                    scaled[:, [1, 3]] *= scale_y
                    bboxes = scaled.tolist()
                else:
                    bboxes = []

                footpoints = [((x1 + x2) / 2, y2) for x1, y1, x2, y2 in bboxes]

                inference_ms = (t1 - t0) * 1000 if frame_idx > WARMUP_FRAMES else float("nan")

                records.append({
                    "frame_id":        frame_id,
                    "inference_ms":    inference_ms,
                    "n_detections":    len(track_ids),
                    "track_ids":       json.dumps(track_ids),
                    "bboxes_xyxy":     json.dumps(bboxes),
                    "confs":           json.dumps(track_confs),
                    "footpoints":      json.dumps(footpoints),
                    "mem_total_bytes": mem_total_bytes,
                    "mem_delta_bytes": mem_delta_bytes,
                    "mem_bytes":       mem_total_bytes,   # backward-compat alias
                    "imgsz":           engine_imgsz,
                    "model":           model_name,
                    "seq":             seq_name,
                })

            # Single-pass mode
            if max_duration_s is None:
                break

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return df
