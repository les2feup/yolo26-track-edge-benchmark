from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import pandas as pd
import psutil
import torch

from benchmark.config import CLASSES, CONF, TRACKER, WARMUP_FRAMES

def run_sequence(
    model,
    seq_dir: Path,
    imgsz: int,
    out_csv: Path,
    tracker: str | None = None,
    max_duration_s: float | None = None,
    mem_total_bytes: int | None = None,
    mem_delta_bytes: int | None = None,
) -> pd.DataFrame:
    """Per-frame YOLO tracking loop for one MOT17 sequence.

    The caller creates and owns the YOLO model instance. ByteTrack state
    accumulated via persist=True must survive across all frames in one
    sequence, so the model must not be re-instantiated mid-sequence.

    Timing excludes I/O: only the model.track() wall-clock time is measured.
    The first WARMUP_FRAMES are run but their timing is discarded, matching
    the warm-up protocol described in the methodology.

    Memory columns written to the CSV:
    - mem_total_bytes: absolute process RSS at model-loaded steady state (platform view).
    - mem_delta_bytes: isolated model footprint = total − baseline before any framework imports.
    - mem_bytes:       alias for mem_total_bytes, kept for backward compatibility with
                       downstream aggregation cells that read df["mem_bytes"].max().

    Both values are constants supplied by the multiprocessing worker (worker.py) which
    measures them in isolation before and after framework/model load. Passing None falls
    back to a live RSS snapshot, which is less accurate but maintains single-process compat.

    Args:
        model:           Ultralytics YOLO instance, already loaded and on target device.
        seq_dir:         Path to one MOT17 sequence directory (img1/, gt/, seqinfo.ini).
        imgsz:           Inference resolution passed to model.track().
        out_csv:         Destination path for the per-frame CSV output.
        tracker:         Path to a custom tracker YAML. Defaults to TRACKER from config.
        mem_total_bytes: Absolute peak process RSS supplied by the worker.
        mem_delta_bytes: Isolated model footprint (total − pre-import baseline).

    Returns:
        DataFrame with one row per frame, same schema as the output CSV.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_paths  = sorted((seq_dir / "img1").glob("*.jpg"))
    seq_name     = seq_dir.name.rsplit("-", 1)[0]   # "MOT17-09-SDP" → "MOT17-09"
    model_name   = Path(model.model_name).name
    tracker_path = tracker if tracker is not None else TRACKER

    # TensorRT/engine models don't expose PyTorch parameters — model.model
    # is the engine path string, not an nn.Module.  Fall back to model.device.
    try:
        _on_cuda = next(model.model.parameters()).is_cuda
    except (StopIteration, AttributeError):
        _on_cuda = hasattr(model, "device") and "cuda" in str(model.device)
    use_cuda = torch.cuda.is_available() and _on_cuda
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Explicit device for model.track() — ensures the predictor's AutoBackend
    # runs on the correct device.  Passing device=None explicitly (rather than
    # omitting the kwarg) triggers silent inference corruption on Qualcomm
    # torch 2.0.0.post4, so we only set device_arg when we have a real value.
    if hasattr(model, "device") and str(model.device) != "cpu":
        device_arg = model.device
    elif use_cuda:
        device_arg = "cuda:0"
    else:
        device_arg = None

    process = psutil.Process()
    records = []

    # Fallback: if the worker didn't supply pre-measured values, snapshot live RSS.
    if mem_total_bytes is None:
        mem_total_bytes = process.memory_info().rss
    if mem_delta_bytes is None:
        mem_delta_bytes = 0

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

            frame_id  = frame_idx + 1   # MOT17 uses 1-indexed frame IDs
            frame_idx += 1
            frame_bgr = cv2.imread(str(img_path))

            t0      = time.perf_counter()
            track_kwargs = dict(
                persist=True,
                tracker=tracker_path,
                conf=CONF,
                classes=CLASSES,
                imgsz=imgsz,
                verbose=False,
            )
            if device_arg is not None:
                track_kwargs["device"] = device_arg
            results = model.track(frame_bgr, **track_kwargs)
            t1 = time.perf_counter()

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                track_ids  = boxes.id.int().cpu().tolist()
                bboxes     = boxes.xyxy.cpu().tolist()
                confs      = boxes.conf.cpu().tolist()
                footpoints = [((x1 + x2) / 2, y2) for x1, y1, x2, y2 in bboxes]
            else:
                track_ids = bboxes = confs = footpoints = []

            # Skip warm-up frames for timing records but still run inference
            # to allow ByteTrack to build its internal track state.
            inference_ms = (t1 - t0) * 1000 if frame_idx > WARMUP_FRAMES else float("nan")

            records.append({
                "frame_id":        frame_id,
                "inference_ms":    inference_ms,
                "n_detections":    len(track_ids),
                "track_ids":       json.dumps(track_ids),
                "bboxes_xyxy":     json.dumps(bboxes),
                "confs":           json.dumps(confs),
                "footpoints":      json.dumps(footpoints),
                "mem_total_bytes": mem_total_bytes,
                "mem_delta_bytes": mem_delta_bytes,
                "mem_bytes":       mem_total_bytes,   # backward-compat alias
                "imgsz":           imgsz,
                "model":           model_name,
                "seq":             seq_name,
            })

        # Single-pass mode: no time budget → exit after one pass
        if max_duration_s is None:
            break

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return df
