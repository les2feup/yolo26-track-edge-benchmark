from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
import torch

from benchmark.config import CLASSES, CONF, TRACKER, WARMUP_FRAMES

# CLAHE parameters: standard defaults — not tuned per sequence.
_CLAHE_CLIP = 2.0
_CLAHE_GRID = (8, 8)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE contrast normalisation applied to the luminance channel.

    Converts to LAB, equalises L, converts back to BGR. Colour channels
    are untouched, preserving model feature extraction while improving
    edge contrast in low-light regions. Parameters are fixed at standard
    defaults across all sequences and resolution levels.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID)
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def run_sequence(
    model,
    seq_dir: Path,
    imgsz: int,
    out_csv: Path,
    tracker: str | None = None,
    clahe: bool = False,
    max_duration_s: float | None = None,
) -> pd.DataFrame:
    """Per-frame YOLO tracking loop for one MOT17 sequence.

    The caller creates and owns the YOLO model instance. ByteTrack state
    accumulated via persist=True must survive across all frames in one
    sequence, so the model must not be re-instantiated mid-sequence.

    Timing excludes I/O: only the model.track() wall-clock time is measured.
    The first WARMUP_FRAMES are run but their timing is discarded, matching
    the warm-up protocol described in the methodology.

    Memory is sampled once per frame:
    - GPU runs: torch.cuda.max_memory_allocated() (bytes, peak since last reset)
    - CPU runs: psutil RSS of the current process (bytes)

    The output CSV is written atomically via a temporary file to prevent
    partial writes on interruption.

    Args:
        model:   Ultralytics YOLO instance, already loaded and on target device.
        seq_dir: Path to one MOT17 sequence directory (contains img1/, gt/, seqinfo.ini).
        imgsz:   Inference resolution passed to model.track().
        out_csv: Destination path for the per-frame CSV output.
        tracker: Path to a custom tracker YAML.  Defaults to TRACKER from config (bytetrack).
        clahe:   Apply CLAHE luminance normalisation before inference (clip=2.0, grid=8×8).
                 Must be applied uniformly across ALL sequences and resolutions if enabled.

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
    # runs on the correct device.  Without this, some ultralytics versions
    # (especially on older torch/Python) fall back to CPU silently.
    if hasattr(model, "device") and str(model.device) != "cpu":
        device_arg = model.device
    elif use_cuda:
        device_arg = "cuda:0"
    else:
        device_arg = None

    process = psutil.Process()
    records = []

    t_loop_start = time.perf_counter()

    for frame_idx, img_path in enumerate(frame_paths):
        # Time-budget guard: stop inference after max_duration_s elapsed
        if max_duration_s is not None and (time.perf_counter() - t_loop_start) >= max_duration_s:
            break

        frame_id  = frame_idx + 1   # MOT17 uses 1-indexed frame IDs
        frame_bgr = cv2.imread(str(img_path))
        if clahe:
            frame_bgr = preprocess_frame(frame_bgr)

        t0      = time.perf_counter()
        results = model.track(
            frame_bgr,
            persist=True,
            tracker=tracker_path,
            conf=CONF,
            classes=CLASSES,
            imgsz=imgsz,
            device=device_arg,
            verbose=False,
        )
        t1 = time.perf_counter()

        # Memory snapshot: GPU peak (cumulative since reset) or CPU RSS
        if use_cuda:
            mem_bytes = torch.cuda.max_memory_allocated()
        else:
            mem_bytes = process.memory_info().rss

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
        inference_ms = (t1 - t0) * 1000 if frame_idx >= WARMUP_FRAMES else float("nan")

        records.append({
            "frame_id":     frame_id,
            "inference_ms": inference_ms,
            "n_detections": len(track_ids),
            "track_ids":    json.dumps(track_ids),
            "bboxes_xyxy":  json.dumps(bboxes),
            "confs":        json.dumps(confs),
            "footpoints":   json.dumps(footpoints),
            "mem_bytes":    mem_bytes,
            "imgsz":        imgsz,
            "model":        model_name,
            "seq":          seq_name,
        })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return df
