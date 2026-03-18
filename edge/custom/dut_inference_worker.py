#!/usr/bin/env python3
"""
DUT-side inference worker for power profiling.

Runs on the Arduino Uno Q (or any edge device). Loads a single model,
applies the standard thread-pinning from device_profile, and runs
inference for a timed window. Designed to be called by the host-side
remote_power_profile.py via SSH.

Usage
-----
    python edge/custom/dut_inference_worker.py \
        --profile edge/profiles/arduino_uno_q.yaml \
        --model yolo26n_640_ncnn_model \
        --imgsz 640 \
        --duration 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── libgomp preload for ultralytics on aarch64 ──────────────────────────
_LIBGOMP = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"
if os.path.exists(_LIBGOMP):
    os.environ.setdefault("LD_PRELOAD", _LIBGOMP)

# Project root on sys.path for editable-install-free usage
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from benchmark.config import DATA_ROOT, SEQ_SUFFIX
from benchmark.device_profile import (
    load_profile,
    apply_thread_pinning,
    try_load_model,
    resolve_model_path,
)
from benchmark.runner import run_sequence


# Workload sequence (shortest MOT17 static-camera, 525 frames)
POWER_SEQ = "MOT17-09"


def main():
    ap = argparse.ArgumentParser(description="DUT inference worker for power profiling")
    ap.add_argument("--profile", required=True, help="Path to device profile YAML")
    ap.add_argument("--model", required=True, help="Model variant name (e.g. yolo26n_640_ncnn_model)")
    ap.add_argument("--imgsz", type=int, required=True, help="Input resolution")
    ap.add_argument("--duration", type=float, required=True, help="Total inference duration (seconds)")

    args = ap.parse_args()

    # ── Load profile and apply thread pinning (matches notebook setup) ──
    profile = load_profile(args.profile)
    apply_thread_pinning(profile)

    print(f"Device  : {profile.device_label}", flush=True)
    print(f"Backend : {profile.backend}", flush=True)
    print(f"Threads : OMP={os.environ.get('OMP_NUM_THREADS', '?')}", flush=True)

    # ── Load model ──────────────────────────────────────────────────────
    model, err = try_load_model(args.model, profile.torch_device)
    if err:
        print(f"FATAL: {err}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"Model   : {args.model} (loaded)", flush=True)

    # ── Run inference loop ──────────────────────────────────────────────
    seq_dir = DATA_ROOT / f"{POWER_SEQ}-{SEQ_SUFFIX}"
    out_csv = Path(f"/tmp/power_scratch_{args.model}_{args.imgsz}.csv")

    print(f"Sequence: {seq_dir.name}", flush=True)
    print(f"Duration: {args.duration:.0f}s", flush=True)
    print(f"Starting inference ...", flush=True)

    t0 = time.monotonic()
    run_sequence(
        model, seq_dir,
        imgsz=args.imgsz,
        out_csv=out_csv,
        max_duration_s=args.duration,
    )
    elapsed = time.monotonic() - t0

    print(f"Inference complete ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
