"""
Isolated evaluation worker for per-run memory measurement.

Each call runs in a spawned subprocess so that Python + framework RSS is
measured from a clean slate. This gives mem_total_bytes (platform view: how
much RAM the full stack consumes) and mem_delta_bytes (model-only footprint:
total minus pre-import baseline). Without isolation, accumulated imports from
previous runs contaminate the baseline and underreport model cost.
"""

from __future__ import annotations

import os
from pathlib import Path

import psutil as _psutil


def run_evaluation_isolated(
    backend: str,
    model_path: str,
    imgsz: int,
    seq_name: str,
    seq_dir: Path,
    out_csv: Path,
    torch_device: str,
) -> None:
    """Entry point for the spawned subprocess.

    Measures baseline RSS before any heavy imports, loads the model, then
    passes both pre- and post-load RSS values into the runner so that the
    CSV records platform-accurate memory figures.
    """
    process = _psutil.Process(os.getpid())
    baseline_ram = process.memory_info().rss

    try:
        from benchmark.device_profile import (
            try_load_model,
            resolve_model_path,
            _BAKED_RESOLUTION_BACKENDS,
        )

        if backend == "hailo":
            from benchmark.hailo_runner import run_sequence_hailo

            resolved = resolve_model_path(model_path)
            # Hailo runner measures its own RSS delta internally around HailoInfer init.
            # baseline_ram is not forwarded because the Hailo driver load is part of the
            # model cost (it only starts when HailoInfer opens the device).
            run_sequence_hailo(resolved, seq_dir, imgsz=imgsz, out_csv=out_csv)

        elif backend == "tensorrt_hq":
            from benchmark.trt_runner import run_sequence_trt

            resolved = resolve_model_path(model_path)
            run_sequence_trt(
                resolved, seq_dir, imgsz=imgsz, out_csv=out_csv,
                baseline_ram=baseline_ram,
            )

        else:
            # Dynamic backends (cpu, cuda, tensorrt via ultralytics) and other
            # baked-resolution backends all use the standard runner.
            from benchmark.runner import run_sequence

            model, err = try_load_model(model_path, torch_device)
            if err:
                print(f"  SKIP — could not load: {err}")
                return

            mem_total_bytes = process.memory_info().rss
            mem_delta_bytes = max(mem_total_bytes - baseline_ram, 0)

            run_sequence(
                model, seq_dir, imgsz=imgsz, out_csv=out_csv,
                mem_total_bytes=mem_total_bytes,
                mem_delta_bytes=mem_delta_bytes,
            )

    except Exception as exc:
        print(f"  FAILED in subprocess: {exc}")
        raise
