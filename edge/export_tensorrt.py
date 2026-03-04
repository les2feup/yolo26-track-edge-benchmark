"""
TensorRT .engine export for Jetson Nano — two-stage pipeline.

Stage 1: ultralytics YOLO .pt → ONNX  (Python, runs in-process)
Stage 2: trtexec CLI .onnx → .engine  (C++ binary, bypasses Python tensorrt bindings)

JetPack 4.6.x ships TensorRT 8.2 with Python 3.6 bindings only. Since we run
Python 3.8, the tensorrt Python module cannot be imported. The trtexec binary
(/usr/src/tensorrt/bin/trtexec) links directly to libnvinfer.so and has no
Python version dependency.

TensorRT engines have a fixed input shape baked in at compile time. Each
model×resolution pair produces a separate .engine file. The naming convention
matches the Hailo export pattern:
    yolo26n.engine       ← 640px (canonical, no suffix)
    yolo26n_576.engine   ← 576px

This script MUST run on the Jetson Nano itself — TensorRT engines are compiled
for the device's specific GPU architecture (Maxwell, sm_53) and TRT version.

Usage:
    source .venv/bin/activate
    cd /media/les2/SD-128-J10/yolo26-track-edge-benchmark

    python edge/export_tensorrt.py                  # all variants × all resolutions
    python edge/export_tensorrt.py --model yolo26n  # single variant, all resolutions
    python edge/export_tensorrt.py --imgsz 576      # all variants, 576px only

Outputs:
    models/yolo26n.engine, models/yolo26n_576.engine, ...
    edge/export_results_jetson.csv — success/failure summary
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT       = Path(__file__).parents[1]
_MODELS_DIR = _ROOT / "models"

# Default variants and resolutions matching the jetson_nano.yaml profile
_DEFAULT_VARIANTS   = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]
_DEFAULT_RESOLUTIONS = [640, 576]

# trtexec binary shipped with JetPack
_TRTEXEC_PATHS = [
    "/usr/src/tensorrt/bin/trtexec",
    shutil.which("trtexec") or "",
]


def _engine_stem(variant: str, imgsz: int) -> str:
    """Resolution-suffixed stem: yolo26n @ 640 → yolo26n, yolo26n @ 576 → yolo26n_576."""
    return variant if imgsz == 640 else f"{variant}_{imgsz}"


def _find_trtexec() -> str:
    """Locate the trtexec binary on the system."""
    for p in _TRTEXEC_PATHS:
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError(
        "trtexec not found. Expected at /usr/src/tensorrt/bin/trtexec "
        "(shipped with JetPack). Is TensorRT installed?"
    )


def _export_onnx(pt_path: Path, imgsz: int) -> Path:
    """Stage 1: Export .pt → .onnx via ultralytics at the given resolution."""
    stem      = _engine_stem(pt_path.stem, imgsz)
    onnx_path = _MODELS_DIR / f"{stem}.onnx"
    if onnx_path.exists():
        print(f"  [onnx] reusing existing {onnx_path.name}")
        return onnx_path

    from ultralytics import YOLO

    print(f"  [onnx] {pt_path.name} → {onnx_path.name} (imgsz={imgsz})")
    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=14,
        dynamic=False,
        simplify=True,
    )
    # ultralytics places the .onnx next to the .pt with its original stem
    exported = pt_path.with_suffix(".onnx")
    if exported != onnx_path:
        exported.rename(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX export did not produce {onnx_path}")
    return onnx_path


def _compile_engine(
    onnx_path: Path, engine_path: Path, half: bool, workspace_mb: int,
) -> None:
    """Stage 2: Compile .onnx → .engine via trtexec CLI."""
    trtexec = _find_trtexec()

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if half:
        cmd.append("--fp16")

    print(f"  [trtexec] {onnx_path.name} → {engine_path.name} (fp16={half})")
    print(f"  [trtexec] cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        output = result.stdout + result.stderr
        error_lines = [
            ln for ln in output.splitlines()
            if any(tag in ln for tag in ["[E]", "ERROR", "FAILED", "error"])
        ]
        detail = "\n".join(error_lines[-10:]) if error_lines else output[-500:]
        raise RuntimeError(f"trtexec failed (rc={result.returncode}):\n{detail}")


def export_one(variant: str, imgsz: int, half: bool, workspace_mb: int, force: bool = False) -> dict:
    """Export a single .pt model at a given resolution to a TensorRT .engine file.

    Returns a result record dict with keys: model, imgsz, status, engine_mb, notes.
    """
    pt_path     = _MODELS_DIR / f"{variant}.pt"
    stem        = _engine_stem(variant, imgsz)
    engine_path = _MODELS_DIR / f"{stem}.engine"
    record      = {
        "model": stem, "imgsz": imgsz, "half": half,
        "status": "unknown", "engine_mb": "", "elapsed_s": "", "notes": "",
    }

    if not pt_path.exists():
        record["status"] = "skipped"
        record["notes"]  = f".pt not found: {pt_path}"
        return record

    if engine_path.exists() and not force:
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"  [skip] {engine_path.name} already exists ({size_mb:.1f} MB)")
        record["status"]    = "ok"
        record["engine_mb"] = f"{size_mb:.1f}"
        record["notes"]     = "reused existing engine"
        return record

    try:
        t0 = time.time()

        # Stage 1: .pt → .onnx (resolution-specific)
        onnx_path = _export_onnx(pt_path, imgsz)

        # Stage 2: .onnx → .engine
        _compile_engine(onnx_path, engine_path, half, workspace_mb)

        elapsed = time.time() - t0

        if not engine_path.exists():
            raise FileNotFoundError(f"trtexec did not produce {engine_path}")

        size_mb = engine_path.stat().st_size / (1024 * 1024)
        record["status"]    = "ok"
        record["engine_mb"] = f"{size_mb:.1f}"
        record["elapsed_s"] = f"{elapsed:.0f}"
        print(f"  OK: {engine_path.name} ({size_mb:.1f} MB, {elapsed:.0f}s)")

    except MemoryError as exc:
        record["status"] = "oom"
        record["notes"]  = str(exc)
        print(f"  OOM: {exc}")
    except Exception as exc:
        record["status"] = "failed"
        record["notes"]  = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {exc}")

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt models to TensorRT .engine on Jetson Nano"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single variant stem (e.g. yolo26n). Default: all variants in profile.",
    )
    parser.add_argument(
        "--imgsz", type=int, nargs="+", default=None,
        help="Input resolution(s) in pixels (default: 640 576). Multiple values supported.",
    )
    parser.add_argument(
        "--no-half", action="store_true",
        help="Disable FP16 (use FP32). FP16 is default and recommended on Maxwell.",
    )
    parser.add_argument(
        "--workspace", type=int, default=1024,
        help="TensorRT workspace size in MB (default: 1024)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-export even if .engine file already exists.",
    )
    args = parser.parse_args()

    # Verify trtexec is available before starting
    trtexec = _find_trtexec()
    print(f"Using trtexec: {trtexec}")

    variants    = [args.model] if args.model else _DEFAULT_VARIANTS
    resolutions = args.imgsz if args.imgsz else _DEFAULT_RESOLUTIONS
    half        = not args.no_half
    results     = []

    for variant in variants:
        for imgsz in resolutions:
            stem = _engine_stem(variant, imgsz)
            print(f"\n{'='*60}")
            print(f"Exporting {stem} @ {imgsz}px (half={half})")
            print(f"{'='*60}")
            rec = export_one(variant, imgsz, half, args.workspace, args.force)
            results.append(rec)

    # Append results to CSV log
    summary_path = _ROOT / "edge" / "export_results_jetson.csv"
    file_exists  = summary_path.exists()
    fieldnames   = ["model", "imgsz", "half", "status", "engine_mb", "elapsed_s", "notes"]
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nExport summary appended to {summary_path}")

    ok     = [r["model"] for r in results if r["status"] == "ok"]
    failed = [r["model"] for r in results if r["status"] not in ("ok", "skipped")]
    print(f"  OK: {ok}")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
