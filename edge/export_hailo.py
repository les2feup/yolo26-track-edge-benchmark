"""
Hailo .hef export pipeline: .pt → .onnx → .har → .hef

REQUIREMENTS — This script must run in a SEPARATE environment from the project venv.
It requires Ubuntu 22.04 x86_64, Python 3.10, and Hailo DFC v3.32.

Quick setup:
    sudo apt install python3.10 python3.10-venv python3.10-dev graphviz
    python3.10 -m venv ~/hailodfc && source ~/hailodfc/bin/activate
    pip install ultralytics               # for ONNX export
    pip install pyyaml
    # Download from https://hailo.ai/developer-zone/software-downloads/
    pip install hailo_dataflow_compiler-3.32.0-py3-none-linux_x86_64.whl
    pip install hailort-4.22.0-cp310-cp310-linux_x86_64.whl
    git clone -b v2.16 https://github.com/hailo-ai/hailo_model_zoo.git
    pip install -e hailo_model_zoo/

Usage:
    python edge/export_hailo.py --model yolo26n.pt --calib-dir edge/calib_imgs/ [--hw-arch hailo8l]
    python edge/export_hailo.py --all --calib-dir edge/calib_imgs/   # export all 5 variants

    # Batch export — runs all variants and reports success/failure per model:
    python edge/export_hailo.py --all --calib-dir edge/calib_imgs/ 2>&1 | tee edge/export_log.txt

Outputs:
    models/yolo26n.hef (and .har intermediate)
    edge/export_results.csv — success/failure summary for each variant

Notes:
    - DFC v3.32 / HailoRT 4.22 is the LAST stable version supporting Hailo-8L.
      hailo_model_zoo v5.x dropped Hailo-8/8L support — DO NOT use v5.x.
    - hailo8l is the correct --hw-arch for the Raspberry Pi AI Kit M.2 hat.
    - ONNX opset 11 is required; higher opsets have known DFC parser issues.
    - If yolo26 ONNX graph is not recognised by hailomz, use --fallback-parser
      which invokes the three-step DFC API (parse → optimize → compile) directly.
    - Calibration images: 64–200 unlabelled representative frames are sufficient.
      Collect them from MOT17 img1/ directories: edge/collect_calib.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT       = Path(__file__).parents[1]
_MODELS_DIR = _ROOT / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_ALL_VARIANTS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

# Models likely to exceed Hailo-8L SRAM (~8 MB) — attempted but expected to fail
_EXPECTED_OVERSIZE = {"yolo26l.pt", "yolo26x.pt"}


def export_onnx(pt_path: Path) -> Path:
    """Export PyTorch .pt to ONNX opset 11 with static shapes."""
    onnx_path = pt_path.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"[onnx] reusing existing {onnx_path.name}")
        return onnx_path

    print(f"[onnx] exporting {pt_path.name} → {onnx_path.name}")
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=640,
        opset=11,       # DFC v3.x parser is validated on opset 11
        dynamic=False,  # static shapes required for Hailo compilation
        simplify=False, # Hailo parser handles simplification internally
    )
    # ultralytics exports to the same directory as the .pt file
    exported = pt_path.parent / (pt_path.stem + ".onnx")
    if exported != onnx_path:
        exported.rename(onnx_path)
    return onnx_path


def compile_hef(
    onnx_path: Path,
    calib_dir: Path,
    hw_arch: str,
    fallback_parser: bool,
) -> tuple[Path | None, str]:
    """
    Compile ONNX → HEF using the hailo_sdk_client Python API directly.

    The three-step DFC pipeline (parse → optimize → compile) is driven via
    ClientRunner, bypassing the `hailo` CLI which requires the HailoRT runtime
    library (libhailort.so) — a separate .deb not bundled in the Python wheel.

    The --fallback-parser flag is retained for API compatibility but is now a
    no-op: the Python API path is always used.

    Returns (hef_path, "") on success or (None, error_message) on failure.
    """
    stem     = onnx_path.stem
    hef_path = _MODELS_DIR / f"{stem}.hef"

    if hef_path.exists():
        print(f"[hailo] reusing existing {hef_path.name}")
        return hef_path, ""

    try:
        from hailo_sdk_client import ClientRunner
        from hailo_sdk_client.exposed_definitions import States
        from hailo_sdk_common.paths_manager.paths import SDKPaths
    except ImportError as exc:
        return None, f"hailo_sdk_client import failed: {exc}"

    # Ubuntu/Debian installs to dist-packages, not site-packages.
    # SDKPaths.is_release checks for "site-packages" in the path and returns False
    # on dist-packages, making join_hailo_tools_path() return None and silently
    # breaking compile().  Force is_release=True so the path resolves correctly.
    SDKPaths()._is_release = True

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir  = Path(_tmp)
        har_raw = tmpdir / f"{stem}.har"
        har_opt = tmpdir / f"{stem}_quantized.har"

        # Step 1: parse ONNX → HAR
        # YOLO26's detection head (/model.23) contains ops unsupported by Hailo-8L
        # (GatherElements, TopK, Mod, ReduceMax on batch axis). The DFC recommends
        # cutting at the six one2one_cv2/cv3 Conv outputs — these are the raw
        # box-regression and class-score feature maps before NMS decoding. NMS runs
        # on the host CPU via HailoRT post-processing or a custom wrapper.
        # These node names are consistent across all YOLO26 n/s/m/l/x variants.
        _YOLO26_END_NODES = [
            f"/model.23/one2one_cv2.{i}/one2one_cv2.{i}.2/Conv" for i in range(3)
        ] + [
            f"/model.23/one2one_cv3.{i}/one2one_cv3.{i}.2/Conv" for i in range(3)
        ]
        print(f"[hailo] parse  {onnx_path.name} → {har_raw.name}  (6 cv2/cv3 end nodes)")
        try:
            runner = ClientRunner(hw_arch=hw_arch)
            hn, params = runner.translate_onnx_model(
                str(onnx_path),
                net_name=stem,
                end_node_names=_YOLO26_END_NODES,
            )
            runner.save_har(str(har_raw))
        except Exception as exc:
            return None, f"parse failed: {exc}"

        # Step 2: quantize / optimize using calibration images
        # DFC optimize() with data_type=CalibrationDataType.np_array expects a single
        # numpy.ndarray shaped (N, H, W, C) float32 in range [0, 1].
        print(f"[hailo] optimize {har_raw.name}  calib={calib_dir}")
        try:
            import numpy as np
            import cv2
            from hailo_sdk_client.exposed_definitions import CalibrationDataType

            calib_imgs = sorted(calib_dir.glob("*.jpg"))
            if not calib_imgs:
                return None, f"no calibration images found in {calib_dir}"

            calib_array = np.stack([
                cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (640, 640)),
                             cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                for p in calib_imgs
            ])  # shape: (N, 640, 640, 3)

            runner = ClientRunner(hw_arch=hw_arch, har=str(har_raw))
            runner.optimize(calib_array, data_type=CalibrationDataType.np_array)
            runner.save_har(str(har_opt))
        except Exception as exc:
            return None, f"optimize failed: {exc}"

        # Step 3: compile quantized HAR → HEF
        print(f"[hailo] compile {har_opt.name} → {hef_path.name}")
        try:
            runner = ClientRunner(hw_arch=hw_arch, har=str(har_opt))
            hef_bytes = runner.compile()
            with open(hef_path, "wb") as f:
                f.write(hef_bytes)
        except Exception as exc:
            return None, f"compile failed: {exc}"

    return hef_path, ""


def export_one(
    model_name: str,
    calib_dir: Path,
    hw_arch: str,
    fallback_parser: bool,
) -> dict:
    """Export a single model variant. Returns a result record dict."""
    pt_path = _MODELS_DIR / model_name
    record  = {"model": model_name, "status": "unknown", "hef_path": "", "notes": ""}

    if not pt_path.exists():
        record["status"] = "skipped"
        record["notes"]  = f".pt file not found at {pt_path}"
        return record

    if model_name in _EXPECTED_OVERSIZE:
        record["notes"] = "expected to exceed Hailo-8L SRAM — attempting anyway"

    try:
        onnx_path = export_onnx(pt_path)
    except Exception as exc:
        record["status"] = "failed"
        record["notes"]  = f"ONNX export error: {exc}"
        return record

    hef_path, err = compile_hef(onnx_path, calib_dir, hw_arch, fallback_parser)
    if err:
        record["status"] = "failed"
        record["notes"]  = err
    else:
        record["status"]   = "ok"
        record["hef_path"] = str(hef_path)

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt models to Hailo .hef for Hailo-8L deployment"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model",  help="Single .pt filename, e.g. yolo26n.pt")
    group.add_argument("--all",    action="store_true", help="Export all 5 variants")

    parser.add_argument("--calib-dir",       required=True, type=Path,
                        help="Directory of calibration images (64–200 JPEG frames)")
    parser.add_argument("--hw-arch",         default="hailo8l",
                        choices=["hailo8", "hailo8l"],
                        help="Target Hailo architecture (default: hailo8l for RPi AI Kit)")
    parser.add_argument("--fallback-parser", action="store_true",
                        help="Skip hailomz and use DFC three-step API directly")
    args = parser.parse_args()

    if not args.calib_dir.is_dir():
        print(f"ERROR: calibration directory not found: {args.calib_dir}", file=sys.stderr)
        sys.exit(1)

    variants = _ALL_VARIANTS if args.all else [args.model]
    results  = []

    for variant in variants:
        print(f"\n{'='*60}\nExporting {variant}\n{'='*60}")
        rec = export_one(variant, args.calib_dir, args.hw_arch, args.fallback_parser)
        results.append(rec)
        status_tag = "OK" if rec["status"] == "ok" else rec["status"].upper()
        print(f"[{status_tag}] {variant}  {rec.get('hef_path', '')}  {rec.get('notes', '')}")

    # Write summary CSV
    summary_path = _ROOT / "edge" / "export_results.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "status", "hef_path", "notes"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nExport summary written to {summary_path}")
    failed = [r["model"] for r in results if r["status"] == "failed"]
    if failed:
        print(f"FAILED variants: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
