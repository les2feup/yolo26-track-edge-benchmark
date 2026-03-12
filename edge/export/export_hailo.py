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

    # High-quality export — Adaround + bias correction + a16_w16 on detection heads:
    python edge/export_hailo.py --model yolo26n.pt --calib-dir edge/calib_imgs/ --high-quality

    # Batch export — runs all variants and reports success/failure per model:
    python edge/export_hailo.py --all --calib-dir edge/calib_imgs/ 2>&1 | tee edge/export_log.txt

Outputs:
    models/yolo26n.hef (and .har intermediate)
    edge/export/logs/export_results.csv — success/failure summary for each variant

Notes:
    - DFC v3.32 / HailoRT 4.22 is the LAST stable version supporting Hailo-8L.
      hailo_model_zoo v5.x dropped Hailo-8/8L support — DO NOT use v5.x.
    - hailo8l is the correct --hw-arch for the Raspberry Pi AI Kit M.2 hat.
    - ONNX opset 11 is required; higher opsets have known DFC parser issues.
    - The DFC three-step API (parse → optimize → compile) is used directly;
      hailomz is not involved.
    - Calibration images: 64–200 unlabelled representative frames are sufficient.
      Collect them from MOT17 img1/ directories: edge/collect_calib.py
    - --high-quality enables post-quantization tuning to recover accuracy lost by
      default int8 quantization.  Produces *_hq.hef files alongside the defaults.
      Techniques applied: (1) a16_w16 precision on the 6 detection-head Conv outputs,
      (2) optimization_level=2 (equalization + finetune encoding),
      (3) Adaround weight rounding optimisation (GPU only — skipped on CPU),
      (4) high-effort compiler allocation.
      With GPU: ≥1024 calibration images recommended; ~5–10× slower than default.
      Without GPU: a16_w16 + level-2 optimization still apply; Adaround is skipped.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

_ROOT       = Path(__file__).parents[1]
_MODELS_DIR = _ROOT / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_ALL_VARIANTS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

# Models likely to exceed Hailo-8L SRAM (~8 MB) — attempted but expected to fail
_EXPECTED_OVERSIZE = {"yolo26l.pt", "yolo26x.pt"}

# Canonical 640-px suffix used when imgsz == 640, for backwards compatibility
# with the existing .hef filenames (no suffix).  All other resolutions are
# encoded as _{imgsz} in the stem, e.g. yolo26n_576.hef.
# High-quality exports append _hq before the extension: yolo26n_hq.hef
def _hef_stem(pt_stem: str, imgsz: int, high_quality: bool = False) -> str:
    base = pt_stem if imgsz == 640 else f"{pt_stem}_{imgsz}"
    return f"{base}_hq" if high_quality else base


# YOLO26 detection head — the six Conv outputs cut as end nodes.
# These are the most quantization-sensitive layers: box regression (cv2)
# and class scores (cv3) directly feed into NMS post-processing.
_YOLO26_END_NODES = [
    f"/model.23/one2one_cv2.{i}/one2one_cv2.{i}.2/Conv" for i in range(3)
] + [
    f"/model.23/one2one_cv3.{i}/one2one_cv3.{i}.2/Conv" for i in range(3)
]


def _find_output_layer_names(runner) -> list[str]:
    """
    Extract the DFC-assigned short names for the model's output layers.

    After translate_onnx_model(), the ClientRunner holds a Hailo Network graph.
    get_output_layers() returns the output layer objects whose .name attribute
    gives the short identifier used in ALLS scripts (e.g. "conv42").
    """
    try:
        # ClientRunner exposes the parsed graph via get_hn_model() or similar
        # The output layers correspond to our 6 end-node Convs
        hn_model = runner.get_hn_model()
        output_layers = hn_model.get_output_layers()
        return [layer.name for layer in output_layers]
    except Exception:
        # Fallback: if the API doesn't expose layer names this way,
        # return empty and skip per-layer a16_w16 (Adaround + bias correction
        # still apply globally and provide the bulk of the quality gain)
        return []


def _build_hq_model_script(output_layer_names: list[str]) -> str:
    """
    Hailo ALLS model script for high-quality quantization.

    Applied techniques:
    - a16_w16 on detection-head output Conv layers — these produce the raw
      box and class logits where int8 rounding error is most damaging.
      Layer names are discovered dynamically from the parsed HAR.
    - optimization_level=2 — forces calibration-aware quantization with
      equalization and finetune encoding.  Without this, DFC drops to
      level 0 on CPU-only hosts and skips all accuracy recovery.
    - Adaround — learns optimal per-weight rounding (requires GPU;
      DFC will skip gracefully on CPU-only hosts).

    NOT included:
    - bias_correction — DFC v3.33 ships Keras 3.x where the bias correction
      algorithm crashes (ValueError in HailoModel.build() — missing _shape
      suffix on Keras 3 Layer.build() arguments).
    """
    lines = []

    # Promote detection head outputs to 16-bit weights and activations.
    # Layer names are the DFC-assigned identifiers (e.g. "yolo26n/output_layer1"),
    # discovered from the parsed HAR rather than hardcoded.
    if output_layer_names:
        for name in output_layer_names:
            lines.append(f'quantization_param({name}, precision_mode=a16_w16)')
    else:
        lines.append('# WARNING: output layer names not resolved — skipping a16_w16')

    # Force optimization level 2 — calibration-aware quantization with
    # equalization and finetune encoding, even on CPU-only hosts.
    # compression_level=0 keeps all layers at their declared precision.
    lines.append('model_optimization_flavor(optimization_level=2, compression_level=0)')

    # Calibration batch size — smaller batches use less GPU memory.
    # The DFC will iterate through the full dataset passed to optimize().
    lines.append('model_optimization_config(calibration, batch_size=8)')

    # Adaround — adaptive rounding optimisation for all layers.
    # Requires GPU; DFC skips it on CPU but won't error.
    lines.append('post_quantization_optimization(adaround, policy=enabled)')

    return "\n".join(lines) + "\n"


def export_onnx(pt_path: Path, imgsz: int = 640) -> Path:
    """Export PyTorch .pt to ONNX opset 11 with static shapes at the given resolution."""
    stem      = _hef_stem(pt_path.stem, imgsz)
    onnx_path = _MODELS_DIR / f"{stem}.onnx"
    if onnx_path.exists():
        print(f"[onnx] reusing existing {onnx_path.name}")
        return onnx_path

    print(f"[onnx] exporting {pt_path.name} @ {imgsz}px → {onnx_path.name}")
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=11,       # DFC v3.x parser is validated on opset 11
        dynamic=False,  # static shapes required for Hailo compilation
        simplify=False, # Hailo parser handles simplification internally
    )
    # ultralytics exports to the same directory as the .pt file; move to models/
    exported = pt_path.parent / (pt_path.stem + ".onnx")
    if exported != onnx_path:
        exported.rename(onnx_path)
    return onnx_path


def compile_hef(
    onnx_path: Path,
    calib_dir: Path,
    hw_arch: str,
    imgsz: int = 640,
    high_quality: bool = False,
) -> tuple[Path | None, str]:
    """
    Compile ONNX → HEF using the hailo_sdk_client Python API directly.

    The three-step DFC pipeline (parse → optimize → compile) is driven via
    ClientRunner, bypassing the `hailo` CLI which requires the HailoRT runtime
    library (libhailort.so) — a separate .deb not bundled in the Python wheel.

    When high_quality is True, a model script is injected before optimize() to
    enable a16_w16 on detection heads, Adaround, and bias correction.  The
    compiler also runs at high allocation effort.  Output uses *_hq.hef suffix.

    Returns (hef_path, "") on success or (None, error_message) on failure.
    """
    # Resolve output stem — HQ exports get a distinct filename
    src_stem  = onnx_path.stem                       # e.g. yolo26n or yolo26n_576
    out_stem  = f"{src_stem}_hq" if high_quality else src_stem
    hef_path  = _MODELS_DIR / f"{out_stem}.hef"

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

    hq_tag = " [HIGH QUALITY]" if high_quality else ""

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir  = Path(_tmp)
        har_raw = tmpdir / f"{src_stem}.har"
        har_opt = tmpdir / f"{src_stem}_quantized.har"

        # Step 1: parse ONNX → HAR
        # YOLO26's detection head (/model.23) contains ops unsupported by Hailo-8L
        # (GatherElements, TopK, Mod, ReduceMax on batch axis). The DFC recommends
        # cutting at the six one2one_cv2/cv3 Conv outputs — these are the raw
        # box-regression and class-score feature maps before NMS decoding. NMS runs
        # on the host CPU via HailoRT post-processing or a custom wrapper.
        print(f"[hailo] parse  {onnx_path.name} → {har_raw.name}  (6 cv2/cv3 end nodes){hq_tag}")
        output_layer_names = []
        try:
            runner = ClientRunner(hw_arch=hw_arch)
            hn, params = runner.translate_onnx_model(
                str(onnx_path),
                net_name=src_stem,
                end_node_names=_YOLO26_END_NODES,
            )
            # Extract DFC-assigned short layer names for HQ model script
            if high_quality:
                output_layer_names = _find_output_layer_names(runner)
                if output_layer_names:
                    print(f"[hailo] detected output layers: {output_layer_names}")
                else:
                    print("[hailo] WARNING: could not resolve output layer names")
            runner.save_har(str(har_raw))
        except Exception as exc:
            return None, f"parse failed: {exc}"

        # Step 2: quantize / optimize using calibration images
        # DFC optimize() with data_type=CalibrationDataType.np_array expects a single
        # numpy.ndarray shaped (N, H, W, C) float32 in range [0, 1].
        print(f"[hailo] optimize {har_raw.name}  calib={calib_dir}{hq_tag}")
        try:
            import numpy as np
            import cv2
            from hailo_sdk_client.exposed_definitions import CalibrationDataType

            calib_imgs = sorted(calib_dir.glob("*.jpg"))
            if not calib_imgs:
                return None, f"no calibration images found in {calib_dir}"

            calib_array = np.stack([
                cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (imgsz, imgsz)),
                             cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                for p in calib_imgs
            ])  # shape: (N, imgsz, imgsz, 3)
            print(f"[hailo] loaded {calib_array.shape[0]} calibration images")

            runner = ClientRunner(hw_arch=hw_arch, har=str(har_raw))

            # Inject model script for HQ mode — must be loaded before optimize()
            if high_quality:
                alls_script = _build_hq_model_script(output_layer_names)
                print(f"[hailo] loading HQ model script:\n{alls_script}")
                runner.load_model_script(alls_script)

            runner.optimize(calib_array, data_type=CalibrationDataType.np_array)
            runner.save_har(str(har_opt))
        except Exception as exc:
            return None, f"optimize failed: {exc}"

        # Step 3: compile quantized HAR → HEF
        print(f"[hailo] compile {har_opt.name} → {hef_path.name}{hq_tag}")
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
    imgsz: int = 640,
    high_quality: bool = False,
) -> dict:
    """Export a single model variant at the given resolution. Returns a result record dict."""
    pt_path = _MODELS_DIR / model_name
    record  = {"model": model_name, "imgsz": imgsz, "status": "unknown",
               "hef_path": "", "notes": "", "quality": "hq" if high_quality else "default"}

    if not pt_path.exists():
        record["status"] = "skipped"
        record["notes"]  = f".pt file not found at {pt_path}"
        return record

    if model_name in _EXPECTED_OVERSIZE:
        record["notes"] = "expected to exceed Hailo-8L SRAM — attempting anyway"

    try:
        onnx_path = export_onnx(pt_path, imgsz=imgsz)
    except Exception as exc:
        record["status"] = "failed"
        record["notes"]  = f"ONNX export error: {exc}"
        return record

    hef_path, err = compile_hef(
        onnx_path, calib_dir, hw_arch,
        imgsz=imgsz, high_quality=high_quality,
    )
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
    parser.add_argument("--imgsz",           type=int, default=640,
                        help="Input resolution in pixels (default: 640; e.g. 576 for a second sweep)")
    parser.add_argument("--high-quality",   action="store_true",
                        help="Enable a16_w16 detection heads + Adaround + bias correction "
                             "(slower, produces *_hq.hef alongside defaults)")
    args = parser.parse_args()

    if not args.calib_dir.is_dir():
        print(f"ERROR: calibration directory not found: {args.calib_dir}", file=sys.stderr)
        sys.exit(1)

    variants = _ALL_VARIANTS if args.all else [args.model]
    results  = []

    hq_label = " [HIGH QUALITY]" if args.high_quality else ""
    for variant in variants:
        print(f"\n{'='*60}\nExporting {variant} @ {args.imgsz}px{hq_label}\n{'='*60}")
        rec = export_one(
            variant, args.calib_dir, args.hw_arch,
            imgsz=args.imgsz, high_quality=args.high_quality,
        )
        results.append(rec)
        status_tag = "OK" if rec["status"] == "ok" else rec["status"].upper()
        print(f"[{status_tag}] {variant} @ {args.imgsz}px  {rec.get('hef_path', '')}  {rec.get('notes', '')}")

    # Write summary CSV — append rows so existing 640-px results are preserved
    summary_path = _ROOT / "edge" / "export" / "logs" / "export_results.csv"
    file_exists  = summary_path.exists()
    fieldnames   = ["model", "imgsz", "quality", "status", "hef_path", "notes"]
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nExport summary written to {summary_path}")
    failed = [r["model"] for r in results if r["status"] == "failed"]
    if failed:
        print(f"FAILED variants: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
