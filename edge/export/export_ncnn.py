"""
NCNN export pipeline: .pt → <model>_<res>_ncnn/ folder

Runs in the project venv (ultralytics 8.4.x + torch 2.10.x).
Must run on an x86 desktop — PNNX does not have an ARM binary.

Setup:
    source .venv/bin/activate
    pip install ncnn pnnx

Usage:
    # Export default set (n/s/m at 640 and 576):
    python edge/export_ncnn.py

    # Export a single model at one resolution:
    python edge/export_ncnn.py --model yolo26n.pt --imgsz 640

    # Export larger variants (l/x — not in the default set):
    python edge/export_ncnn.py --models yolo26n.pt yolo26s.pt yolo26m.pt yolo26l.pt

    # Override resolutions:
    python edge/export_ncnn.py --resolutions 640 576 512

Outputs:
    models/yolo26n_640_ncnn/model.ncnn.param
    models/yolo26n_640_ncnn/model.ncnn.bin
    models/yolo26n_576_ncnn/...
    (one folder per model × resolution combination)

Notes:
    - PNNX is an x86-only binary. This script cannot run on ARM devices.
    - The input resolution is baked into model.ncnn.param at export time.
      Each output folder is valid for exactly one resolution. Passing a different
      imgsz= at inference is a silent correctness error.
    - Ultralytics cleans up PNNX artifacts automatically since 8.4.x; this
      script asserts the two inference-critical files survive regardless.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT       = Path(__file__).parents[1]
_MODELS_DIR = _ROOT / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default export set — conservative for 4 GB ARM devices (RPi 4).
# Extend via --models to add l/x variants for RPi 5 (8 GB).
_DEFAULT_MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"]

_DEFAULT_RESOLUTIONS = [640, 576]

# Artifacts produced alongside the two inference-critical files (param + bin).
# Not needed on the device; removing them reduces transfer size.
_ARTIFACTS_TO_REMOVE = [
    "model.pnnx.param",
    "model.pnnx.bin",
    "model.pnnx.onnx",
    "model_ncnn.py",
    "model_pnnx.py",
    "model.pt",
    "__pycache__",
]


def _check_filesystem() -> None:
    """
    Guard against cross-device link errors during PNNX binary download.

    Ultralytics downloads the PNNX binary at export time and installs it via
    os.rename(). If the working directory and the Python environment are on
    different filesystems, os.rename() raises [Errno 18] and the export fails.
    """
    import site
    site_pkg = site.getsitepackages()[0]

    result_cwd  = subprocess.run(["df", "--output=source", "."],
                                 capture_output=True, text=True)
    result_site = subprocess.run(["df", "--output=source", site_pkg],
                                 capture_output=True, text=True)

    dev_cwd  = result_cwd.stdout.strip().splitlines()[-1]
    dev_site = result_site.stdout.strip().splitlines()[-1]

    if dev_cwd != dev_site:
        print(
            f"ERROR: cross-device link risk.\n"
            f"  Working dir filesystem : {dev_cwd}\n"
            f"  Python venv filesystem : {dev_site}\n"
            f"  Run the export from a directory on the same filesystem as the venv.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[env] filesystem check OK  ({dev_cwd})")


def export_ncnn(model_name: str, res: int) -> Path | None:
    """
    Export a single YOLO26 .pt model to NCNN format for ARM Cortex-A Linux targets.

    The input resolution is baked into model.ncnn.param at export time. Every
    inference call against the resulting folder must pass imgsz=res to match.
    Passing a different resolution is a silent correctness error.

    Returns the output directory path on success, None if the .pt is not found.
    """
    from ultralytics import YOLO

    pt_path = _MODELS_DIR / model_name
    if not pt_path.exists():
        print(f"[SKIP] {model_name} not found at {pt_path}")
        return None

    stem = pt_path.stem                           # e.g. "yolo26n"
    dst  = _MODELS_DIR / f"{stem}_{res}_ncnn_model"

    if dst.exists() and (dst / "model.ncnn.param").exists() and (dst / "model.ncnn.bin").exists():
        print(f"[skip] {dst.name} already exists — delete to re-export")
        return dst

    print(f"\n--- {model_name} @ {res}px ---")
    model = YOLO(str(pt_path))

    # Ultralytics names the exported folder <stem>_ncnn_model/ in the working dir.
    # Using imgsz as int (not list) produces a square input, which is what we need.
    exported_raw = Path(model.export(format="ncnn", imgsz=res, half=False))

    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(exported_raw), dst)

    # Strip artifacts not required for inference
    for name in _ARTIFACTS_TO_REMOVE:
        p = dst / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    # Hard assertion — if these are missing the export was silently broken
    assert (dst / "model.ncnn.param").exists(), f"model.ncnn.param missing in {dst}"
    assert (dst / "model.ncnn.bin").exists(),   f"model.ncnn.bin missing in {dst}"

    remaining = sorted(f.name for f in dst.iterdir())
    print(f"  OK → {dst.name}  {remaining}")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO26 .pt models to NCNN folders for ARM Cortex-A targets"
    )
    parser.add_argument(
        "--model",
        help="Single .pt filename, e.g. yolo26n.pt (overrides --models)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=_DEFAULT_MODELS,
        metavar="MODEL",
        help=f"List of .pt filenames (default: {_DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Single resolution override (overrides --resolutions)",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=_DEFAULT_RESOLUTIONS,
        metavar="RES",
        help=f"List of resolutions to export (default: {_DEFAULT_RESOLUTIONS})",
    )
    args = parser.parse_args()

    _check_filesystem()

    models      = [args.model] if args.model else args.models
    resolutions = [args.imgsz] if args.imgsz else args.resolutions

    results: list[tuple[str, int, str]] = []   # (model, res, status)

    for model_name in models:
        for res in resolutions:
            try:
                out = export_ncnn(model_name, res)
                results.append((model_name, res, "ok" if out else "skipped"))
            except AssertionError as exc:
                print(f"  FAIL assertion: {exc}", file=sys.stderr)
                results.append((model_name, res, f"failed: {exc}"))
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
                results.append((model_name, res, f"failed: {type(exc).__name__}: {exc}"))

    print("\n--- Export summary ---")
    for model_name, res, status in results:
        print(f"  {model_name} @ {res}px  →  {status}")

    failed = [(m, r) for m, r, s in results if s.startswith("failed")]
    if failed:
        print(f"\nFAILED: {failed}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nTransfer models/ to each ARM device.\n"
        f"Only the *_ncnn/ folders are needed — each is valid for its baked resolution."
    )


if __name__ == "__main__":
    main()
