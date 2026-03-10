"""
TensorRT HQ export pipeline for Jetson Nano: .pt → .onnx → graph surgery → .engine

Analogous to export_hailo.py --high-quality but targeting TensorRT 8.2 on Maxwell
(sm_53, JetPack 4.6.x). The standard TRT export fails because:

  1. ultralytics ONNX export emits Mod ops unsupported by TRT 8.2
  2. The full detection-head graph (DFL + NMS) compiles incorrectly,
     producing ~1/3 of expected detections

Strategy (mirrors Hailo HQ pipeline):
  - Cut the ONNX graph at the 6 detection-head Conv outputs (cv2/cv3)
  - Replace any remaining Mod ops with Sub/Mul/Floor/Div decomposition
  - Compile the clean backbone+neck+head-conv subgraph via trtexec
  - NMS and DFL decode run on CPU (trt_postprocess.py)

This script MUST run on the Jetson Nano itself — TRT engines are architecture-specific.

Dependencies (add to requirements-jetson.txt):
    onnx>=1.12.0,<1.15.0
    onnx-graphsurgeon       (NVIDIA, pure Python — pip install onnx_graphsurgeon)

Usage:
    source .venv/bin/activate
    cd /media/les2/SD-128-J10/yolo26-track-edge-benchmark

    python edge/export_tensorrt_hq.py                          # all variants × all resolutions
    python edge/export_tensorrt_hq.py --model yolo26n          # single variant
    python edge/export_tensorrt_hq.py --imgsz 576              # all variants, 576px only
    python edge/export_tensorrt_hq.py --model yolo26n --fp16   # FP16 engine (half bandwidth)

Outputs:
    models/yolo26n_hq.onnx       — surgered ONNX (6 raw Conv outputs, no Mod ops)
    models/yolo26n_hq.engine     — TRT engine compiled from surgered ONNX
    edge/export_results_jetson_hq.csv — success/failure summary
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

_DEFAULT_VARIANTS    = ["yolo26n", "yolo26s", "yolo26m"]
_DEFAULT_RESOLUTIONS = [640, 576]

# Detection-head end-node identification by weight name pattern.
#
# onnxslim (run internally by ultralytics) flattens node names to Conv_NNN,
# but weight tensor names preserve the PyTorch module hierarchy. We identify
# the 6 end-node Convs by matching their weight (2nd input) against these
# patterns. Each pattern matches the final 1×1 Conv in the one2one head:
#
#   cv2 (box regression): out_ch=4, ltrb distances (post-DFL, NO DFL decode needed)
#   cv3 (classification): out_ch=80, raw class logits (sigmoid applied in postprocess)
_END_NODE_WEIGHT_PATTERNS = [
    f"model.23.one2one_cv2.{i}.2.weight" for i in range(3)
] + [
    f"model.23.one2one_cv3.{i}.2.weight" for i in range(3)
]

# trtexec binary shipped with JetPack
_TRTEXEC_PATHS = [
    "/usr/src/tensorrt/bin/trtexec",
    shutil.which("trtexec") or "",
]


def _engine_stem(variant: str, imgsz: int) -> str:
    """Resolution-suffixed stem: yolo26n @ 640 → yolo26n_hq, yolo26n @ 576 → yolo26n_576_hq."""
    base = variant if imgsz == 640 else f"{variant}_{imgsz}"
    return f"{base}_hq"


def _find_trtexec() -> str:
    for p in _TRTEXEC_PATHS:
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError(
        "trtexec not found. Expected at /usr/src/tensorrt/bin/trtexec "
        "(shipped with JetPack)."
    )


# ---------------------------------------------------------------------------
# Stage 1: .pt → .onnx (standard ultralytics export)
# ---------------------------------------------------------------------------

def export_onnx(pt_path: Path, imgsz: int) -> Path:
    """Export PyTorch .pt to full ONNX graph (before surgery).

    Uses simplify=False to preserve the hierarchical node names from the
    PyTorch module tree (e.g. /model.23/one2one_cv2.0/...). When simplify=True,
    onnxsim flattens all names to generic Conv_NNN, making end-node identification
    impossible. Graph cleanup happens in the surgery step instead.
    """
    stem      = pt_path.stem if imgsz == 640 else f"{pt_path.stem}_{imgsz}"
    onnx_path = _MODELS_DIR / f"{stem}_full.onnx"
    if onnx_path.exists():
        print(f"  [onnx] reusing existing {onnx_path.name}")
        return onnx_path

    from ultralytics import YOLO

    print(f"  [onnx] {pt_path.name} → {onnx_path.name} (imgsz={imgsz}, opset=11, simplify=False)")
    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=11,        # opset 11 avoids some ops TRT 8.2 can't handle
        dynamic=False,   # static shapes for TRT compilation
        simplify=False,  # preserve hierarchical node names for end-node surgery
    )
    # ultralytics places .onnx next to the .pt with its original stem
    exported = pt_path.with_suffix(".onnx")
    if exported != onnx_path:
        exported.rename(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX export did not produce {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# Stage 2: ONNX graph surgery — cut at end nodes + replace Mod ops
# ---------------------------------------------------------------------------

def _replace_mod_nodes(graph):
    """Replace all Mod nodes with the equivalent Sub/Mul/Floor/Div subgraph.

    Mathematical identity: a % b = a - b * floor(a / b)
    All replacement ops (Sub, Mul, Floor, Div) are supported by TRT 8.2.
    """
    import onnx_graphsurgeon as gs
    import numpy as np

    mod_nodes = [n for n in graph.nodes if n.op == "Mod"]
    if not mod_nodes:
        print("  [surgery] no Mod nodes found — skipping replacement")
        return

    for node in mod_nodes:
        a_tensor = node.inputs[0]
        b_tensor = node.inputs[1]
        out_tensor = node.outputs[0]

        # a / b
        div_out = gs.Variable(name=f"{node.name}_div", dtype=np.float32)
        div_node = gs.Node(op="Div", name=f"{node.name}_Div",
                           inputs=[a_tensor, b_tensor], outputs=[div_out])

        # floor(a / b)
        floor_out = gs.Variable(name=f"{node.name}_floor", dtype=np.float32)
        floor_node = gs.Node(op="Floor", name=f"{node.name}_Floor",
                             inputs=[div_out], outputs=[floor_out])

        # b * floor(a / b)
        mul_out = gs.Variable(name=f"{node.name}_mul", dtype=np.float32)
        mul_node = gs.Node(op="Mul", name=f"{node.name}_Mul",
                           inputs=[b_tensor, floor_out], outputs=[mul_out])

        # a - b * floor(a / b)
        sub_node = gs.Node(op="Sub", name=f"{node.name}_Sub",
                           inputs=[a_tensor, mul_out], outputs=[out_tensor])

        # Splice into graph: remove original Mod, add decomposition nodes
        graph.nodes.remove(node)
        graph.nodes.extend([div_node, floor_node, mul_node, sub_node])

        print(f"  [surgery] replaced Mod node '{node.name}' with Div→Floor→Mul→Sub")

    graph.cleanup().toposort()


def _cut_at_end_nodes(graph):
    """Truncate the graph at the 6 detection-head Conv outputs.

    Identifies end-node Convs by matching their weight tensor name against
    _END_NODE_WEIGHT_PATTERNS (onnxslim flattens node names but preserves
    weight names from the PyTorch module tree).

    Removes all downstream nodes (Reshape, Concat, TopK, NMS, etc.) and marks
    the 6 Conv output tensors as the new graph outputs:
      - 3 cv2 outputs: (1, 4, H, W)  — ltrb distances (post-DFL, in stride units)
      - 3 cv3 outputs: (1, 80, H, W) — raw class logits
    """
    import numpy as np

    # Build weight_name → Conv node map
    weight_to_node = {}
    for n in graph.nodes:
        if n.op == "Conv" and len(n.inputs) > 1:
            w_name = n.inputs[1].name
            weight_to_node[w_name] = n

    new_outputs = []
    found = []
    for pattern in _END_NODE_WEIGHT_PATTERNS:
        if pattern not in weight_to_node:
            print(f"  [surgery] WARNING: no Conv with weight '{pattern}' found in graph")
            continue
        node = weight_to_node[pattern]
        out_tensor = node.outputs[0]
        out_tensor.dtype = np.float32
        new_outputs.append(out_tensor)
        found.append(f"{node.name} (weight={pattern})")

    if len(new_outputs) != 6:
        raise RuntimeError(
            f"Expected 6 end-node outputs, found {len(new_outputs)}: {found}\n"
            "The ONNX graph structure may differ from expected YOLO26 architecture."
        )

    # Replace graph outputs with only the 6 Conv tensors
    graph.outputs = new_outputs

    # Remove all now-unreachable nodes downstream of the cut
    graph.cleanup().toposort()

    # Count removed nodes
    remaining = len(graph.nodes)
    print(f"  [surgery] cut graph at 6 end nodes — {remaining} nodes remaining")


def surgery(onnx_full: Path, variant: str, imgsz: int) -> Path:
    """Apply graph surgery: cut end nodes + replace Mod ops. Returns path to surgered ONNX."""
    import onnx
    import onnx_graphsurgeon as gs

    stem     = _engine_stem(variant, imgsz)
    out_path = _MODELS_DIR / f"{stem}.onnx"

    if out_path.exists():
        print(f"  [surgery] reusing existing {out_path.name}")
        return out_path

    print(f"  [surgery] loading {onnx_full.name}")
    model = onnx.load(str(onnx_full))
    graph = gs.import_onnx(model)

    print(f"  [surgery] original graph: {len(graph.nodes)} nodes, "
          f"{len(graph.outputs)} outputs")

    # Step 1: cut at detection-head Conv outputs (removes NMS/DFL downstream)
    _cut_at_end_nodes(graph)

    # Step 2: replace any remaining Mod ops in the backbone/neck
    _replace_mod_nodes(graph)

    # Export surgered ONNX
    model_out = gs.export_onnx(graph)
    onnx.save(model_out, str(out_path))
    print(f"  [surgery] saved {out_path.name}")

    return out_path


# ---------------------------------------------------------------------------
# Stage 3: trtexec compilation
# ---------------------------------------------------------------------------

def compile_engine(onnx_path: Path, variant: str, imgsz: int,
                   fp16: bool, workspace_mb: int, force: bool = False) -> Path:
    """Compile surgered ONNX → TRT engine via trtexec CLI."""
    stem        = _engine_stem(variant, imgsz)
    engine_path = _MODELS_DIR / f"{stem}.engine"

    if engine_path.exists() and not force:
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"  [trtexec] reusing existing {engine_path.name} ({size_mb:.1f} MB)")
        return engine_path

    trtexec = _find_trtexec()
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")

    prec = "FP16" if fp16 else "FP32"
    print(f"  [trtexec] {onnx_path.name} → {engine_path.name} ({prec})")
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

    if not engine_path.exists():
        raise FileNotFoundError(f"trtexec did not produce {engine_path}")

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"  [trtexec] OK: {engine_path.name} ({size_mb:.1f} MB)")
    return engine_path


# ---------------------------------------------------------------------------
# Full pipeline: export + surgery + compile
# ---------------------------------------------------------------------------

def export_one(variant: str, imgsz: int, fp16: bool, workspace_mb: int,
               force: bool = False) -> dict:
    """Run the full HQ export pipeline for one model×resolution pair."""
    pt_path = _MODELS_DIR / f"{variant}.pt"
    stem    = _engine_stem(variant, imgsz)
    record  = {
        "model": stem, "imgsz": imgsz, "fp16": fp16,
        "status": "unknown", "engine_mb": "", "elapsed_s": "", "notes": "",
    }

    if not pt_path.exists():
        record["status"] = "skipped"
        record["notes"]  = f".pt not found: {pt_path}"
        return record

    engine_path = _MODELS_DIR / f"{stem}.engine"
    if engine_path.exists() and not force:
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        record["status"]    = "ok"
        record["engine_mb"] = f"{size_mb:.1f}"
        record["notes"]     = "reused existing engine"
        return record

    try:
        t0 = time.time()

        # Stage 1: .pt → full ONNX
        onnx_full = export_onnx(pt_path, imgsz)

        # Stage 2: graph surgery (cut end nodes + replace Mod)
        onnx_hq = surgery(onnx_full, variant, imgsz)

        # Stage 3: trtexec → .engine
        engine = compile_engine(onnx_hq, variant, imgsz, fp16, workspace_mb, force)

        elapsed = time.time() - t0
        size_mb = engine.stat().st_size / (1024 * 1024)

        record["status"]    = "ok"
        record["engine_mb"] = f"{size_mb:.1f}"
        record["elapsed_s"] = f"{elapsed:.0f}"

    except Exception as exc:
        record["status"] = "failed"
        record["notes"]  = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {exc}")

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HQ TensorRT export: graph surgery + trtexec on Jetson Nano"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single variant stem (e.g. yolo26n). Default: all variants.",
    )
    parser.add_argument(
        "--imgsz", type=int, nargs="+", default=None,
        help="Input resolution(s) (default: 640 576).",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable FP16 precision (halves memory bandwidth on Maxwell).",
    )
    parser.add_argument(
        "--workspace", type=int, default=1024,
        help="TRT workspace size in MB (default: 1024).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-export even if .engine file already exists.",
    )
    args = parser.parse_args()

    # Verify trtexec is available early
    trtexec = _find_trtexec()
    print(f"Using trtexec: {trtexec}")

    variants    = [args.model] if args.model else _DEFAULT_VARIANTS
    resolutions = args.imgsz if args.imgsz else _DEFAULT_RESOLUTIONS
    results     = []

    for variant in variants:
        for imgsz in resolutions:
            stem = _engine_stem(variant, imgsz)
            prec = "FP16" if args.fp16 else "FP32"
            print(f"\n{'='*60}")
            print(f"HQ Export: {stem} @ {imgsz}px ({prec})")
            print(f"{'='*60}")
            rec = export_one(variant, imgsz, args.fp16, args.workspace, args.force)
            results.append(rec)
            tag = "OK" if rec["status"] == "ok" else rec["status"].upper()
            print(f"  [{tag}] {stem}  {rec.get('engine_mb', '')} MB  {rec.get('notes', '')}")

    # Append results to CSV log
    summary_path = _ROOT / "edge" / "export_results_jetson_hq.csv"
    file_exists  = summary_path.exists()
    fieldnames   = ["model", "imgsz", "fp16", "status", "engine_mb", "elapsed_s", "notes"]
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
