"""
QNN (Qualcomm AI Engine Direct) export pipeline: .pt → .onnx → graph surgery

Prepares ONNX models for inference on the Arduino UNO Q's Adreno 702 GPU via
ONNX Runtime's QNNExecutionProvider. Unlike the Hailo and TensorRT HQ pipelines,
there is no offline compilation step — the QNN EP compiles the ONNX graph
on-device at session creation time and caches the result.

Strategy (mirrors Hailo/TRT HQ graph-surgery approach):
  - Export .pt → full ONNX graph (opset 17, static shapes)
  - Cut at the 6 detection-head Conv outputs (cv2/cv3)
  - Replace any Mod ops with Sub/Mul/Floor/Div decomposition
  - The clean backbone+neck+head-conv subgraph runs on Adreno via QNN EP
  - NMS and anchor decode run on CPU (trt_postprocess.py — same NCHW format)

This script runs on the DESKTOP — the resulting .onnx files are transferred
to the Arduino UNO Q for inference via qnn_runner.py.

Dependencies (desktop):
    pip install ultralytics onnx onnx-graphsurgeon

Usage:
    python edge/export_qnn.py                          # all variants × all resolutions
    python edge/export_qnn.py --model yolo26n          # single variant
    python edge/export_qnn.py --imgsz 576              # all variants, 576px only

Outputs:
    models/yolo26n_qnn.onnx       — surgered ONNX (6 raw Conv outputs, QNN-ready)
    edge/export/logs/export_results_qnn.csv — success/failure summary
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_ROOT       = Path(__file__).parents[1]
_MODELS_DIR = _ROOT / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_VARIANTS    = ["yolo26n", "yolo26s", "yolo26m"]
_DEFAULT_RESOLUTIONS = [640, 576]

# Detection-head end-node identification by weight tensor name.
# onnxslim (run by ultralytics) flattens node names but preserves weight names
# from the PyTorch module hierarchy.
_END_NODE_WEIGHT_PATTERNS = [
    f"model.23.one2one_cv2.{i}.2.weight" for i in range(3)
] + [
    f"model.23.one2one_cv3.{i}.2.weight" for i in range(3)
]


def _onnx_stem(variant: str, imgsz: int) -> str:
    """Resolution-suffixed stem: yolo26n @ 640 → yolo26n_qnn, @ 576 → yolo26n_576_qnn."""
    base = variant if imgsz == 640 else f"{variant}_{imgsz}"
    return f"{base}_qnn"


# ---------------------------------------------------------------------------
# Stage 1: .pt → full ONNX (standard ultralytics export)
# ---------------------------------------------------------------------------

def export_onnx(pt_path: Path, imgsz: int) -> Path:
    """Export PyTorch .pt to full ONNX graph before surgery.

    Uses opset 17 (QNN EP supports modern opsets) and simplify=False to
    preserve hierarchical node names for end-node identification.

    Ultralytics writes the .onnx next to the .pt file using the original stem.
    If a stale .onnx file exists there (e.g. from a prior TRT export owned by
    root), writing fails.  To avoid this, we symlink the .pt into a temp dir
    and export from there, then move the result to models/.
    """
    import shutil
    import tempfile

    stem      = pt_path.stem if imgsz == 640 else f"{pt_path.stem}_{imgsz}"
    onnx_path = _MODELS_DIR / f"{stem}_full.onnx"
    if onnx_path.exists():
        print(f"  [onnx] reusing existing {onnx_path.name}")
        return onnx_path

    from ultralytics import YOLO

    print(f"  [onnx] {pt_path.name} → {onnx_path.name} (imgsz={imgsz}, opset=17)")

    # Export in a temp dir to avoid permission conflicts with existing .onnx files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pt = Path(tmpdir) / pt_path.name
        shutil.copy2(pt_path, tmp_pt)

        model = YOLO(str(tmp_pt))
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=17,        # QNN EP supports opset 17; higher than TRT 8.2's limit
            dynamic=False,   # static shapes for QNN graph compilation
            simplify=False,  # preserve hierarchical names for surgery
        )
        # ultralytics places .onnx next to the .pt with its original stem
        exported = tmp_pt.with_suffix(".onnx")
        if not exported.exists():
            raise FileNotFoundError(f"ONNX export did not produce {exported}")
        shutil.move(str(exported), str(onnx_path))

    return onnx_path


# ---------------------------------------------------------------------------
# Stage 2: ONNX graph surgery — cut at end nodes + replace Mod ops
# ---------------------------------------------------------------------------

def _replace_mod_nodes(graph):
    """Replace all Mod nodes with Sub/Mul/Floor/Div decomposition.

    Mathematical identity: a % b = a - b * floor(a / b)
    All replacement ops are universally supported by QNN and TRT backends.
    """
    import onnx_graphsurgeon as gs
    import numpy as np

    mod_nodes = [n for n in graph.nodes if n.op == "Mod"]
    if not mod_nodes:
        print("  [surgery] no Mod nodes found — skipping replacement")
        return

    for node in mod_nodes:
        a_tensor   = node.inputs[0]
        b_tensor   = node.inputs[1]
        out_tensor = node.outputs[0]

        div_out = gs.Variable(name=f"{node.name}_div", dtype=np.float32)
        div_node = gs.Node(op="Div", name=f"{node.name}_Div",
                           inputs=[a_tensor, b_tensor], outputs=[div_out])

        floor_out = gs.Variable(name=f"{node.name}_floor", dtype=np.float32)
        floor_node = gs.Node(op="Floor", name=f"{node.name}_Floor",
                             inputs=[div_out], outputs=[floor_out])

        mul_out = gs.Variable(name=f"{node.name}_mul", dtype=np.float32)
        mul_node = gs.Node(op="Mul", name=f"{node.name}_Mul",
                           inputs=[b_tensor, floor_out], outputs=[mul_out])

        sub_node = gs.Node(op="Sub", name=f"{node.name}_Sub",
                           inputs=[a_tensor, mul_out], outputs=[out_tensor])

        graph.nodes.remove(node)
        graph.nodes.extend([div_node, floor_node, mul_node, sub_node])
        print(f"  [surgery] replaced Mod node '{node.name}' with Div→Floor→Mul→Sub")

    graph.cleanup().toposort()


def _cut_at_end_nodes(graph):
    """Truncate graph at the 6 detection-head Conv outputs.

    Identifies end-node Convs by matching their weight tensor name against
    _END_NODE_WEIGHT_PATTERNS. Removes all downstream nodes (Reshape, Concat,
    TopK, NMS, etc.) and marks the 6 Conv output tensors as new graph outputs:
      - 3 cv2 outputs: (1, 4, H, W)  — ltrb distances in stride units
      - 3 cv3 outputs: (1, 80, H, W) — raw class logits
    """
    import numpy as np

    weight_to_node = {}
    for n in graph.nodes:
        if n.op == "Conv" and len(n.inputs) > 1:
            weight_to_node[n.inputs[1].name] = n

    new_outputs = []
    found = []
    for pattern in _END_NODE_WEIGHT_PATTERNS:
        if pattern not in weight_to_node:
            print(f"  [surgery] WARNING: no Conv with weight '{pattern}' found")
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

    graph.outputs = new_outputs
    graph.cleanup().toposort()

    print(f"  [surgery] cut graph at 6 end nodes — {len(graph.nodes)} nodes remaining")


def surgery(onnx_full: Path, variant: str, imgsz: int) -> Path:
    """Apply graph surgery: cut end nodes + replace Mod ops. Returns QNN-ready ONNX."""
    import onnx
    import onnx_graphsurgeon as gs

    stem     = _onnx_stem(variant, imgsz)
    out_path = _MODELS_DIR / f"{stem}.onnx"

    if out_path.exists():
        print(f"  [surgery] reusing existing {out_path.name}")
        return out_path

    print(f"  [surgery] loading {onnx_full.name}")
    model = onnx.load(str(onnx_full))
    graph = gs.import_onnx(model)

    print(f"  [surgery] original graph: {len(graph.nodes)} nodes, "
          f"{len(graph.outputs)} outputs")

    # Cut at detection-head Conv outputs (removes NMS/DFL downstream)
    _cut_at_end_nodes(graph)

    # Replace any remaining Mod ops in backbone/neck
    _replace_mod_nodes(graph)

    model_out = gs.export_onnx(graph)
    onnx.save(model_out, str(out_path))
    size_kb = out_path.stat().st_size / 1024
    print(f"  [surgery] saved {out_path.name} ({size_kb:.0f} KB)")

    return out_path


# ---------------------------------------------------------------------------
# Full pipeline: export + surgery
# ---------------------------------------------------------------------------

def export_one(variant: str, imgsz: int) -> dict:
    """Run the full QNN export pipeline for one model×resolution pair."""
    pt_path = _MODELS_DIR / f"{variant}.pt"
    stem    = _onnx_stem(variant, imgsz)
    record  = {
        "model": stem, "imgsz": imgsz,
        "status": "unknown", "onnx_kb": "", "elapsed_s": "", "notes": "",
    }

    if not pt_path.exists():
        record["status"] = "skipped"
        record["notes"]  = f".pt not found: {pt_path}"
        return record

    try:
        t0 = time.time()

        # Stage 1: .pt → full ONNX
        onnx_full = export_onnx(pt_path, imgsz)

        # Stage 2: graph surgery (cut end nodes + replace Mod)
        onnx_qnn = surgery(onnx_full, variant, imgsz)

        elapsed = time.time() - t0
        size_kb = onnx_qnn.stat().st_size / 1024

        record["status"]    = "ok"
        record["onnx_kb"]   = f"{size_kb:.0f}"
        record["elapsed_s"] = f"{elapsed:.1f}"

    except Exception as exc:
        record["status"] = "failed"
        record["notes"]  = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {exc}")

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QNN ONNX export: graph surgery for Adreno GPU deployment"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single variant stem (e.g. yolo26n). Default: all variants.",
    )
    parser.add_argument(
        "--imgsz", type=int, nargs="+", default=None,
        help="Input resolution(s) (default: 640 576).",
    )
    args = parser.parse_args()

    variants    = [args.model] if args.model else _DEFAULT_VARIANTS
    resolutions = args.imgsz if args.imgsz else _DEFAULT_RESOLUTIONS
    results     = []

    for variant in variants:
        for imgsz in resolutions:
            stem = _onnx_stem(variant, imgsz)
            print(f"\n{'='*60}")
            print(f"QNN Export: {stem} @ {imgsz}px")
            print(f"{'='*60}")
            rec = export_one(variant, imgsz)
            results.append(rec)
            tag = "OK" if rec["status"] == "ok" else rec["status"].upper()
            print(f"  [{tag}] {stem}  {rec.get('onnx_kb', '')} KB  {rec.get('notes', '')}")

    # Append results to CSV log
    summary_path = _ROOT / "edge" / "export" / "logs" / "export_results_qnn.csv"
    file_exists  = summary_path.exists()
    fieldnames   = ["model", "imgsz", "status", "onnx_kb", "elapsed_s", "notes"]
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
