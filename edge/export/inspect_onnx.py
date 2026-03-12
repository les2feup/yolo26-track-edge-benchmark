"""
Diagnostic: dump ONNX graph structure to find detection-head Conv outputs.

Shows output tensor shapes for all Conv nodes (via graph-surgeon shape inference)
so we can identify cv2 (C=64) and cv3 (C=80) outputs by channel dimension.

Usage:
    python edge/inspect_onnx.py models/yolo26n_full.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path


def inspect(onnx_path: str) -> None:
    import onnx

    model = onnx.load(onnx_path)
    graph = model.graph

    print(f"=== {Path(onnx_path).name} ===")
    print(f"Opset: {model.opset_import[0].version}")
    print(f"Total nodes: {len(graph.node)}")

    # Graph outputs
    print(f"\n=== Graph outputs ({len(graph.output)}) ===")
    for o in graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name}  shape={dims}")

    # Build tensor name → shape map from value_info + initializers + inputs
    shape_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        try:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if dims:
                shape_map[vi.name] = dims
        except Exception:
            pass
    for init in graph.initializer:
        shape_map[init.name] = list(init.dims)

    # Build output_name → node_name map
    output_to_node = {}
    for n in graph.node:
        for o in n.output:
            output_to_node[o] = n.name

    # All Conv nodes with their weight shapes (weight = 2nd input)
    convs = [n for n in graph.node if n.op_type == "Conv"]
    print(f"\n=== All Conv nodes ({len(convs)}) ===")
    print(f"{'name':20s} {'weight_shape':20s} {'out_channels':>12s} {'output_tensor':30s} {'output_shape':20s} {'feeds_into':20s}")
    print("-" * 130)

    # Build a map: tensor_name → list of consumer node names
    tensor_consumers = {}
    for n in graph.node:
        for inp in n.input:
            if inp not in tensor_consumers:
                tensor_consumers[inp] = []
            tensor_consumers[inp].append(f"{n.op_type}({n.name})")

    for n in convs:
        # Weight tensor is the 2nd input; shape[0] = out_channels
        w_name = n.input[1] if len(n.input) > 1 else ""
        w_shape = shape_map.get(w_name, [])
        out_ch = w_shape[0] if w_shape else "?"

        out_tensor = n.output[0] if n.output else ""
        out_shape = shape_map.get(out_tensor, [])

        # What ops consume this Conv's output?
        consumers = tensor_consumers.get(out_tensor, ["(graph output or dead)"])
        consumer_str = ", ".join(consumers[:2])
        if len(consumers) > 2:
            consumer_str += f" +{len(consumers)-2}"

        print(f"{n.name:20s} {str(w_shape):20s} {str(out_ch):>12s} {out_tensor:30s} {str(out_shape):20s} {consumer_str}")

    # Specifically highlight Conv nodes with out_channels 64 or 80
    # (cv2 = reg_max*4 = 64, cv3 = num_classes = 80 for COCO)
    print(f"\n=== Candidate detection-head Convs (out_channels=64 or 80) ===")
    for n in convs:
        w_name = n.input[1] if len(n.input) > 1 else ""
        w_shape = shape_map.get(w_name, [])
        if not w_shape:
            continue
        out_ch = w_shape[0]
        if out_ch in (64, 80):
            out_tensor = n.output[0]
            out_shape = shape_map.get(out_tensor, [])
            consumers = tensor_consumers.get(out_tensor, [])
            print(f"  {n.name:20s}  out_ch={out_ch}  output={out_tensor}  shape={out_shape}")
            print(f"    weight={w_name}  weight_shape={w_shape}")
            print(f"    consumers: {consumers}")

    # All Conv nodes whose weight name contains "one2one_cv2"
    print(f"\n=== Conv nodes with weight name containing 'one2one_cv2' ===")
    for n in convs:
        w_name = n.input[1] if len(n.input) > 1 else ""
        if "one2one_cv2" in w_name:
            w_shape = shape_map.get(w_name, [])
            out_tensor = n.output[0]
            out_shape = shape_map.get(out_tensor, [])
            consumers = tensor_consumers.get(out_tensor, [])
            print(f"  {n.name:20s}  out_ch={w_shape[0] if w_shape else '?'}  output={out_tensor}  shape={out_shape}")
            print(f"    weight={w_name}  weight_shape={w_shape}")
            print(f"    consumers: {consumers}")

    # Mod nodes
    mods = [n for n in graph.node if n.op_type == "Mod"]
    print(f"\n=== Mod nodes: {len(mods)} ===")
    for n in mods:
        print(f"  name={n.name}  inputs={list(n.input)}  outputs={list(n.output)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_onnx>")
        sys.exit(1)
    inspect(sys.argv[1])
