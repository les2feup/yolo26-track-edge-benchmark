"""
Quick sanity check for TRT HQ engine: load engine, run one frame, print detections.

Usage (on Jetson Nano):
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
    source .venv/bin/activate
    python edge/test_trt_hq.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT / "src"))

ENGINE = _ROOT / "models" / "yolo26n_hq.engine"
FRAME  = _ROOT / "data" / "MOT17" / "train" / "MOT17-09-SDP" / "img1" / "000001.jpg"


def main() -> None:
    import cv2
    from benchmark.trt_infer import TrtInfer
    from benchmark.trt_postprocess import decode_detections

    if not ENGINE.exists():
        print(f"Engine not found: {ENGINE}")
        sys.exit(1)
    if not FRAME.exists():
        print(f"Frame not found: {FRAME}")
        sys.exit(1)

    print(f"Engine: {ENGINE.name}")
    print(f"Frame:  {FRAME.name}")

    # Load engine
    infer = TrtInfer(str(ENGINE))
    print(f"Input shape: (1, 3, {infer.input_h}, {infer.input_w})")
    print(f"Output bindings: {infer._output_names}")

    # Read frame
    frame = cv2.imread(str(FRAME))
    print(f"Frame shape: {frame.shape}")

    # Run inference
    raw = infer.infer(frame)

    # Print raw output shapes
    print("\nRaw outputs:")
    for name, arr in raw.items():
        print(f"  {name}: shape={arr.shape}  dtype={arr.dtype}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")

    # Decode detections (person class = 0)
    boxes, scores, cls_ids = decode_detections(
        raw, imgsz=infer.input_h, conf_thres=0.25, iou_thres=0.45,
        target_classes=[0],
    )

    print(f"\nDetections (person, conf>=0.25): {len(boxes)}")
    for i, (box, score, cid) in enumerate(zip(boxes, scores, cls_ids)):
        x1, y1, x2, y2 = box
        print(f"  [{i}] class={cid} conf={score:.3f} "
              f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # Also show all-class count for comparison
    boxes_all, scores_all, cls_all = decode_detections(
        raw, imgsz=infer.input_h, conf_thres=0.25, iou_thres=0.45,
    )
    print(f"\nDetections (all classes, conf>=0.25): {len(boxes_all)}")

    # Low-threshold sweep to find borderline detections
    boxes_low, scores_low, cls_low = decode_detections(
        raw, imgsz=infer.input_h, conf_thres=0.10, iou_thres=0.45,
        target_classes=[0],
    )
    print(f"\nDetections (person, conf>=0.10): {len(boxes_low)}")
    for i, (box, score, cid) in enumerate(zip(boxes_low, scores_low, cls_low)):
        x1, y1, x2, y2 = box
        print(f"  [{i}] class={cid} conf={score:.3f} "
              f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # Reference: .pt FP32 on this frame gives 6 person detections
    print(f"\nExpected ~6 person detections (from .pt FP32 reference)")

    # Timing: 50 iterations (5 warmup + 45 timed) of infer + decode
    import time
    n_warmup, n_timed = 5, 45
    print(f"\nTiming: {n_warmup} warmup + {n_timed} timed iterations...")

    for _ in range(n_warmup):
        r = infer.infer(frame)
        decode_detections(r, imgsz=infer.input_h, conf_thres=0.25,
                          iou_thres=0.45, target_classes=[0])

    t0 = time.time()
    for _ in range(n_timed):
        r = infer.infer(frame)
        decode_detections(r, imgsz=infer.input_h, conf_thres=0.25,
                          iou_thres=0.45, target_classes=[0])
    elapsed = time.time() - t0

    avg_ms = (elapsed / n_timed) * 1000
    fps = n_timed / elapsed
    print(f"  Average: {avg_ms:.1f} ms/frame  ({fps:.1f} FPS)")
    print(f"  Engine:  {ENGINE.name}")

    infer.close()


if __name__ == "__main__":
    main()
