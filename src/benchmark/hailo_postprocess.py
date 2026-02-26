"""
Post-processing for YOLO26 raw HEF output tensors.

The DFC compiler fuses the DFL decode into the HEF, so the 6 output tensors
are already decoded feature maps — not raw DFL logits:

    bbox tensors  — shape (H, W, 4):  ltrb distances in stride units (DFL fused by DFC)
    class tensors — shape (H, W, C):  raw logits (sigmoid NOT applied — done here)

Tensors are paired by channel count: C=4 → bbox, C>4 → class.
Scales are inferred from the spatial dimensions: 80×80→stride 8, 40×40→16, 20×20→32.

Decoding steps (all on CPU / numpy):
  1. Split tensors into bbox and class groups; sort both by spatial area (descending).
  2. Build anchor grids per scale.
  3. Convert ltrb (stride units) → xyxy pixel coordinates via anchor centres.
  4. Take max-class score as confidence. Filter by conf_thres.
  5. Greedy per-class NMS.
"""

from __future__ import annotations

import numpy as np


def _make_anchor_grid(h: int, w: int, stride: int) -> np.ndarray:
    """
    Anchor point grid for one feature-map scale.
    Returns (H*W, 2) of (cx, cy) pixel-space centres at stride resolution.
    """
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    anchors = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    return (anchors + 0.5) * stride


def _decode_scale(
    bbox_hwc: np.ndarray,   # (H, W, 4)  — ltrb in stride units
    cls_hwc:  np.ndarray,   # (H, W, C)  — class scores, sigmoid already applied
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one scale into xyxy boxes, confidence scores, and class IDs."""
    H, W, _ = bbox_hwc.shape
    N = H * W

    anchors = _make_anchor_grid(H, W, stride)          # (N, 2)
    ltrb    = bbox_hwc.reshape(N, 4) * stride          # convert stride units → pixels

    x1 = anchors[:, 0] - ltrb[:, 0]
    y1 = anchors[:, 1] - ltrb[:, 1]
    x2 = anchors[:, 0] + ltrb[:, 2]
    y2 = anchors[:, 1] + ltrb[:, 3]
    boxes = np.stack([x1, y1, x2, y2], axis=1)        # (N, 4)

    # Class outputs are raw logits (not sigmoid'd by DFC) — apply sigmoid now.
    # np.exp overflows for large negative values; clip to [-88, 88] (float32 safe range).
    logits    = cls_hwc.reshape(N, -1).clip(-88.0, 88.0)
    cls_probs = 1.0 / (1.0 + np.exp(-logits))                  # (N, C)
    class_ids = cls_probs.argmax(axis=1).astype(np.int32)
    scores    = cls_probs[np.arange(N), class_ids]    # (N,)

    return boxes, scores, class_ids


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS. Returns indices of kept boxes sorted by descending score."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    order  = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1  = np.maximum(x1[i], x1[rest])
        iy1  = np.maximum(y1[i], y1[rest])
        ix2  = np.minimum(x2[i], x2[rest])
        iy2  = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou  = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thresh]

    return np.array(keep, dtype=np.int32)


def decode_detections(
    raw_outputs: dict[str, np.ndarray],
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    target_classes: list[int] | None = None,
) -> tuple[list[list[float]], list[float], list[int]]:
    """
    Decode YOLO26 HEF raw outputs into filtered, NMS-suppressed detections.

    Pairing logic: tensors with C=4 are bbox (ltrb), tensors with C>4 are class.
    Scales are matched by spatial size — largest area is P3 (stride 8).

    Args:
        raw_outputs:    Dict from HailoInfer.infer() — {tensor_name: ndarray}.
        imgsz:          Model input resolution (assumed square, e.g. 640).
        conf_thres:     Minimum confidence score to retain a detection.
        iou_thres:      IoU threshold for NMS suppression.
        target_classes: If set, discard all other class IDs (e.g. [0] for person).

    Returns:
        boxes_xyxy : list of [x1, y1, x2, y2] in pixel coordinates
        confs      : list of float confidence scores
        class_ids  : list of int class IDs
    """
    # Partition tensors into bbox (C=4) and class (C>4) groups.
    bbox_tensors = {k: v for k, v in raw_outputs.items() if v.shape[-1] == 4}
    cls_tensors  = {k: v for k, v in raw_outputs.items() if v.shape[-1] > 4}

    if not bbox_tensors or not cls_tensors:
        raise ValueError(
            f"Cannot identify bbox/class tensors from shapes: "
            f"{ {k: v.shape for k, v in raw_outputs.items()} }"
        )

    # Sort both groups by spatial area descending: P3 (80×80) first, P5 (20×20) last.
    def _area(arr): return arr.shape[0] * arr.shape[1]

    bbox_sorted = sorted(bbox_tensors.values(), key=_area, reverse=True)
    cls_sorted  = sorted(cls_tensors.values(),  key=_area, reverse=True)
    strides     = [8, 16, 32]   # P3, P4, P5 for 640-px input

    all_boxes, all_scores, all_cls = [], [], []

    for bbox_hwc, cls_hwc, stride in zip(bbox_sorted, cls_sorted, strides):
        boxes, scores, cls_ids = _decode_scale(bbox_hwc, cls_hwc, stride)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_cls.append(cls_ids)

    boxes  = np.concatenate(all_boxes,  axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls    = np.concatenate(all_cls,    axis=0)

    # Confidence + class filter
    mask = scores >= conf_thres
    if target_classes is not None:
        mask &= np.isin(cls, target_classes)

    boxes, scores, cls = boxes[mask], scores[mask], cls[mask]

    if len(boxes) == 0:
        return [], [], []

    # Per-class greedy NMS
    keep_idx = []
    for cid in np.unique(cls):
        c_mask = cls == cid
        k = _nms(boxes[c_mask], scores[c_mask], iou_thres)
        keep_idx.extend(np.where(c_mask)[0][k].tolist())

    boxes  = boxes[keep_idx]
    scores = scores[keep_idx]
    cls    = cls[keep_idx]

    # Clip to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, imgsz)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, imgsz)

    return boxes.tolist(), scores.tolist(), cls.tolist()
