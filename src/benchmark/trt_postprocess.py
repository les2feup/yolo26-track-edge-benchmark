"""
Post-processing for YOLO26 raw TensorRT HQ engine outputs.

The HQ engine is cut at the 6 detection-head Conv outputs (3 cv2 + 3 cv3).
YOLO26's one2one head performs DFL decode internally, so the outputs are
already decoded — same format as Hailo DFC post-DFL outputs:

    cv2 (box regression) — shape (1, 4, H, W): ltrb distances in stride units
    cv3 (classification) — shape (1, 80, H, W): raw class logits (sigmoid here)

Tensors are paired by channel count: C=4 → bbox, C>4 → class.
Scales are inferred from the spatial dimensions: 80×80→stride 8, 40×40→16, 20×20→32
(at 640px input; scales adjust proportionally with resolution).

Decoding steps (all on CPU / numpy):
  1. Pair cv2/cv3 tensors by spatial resolution (largest area = P3).
  2. Build anchor grids per scale.
  3. Convert ltrb (stride units) → xyxy pixel coordinates via anchor centres.
  4. Sigmoid on class logits → max-class confidence. Filter by conf_thres.
     Fast path: when target_classes=[0], only channel 0 is sigmoid'd (no argmax).
  5. Per-class greedy NMS (single pass for person-only).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _make_anchor_grid(h: int, w: int, stride: int) -> np.ndarray:
    """Anchor point grid for one feature-map scale.

    Returns (H*W, 2) of (cx, cy) pixel-space centres at stride resolution.
    Cached — shapes and strides are fixed per model, so this builds once per
    (h, w, stride) triple across the full benchmark run.
    """
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    anchors = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    return (anchors + 0.5) * stride


def _decode_scale(
    box_nchw: np.ndarray,   # (1, 4, H, W) — ltrb in stride units
    cls_nchw: np.ndarray,   # (1, C, H, W) — raw class logits
    stride: int,
    person_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one scale from raw TRT Conv outputs into xyxy boxes + scores.

    When person_only is True, only channel 0 is sigmoid'd and no argmax is
    needed — skips ~98% of the classification work for single-class use.
    """
    _, _, H, W = box_nchw.shape
    N = H * W

    # Transpose NCHW → (N, C)
    ltrb     = box_nchw[0].reshape(4, N).T              # (N, 4)

    # Anchor grid → pixel coordinates
    anchors = _make_anchor_grid(H, W, stride)            # (N, 2)

    # ltrb → xyxy: anchor_centre ∓ distance*stride
    x1 = anchors[:, 0] - ltrb[:, 0] * stride
    y1 = anchors[:, 1] - ltrb[:, 1] * stride
    x2 = anchors[:, 0] + ltrb[:, 2] * stride
    y2 = anchors[:, 1] + ltrb[:, 3] * stride
    boxes = np.stack([x1, y1, x2, y2], axis=1)          # (N, 4)

    if person_only:
        # Sigmoid only the person channel (index 0) — no argmax needed.
        logit0 = cls_nchw[0, 0].reshape(N).clip(-88.0, 88.0)
        scores    = 1.0 / (1.0 + np.exp(-logit0))       # (N,)
        class_ids = np.zeros(N, dtype=np.int32)
    else:
        cls_flat  = cls_nchw[0].reshape(-1, N).T        # (N, num_classes)
        logits    = cls_flat.clip(-88.0, 88.0)
        cls_probs = 1.0 / (1.0 + np.exp(-logits))       # (N, C)
        class_ids = cls_probs.argmax(axis=1).astype(np.int32)
        scores    = cls_probs[np.arange(N), class_ids]

    return boxes, scores, class_ids


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS. Returns indices of kept boxes sorted by descending score."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

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
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thresh]

    return np.array(keep, dtype=np.int32)


def decode_detections(
    raw_outputs: dict[str, np.ndarray],
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    target_classes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode YOLO26 TRT HQ engine raw outputs into filtered, NMS-suppressed detections.

    Pairing logic: tensors with C=4 are box regression (cv2, post-DFL ltrb),
    tensors with C>4 are classification (cv3). Scales are matched by spatial
    area — largest area is P3 (stride 8).

    Args:
        raw_outputs:    Dict mapping output tensor name → ndarray (1, C, H, W).
        imgsz:          Model input resolution (assumed square).
        conf_thres:     Minimum confidence score.
        iou_thres:      IoU threshold for NMS.
        target_classes: If set, retain only these class IDs (e.g. [0] for person).

    Returns:
        boxes_xyxy : float32 ndarray (N, 4) — xyxy pixel coordinates
        confs      : float32 ndarray (N,)   — confidence scores
        class_ids  : int32   ndarray (N,)   — class IDs
    """
    # Partition tensors: C=4 → box regression, C>4 → classification
    box_tensors = {}
    cls_tensors = {}

    for name, arr in raw_outputs.items():
        C = arr.shape[1]   # NCHW format
        if C == 4:
            box_tensors[name] = arr
        else:
            cls_tensors[name] = arr

    if not box_tensors or not cls_tensors:
        raise ValueError(
            f"Cannot identify box/class tensors from shapes: "
            f"{ {k: v.shape for k, v in raw_outputs.items()} }"
        )

    if len(box_tensors) != 3 or len(cls_tensors) != 3:
        raise ValueError(
            f"Expected 3 box + 3 class tensors, got {len(box_tensors)} + {len(cls_tensors)}. "
            f"Shapes: { {k: v.shape for k, v in raw_outputs.items()} }"
        )

    # Sort by spatial area descending: P3 (80×80) → P4 (40×40) → P5 (20×20)
    def _area(arr):
        return arr.shape[2] * arr.shape[3]

    box_sorted = sorted(box_tensors.values(), key=_area, reverse=True)
    cls_sorted = sorted(cls_tensors.values(), key=_area, reverse=True)

    # Infer strides from spatial dimensions relative to input resolution
    strides = [imgsz // arr.shape[2] for arr in box_sorted]

    person_only = target_classes == [0]
    all_boxes, all_scores, all_cls = [], [], []

    for box_nchw, cls_nchw, stride in zip(box_sorted, cls_sorted, strides):
        boxes, scores, cls_ids = _decode_scale(box_nchw, cls_nchw, stride, person_only)
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

    _empty = np.empty((0, 4), dtype=np.float32)
    if len(boxes) == 0:
        return _empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

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

    return boxes, scores, cls
