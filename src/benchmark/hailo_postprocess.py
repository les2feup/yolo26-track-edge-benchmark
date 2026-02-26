"""
Post-processing for YOLO26 raw HEF output tensors.

The HEF is compiled without fused NMS, so the 6 output tensors are raw
feature maps from the detection head's two branches at three scales:

    cv2.{0,1,2}  — DFL bbox regression:  shape (H, W, 4*reg_max)
    cv3.{0,1,2}  — class logits:          shape (H, W, num_classes)

Scales correspond to P3 (80×80), P4 (40×40), P5 (20×20) for a 640-px input.

Decoding steps (all on CPU / numpy):
  1. Pair cv2 and cv3 tensors per scale.
  2. Softmax over the DFL dimension of cv2, then dot with [0..reg_max-1]
     to get expected ltrb distances from each anchor point.
  3. Convert ltrb → xyxy in normalised [0,1] pixel space using anchor grids.
  4. Sigmoid on cv3 → class probabilities. Multiply by best-class score → conf.
  5. Threshold by conf_thres, then batched NMS (torchvision-free implementation).
"""

from __future__ import annotations

import numpy as np


# YOLO26 DFL regularisation maximum — matches the compiled model.
_REG_MAX = 16
_REG_PROJ = np.arange(_REG_MAX, dtype=np.float32)   # [0, 1, ..., 15]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_anchor_grid(h: int, w: int, stride: int) -> np.ndarray:
    """
    Anchor point grid for one feature-map scale.

    Returns (H*W, 2) array of (cx, cy) pixel-space anchor centres,
    offset by 0.5 to place the point at the cell centre.
    """
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    anchors = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    return (anchors + 0.5) * stride   # pixel coordinates


def _decode_scale(
    cv2_hwc: np.ndarray,   # (H, W, 4*reg_max)  — DFL bbox branch
    cv3_hwc: np.ndarray,   # (H, W, num_classes) — class branch
    stride: int,
    imgsz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode one feature-map scale into boxes and scores.

    Returns:
        boxes_xyxy : (N, 4) float32, pixel coordinates in [0, imgsz]
        scores     : (N,)   float32, max-class confidence
        class_ids  : (N,)   int32
    """
    H, W, _ = cv2_hwc.shape
    N = H * W

    anchors = _make_anchor_grid(H, W, stride)   # (N, 2) cx, cy

    # DFL decode: softmax → expected value → ltrb distances (in pixels)
    dfl = cv2_hwc.reshape(N, 4, _REG_MAX)       # (N, 4, reg_max)
    dfl = _softmax(dfl, axis=-1)
    ltrb = (dfl * _REG_PROJ).sum(axis=-1) * stride  # (N, 4) in pixels

    # ltrb → xyxy
    x1 = anchors[:, 0] - ltrb[:, 0]
    y1 = anchors[:, 1] - ltrb[:, 1]
    x2 = anchors[:, 0] + ltrb[:, 2]
    y2 = anchors[:, 1] + ltrb[:, 3]
    boxes = np.stack([x1, y1, x2, y2], axis=1)  # (N, 4)

    # Class scores: sigmoid → max class
    cls_probs = _sigmoid(cv3_hwc.reshape(N, -1))  # (N, num_classes)
    class_ids = cls_probs.argmax(axis=1).astype(np.int32)
    scores = cls_probs[np.arange(N), class_ids]   # (N,)

    return boxes, scores, class_ids


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    Greedy NMS. Returns indices of kept boxes, sorted by descending score.
    """
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
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
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

    Args:
        raw_outputs:    Dict from HailoInfer.infer() — {tensor_name: ndarray}.
        imgsz:          Model input resolution (assumed square).
        conf_thres:     Minimum confidence to keep a detection.
        iou_thres:      IoU threshold for NMS.
        target_classes: If set, keep only these class IDs (e.g. [0] for person).

    Returns:
        boxes_xyxy : list of [x1, y1, x2, y2] in pixel coordinates
        confs      : list of float confidence scores
        class_ids  : list of int class IDs
    """
    # Identify cv2/cv3 pairs by matching tensor names at each scale index.
    # Expected names: one2one_cv2.0 / one2one_cv3.0, etc.
    # Sort by name so indices 0,1,2 map to strides 8,16,32 (P3→P5).
    cv2_names = sorted(k for k in raw_outputs if "cv2" in k)
    cv3_names = sorted(k for k in raw_outputs if "cv3" in k)

    strides = [8, 16, 32]   # P3, P4, P5 for 640-px input

    all_boxes, all_scores, all_cls = [], [], []

    for cv2_name, cv3_name, stride in zip(cv2_names, cv3_names, strides):
        boxes, scores, cls_ids = _decode_scale(
            raw_outputs[cv2_name],
            raw_outputs[cv3_name],
            stride=stride,
            imgsz=imgsz,
        )
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_cls.append(cls_ids)

    boxes  = np.concatenate(all_boxes,  axis=0)   # (total_anchors, 4)
    scores = np.concatenate(all_scores, axis=0)   # (total_anchors,)
    cls    = np.concatenate(all_cls,    axis=0)   # (total_anchors,)

    # Confidence threshold filter
    mask = scores >= conf_thres
    if target_classes is not None:
        mask &= np.isin(cls, target_classes)

    boxes, scores, cls = boxes[mask], scores[mask], cls[mask]

    if len(boxes) == 0:
        return [], [], []

    # Per-class NMS
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
