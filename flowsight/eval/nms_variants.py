"""NMS variants for crowd recall — no retraining (recall-lab hypothesis #3).

hard-NMS suppresses overlapping boxes outright, which deletes a real neighbour in
a dense crowd (IoU high between *different* people). Soft-NMS / DIoU-NMS decay or
gate the score instead of deleting, recovering crowd recall. WBF fuses boxes from
several sources (multi-scale / tiling / ensemble).

All numpy, operate on boxes (N,4) xyxy + scores (N,). Return kept indices (or
fused boxes for WBF). Drop-in for the inference-option experiments (cost=infer).
"""
from __future__ import annotations

import numpy as np

from .slice_metrics import iou_matrix


def hard_nms(boxes, scores, iou_thr=0.5):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return np.array(keep, int)


def soft_nms(boxes, scores, iou_thr=0.5, sigma=0.5, method="gaussian", score_thr=0.001):
    """Soft-NMS (Bodla 2017): decay overlapping scores instead of deleting.
    Returns (kept_idx, new_scores_for_kept)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    s = np.asarray(scores, float).reshape(-1).copy()
    idx = np.arange(len(s))
    keep, keep_s = [], []
    while len(idx):
        m = np.argmax(s[idx])
        i = idx[m]
        keep.append(int(i)); keep_s.append(float(s[i]))
        idx = np.delete(idx, m)
        if not len(idx):
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[idx])[0]
        if method == "linear":
            decay = np.where(ious >= iou_thr, 1 - ious, 1.0)
        else:  # gaussian
            decay = np.exp(-(ious ** 2) / sigma)
        s[idx] = s[idx] * decay
        live = s[idx] >= score_thr
        idx = idx[live]
    keep = np.array(keep, int); keep_s = np.array(keep_s, float)
    return keep, keep_s


def _diou(a, b):
    iou = iou_matrix(a, b)
    ca = (a[:, None, :2] + a[:, None, 2:]) / 2
    cb = (b[None, :, :2] + b[None, :, 2:]) / 2
    cdist = np.sum((ca - cb) ** 2, axis=2)
    tl = np.minimum(a[:, None, :2], b[None, :, :2])
    br = np.maximum(a[:, None, 2:], b[None, :, 2:])
    cdiag = np.sum((br - tl) ** 2, axis=2) + 1e-9
    return iou - cdist / cdiag


def diou_nms(boxes, scores, iou_thr=0.5):
    """DIoU-NMS: penalise overlap by centre distance -> keeps close-but-distinct people."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        d = _diou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][d < iou_thr]
    return np.array(keep, int)


def weighted_boxes_fusion(boxes_list, scores_list, iou_thr=0.55):
    """Simplified WBF across sources (multi-scale / tiling / ensemble).
    Clusters boxes by IoU and returns score-weighted averaged boxes + scores."""
    boxes = np.concatenate([np.asarray(b, float).reshape(-1, 4) for b in boxes_list]) \
        if boxes_list else np.zeros((0, 4))
    scores = np.concatenate([np.asarray(s, float).reshape(-1) for s in scores_list]) \
        if scores_list else np.zeros((0,))
    if not len(boxes):
        return np.zeros((0, 4)), np.zeros((0,))
    order = np.argsort(-scores)
    used = np.zeros(len(boxes), bool)
    fb, fs = [], []
    for i in order:
        if used[i]:
            continue
        rest = order[(~used[order]) & (order != i)]
        ious = iou_matrix(boxes[i:i + 1], boxes[rest])[0] if len(rest) else np.array([])
        cluster = [i] + [int(r) for r, v in zip(rest, ious) if v >= iou_thr]
        for c in cluster:
            used[c] = True
        w = scores[cluster]
        fb.append(np.average(boxes[cluster], axis=0, weights=w))
        fs.append(float(w.max()))
    return np.array(fb), np.array(fs)
