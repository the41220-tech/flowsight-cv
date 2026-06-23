"""Detector-injectable tiling / multi-scale inference + WBF (recall-lab H5).

The detector is injected as a callable, so ANY model (in-domain, head, pose, …)
can be A/B'd in the recall lab without retraining — this is the experiment
infrastructure for the Cycle-3 finding "recall is a detector-domain problem".

Contract:
    detect(region_xyxy) -> (K,5) [x1,y1,x2,y2,score] in LOCAL coords
                           (relative to the region's top-left).
On Colab the adapter is e.g.
    detect = lambda r: _to_local(yolo(img[r[1]:r[3], r[0]:r[2]]) , r)
Here `detect` is mocked for unit tests (no GPU / no image).

`run_tiled` tiles the frame, remaps each tile's boxes to global coords, and fuses
overlapping cross-tile boxes with Weighted Box Fusion.
"""
from __future__ import annotations

import numpy as np

from .nms_variants import weighted_boxes_fusion


def tile_grid(W, H, slice=512, overlap=0.2):
    """Overlapping slice-sized tiles covering WxH. Returns sorted unique (x0,y0,x1,y1)."""
    step = max(1, int(slice * (1 - overlap)))
    xs = list(range(0, max(1, W), step))
    ys = list(range(0, max(1, H), step))
    tiles = set()
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + slice, W)
            y1 = min(y0 + slice, H)
            x0a = max(0, x1 - slice)          # keep full slice size at right/bottom edges
            y0a = max(0, y1 - slice)
            tiles.add((x0a, y0a, x1, y1))
    return sorted(tiles)


def run_tiled(detect, img_wh, slice=512, overlap=0.2, wbf_iou=0.55, fuse=True):
    """detect(region)->local (K,5) ; returns fused global (M,5)."""
    W, H = img_wh
    boxes_list, scores_list = [], []
    for (x0, y0, x1, y1) in tile_grid(W, H, slice, overlap):
        loc = np.asarray(detect((x0, y0, x1, y1)), float).reshape(-1, 5)
        if not len(loc):
            continue
        g = loc.copy()
        g[:, [0, 2]] += x0
        g[:, [1, 3]] += y0
        boxes_list.append(g[:, :4])
        scores_list.append(g[:, 4])
    if not boxes_list:
        return np.zeros((0, 5))
    if fuse:
        fb, fs = weighted_boxes_fusion(boxes_list, scores_list, wbf_iou)
        return np.column_stack([fb, fs]) if len(fb) else np.zeros((0, 5))
    b = np.concatenate(boxes_list)
    s = np.concatenate(scores_list)
    return np.column_stack([b, s])


def tiled_dataset(detect_for, img_whs, **kw):
    """Build per-image tiled preds for a whole dataset.
    detect_for(i) -> a detect(region) callable for image i; img_whs[i]=(W,H)."""
    return [run_tiled(detect_for(i), img_whs[i], **kw) for i in range(len(img_whs))]


def to_local(global_boxes, region):
    """Helper for real adapters: shift global (K,5) boxes into a region's local frame."""
    g = np.asarray(global_boxes, float).reshape(-1, 5).copy()
    x0, y0 = region[0], region[1]
    g[:, [0, 2]] -= x0
    g[:, [1, 3]] -= y0
    return g
