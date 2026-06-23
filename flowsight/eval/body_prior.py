"""Head / upper-body -> foot ground-anchor recovery (recall-lab H1/H10 inference).

Tiling (Cycle 5) helps SMALL people; it does nothing for FOOT-OCCLUDED people in a
dense crowd, whose ground anchor (bbox bottom) is wrong -> matching fails. But the
HEAD is usually visible. So: detect heads (injected callable, like the tiling
detector), then a geometric prior maps each head to a full-body box / foot anchor:

    body_height = k * head_height,  body_width = w_ratio * head_width,
    foot = (head_center_x,  head_top + k * head_height)

Head proposals that don't already overlap a person detection are merged in,
recovering people the body detector missed. The real head detector plugs into the
lab the same way the tiling detector does; here it is mockable for unit tests.
"""
from __future__ import annotations

import numpy as np

from .slice_metrics import iou_matrix

K_DEFAULT = 7.5        # body height ≈ 7–8 head-heights
W_RATIO = 2.0          # body width ≈ 2 × head width


def head_to_body(head_boxes, k=K_DEFAULT, w_ratio=W_RATIO):
    """Head boxes (N,4) -> estimated full-body boxes (N,4), top-aligned to the head."""
    h = np.asarray(head_boxes, float).reshape(-1, 4)
    if not len(h):
        return np.zeros((0, 4))
    cx = (h[:, 0] + h[:, 2]) / 2
    hw = h[:, 2] - h[:, 0]
    hh = h[:, 3] - h[:, 1]
    top = h[:, 1]
    bw = w_ratio * hw
    bh = k * hh
    return np.column_stack([cx - bw / 2, top, cx + bw / 2, top + bh])


def head_to_foot(head_boxes, k=K_DEFAULT):
    """Head boxes (N,4) -> ground foot anchor (N,2) = (head_center_x, head_top + k*head_h)."""
    h = np.asarray(head_boxes, float).reshape(-1, 4)
    if not len(h):
        return np.zeros((0, 2))
    cx = (h[:, 0] + h[:, 2]) / 2
    hh = h[:, 3] - h[:, 1]
    return np.column_stack([cx, h[:, 1] + k * hh])


def merge_head_proposals(person_dets, head_dets, k=K_DEFAULT, w_ratio=W_RATIO,
                         dedup_iou=0.4, score_scale=0.9):
    """Add body proposals from heads not already covered by a person detection.

    person_dets (N,5) [x1,y1,x2,y2,score]; head_dets (M,5). Robust to empty inputs
    (empty heads -> person_dets unchanged; empty persons -> pure proposals).
    Returns merged (P,5)."""
    p = np.asarray(person_dets, float).reshape(-1, 5) if len(person_dets) else np.zeros((0, 5))
    h = np.asarray(head_dets, float).reshape(-1, 5) if len(head_dets) else np.zeros((0, 5))
    if not len(h):
        return p
    body = head_to_body(h[:, :4], k, w_ratio)
    prop = np.column_stack([body, h[:, 4] * score_scale])
    if not len(p):
        return prop
    ious = iou_matrix(prop[:, :4], p[:, :4])
    keep = ious.max(axis=1) < dedup_iou if ious.size else np.ones(len(prop), bool)
    return np.vstack([p, prop[keep]])
