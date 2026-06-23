"""Ground-anchor projection experiment (recall lab Cycle 8).

Cycle 7 showed the bottleneck is the bbox-foot -> ground projection, not detection.
This module compares WHICH bbox feature, projected through the real camera
calibration, lands closest to the true ground position — instead of always using
the foot (bbox bottom), which is wrong when feet are occluded.

A single parameter `alpha` selects the vertical anchor on the bbox:
    anchor_y = y1 + alpha * (y2 - y1)   (alpha=1 foot/bottom, 0 head/top, 0.5 centre,
                                         >1 extrapolated below the box for occluded feet)
`calibrate_alpha` FITS alpha on train data (the "fine-tune") to minimise world
localisation error; evaluation is on held-out frames (no leakage). The detector is
isolated by using GT boxes, so this measures the projection/anchor accuracy alone.

cam must expose `to_ground(uv)->(N,2) world metres` (flowsight.geometry.wildtrack
.WildtrackCamera). Pure numpy otherwise.
"""
from __future__ import annotations

import numpy as np


def bbox_anchor(boxes, alpha=1.0):
    """(N,4) xyxy -> (N,2) anchor pixels at vertical fraction `alpha`."""
    b = np.atleast_2d(np.asarray(boxes, float))
    if not len(b):
        return np.zeros((0, 2))
    cx = (b[:, 0] + b[:, 2]) / 2.0
    y = b[:, 1] + alpha * (b[:, 3] - b[:, 1])
    return np.column_stack([cx, y])


def project_anchor(cam, boxes, alpha=1.0):
    """bbox -> anchor(alpha) px -> world (m)."""
    a = bbox_anchor(boxes, alpha)
    if not len(a):
        return np.zeros((0, 2))
    return cam.to_ground(a)


def loc_errors(cam, boxes, gt_world, alpha=1.0):
    """Per-box ground localisation error (m): ||project(box,alpha) - gt_world||.
    boxes[i] is paired with gt_world[i]."""
    w = project_anchor(cam, boxes, alpha)
    g = np.atleast_2d(np.asarray(gt_world, float))
    n = min(len(w), len(g))
    if n == 0:
        return np.zeros(0)
    return np.linalg.norm(w[:n] - g[:n], axis=1)


def median_err(cam, boxes, gt_world, alpha=1.0):
    e = loc_errors(cam, boxes, gt_world, alpha)
    return float(np.median(e)) if len(e) else float("nan")


def calibrate_alpha(cam, boxes, gt_world, grid=None):
    """FIT the anchor fraction alpha minimising median world error (the 'fine-tune').
    Returns (best_alpha, best_median_err)."""
    grid = np.linspace(0.5, 1.5, 41) if grid is None else np.asarray(grid, float)
    errs = [median_err(cam, boxes, gt_world, a) for a in grid]
    i = int(np.nanargmin(errs))
    return float(grid[i]), float(errs[i])


def occlude_boxes(boxes, frac=0.2):
    """Simulate foot-occlusion (modal/truncated boxes): raise the bbox bottom by
    `frac` of its height, so the bottom is no longer the true feet."""
    b = np.atleast_2d(np.asarray(boxes, float)).copy()
    if not len(b):
        return b
    b[:, 3] = b[:, 3] - frac * (b[:, 3] - b[:, 1])
    return b


def head_anchor(boxes):
    """(N,4) xyxy -> (N,2) head pixel = top-centre of the bbox."""
    b = np.atleast_2d(np.asarray(boxes, float))
    if not len(b):
        return np.zeros((0, 2))
    cx = (b[:, 0] + b[:, 2]) / 2.0
    return np.column_stack([cx, b[:, 1]])


def project_head(cam, boxes, height_m=1.7):
    """Head pixel -> intersect ray with the Z=height_m plane -> world (X,Y) (m).

    Height-prior vertical-segment anchor: a standing person's head sits ~height_m
    above their ground contact and shares its (X,Y), so the head ray meeting the
    Z=height_m plane recovers the ground position. Unlike the bbox fraction alpha it
    needs NO per-camera fit and is robust to FOOT OCCLUSION (head stays visible).
    cam must expose ``to_plane`` (flowsight.geometry.wildtrack.WildtrackCamera).
    (Zhang & Ye 2024: head localisation beats ankle; Niu 2021 vertical segment.)"""
    h = head_anchor(boxes)
    if not len(h):
        return np.zeros((0, 2))
    return cam.to_plane(h, height_m)


def head_loc_errors(cam, boxes, gt_world, height_m=1.7):
    """Per-box ground error (m) of the head-anchor projection vs gt_world[i]."""
    w = project_head(cam, boxes, height_m)
    g = np.atleast_2d(np.asarray(gt_world, float))
    n = min(len(w), len(g))
    if n == 0:
        return np.zeros(0)
    return np.linalg.norm(w[:n] - g[:n], axis=1)


def head_extrapolated_anchor(boxes_full_h, trunc_boxes):
    """Given a truncated box + a known full-body pixel height, estimate the foot
    pixel from the (visible) head/top: foot_y = top + full_height. Returns (N,2)."""
    t = np.atleast_2d(np.asarray(trunc_boxes, float))
    fh = np.atleast_2d(np.asarray(boxes_full_h, float))
    if not len(t):
        return np.zeros((0, 2))
    cx = (t[:, 0] + t[:, 2]) / 2.0
    full_h = fh[:, 3] - fh[:, 1]
    return np.column_stack([cx, t[:, 1] + full_h])
