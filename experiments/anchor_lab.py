"""Ground-anchor experiment on real WILDTRACK (recall lab Cycle 8).

Isolates the projection/anchor question from detection by using GT 2D boxes paired
with GT world positions (positionID). For a camera view it: auto-picks the matching
calibration (min foot-error, removes the folder<->calib ambiguity), fits the anchor
fraction alpha on TRAIN frames, and reports world localisation error for
foot(alpha=1) / centre(0.5) / calibrated(alpha*) on held-out TEST frames, overall
and on the occluded slice. Also a foot-occlusion simulation: foot vs head-
extrapolated anchor on truncated boxes (shows the head's value when feet are hidden).

Run on Colab (real calib needs the FIXED wildtrack.py):
    PYTHONPATH=. python experiments/anchor_lab.py --root /content/WTx/Wildtrack_dataset --view C1 --n 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from flowsight.eval.anchor_proj import (calibrate_alpha, head_extrapolated_anchor,
                                        loc_errors, median_err, occlude_boxes)
from flowsight.eval.slice_metrics import iou_matrix
from flowsight.geometry.wildtrack import load_camera, positionid_to_world

NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]


def _cam(root, name):
    return load_camera(os.path.join(root, "calibrations", "intrinsic_zero", "intr_%s.xml" % name),
                       os.path.join(root, "calibrations", "extrinsic", "extr_%s.xml" % name))


def load_pairs(root, view_idx, n):
    """List of (boxes (M,4), gt_world (M,2)) for persons visible in the view."""
    out = []
    for af in sorted(glob.glob(os.path.join(root, "annotations_positions", "*.json")))[:n]:
        boxes, pids = [], []
        for d in json.load(open(af)):
            for v in d.get("views", []):
                if v.get("viewNum") == view_idx and v.get("xmax", -1) != -1:
                    boxes.append([v["xmin"], v["ymin"], v["xmax"], v["ymax"]])
                    pids.append(d["positionID"])
        b = np.array(boxes, float).reshape(-1, 4)
        g = positionid_to_world(np.array(pids)) if pids else np.zeros((0, 2))
        out.append((b, g))
    return out


def _occ(boxes, thr=0.3):
    if not len(boxes):
        return np.zeros(0, bool)
    m = iou_matrix(boxes, boxes)
    np.fill_diagonal(m, 0.0)
    return (m > thr).any(axis=1)


def _stack(frames):
    bs = [b for b, g in frames if len(b)]
    gs = [g for b, g in frames if len(b)]
    if not bs:
        return np.zeros((0, 4)), np.zeros((0, 2))
    return np.vstack(bs), np.vstack(gs)


def main(a):
    view_idx = int(a.view[1:]) - 1
    frames = load_pairs(a.root, view_idx, a.n)
    cams = {n: _cam(a.root, n) for n in NAMES}

    def foot_err(cam, fr):
        e = []
        for b, g in fr:
            e += list(loc_errors(cam, b, g, 1.0))
        return float(np.median(e)) if e else 1e9
    pick = min(NAMES, key=lambda n: foot_err(cams[n], frames[:5]))
    cam = cams[pick]
    print("[anchor] view=%s -> calib=%s (auto, min foot-err) frames=%d" %
          (a.view, pick, len(frames)), flush=True)

    k = max(1, len(frames) // 2)
    Bt, Gt = _stack(frames[:k])
    Bv, Gv = _stack(frames[k:])
    occ = _occ_all = np.concatenate([_occ(b) for b, g in frames[k:] if len(b)]) \
        if any(len(b) for b, g in frames[k:]) else np.zeros(0, bool)
    a_star, _ = calibrate_alpha(cam, Bt, Gt)
    print("=== ANCHOR_RESULT (test frames, GT boxes) ===", flush=True)
    print("calibrated alpha* = %.3f (fit on train)" % a_star, flush=True)
    for name, al in [("foot(1.0)", 1.0), ("center(0.5)", 0.5), ("calib(%.2f)" % a_star, a_star)]:
        eo = median_err(cam, Bv, Gv, al)
        ev = loc_errors(cam, Bv, Gv, al)
        eocc = float(np.median(ev[occ])) if occ.any() and len(ev) == len(occ) else float("nan")
        print("  %-12s loc_err median %.3f m | occluded %.3f m" % (name, eo, eocc), flush=True)

    # foot-occlusion simulation: truncate feet, foot vs head-extrapolated
    Btr = occlude_boxes(Bv, 0.25)
    foot_tr = median_err(cam, Btr, Gv, 1.0)
    head_px = head_extrapolated_anchor(Bv, Btr)
    hw = cam.to_ground(head_px) if len(head_px) else np.zeros((0, 2))
    he = float(np.median(np.linalg.norm(hw[:len(Gv)] - Gv[:len(hw)], axis=1))) if len(hw) else float("nan")
    print("[occlusion-sim feet truncated 25%%] foot-anchor %.3f m  vs  head-extrapolated %.3f m"
          % (foot_tr, he), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/WTx/Wildtrack_dataset")
    ap.add_argument("--view", default="C1")
    ap.add_argument("--n", type=int, default=40)
    main(ap.parse_args())
