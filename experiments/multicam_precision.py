"""Multi-camera BEV PRECISION: greedy fuse vs world-NMS x (calib / head) anchor (Cycle 13).

Cycle 12 found multi-cam calibrated fusion lifts recall (single-view @2m 0.34 -> 4-cam 0.54)
but HALVES precision (0.189 -> 0.091): the same person seen by several cameras projects to
points ~2 m apart (calibrated-anchor error) > the greedy assoc radius, so the duplicates are
never merged and inflate the false-positive count. sigma-gating (Cycle 11 H3) did NOT fix it
on real data because real-detector FPs are normal-sigma, not near-horizon noise.

This runner adds:
  * world-space confidence NMS (flowsight.geometry.multicam.world_nms) -- keep the highest-
    confidence detection, suppress all others within `radius` m regardless of source camera,
    collapsing cross-view duplicates -> the precision fix.
  * a per-detection HEAD anchor (project_head) alternative to the per-camera calibrated alpha.

It compares, on real WILDTRACK, the Cycle-12 greedy baseline vs world-NMS (radius sweep) for
both anchors, reporting recall AND precision @1m/2m against the full-frame GT.

Testable core `eval_fused` takes detection arrays (mockable, no YOLO). `main` wires ultralytics.
Run on Colab after a clean `git clone` (no base64 hacks needed):
    PYTHONPATH=. python experiments/multicam_precision.py --root /content/WTx/Wildtrack_dataset --n 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from flowsight.eval.anchor_proj import bbox_anchor, calibrate_alpha, project_head
from flowsight.geometry.multicam import world_nms
from flowsight.geometry.wildtrack import load_camera, match_to_gt, positionid_to_world
from experiments.anchor_lab import NAMES, _cam, load_pairs

# C3/C6/C7 zips are historically corrupt; use the four clean views.
VIEWS = [("C1", 0), ("C2", 1), ("C4", 3), ("C5", 4)]


def gt_full(af):
    """All people in a frame -> world (M,2) (full-frame GT, not per-view-visible)."""
    pid = [d["positionID"] for d in json.load(open(af)) if "positionID" in d]
    return positionid_to_world(np.array(pid)) if pid else np.zeros((0, 2))


def auto_calib(root, frames):
    """Pick the calibration that minimises foot-anchor world error on a few frames."""
    best = (None, 1e9)
    for nm in NAMES:
        c = _cam(root, nm)
        e = []
        for b, g in frames[:5]:
            w = c.to_ground(bbox_anchor(b, 1.0))
            n = min(len(w), len(g))
            if n:
                e += list(np.linalg.norm(w[:n] - g[:n], axis=1))
        me = float(np.median(e)) if e else 1e9
        if me < best[1]:
            best = (nm, me)
    return _cam(root, best[0])


def _project(cam, boxes, anchor, alpha):
    """boxes (N,>=4) -> world (N,2) via calibrated vertical fraction or height-prior head."""
    if not len(boxes):
        return np.zeros((0, 2))
    if anchor == "head":
        return project_head(cam, boxes[:, :4], 1.7)
    return cam.to_ground(bbox_anchor(boxes[:, :4], alpha))


def _greedy(items, r0):
    """Cycle-12 baseline: unweighted centroid clustering that never merges two dets from the
    same camera. items = list of (view_name, world_xy)."""
    clusters = []
    for nm, p in items:
        best, bd = None, r0
        for c in clusters:
            d = float(np.linalg.norm(c["cen"] - p))
            if d <= bd and nm not in c["v"]:
                best, bd = c, d
        if best is None:
            clusters.append({"pts": [p], "v": {nm}, "cen": p.copy()})
        else:
            best["pts"].append(p)
            best["v"].add(nm)
            best["cen"] = np.mean(best["pts"], axis=0)
    return np.array([c["cen"] for c in clusters]) if clusters else np.zeros((0, 2))


def eval_fused(cams, astar, dets_by_view, gtw_list, anchor="calib", method="nms",
               radius=1.5, assoc=1.5):
    """Fuse all cameras' detections per frame and score vs full GT.

    cams: {view: WildtrackCamera}; astar: {view: alpha}; dets_by_view: {view: [ (Ni,5) per frame ]}
    (last column = confidence; if absent, 1.0); gtw_list: [ (Mi,2) full GT world per frame ].
    method 'greedy' = same-view-aware unweighted cluster (Cycle 12); 'nms' = world_nms (Cycle 13).
    Returns {'@1': (recall, precision), '@2': (recall, precision)}."""
    out = {}
    for rad in (1.0, 2.0):
        tp = fn = fp = 0
        for i in range(len(gtw_list)):
            g = gtw_list[i]
            if not len(g):
                continue
            items = []
            for vw, _vi in VIEWS:
                det = np.atleast_2d(np.asarray(dets_by_view[vw][i], float))
                if not len(det):
                    continue
                w = _project(cams[vw], det, anchor, astar[vw])
                s = det[:, 4] if det.shape[1] > 4 else np.ones(len(w))
                for p, sv in zip(w, s):
                    items.append((vw, p, float(sv)))
            if method == "nms":
                P = np.array([p for _n, p, _s in items]) if items else np.zeros((0, 2))
                S = np.array([s for _n, _p, s in items]) if items else np.zeros(0)
                fused = P[world_nms(P, S, radius)] if len(P) else np.zeros((0, 2))
            else:
                fused = _greedy([(n, p) for n, p, _s in items], assoc)
            m = match_to_gt(fused, g, rad)
            tp += m["tp"]; fn += m["fn"]; fp += m["fp"]
        out["@%d" % int(rad)] = (tp / (tp + fn + 1e-9), tp / (tp + fp + 1e-9))
    return out


def main(a):
    allf = sorted(glob.glob(os.path.join(a.root, "annotations_positions", "*.json")))[:a.n]
    fids = [os.path.splitext(os.path.basename(p))[0] for p in allf]
    k = max(1, a.n // 2)
    test_fid = fids[k:]
    gtw = [gt_full(f) for f in allf[k:]]

    cams, astar = {}, {}
    for vw, vi in VIEWS:
        fr = load_pairs(a.root, vi, a.n)
        cams[vw] = auto_calib(a.root, fr[:5])
        Bt = np.vstack([b for b, g in fr[:k] if len(b)])
        Gt = np.vstack([g for b, g in fr[:k] if len(b)])
        astar[vw] = calibrate_alpha(cams[vw], Bt, Gt)[0]

    from ultralytics import YOLO
    mdl = YOLO(a.weights)
    dets = {vw: [] for vw, _ in VIEWS}
    for vw, _vi in VIEWS:
        for fid in test_fid:
            ip = os.path.join(a.root, "Image_subsets", vw, fid + ".png")
            if not os.path.exists(ip):
                dets[vw].append(np.zeros((0, 5))); continue
            r = mdl.predict(ip, classes=[0], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                dets[vw].append(np.zeros((0, 5))); continue
            dets[vw].append(np.column_stack([r.boxes.xyxy.cpu().numpy(),
                                             r.boxes.conf.cpu().numpy()]))

    print("=== MULTICAM_PRECISION (real) cams=%s astar=%s ntest=%d ===" %
          ([v for v, _ in VIEWS], {x: round(y, 2) for x, y in astar.items()}, len(test_fid)),
          flush=True)
    g = eval_fused(cams, astar, dets, gtw, "calib", "greedy")
    print("greedy(calib,Cycle12) @1m r%.3f p%.3f | @2m r%.3f p%.3f" % (*g["@1"], *g["@2"]), flush=True)
    for anchor in ("calib", "head"):
        for rad in (1.0, 1.5, 2.0):
            R = eval_fused(cams, astar, dets, gtw, anchor, "nms", rad)
            print("nms-%-5s r%.1f      @1m r%.3f p%.3f | @2m r%.3f p%.3f" %
                  (anchor, rad, *R["@1"], *R["@2"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/WTx/Wildtrack_dataset")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1280)
    main(ap.parse_args())
