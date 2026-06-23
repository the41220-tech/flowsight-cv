"""Operationalised calibrated-anchor BEV recall (recall lab Cycle 9).

Cycle 8 found the bbox-bottom (foot) anchor is ~8 m off; a data-calibrated vertical
fraction alpha* cuts it to ~2 m. This puts that into the END-TO-END pipeline with a
REAL detector: fit alpha* per camera on TRAIN frames (GT boxes <-> GT world), then on
held-out TEST frames run the detector, project each box's alpha-anchor to the ground,
and measure BEV recall (match to GT world @1m/2m) for foot(alpha=1) vs calibrated(alpha*).

Testable core `bev_recall` takes detection arrays (mockable, no YOLO). `main` wires
ultralytics YOLO. Run on Colab:
    PYTHONPATH=. python experiments/recall_bev.py --root /content/WTx/Wildtrack_dataset --view C1 --n 40
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from flowsight.eval.anchor_proj import calibrate_alpha, loc_errors, project_anchor
from flowsight.geometry.wildtrack import match_to_gt
from experiments.anchor_lab import NAMES, _cam, load_pairs


def bev_recall(cam, det_list, gtw_list, alpha, radius=1.0):
    """det_list[i] = (Ni,5|4) detector boxes; gtw_list[i] = (Mi,2) GT world. Project
    each box's alpha-anchor -> ground -> match to GT world within `radius`."""
    tp = fn = fp = 0
    errs = []
    for det, g in zip(det_list, gtw_list):
        if not len(g):
            continue
        det = np.atleast_2d(np.asarray(det, float))
        w = project_anchor(cam, det[:, :4], alpha) if len(det) else np.zeros((0, 2))
        m = match_to_gt(w, g, radius)
        tp += m["tp"]; fn += m["fn"]; fp += m["fp"]
        if m["mean_loc_err_m"] is not None:
            errs.append(m["mean_loc_err_m"])
    return {"recall": tp / (tp + fn + 1e-9), "loc_err": float(np.mean(errs)) if errs else float("nan"),
            "tp": tp, "fn": fn, "fp": fp}


def auto_calib(root, frames):
    cams = {n: _cam(root, n) for n in NAMES}

    def fe(cam):
        e = []
        for b, g in frames[:5]:
            e += list(loc_errors(cam, b, g, 1.0))
        return float(np.median(e)) if e else 1e9
    pick = min(NAMES, key=lambda n: fe(cams[n]))
    return pick, cams[pick]


def main(a):
    view_idx = int(a.view[1:]) - 1
    frames = load_pairs(a.root, view_idx, a.n)
    fids = [os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(a.root, "annotations_positions", "*.json")))[:a.n]]
    pick, cam = auto_calib(a.root, frames)
    k = max(1, len(frames) // 2)
    Bt = np.vstack([b for b, g in frames[:k] if len(b)])
    Gt = np.vstack([g for b, g in frames[:k] if len(b)])
    a_star, fit_err = calibrate_alpha(cam, Bt, Gt)
    test = frames[k:]; test_fids = fids[k:]
    gtw_test = [g for b, g in test]
    print("[bev] view=%s calib=%s alpha*=%.3f (fit_err %.2fm) train=%d test=%d" %
          (a.view, pick, a_star, fit_err, k, len(test)), flush=True)

    from ultralytics import YOLO
    import cv2
    model = YOLO(a.weights)
    dets = []
    for fid in test_fids:
        ip = os.path.join(a.root, "Image_subsets", a.view, fid + ".png")
        if not os.path.exists(ip):
            dets.append(np.zeros((0, 5))); continue
        r = model.predict(ip, classes=[0], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            dets.append(np.zeros((0, 5))); continue
        b = r.boxes.xyxy.cpu().numpy(); s = r.boxes.conf.cpu().numpy()
        dets.append(np.column_stack([b, s]))

    print("=== BEV_RECALL_RESULT (real detector boxes) ===", flush=True)
    for r in (1.0, 2.0):
        foot = bev_recall(cam, dets, gtw_test, 1.0, r)
        cal = bev_recall(cam, dets, gtw_test, a_star, r)
        print("  @%.0fm  foot recall %.3f (locerr %.2f)  |  calib(a=%.2f) recall %.3f (locerr %.2f)  Δ%+.3f"
              % (r, foot["recall"], foot["loc_err"], a_star, cal["recall"], cal["loc_err"],
                 cal["recall"] - foot["recall"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/WTx/Wildtrack_dataset")
    ap.add_argument("--view", default="C1")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1280)
    main(ap.parse_args())
