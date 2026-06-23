"""WILDTRACK validation (Phase E + F) — multi-camera fusion accuracy + absolute scale.

Uses WILDTRACK's REAL OpenCV calibration (intrinsic_zero + extrinsic) and ground-
truth positions to validate, on a true metric 12x36 m plaza:
  * MULTI-CAMERA FUSION: detect people in N cameras -> project each view's foot
    points to the common world ground via calibration -> fuse (world-space
    clustering, flowsight.geometry.multicam) -> match to GT (precision / recall /
    localisation error). Compares 1-camera vs 2-camera to show occlusion fill.
  * ABSOLUTE SCALE: density (persons/m^2) on the real metric ground — a physical
    sanity check that no longer depends on a guessed FOV.

Self-detecting layout (so it runs first try): finds the dataset root, the per-
camera image-subset dirs, and the calibration files by globbing. Auto-detects the
annotated frame numbers from annotations_positions/*.json.

Run on Colab (after extracting Image_subsets for >=2 cameras):
  !PYTHONPATH=. python -u experiments/wildtrack_validate.py --root /content/WT/Wildtrack_dataset \
      --cams 2 --frames 20 --weights /content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from flowsight.geometry.multicam import CameraView, MultiCameraFusion
from flowsight.geometry.wildtrack import WildtrackCamera, load_camera, match_to_gt
from flowsight.physics.density import DensityField


def _find(root, *parts):
    hits = glob.glob(os.path.join(root, *parts))
    return sorted(h for h in hits if "__MACOSX" not in h)


def discover(root: str):
    """Return (cam_names, {cam:intr}, {cam:extr}, {cam:image_dir}, ann_dir)."""
    intr = {}
    extr = {}
    for p in _find(root, "calibrations", "intrinsic_zero", "intr_*.xml"):
        intr[os.path.basename(p)[5:-4]] = p          # intr_<NAME>.xml
    for p in _find(root, "calibrations", "extrinsic", "extr_*.xml"):
        extr[os.path.basename(p)[5:-4]] = p
    cams = [c for c in intr if c in extr]             # calibration order
    # image subset dirs (e.g. Image_subsets/C1 ...); map to calib cams by order
    img_dirs = [d for d in _find(root, "Image_subsets", "*") if os.path.isdir(d)]
    if not img_dirs:
        img_dirs = [d for d in _find(root, "*") if os.path.isdir(d)
                    and os.path.basename(d).lower().startswith("c")]
    cam_img = dict(zip(cams, img_dirs))               # CVLab1->C1, ... by order
    ann_dir = (_find(root, "annotations_positions") or [None])[0]
    return cams, intr, extr, cam_img, ann_dir


def frames_for(img_dir):
    fs = _find(img_dir, "*.png") + _find(img_dir, "*.jpg")
    return {os.path.splitext(os.path.basename(f))[0]: f for f in fs}


def main(a) -> None:
    from ultralytics import YOLO

    cams, intr, extr, cam_img, ann_dir = discover(a.root)
    print("[wt] cams=%s" % cams, flush=True)
    print("[wt] image dirs: %s" % {c: cam_img.get(c) for c in cams[:a.cams]}, flush=True)
    use = cams[:a.cams]
    wcams = {c: load_camera(intr[c], extr[c], unit_scale=a.unit_scale) for c in use}
    views = [CameraView(c, wcams[c]) for c in use]    # world frame is common
    fusion = MultiCameraFusion(views, assoc_radius_m=a.assoc_m)
    model = YOLO(a.weights)
    per_cam_frames = {c: frames_for(cam_img[c]) for c in use}
    ann_files = sorted(_find(ann_dir, "*.json")) if ann_dir else []
    print("[wt] %d annotated frames; validating first %d" % (len(ann_files), a.frames),
          flush=True)

    single, multi, gtN = {"tp": 0, "fp": 0, "fn": 0, "err": []}, \
        {"tp": 0, "fp": 0, "fn": 0, "err": []}, 0
    denss = []
    for af in ann_files[:a.frames]:
        fid = os.path.splitext(os.path.basename(af))[0]
        gt = _load_gt(af, a)
        gtN += len(gt)
        dets_px = {}
        for c in use:
            fp = per_cam_frames[c].get(fid)
            if fp is None:
                continue
            r = model.predict(fp, classes=[0], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                dets_px[c] = np.zeros((0, 2))
                continue
            b = r.boxes.xyxy.cpu().numpy()
            dets_px[c] = np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])  # foot pixels
        # MultiCameraFusion.fuse() projects pixels -> world itself; pass PIXELS (not
        # pre-projected world) or it would apply to_ground twice. Single-cam = first
        # view projected directly.
        s_pred = (wcams[use[0]].to_ground(dets_px[use[0]])
                  if use[0] in dets_px else np.zeros((0, 2)))
        m_pred = fusion.fuse(dets_px)["fused"]
        _acc(single, match_to_gt(s_pred, gt, a.match_m))
        _acc(multi, match_to_gt(m_pred, gt, a.match_m))
        if len(m_pred):
            denss.append(_peak_density(m_pred, a))

    print("=== WT_RESULT ===", flush=True)
    print("GT persons total: %d over %d frames" % (gtN, min(a.frames, len(ann_files))), flush=True)
    _report("1-camera ", single)
    _report("%d-camera" % a.cams, multi)
    if denss:
        print("[abs-scale] peak density on real 12x36m ground: mean %.2f /m^2 (max %.2f) "
              "-- physical (Itaewon crush ~6)" % (np.mean(denss), np.max(denss)), flush=True)


def _load_gt(af, a):
    data = json.load(open(af))
    pts = []
    for d in data:
        pid = d.get("positionID")
        if pid is None:
            continue
        gx = pid % a.grid_w
        gy = pid // a.grid_w
        pts.append([(a.origin_x + gx * a.step_cm) * a.unit_scale,
                    (a.origin_y + gy * a.step_cm) * a.unit_scale])
    return np.array(pts) if pts else np.zeros((0, 2))


def _peak_density(xy, a):
    x0, y0 = xy.min(0) - 1
    x1, y1 = xy.max(0) + 1
    df = DensityField((x0, y0, x1, y1), cell=1.0, sigma_m=1.0)
    return float(df.compute(xy).max())


def _acc(d, m):
    d["tp"] += m["tp"]
    d["fp"] += m["fp"]
    d["fn"] += m["fn"]
    if m["mean_loc_err_m"] is not None:
        d["err"].append(m["mean_loc_err_m"])


def _report(name, d):
    prec = d["tp"] / (d["tp"] + d["fp"] + 1e-9)
    rec = d["tp"] / (d["tp"] + d["fn"] + 1e-9)
    err = np.mean(d["err"]) if d["err"] else float("nan")
    print("[%s] precision %.3f  recall %.3f  loc_err %.2f m  (tp%d fp%d fn%d)"
          % (name, prec, rec, err, d["tp"], d["fp"], d["fn"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/WT/Wildtrack_dataset")
    ap.add_argument("--cams", type=int, default=2)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--assoc-m", type=float, default=1.0)
    ap.add_argument("--match-m", type=float, default=1.0)
    ap.add_argument("--unit-scale", type=float, default=0.01)  # cm -> m
    ap.add_argument("--grid-w", type=int, default=480)
    ap.add_argument("--step-cm", type=float, default=2.5)
    ap.add_argument("--origin-x", type=float, default=-300.0)
    ap.add_argument("--origin-y", type=float, default=-90.0)  # official WILDTRACK frame
    main(ap.parse_args())
