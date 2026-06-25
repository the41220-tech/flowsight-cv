"""Train the MVDet-style multi-view BEV occupancy detector (H2, Cycle 16) on WILDTRACK.

This is the learned detector that aims to break the precision-recall TRADE of late fusion
(greedy/NMS/bev-vote span a frontier; a trained BEV net should lift BOTH). Per-view ResNet-18
features are warped to the shared BEV ground plane via precomputed calibration grids, fused,
and a BEV head predicts an occupancy heatmap; peaks are matched to GT world positions.

T4 scope: ResNet-18 backbone, input 360x640, batch 1, ~20-40 epochs. Works on whatever views
are present (the Drive subset has C1/C2/C4/C5 x 40 frames -> PIPELINE SMOKE TEST; a real
benchmark needs the full 7-view x 400-frame re-extract). Requires torch+torchvision (Colab GPU).

Run on Colab (data Drive-cached):
    PYTHONPATH=. python experiments/train_mvdet.py --root /content/WTx/Wildtrack_dataset \
        --views C1,C2,C4,C5 --epochs 30 --hw 360,640
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from flowsight.eval.anchor_proj import bbox_anchor
from flowsight.eval.bev_gt import bev_grid_centres, bev_gt_heatmap, bev_projection_grid
from flowsight.geometry.multicam import world_nms
from flowsight.geometry.wildtrack import load_camera, match_to_gt, positionid_to_world
from flowsight.models.mvdet import build_mvdet, focal_bev_loss

NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
VIEW_IDX = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4, "C6": 5, "C7": 6}
BOUNDS = (-3.0, -0.9, 9.0, 35.1)
CELL = 0.1   # 10 cm BEV grid -> ~120x360


def _cam(root, nm):
    return load_camera(os.path.join(root, "calibrations", "intrinsic_zero", "intr_%s.xml" % nm),
                       os.path.join(root, "calibrations", "extrinsic", "extr_%s.xml" % nm))


def auto_calib(root, view, frames_pairs):
    """Pick the calibration with min foot-anchor world error for this view's GT boxes."""
    best = (None, 1e9)
    for nm in NAMES:
        c = _cam(root, nm)
        e = []
        for b, g in frames_pairs[:5]:
            if not len(b):
                continue
            w = c.to_ground(bbox_anchor(b, 1.0))
            n = min(len(w), len(g))
            if n:
                e += list(np.linalg.norm(w[:n] - g[:n], axis=1))
        me = float(np.median(e)) if e else 1e9
        if me < best[1]:
            best = (nm, me)
    return _cam(root, best[0])


def view_pairs(root, vi, fl):
    out = []
    for af in fl:
        bx, pid = [], []
        for d in json.load(open(af)):
            for v in d.get("views", []):
                if v.get("viewNum") == vi and v.get("xmax", -1) != -1:
                    bx.append([v["xmin"], v["ymin"], v["xmax"], v["ymax"]]); pid.append(d["positionID"])
        b = np.array(bx, float).reshape(-1, 4)
        g = positionid_to_world(np.array(pid)) if pid else np.zeros((0, 2))
        out.append((b, g))
    return out


def gt_world(af):
    pid = [d["positionID"] for d in json.load(open(af)) if "positionID" in d]
    return positionid_to_world(np.array(pid)) if pid else np.zeros((0, 2))


def peaks(heat, bounds, cell, thr, radius=1.0):
    """BEV heatmap -> world peaks: 3x3 local maxima above `thr`, then world-space NMS
    (suppress lower peaks within `radius` m) to collapse the diffuse multi-maxima blobs an
    undertrained net produces -> far fewer false positives. radius<=0 disables NMS."""
    x0, y0 = bounds[0], bounds[1]
    gh, gw = heat.shape
    p = np.pad(heat, 1, mode="constant", constant_values=-1e18)
    mx = np.maximum.reduce([p[i:i + gh, j:j + gw] for i in range(3) for j in range(3)])
    ys, xs = np.where((heat >= mx - 1e-9) & (heat > thr))
    if not len(xs):
        return np.zeros((0, 2))
    cand = np.column_stack([x0 + (xs + 0.5) * cell, y0 + (ys + 0.5) * cell])
    if radius and radius > 0:
        cand = cand[world_nms(cand, heat[ys, xs], radius)]
    return cand


def main(a):
    import torch
    import torch.nn.functional as F
    import cv2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    views = a.views.split(",")
    H, W = (int(x) for x in a.hw.split(","))
    allf = sorted(glob.glob(os.path.join(a.root, "annotations_positions", "*.json")))[:a.n]
    fids = [os.path.splitext(os.path.basename(p))[0] for p in allf]
    k = max(1, len(allf) // 2)
    tr, te = slice(0, k), slice(k, len(allf))

    cams, grids, valids = {}, [], []
    img0 = cv2.imread(os.path.join(a.root, "Image_subsets", views[0], fids[0] + ".png"))
    OW, OH = img0.shape[1], img0.shape[0]
    for vw in views:
        cams[vw] = auto_calib(a.root, vw, view_pairs(a.root, VIEW_IDX[vw], allf))
        g, val = bev_projection_grid(cams[vw], BOUNDS, CELL, (OW, OH))   # grid in ORIGINAL px -> [-1,1]
        grids.append(torch.from_numpy(g).to(dev)); valids.append(val)
    grids = torch.stack(grids)                                          # [V,Hg,Wg,2]
    centres, Hg, Wg = bev_grid_centres(BOUNDS, CELL)
    coord = torch.from_numpy(np.stack([
        (centres[..., 0] - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]),
        (centres[..., 1] - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1])], 0).astype(np.float32)).to(dev)

    def load_imgs(fid):
        ims = []
        for vw in views:
            im = cv2.imread(os.path.join(a.root, "Image_subsets", vw, fid + ".png"))
            im = cv2.cvtColor(cv2.resize(im, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ims.append(im.transpose(2, 0, 1))
        return torch.from_numpy(np.stack(ims)).unsqueeze(0).to(dev)      # [1,V,3,H,W]

    net = build_mvdet(n_views=len(views)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for af, fid in zip(allf[tr], fids[tr]):
            tgt = torch.from_numpy(bev_gt_heatmap(gt_world(af), BOUNDS, CELL, 0.5)).to(dev)[None, None]
            logit = net(load_imgs(fid), grids, coord)
            logit = F.interpolate(logit, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
            loss = focal_bev_loss(logit, tgt)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        print("ep %d loss %.4f" % (ep, tot / max(1, k)), flush=True)

    net.eval()
    cache = []
    with torch.no_grad():
        for af, fid in zip(allf[te], fids[te]):
            logit = net(load_imgs(fid), grids, coord)
            logit = F.interpolate(logit, size=(Hg, Wg), mode="bilinear", align_corners=False)
            cache.append((torch.sigmoid(logit)[0, 0].cpu().numpy(), gt_world(af)))
    print("=== MVDET_RESULT views=%s peak-NMS=%.1fm (thr sweep) ===" % (views, a.nms), flush=True)
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
        tp = fn = fp = 0
        for heat, g in cache:
            m = match_to_gt(peaks(heat, BOUNDS, CELL, thr, a.nms), g, 2.0)
            tp += m["tp"]; fn += m["fn"]; fp += m["fp"]
        rec = tp / (tp + fn + 1e-9); prec = tp / (tp + fp + 1e-9)
        f1 = 2 * rec * prec / (rec + prec + 1e-9)
        print("thr %.2f @2m recall %.3f precision %.3f f1 %.3f (tp%d fn%d fp%d)" %
              (thr, rec, prec, f1, tp, fn, fp), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/WTx/Wildtrack_dataset")
    ap.add_argument("--views", default="C1,C2,C4,C5")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hw", default="360,640")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=1.0, help="peak-NMS radius (m); 0 disables")
    main(ap.parse_args())
