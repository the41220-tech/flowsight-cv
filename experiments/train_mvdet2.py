"""Cycle 17 — MVDet 4-view CEILING-BREAK experiment harness (5 fugu-passed hypotheses).

Baseline (Cycle 16b, 40ep, peak-NMS): best-F1 0.403 @thr0.05 on the 40-frame 4-view
WILDTRACK subset (20 train / 20 test). This script A/B-tests 5 levers as flags, each
vs the same baseline, reporting the recall/precision frontier (thr sweep) + best-F1.

Hypotheses (Opus proposed 10-ish -> fugu critique -> these 5 survive a tech+commercial bar):
  HA  --deep         : add ResNet layer3 (stride16, upsampled) to layer2 -> richer BEV semantics
  HB  --aug          : train-time photometric jitter + per-view dropout -> anti-overfit / missing-cam robust
  HC  --freeze --wd W: freeze backbone + weight decay -> regularise the 20-frame regime
  HD  --robust-calib : pick each view's calibration over ALL frames (not 5) -> better BEV alignment
  HE  --cv-thr       : choose thr+NMS on TRAIN heatmaps, then REPORT that single op-point on TEST (honest)

Run (Colab, Drive data, no download):
  PYTHONPATH=. python experiments/train_mvdet2.py --root /content/WTx/Wildtrack_dataset \
      --views C1,C2,C4,C5 --epochs 40 [--deep] [--aug] [--freeze --wd 5e-4] [--robust-calib] [--cv-thr]
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

NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
VIEW_IDX = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4, "C6": 5, "C7": 6}
BOUNDS = (-3.0, -0.9, 9.0, 35.1)
CELL = 0.1
SWEEP = (0.02, 0.03, 0.05, 0.07, 0.10, 0.15)


def _cam(root, nm):
    return load_camera(os.path.join(root, "calibrations", "intrinsic_zero", "intr_%s.xml" % nm),
                       os.path.join(root, "calibrations", "extrinsic", "extr_%s.xml" % nm))


def auto_calib(root, view, frames_pairs, n_probe):
    """Pick the calibration with min foot-anchor world error. HD: widen n_probe -> all frames."""
    best = (None, 1e9)
    for nm in NAMES:
        c = _cam(root, nm)
        e = []
        for b, g in frames_pairs[:n_probe]:
            if not len(b):
                continue
            w = c.to_ground(bbox_anchor(b, 1.0), bounds=BOUNDS)
            n = min(len(w), len(g))
            if n:
                e += list(np.linalg.norm(w[:n] - g[:n], axis=1))
        me = float(np.median(e)) if e else 1e9
        if me < best[1]:
            best = (nm, me)
    return _cam(root, best[0]), best[0]


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


def peaks(heat, bounds, cell, thr, radius):
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


def prf(cache, thr, nms):
    tp = fn = fp = 0
    for heat, g in cache:
        m = match_to_gt(peaks(heat, BOUNDS, CELL, thr, nms), g, 2.0)
        tp += m["tp"]; fn += m["fn"]; fp += m["fp"]
    rec = tp / (tp + fn + 1e-9); prec = tp / (tp + fp + 1e-9)
    f1 = 2 * rec * prec / (rec + prec + 1e-9)
    return rec, prec, f1, tp, fn, fp


def build_mvdet2(n_views, deep, feat_ch=64, head_ch=128, freeze=False, pretrained=True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    class MVDet2(nn.Module):
        def __init__(self):
            super().__init__()
            bb = torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
            self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2)
            self.deep = deep
            in_ch = 128
            if deep:
                self.layer3 = bb.layer3  # stride16, 256ch
                in_ch = 128 + 256
            self.compress = nn.Conv2d(in_ch, feat_ch, 1)
            gn = lambda c: nn.GroupNorm(min(32, c), c)   # GroupNorm: batch=1-safe (BN is not)
            self.head = nn.Sequential(
                nn.Conv2d(n_views * feat_ch + 2, head_ch, 3, padding=2, dilation=2), gn(head_ch), nn.ReLU(True),
                nn.Conv2d(head_ch, head_ch, 3, padding=2, dilation=2), gn(head_ch), nn.ReLU(True),
                nn.Conv2d(head_ch, head_ch, 3, padding=1), gn(head_ch), nn.ReLU(True),
                nn.Conv2d(head_ch, 1, 1),
            )
            if freeze:
                for p in self.stem.parameters():
                    p.requires_grad = False
                if deep:
                    for p in self.layer3.parameters():
                        p.requires_grad = False

        def train(self, mode=True):
            # FrozenBN: keep pretrained backbone BatchNorm in eval (batch=1 makes BN stats unstable)
            super().train(mode)
            for mod in self.stem.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
            if self.deep:
                for mod in self.layer3.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.eval()
            return self

        def forward(self, imgs, grids, coord):
            B, V, C, H, W = imgs.shape
            f = self.stem(imgs.view(B * V, C, H, W))
            if self.deep:
                f3 = self.layer3(f)
                f3 = F.interpolate(f3, size=f.shape[-2:], mode="bilinear", align_corners=False)
                f = torch.cat([f, f3], dim=1)
            f = self.compress(f)
            _, fc, h, w = f.shape
            f = f.view(B, V, fc, h, w)
            Hg, Wg = grids.shape[1], grids.shape[2]
            proj = []
            for v in range(V):
                g = grids[v].unsqueeze(0).expand(B, Hg, Wg, 2)
                proj.append(F.grid_sample(f[:, v], g, align_corners=False, padding_mode="zeros"))
            bev = torch.cat(proj, dim=1)
            cc = coord.unsqueeze(0).expand(B, 2, Hg, Wg)
            bev = torch.cat([bev, cc], dim=1)
            return self.head(bev)

    return MVDet2()


def focal_bev_loss(logit, target, alpha=2.0, beta=4.0, eps=1e-6):
    import torch
    p = torch.sigmoid(logit).clamp(eps, 1 - eps)
    pos = (target >= 1.0 - 1e-4).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def main(a):
    import torch
    import torch.nn.functional as F
    import cv2

    import random
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    views = a.views.split(",")
    H, W = (int(x) for x in a.hw.split(","))
    allf = sorted(glob.glob(os.path.join(a.root, "annotations_positions", "*.json")))[:a.n]
    fids = [os.path.splitext(os.path.basename(p))[0] for p in allf]
    k = max(1, len(allf) // 2)
    tr, te = slice(0, k), slice(k, len(allf))
    n_probe = len(allf) if a.robust_calib else 5

    cams, grids = {}, []
    img0 = cv2.imread(os.path.join(a.root, "Image_subsets", views[0], fids[0] + ".png"))
    OW, OH = img0.shape[1], img0.shape[0]
    mapping = {}
    for vw in views:
        cams[vw], nm = auto_calib(a.root, vw, view_pairs(a.root, VIEW_IDX[vw], allf), n_probe)
        mapping[vw] = nm
        g, _ = bev_projection_grid(cams[vw], BOUNDS, CELL, (OW, OH))
        grids.append(torch.from_numpy(g).to(dev))
    grids = torch.stack(grids)
    centres, Hg, Wg = bev_grid_centres(BOUNDS, CELL)
    coord = torch.from_numpy(np.stack([
        (centres[..., 0] - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]),
        (centres[..., 1] - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1])], 0).astype(np.float32)).to(dev)

    def load_imgs(fid, train):
        ims = []
        for vw in views:
            im = cv2.imread(os.path.join(a.root, "Image_subsets", vw, fid + ".png"))
            im = cv2.cvtColor(cv2.resize(im, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if train and a.aug:  # HB: per-view photometric jitter
                im = np.clip(im * rng.uniform(0.7, 1.3) + rng.uniform(-0.08, 0.08), 0, 1)
            ims.append(im.transpose(2, 0, 1))
        t = torch.from_numpy(np.stack(ims)).unsqueeze(0).to(dev)
        if train and a.aug and rng.random() < 0.3:  # HB: view dropout
            t[0, int(rng.integers(len(views)))] = 0.0
        return t

    net = build_mvdet2(len(views), a.deep, freeze=a.freeze).to(dev)
    # discriminative LR: backbone (pretrained) lr*0.1, head lr; cosine decay over epochs
    bb, hd = [], []
    for nm, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (bb if ("stem" in nm or "layer3" in nm) else hd).append(p)
    opt = torch.optim.Adam([{"params": bb, "lr": a.lr * 0.1}, {"params": hd, "lr": a.lr}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    for ep in range(a.epochs):
        net.train(); tot = 0.0; opt.zero_grad()
        for i, (af, fid) in enumerate(zip(allf[tr], fids[tr])):
            tgt = torch.from_numpy(bev_gt_heatmap(gt_world(af), BOUNDS, CELL, 0.5)).to(dev)[None, None]
            logit = net(load_imgs(fid, True), grids, coord)
            logit = F.interpolate(logit, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
            loss = focal_bev_loss(logit, tgt) / a.accum          # grad accumulation -> effective batch
            loss.backward(); tot += float(loss) * a.accum
            if (i + 1) % a.accum == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()                              # flush partial accum
        sched.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print("ep %d loss %.4f" % (ep, tot / max(1, k)), flush=True)

    net.eval()
    def cache_of(sl):
        c = []
        with torch.no_grad():
            for af, fid in zip(allf[sl], fids[sl]):
                logit = net(load_imgs(fid, False), grids, coord)
                logit = F.interpolate(logit, size=(Hg, Wg), mode="bilinear", align_corners=False)
                c.append((torch.sigmoid(logit)[0, 0].cpu().numpy(), gt_world(af)))
        return c
    te_cache = cache_of(te)

    tag = "+".join([n for n, v in [("deep", a.deep), ("aug", a.aug),
                    ("freeze", a.freeze), ("wd", a.wd > 0), ("rcalib", a.robust_calib),
                    ("cvthr", a.cv_thr)] if v]) or "baseline"
    print("=== MVDET2_RESULT cfg=%s views=%s map=%s nms=%.1f (thr sweep) ===" %
          (tag, views, mapping, a.nms), flush=True)
    best = (-1, None)
    for thr in SWEEP:
        rec, prec, f1, tp, fn, fp = prf(te_cache, thr, a.nms)
        if f1 > best[0]:
            best = (f1, (thr, rec, prec))
        print("thr %.2f @2m recall %.3f precision %.3f f1 %.3f (tp%d fn%d fp%d)" %
              (thr, rec, prec, f1, tp, fn, fp), flush=True)
    print("BEST-F1 %.3f @thr%.2f (recall %.3f precision %.3f)" %
          (best[0], best[1][0], best[1][1], best[1][2]), flush=True)

    if a.cv_thr:  # HE: select op-point on TRAIN, report on TEST (no test leakage)
        tr_cache = cache_of(tr)
        bt = (-1, None)
        for thr in SWEEP:
            for nm in (0.5, 1.0, 1.5):
                _, _, f1, _, _, _ = prf(tr_cache, thr, nm)
                if f1 > bt[0]:
                    bt = (f1, (thr, nm))
        thr, nm = bt[1]
        rec, prec, f1, tp, fn, fp = prf(te_cache, thr, nm)
        print("CV-THR picked thr%.2f nms%.1f on TRAIN(f1%.3f) -> TEST recall %.3f precision %.3f f1 %.3f" %
              (thr, nm, bt[0], rec, prec, f1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/WTx/Wildtrack_dataset")
    ap.add_argument("--views", default="C1,C2,C4,C5")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps (effective batch)")
    ap.add_argument("--hw", default="360,640")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nms", type=float, default=1.0)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--robust-calib", dest="robust_calib", action="store_true")
    ap.add_argument("--cv-thr", dest="cv_thr", action="store_true")
    main(ap.parse_args())
