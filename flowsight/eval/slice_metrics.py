"""Slice-aware detection-recall metrics (recall lab — fugu 'Metrics' module).

Pure numpy. Operates on per-image lists:
    preds[i] : (Ni,5) array [x1,y1,x2,y2,score]
    gts[i]   : (Mi,4) array [x1,y1,x2,y2]
    gslices[i]: (Mi,) array of str tags (e.g. "small","occluded","crowd","trunc")

Implements the protocol from the recall-improvement program:
  recall@IoU(0.5/0.75), precision, FPPI, AR@maxDets, MR-2 (log-avg miss-rate over
  FPPI), an FPPI/FROC curve, per-SLICE recall, and the decisive *compare at matched
  FPPI* helper (never compare raw recall — equalise FPPI first).

No training / no data download needed; fully unit-testable on synthetic boxes.
"""
from __future__ import annotations

import numpy as np

_REF_FPPI = np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000])


def iou_matrix(a, b):
    """IoU between boxes a (N,4) and b (M,4) in xyxy -> (N,M)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(pred, gt, iou_thr=0.5, score_thr=None, max_dets=None):
    """Greedy score-sorted matching (COCO-style) for one image.

    Returns (n_gt, n_tp, n_fp, matched_gt_bool, tp_flags_per_kept_pred).
    matched_gt_bool: (M,) which GT were detected (for slice recall)."""
    pred = np.asarray(pred, float).reshape(-1, 5)
    gt = np.asarray(gt, float).reshape(-1, 4)
    if score_thr is not None and len(pred):
        pred = pred[pred[:, 4] >= score_thr]
    order = np.argsort(-pred[:, 4]) if len(pred) else np.array([], int)
    if max_dets is not None:
        order = order[:max_dets]
    ious = iou_matrix(pred[order, :4], gt) if (len(order) and len(gt)) else np.zeros((len(order), len(gt)))
    gt_taken = np.zeros(len(gt), bool)
    tp = np.zeros(len(order), bool)
    for k in range(len(order)):
        if not len(gt):
            break
        j = -1
        best = iou_thr
        for g in range(len(gt)):
            if gt_taken[g]:
                continue
            if ious[k, g] >= best:
                best = ious[k, g]
                j = g
        if j >= 0:
            gt_taken[j] = True
            tp[k] = True
    return len(gt), int(tp.sum()), int((~tp).sum()), gt_taken, tp


def aggregate(preds, gts, iou_thr=0.5, score_thr=None, max_dets=None):
    """Dataset-level recall / precision / FPPI at one operating point."""
    n_gt = n_tp = n_fp = 0
    n_img = len(gts)
    for p, g in zip(preds, gts):
        m = match_image(p, g, iou_thr, score_thr, max_dets)
        n_gt += m[0]; n_tp += m[1]; n_fp += m[2]
    recall = n_tp / (n_gt + 1e-9)
    precision = n_tp / (n_tp + n_fp + 1e-9)
    fppi = n_fp / max(n_img, 1)
    return {"recall": recall, "precision": precision, "fppi": fppi,
            "tp": n_tp, "fp": n_fp, "n_gt": n_gt, "n_img": n_img}


def slice_recall(preds, gts, gslices, slice_name, iou_thr=0.5, score_thr=None):
    """Recall restricted to GT whose slice tag == slice_name."""
    n_gt = n_det = 0
    for p, g, s in zip(preds, gts, gslices):
        if not len(g):
            continue
        _, _, _, taken, _ = match_image(p, g, iou_thr, score_thr)
        s = np.asarray(s)
        sel = s == slice_name
        n_gt += int(sel.sum())
        n_det += int(taken[sel].sum())
    return n_det / (n_gt + 1e-9), n_gt


def fppi_curve(preds, gts, iou_thr=0.5, n_pts=50):
    """Sweep score threshold -> list of (fppi, miss_rate=1-recall)."""
    scores = np.concatenate([np.asarray(p, float).reshape(-1, 5)[:, 4]
                             for p in preds if len(p)]) if any(len(p) for p in preds) else np.array([1.0])
    thrs = np.unique(np.quantile(scores, np.linspace(0, 1, n_pts))) if len(scores) else np.array([0.0])
    out = []
    for t in thrs:
        a = aggregate(preds, gts, iou_thr, score_thr=t)
        out.append((a["fppi"], 1.0 - a["recall"]))
    return sorted(out)


def mr2(preds, gts, iou_thr=0.5):
    """Log-average miss rate over 9 reference FPPI points (Caltech MR-2). Lower=better."""
    curve = fppi_curve(preds, gts, iou_thr, n_pts=100)
    f = np.array([c[0] for c in curve]); mr = np.array([c[1] for c in curve])
    vals = []
    for ref in _REF_FPPI:
        idx = np.where(f <= ref)[0]
        vals.append(mr[idx[-1]] if len(idx) else 1.0)   # miss rate at highest fppi <= ref
    vals = np.clip(vals, 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(vals))))


def ar_at_maxdets(preds, gts, max_dets=100, iou_set=None):
    """Average Recall over IoU 0.5:0.95, top-`max_dets` preds/image."""
    iou_set = iou_set if iou_set is not None else np.arange(0.5, 1.0, 0.05)
    return float(np.mean([aggregate(preds, gts, t, max_dets=max_dets)["recall"] for t in iou_set]))


def recall_at_fppi(preds, gts, target_fppi, iou_thr=0.5):
    """Find the score threshold giving ~target_fppi and return recall there.
    This is THE fair-comparison primitive — equalise FPPI, then read recall."""
    curve = []  # (fppi, recall, thr)
    scores = np.concatenate([np.asarray(p, float).reshape(-1, 5)[:, 4]
                             for p in preds if len(p)]) if any(len(p) for p in preds) else np.array([1.0])
    for t in np.unique(np.quantile(scores, np.linspace(0, 1, 80))) if len(scores) else [0.0]:
        a = aggregate(preds, gts, iou_thr, score_thr=t)
        curve.append((a["fppi"], a["recall"], float(t)))
    curve.sort()
    f = np.array([c[0] for c in curve]); r = np.array([c[1] for c in curve])
    # highest recall achievable without exceeding target_fppi
    ok = f <= target_fppi + 1e-9
    return float(r[ok].max()) if ok.any() else float(r[np.argmin(f)])


def compare_at_matched_fppi(base_preds, var_preds, gts, target_fppi, iou_thr=0.5):
    """ΔRecall at equal FPPI (decision primitive). >0 => variant helps at fixed FP budget."""
    rb = recall_at_fppi(base_preds, gts, target_fppi, iou_thr)
    rv = recall_at_fppi(var_preds, gts, target_fppi, iou_thr)
    return {"recall_base": rb, "recall_var": rv, "delta_recall": rv - rb,
            "target_fppi": target_fppi}


def full_report(preds, gts, gslices=None, iou_thr=0.5, score_thr=None):
    """One-call slice-aware report for a method at a given operating point."""
    rep = {"overall": aggregate(preds, gts, iou_thr, score_thr),
           "recall@0.75": aggregate(preds, gts, 0.75, score_thr)["recall"],
           "AR@100": ar_at_maxdets(preds, gts, 100),
           "MR-2": mr2(preds, gts, iou_thr)}
    if gslices is not None:
        tags = set()
        for s in gslices:
            tags.update(np.asarray(s).tolist())
        rep["slice_recall"] = {t: slice_recall(preds, gts, gslices, t, iou_thr, score_thr)[0]
                               for t in sorted(tags)}
    return rep
