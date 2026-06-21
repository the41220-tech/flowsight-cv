"""Detection recall/precision evaluation (person class) — pure numpy.

Greedy score-ordered matching at IoU>=thr (standard detection protocol).
Micro-averaged over images. Includes NMS for building an ensemble (union of
several detectors' boxes) and size buckets (drone people are mostly 'small').
"""
from __future__ import annotations
import numpy as np

COCO_SMALL = 32 * 32
COCO_MEDIUM = 96 * 96


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """Returns (tp, fp, gt_matched_bool). One pred matches at most one GT."""
    n_gt = len(gt_boxes)
    gt_used = np.zeros(n_gt, bool)
    if len(pred_boxes) == 0:
        return 0, 0, gt_used
    order = np.argsort(-np.asarray(pred_scores, float))
    iou = iou_matrix(pred_boxes, gt_boxes)
    tp = fp = 0
    for i in order:
        if n_gt == 0:
            fp += 1; continue
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= iou_thr and not gt_used[j]:
            gt_used[j] = True; tp += 1
        else:
            fp += 1
    return tp, fp, gt_used


def nms(boxes, scores, iou_thr=0.55):
    boxes = np.asarray(boxes, float).reshape(-1, 4); scores = np.asarray(scores, float)
    if len(boxes) == 0:
        return np.zeros(0, int)
    order = np.argsort(-scores); keep = []
    while len(order):
        i = order[0]; keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        iou = iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        order = rest[iou < iou_thr]
    return np.array(keep, int)


def size_bucket(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < COCO_SMALL else ("medium" if area < COCO_MEDIUM else "large")


class RecallMeter:
    """Accumulate per-image TP/FP/GT (overall, per-domain, per-size)."""
    def __init__(self):
        self.tp = 0; self.fp = 0; self.n_gt = 0
        self.by_domain = {}; self.by_size = {}

    def add(self, pred_boxes, pred_scores, gt_boxes, iou_thr=0.5, domain="all"):
        tp, fp, gt_used = match(pred_boxes, pred_scores, gt_boxes, iou_thr)
        self.tp += tp; self.fp += fp; self.n_gt += len(gt_boxes)
        d = self.by_domain.setdefault(domain, [0, 0, 0])
        d[0] += tp; d[1] += fp; d[2] += len(gt_boxes)
        for k, gb in enumerate(gt_boxes):
            b = self.by_size.setdefault(size_bucket(gb), [0, 0])
            b[1] += 1                       # gt count in bucket
            if gt_used[k]:
                b[0] += 1                   # recalled in bucket

    @staticmethod
    def _prf(tp, fp, n_gt):
        rec = tp / n_gt if n_gt else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"recall": round(rec, 4), "precision": round(prec, 4),
                "f1": round(f1, 4), "tp": tp, "fp": fp, "n_gt": n_gt}

    def summary(self):
        out = {"overall": self._prf(self.tp, self.fp, self.n_gt), "by_domain": {}, "by_size": {}}
        for dom, (tp, fp, n) in self.by_domain.items():
            out["by_domain"][dom] = self._prf(tp, fp, n)
        for sz, (rec, n) in self.by_size.items():
            out["by_size"][sz] = {"recall": round(rec / n, 4) if n else 0.0, "n_gt": n}
        return out
