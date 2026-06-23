"""Recall-improvement experiment lab — hypothesis -> experiment -> verify -> revise.

Implements fugu's design (Registry / Runner / Comparator / Loop orchestrator) for the
person-detection recall problem. Decision rule: a variant is ACCEPTED only if it
raises recall **at matched FPPI** (never compare raw recall) without dropping overall
recall. Cost-ordered: cached-box options (NMS, threshold — run now, no retrain) ->
detector-inference options (tiling/multi-scale/pose — need the live model on Colab) ->
training (amodal, P2, anchor, loss, aug, head-aux).

Runs self-contained on synthetic detections (no data/GPU) so the loop + metrics are
unit-testable; the detector/training tiers emit Colab run-specs instead of executing.

    PYTHONPATH=. python experiments/recall_lab.py     # self-demo (cached tier)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flowsight.eval import nms_variants as nv
from flowsight.eval import slice_metrics as sm

SLICES = ("normal", "small", "occluded", "crowd", "trunc")


# ---------------------------------------------------------------- Registry ----
@dataclass
class Hypothesis:
    id: str
    name: str
    tier: str                 # "cached" | "detector" | "train"
    target_slices: tuple
    params: dict = field(default_factory=dict)
    apply: object = None      # callable(raw_preds_list, **p) -> preds_list  (cached tier only)
    spec: str = ""            # Colab run-spec for detector/train tiers

    @property
    def runnable_now(self):
        return self.tier == "cached" and self.apply is not None


def _nms_dataset(raw, method, iou_thr=0.5, **kw):
    """Apply an NMS variant per image to raw (pre-NMS) preds -> kept preds (N,5)."""
    out = []
    for p in raw:
        p = np.asarray(p, float).reshape(-1, 5)
        if not len(p):
            out.append(p); continue
        if method == "hard":
            keep = nv.hard_nms(p[:, :4], p[:, 4], iou_thr); kept = p[keep]
        elif method == "diou":
            keep = nv.diou_nms(p[:, :4], p[:, 4], iou_thr); kept = p[keep]
        elif method == "soft":
            keep, ns = nv.soft_nms(p[:, :4], p[:, 4], iou_thr, **kw)
            kept = p[keep].copy(); kept[:, 4] = ns
        else:
            kept = p
        out.append(kept)
    return out


def build_registry():
    return [
        Hypothesis("H3_nms", "Soft/DIoU-NMS for crowd overlap", "cached",
                   ("crowd",), {"method": ["soft", "diou"], "iou_thr": [0.5, 0.6]},
                   apply=_nms_dataset),
        Hypothesis("H4_threshold", "Adaptive / lower score threshold", "cached",
                   ("small", "occluded"), {"score_thr": [0.05, 0.1, 0.2, 0.3]},
                   apply=lambda raw, score_thr=0.0: [np.asarray(p, float).reshape(-1, 5)[
                       np.asarray(p, float).reshape(-1, 5)[:, 4] >= score_thr] for p in raw]),
        Hypothesis("H5_tiling", "Hi-res / multi-scale / tiling + WBF", "detector",
                   ("small",), {"imgsz": [960, 1280], "slice": 512, "overlap": 0.2},
                   spec="Colab: re-run detector at imgsz/tiling, fuse with weighted_boxes_fusion."),
        Hypothesis("H10_pose", "Head/upper-body / pose cascade for proposals", "detector",
                   ("occluded",), {"pose_model": "yolo11x-pose"},
                   spec="Colab: run pose/head model, merge head-derived proposals before NMS."),
        Hypothesis("H1_head", "Foot -> head/upper-body anchor (aux head)", "train",
                   ("occluded", "trunc"), spec="train: add head/upper-body regression head."),
        Hypothesis("H2_amodal", "visible + amodal dual labels", "train",
                   ("occluded",), spec="train: multi-task visible+full bbox."),
        Hypothesis("H6_p2", "P2/P1 low-level feature / BiFPN / feat-SR", "train",
                   ("small",), spec="train: add P2 head / BiFPN."),
        Hypothesis("H7_anchor", "k-means anchors / ATSS / SimOTA assign", "train",
                   ("trunc", "small"), spec="train: anchor-free + ATSS/SimOTA assigner."),
        Hypothesis("H8_loss", "focal / varifocal / QFL + IoU-aware", "train",
                   ("occluded", "small"), spec="train: swap loss to varifocal+IoU-aware."),
        Hypothesis("H9_aug", "occlusion / copy-paste / crowd aug + hard-neg", "train",
                   ("occluded", "crowd"), spec="train: copy-paste+random-occlusion aug."),
    ]


# ------------------------------------------------------------------ Runner ----
def evaluate(preds, gts, gslices, iou_thr=0.5):
    return sm.full_report(preds, gts, gslices, iou_thr)


def run_cached_variant(raw, gts, gslices, hyp, params):
    preds = hyp.apply(raw, **params)
    rep = evaluate(preds, gts, gslices)
    return preds, rep


# -------------------------------------------------------------- Comparator ----
def decide(base_preds, var_preds, gts, target_fppi, eps=0.005, iou_thr=0.5):
    cmp = sm.compare_at_matched_fppi(base_preds, var_preds, gts, target_fppi, iou_thr)
    ov_base = sm.aggregate(base_preds, gts, iou_thr)["recall"]
    ov_var = sm.aggregate(var_preds, gts, iou_thr)["recall"]
    accept = (cmp["delta_recall"] > eps) and (ov_var >= ov_base - 0.02)
    cmp.update({"overall_base": ov_base, "overall_var": ov_var, "accept": bool(accept)})
    return cmp


# ---------------------------------------------------------- Loop orchestrator --
class RecallLab:
    def __init__(self, raw, gts, gslices, target_fppi=0.5):
        self.raw, self.gts, self.gslices = raw, gts, gslices
        self.target_fppi = target_fppi
        self.registry = build_registry()
        self.done = set()
        self.log = []
        self.baseline_preds = _nms_dataset(raw, "hard", 0.5)   # current pipeline
        self.baseline = evaluate(self.baseline_preds, gts, gslices)

    def propose(self):
        order = {"cached": 0, "detector": 1, "train": 2}
        pend = [h for h in self.registry if h.id not in self.done]
        return sorted(pend, key=lambda h: order[h.tier])[0] if pend else None

    def run_cycle(self):
        """One hypothesis->experiment->verify->revise step. Returns a record."""
        h = self.propose()
        if h is None:
            return {"done": True}
        rec = {"id": h.id, "name": h.name, "tier": h.tier, "target_slices": h.target_slices}
        if h.runnable_now:
            best = None
            for combo in _grid(h.params):
                try:
                    preds, rep = run_cached_variant(self.raw, self.gts, self.gslices, h, combo)
                except Exception as e:  # noqa: BLE001
                    rec.setdefault("errors", []).append("%s %r" % (combo, e)); continue
                d = decide(self.baseline_preds, preds, self.gts, self.target_fppi)
                cand = {"params": combo, **d, "slice_recall": rep.get("slice_recall", {})}
                if best is None or cand["delta_recall"] > best["delta_recall"]:
                    best = cand
            rec.update({"executed": True, "best": best,
                        "verdict": "ACCEPT" if (best and best["accept"]) else "REJECT"})
        else:
            rec.update({"executed": False, "verdict": "DEFERRED(Colab)", "spec": h.spec,
                        "params": h.params})
        self.done.add(h.id); self.log.append(rec)
        return rec

    def run_all_cached(self):
        while True:
            h = self.propose()
            if h is None or h.tier != "cached":
                break
            self.run_cycle()
        return self.log


def _grid(params):
    if not params:
        return [{}]
    keys = list(params)
    out = [{}]
    for k in keys:
        vals = params[k] if isinstance(params[k], (list, tuple)) else [params[k]]
        out = [dict(o, **{k: v}) for o in out for v in vals]
    return out


# ------------------------------------------------------------- synthetic data --
def synthetic_scene(n_img=40, seed=0):
    """Raw (pre-NMS) preds + GT with slice tags. Simulates low recall: crowd dups,
    small/occluded misses, some FP — so NMS/threshold experiments are meaningful."""
    rng = np.random.default_rng(seed)
    raw, gts, gsl = [], [], []
    for _ in range(n_img):
        m = rng.integers(8, 20)
        cx = rng.uniform(0, 1000, m); cy = rng.uniform(0, 600, m)
        sizes = rng.choice([14, 28, 60], m, p=[0.4, 0.4, 0.2])     # many small
        gt = np.column_stack([cx - sizes/2, cy - sizes, cx + sizes/2, cy + sizes])
        tags = np.where(sizes <= 14, "small", np.where(sizes >= 60, "normal", "occluded")).astype(object)
        # crowd tag where neighbours are close
        for i in range(m):
            if (np.abs(cx - cx[i]) + np.abs(cy - cy[i]) < 40).sum() >= 3:
                tags[i] = "crowd"
        preds = []
        for i in range(m):
            seen_p = 0.85 if tags[i] == "normal" else (0.45 if tags[i] in ("small", "occluded") else 0.6)
            if rng.random() < seen_p:                              # detected
                jit = rng.normal(0, sizes[i] * 0.06, 4)
                sc = rng.uniform(0.3, 0.95) if tags[i] == "normal" else rng.uniform(0.05, 0.5)
                preds.append([*(gt[i] + jit), sc])
                if tags[i] == "crowd" and rng.random() < 0.7:      # duplicate in crowd
                    preds.append([*(gt[i] + rng.normal(0, 4, 4)), sc * rng.uniform(0.7, 1.0)])
        for _ in range(rng.integers(2, 8)):                        # false positives
            x, y, s = rng.uniform(0, 1000), rng.uniform(0, 600), rng.choice([20, 40])
            preds.append([x - s/2, y - s, x + s/2, y + s, rng.uniform(0.05, 0.4)])
        raw.append(np.array(preds, float).reshape(-1, 5))
        gts.append(gt); gsl.append(np.array(tags))
    return raw, gts, gsl


def main():
    raw, gts, gsl = synthetic_scene()
    lab = RecallLab(raw, gts, gsl, target_fppi=1.0)
    print("[baseline] overall recall %.3f  MR-2 %.3f  slice=%s"
          % (lab.baseline["overall"]["recall"], lab.baseline["MR-2"],
             {k: round(v, 2) for k, v in lab.baseline["slice_recall"].items()}), flush=True)
    print("[cached tier — run now, no retrain]", flush=True)
    for rec in lab.run_all_cached():
        b = rec.get("best") or {}
        print("  %-14s %s  ΔRecall@FPPI=%+.3f  %s" %
              (rec["id"], rec["verdict"], b.get("delta_recall", 0.0), b.get("params", "")), flush=True)
    print("[deferred tiers — Colab run-specs]", flush=True)
    while True:
        rec = lab.run_cycle()
        if rec.get("done"):
            break
        print("  %-14s %-16s %s" % (rec["id"], rec["verdict"], rec.get("spec", "")), flush=True)


if __name__ == "__main__":
    main()
