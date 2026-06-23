"""Detector-tier REAL-DATA eval (recall lab Cycle 7): whole vs tiled on WILDTRACK 2D GT.

Answers: does tiled inference raise person-detection recall **at matched FPPI** on
real frames, per slice (small / occluded / normal)? Uses WILDTRACK per-camera 2D
GT boxes (annotations_positions[*].views[viewNum]) + a YOLO detector, scored with
the recall-lab metrics (flowsight.eval.slice_metrics) and the tiling adapter
(flowsight.eval.tiling). The detector is injected, so head/in-domain models slot in
the same way.

Testable core (`eval_view`) takes detector callables -> mockable without YOLO/GPU.
`main()` wires ultralytics YOLO. Run on Colab after extracting WILDTRACK:
    PYTHONPATH=. python experiments/recall_realdata.py --root /content/WTx/Wildtrack_dataset \
        --view C1 --weights yolo11x.pt --n 20
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from flowsight.eval.slice_metrics import compare_at_matched_fppi, full_report, iou_matrix
from flowsight.eval.tiling import run_tiled


# ---- WILDTRACK 2D GT ----------------------------------------------------------
def gt_boxes_for_view(anno_path, view_idx):
    """2D GT boxes (M,4) for camera `view_idx` (0-based) from one annotation JSON."""
    out = []
    for d in json.load(open(anno_path)):
        for v in d.get("views", []):
            if v.get("viewNum") == view_idx and v.get("xmax", -1) != -1 and v.get("xmin", -1) != -1:
                out.append([v["xmin"], v["ymin"], v["xmax"], v["ymax"]])
    return np.array(out, float).reshape(-1, 4)


def slice_tags(boxes, small_h=80.0, overlap=0.3):
    """Per-GT slice tag: small (short bbox) / occluded (overlaps another GT) / normal."""
    boxes = np.atleast_2d(boxes)
    if not len(boxes):
        return np.array([], dtype=object)
    h = boxes[:, 3] - boxes[:, 1]
    tags = np.where(h < small_h, "small", "normal").astype(object)
    iou = iou_matrix(boxes, boxes)
    np.fill_diagonal(iou, 0.0)
    tags[(iou > overlap).any(axis=1)] = "occluded"
    return tags


def load_view(root, view="C1", n=20):
    """Return (fids, gts, gslices, img_paths) for the first n annotated frames of a view."""
    view_idx = int(view[1:]) - 1
    anns = sorted(glob.glob(os.path.join(root, "annotations_positions", "*.json")))[:n]
    fids, gts, gsl, imgs = [], [], [], []
    for af in anns:
        fid = os.path.splitext(os.path.basename(af))[0]
        gt = gt_boxes_for_view(af, view_idx)
        fids.append(fid); gts.append(gt); gsl.append(slice_tags(gt))
        imgs.append(os.path.join(root, "Image_subsets", view, fid + ".png"))
    return fids, gts, gsl, imgs


# ---- testable eval core (detector injected) -----------------------------------
def eval_view(whole_for, tiled_detect_for, img_whs, gts, gslices,
              target_fppis=(0.5, 1.0, 2.0), slice_kw=None):
    """whole_for(i)->(N,5) preds; tiled_detect_for(i)->detect(region) callable."""
    whole, tiled = [], []
    for i in range(len(gts)):
        whole.append(np.asarray(whole_for(i), float).reshape(-1, 5))
        tiled.append(run_tiled(tiled_detect_for(i), img_whs[i], **(slice_kw or {})))
    rep = {"whole": full_report(whole, gts, gslices),
           "tiled": full_report(tiled, gts, gslices),
           "matched_fppi": {}}
    for f in target_fppis:
        rep["matched_fppi"][f] = compare_at_matched_fppi(whole, tiled, gts, f)
    return rep, whole, tiled


def _print(rep):
    for name in ("whole", "tiled"):
        r = rep[name]
        print("[%s] recall@.5 %.3f  AR@100 %.3f  MR-2 %.3f  slice=%s" % (
            name, r["overall"]["recall"], r["AR@100"], r["MR-2"],
            {k: round(v, 3) for k, v in r.get("slice_recall", {}).items()}), flush=True)
    for f, c in rep["matched_fppi"].items():
        print("  @FPPI=%.1f  whole %.3f  tiled %.3f  Δrecall %+.3f" % (
            f, c["recall_base"], c["recall_var"], c["delta_recall"]), flush=True)


# ---- YOLO wiring (Colab) ------------------------------------------------------
def main(a):
    import cv2
    from ultralytics import YOLO

    model = YOLO(a.weights)
    fids, gts, gsl, imgs = load_view(a.root, a.view, a.n)
    print("[realdata] view=%s frames=%d GT_total=%d weights=%s" %
          (a.view, len(fids), int(sum(len(g) for g in gts)), a.weights), flush=True)

    def _yolo(src, imgsz):
        r = model.predict(src, classes=[0], conf=a.conf, imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return np.zeros((0, 5))
        b = r.boxes.xyxy.cpu().numpy(); s = r.boxes.conf.cpu().numpy()
        return np.column_stack([b, s])

    cache = {}

    def img_of(i):
        if i not in cache:
            cache[i] = cv2.imread(imgs[i])
        return cache[i]

    def whole_for(i):
        return _yolo(imgs[i], a.imgsz)

    def tiled_detect_for(i):
        img = img_of(i)

        def detect(region):
            x0, y0, x1, y1 = region
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                return np.zeros((0, 5))
            return _yolo(crop, a.tile)
        return detect

    img_whs = [(im.shape[1], im.shape[0]) if (im := img_of(i)) is not None else (1920, 1080)
               for i in range(len(imgs))]
    rep, _, _ = eval_view(whole_for, tiled_detect_for, img_whs, gts, gsl,
                          slice_kw={"slice": a.slice, "overlap": a.overlap})
    print("=== RECALL_REALDATA_RESULT ===", flush=True)
    _print(rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/WTx/Wildtrack_dataset")
    ap.add_argument("--view", default="C1")
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    main(ap.parse_args())
