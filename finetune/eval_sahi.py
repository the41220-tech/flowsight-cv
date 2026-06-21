"""H7a: does SAHI sliced inference recover drone/small-person recall on top of
the FT-2 detector — with NO additional training?

Same 100-image holdout + RecallMeter as bench/eval_recall (numbers directly
comparable). Compares, on the identical drone+cctv+small split:
  - FT-2 plain (full-frame predict)
  - FT-2 + SAHI (overlapping tiles -> small objects get more pixels)

Research basis: SAHI lifts small-object recall a lot with no retraining
(lit. up to 31.8->86.4%; VisDrone +5-7% AP). H7b (run_h7_sahi_disaster.py)
showed why this matters: density-threshold crush alarm is recall-sensitive.

Run on Colab AFTER FT-2 finishes (needs the trained weights + a GPU):
  !pip -q install sahi
  !PYTHONPATH=. python -u finetune/eval_sahi.py \
      --weights /content/drive/MyDrive/flowsight_ckpt/best.pt \
      --slice 512 --overlap 0.2
"""
from __future__ import annotations
import argparse
import numpy as np

from flowsight.eval.recall import RecallMeter
from experiments.bench_recall import load_visdrone, load_coco128, YOLODetector


class SahiDetector:
    """Same (img)->(xyxy, scores) interface as YOLODetector, via SAHI tiling."""
    def __init__(self, weights, slice_px=512, overlap=0.2, conf=0.05):
        import torch
        from sahi import AutoDetectionModel
        self.slice_px = slice_px
        self.overlap = overlap
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        # newer SAHI uses model_type="ultralytics"; fall back to "yolov8"
        try:
            self.model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=weights,
                confidence_threshold=conf, device=dev)
        except Exception:
            self.model = AutoDetectionModel.from_pretrained(
                model_type="yolov8", model_path=weights,
                confidence_threshold=conf, device=dev)

    def __call__(self, img):
        from sahi.predict import get_sliced_prediction
        res = get_sliced_prediction(
            np.array(img), self.model,
            slice_height=self.slice_px, slice_width=self.slice_px,
            overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap,
            verbose=0)
        b, s = [], []
        for o in res.object_prediction_list:
            if o.category.id == 0:                       # single-class: person
                bb = o.bbox
                b.append([bb.minx, bb.miny, bb.maxx, bb.maxy])
                s.append(o.score.value)
        if not b:
            return np.zeros((0, 4)), np.zeros(0)
        return np.array(b, float), np.array(s, float)


def eval_detector(det, data, thr=0.25):
    m = RecallMeter()
    for im, gt, dom in data:
        b, s = det(im)
        keep = s >= thr
        m.add(b[keep], s[keep], gt, 0.5, domain=dom)
    return m.summary()


def row(name, s):
    ov = s["overall"]
    dr = s["by_domain"].get("drone", {}).get("recall", 0.0)
    cc = s["by_domain"].get("cctv", {}).get("recall", 0.0)
    sm = s["by_size"].get("small", {}).get("recall", 0.0)
    print(f"{name:<22} {ov['recall']:>8.3f} {dr:>8.3f} {cc:>8.3f} {sm:>8.3f} {ov['precision']:>8.3f}")


def main(weights, slice_px, overlap, n_drone=50, n_cctv=50):
    data = ([(im, gt, "drone") for im, gt in load_visdrone(n_drone, 0)]
            + [(im, gt, "cctv") for im, gt in load_coco128(n_cctv, 0)])
    print(f"H7a SAHI eval | imgs={len(data)} thr=0.25 IoU=0.5 slice={slice_px} overlap={overlap}\n")
    print(f"{'model':<22} {'overall':>8} {'drone':>8} {'cctv':>8} {'small':>8} {'prec':>8}")
    row("FT-2 plain", eval_detector(YOLODetector(weights), data))
    row("FT-2 + SAHI", eval_detector(SahiDetector(weights, slice_px, overlap), data))
    print("\nH7a PASS if SAHI lifts drone/small recall vs plain (precision may dip slightly).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--n_drone", type=int, default=50)
    ap.add_argument("--n_cctv", type=int, default=50)
    a = ap.parse_args()
    main(a.weights, a.slice, a.overlap, a.n_drone, a.n_cctv)
