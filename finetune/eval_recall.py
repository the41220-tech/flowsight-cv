"""FT eval: before/after recall on the SAME 100-image bench (no leakage).

Reuses the exact GT loaders + RecallMeter from the benchmark, so numbers are
directly comparable to results/bench_recall_summary.md. Evaluates the baseline
yolo11m.pt vs the fine-tuned weights on the same drone+cctv holdout.

Run on Colab:
  !PYTHONPATH=. python finetune/eval_recall.py --weights runs_ft/ft1/weights/best.pt
"""
from __future__ import annotations
import argparse
from flowsight.eval.recall import RecallMeter
from experiments.bench_recall import load_visdrone, load_coco128, YOLODetector


def eval_model(weights, data, thr=0.25):
    det = YOLODetector(weights)
    m = RecallMeter()
    for im, gt, dom in data:
        b, s = det(im)
        keep = s >= thr
        m.add(b[keep], s[keep], gt, 0.5, domain=dom)
    return m.summary()


def main(weights, n_drone=50, n_cctv=50):
    data = ([(im, gt, "drone") for im, gt in load_visdrone(n_drone, 0)]
            + [(im, gt, "cctv") for im, gt in load_coco128(n_cctv, 0)])
    print(f"eval images: {len(data)} | thr=0.25 IoU=0.5\n")
    print(f"{'model':<24} {'overall':>8} {'drone':>8} {'cctv':>8} {'small':>8} {'prec':>8}")
    for name, w in [("yolo11m baseline", "yolo11m.pt"), ("yolo11m FINETUNED", weights)]:
        s = eval_model(w, data)
        ov = s["overall"]
        dr = s["by_domain"].get("drone", {}).get("recall", 0.0)
        cc = s["by_domain"].get("cctv", {}).get("recall", 0.0)
        sm = s["by_size"].get("small", {}).get("recall", 0.0)
        print(f"{name:<24} {ov['recall']:>8.3f} {dr:>8.3f} {cc:>8.3f} {sm:>8.3f} {ov['precision']:>8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n_drone", type=int, default=50)
    ap.add_argument("--n_cctv", type=int, default=50)
    a = ap.parse_args()
    main(a.weights, a.n_drone, a.n_cctv)
