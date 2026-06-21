"""Multi-model person-detection recall benchmark (Colab GPU).

GT domains (real, with ground-truth boxes):
  - drone : VisDrone-DET val  (ultralytics auto-download; person = {pedestrian, people})
  - cctv  : COCO128           (ultralytics; person = class 0)  -- street/CCTV proxy
            (swap in CrowdHuman/MOT20 later for surveillance-angle; heavier download)

Models (diverse; each -> person boxes+scores):
  rtdetr (PekingU/rtdetr_v2_r50vd), detr (facebook/detr-resnet-50),
  yolov8x, yolo11x (ultralytics), owlv2 (google/owlv2-base-patch16-ensemble, open-vocab),
  ensemble (union of all model boxes + NMS) -> recall ceiling.

Metric: recall/precision/F1 @ IoU>=0.5, micro-averaged, at score thresholds
{0.10, 0.25, 0.50}; broken out by domain and GT box size.

Run on Colab:
  !PYTHONPATH=. python experiments/bench_recall.py --n_drone 50 --n_cctv 50
"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np
from PIL import Image

from flowsight.eval.recall import RecallMeter, nms, iou_matrix  # noqa

HERE = os.path.dirname(__file__)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
EVAL_THRS = (0.10, 0.25, 0.50)
IOU_THR = 0.5


# ----------------------------- GT loaders -----------------------------
def _yolo_items(images_dir, person_classes, n, seed):
    """Yield (PIL RGB, gt_boxes xyxy, ) from a YOLO-format split dir."""
    from pathlib import Path
    images_dir = Path(images_dir)
    if images_dir.is_file():            # a .txt listing image paths
        paths = [Path(p.strip()) for p in images_dir.read_text().splitlines() if p.strip()]
    else:
        paths = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    random.Random(seed).shuffle(paths)
    out = []
    for p in paths:
        lp = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lp.exists():
            continue
        img = Image.open(p).convert("RGB"); W, H = img.size
        boxes = []
        for line in lp.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            c = int(float(t[0]))
            if c not in person_classes:
                continue
            cx, cy, w, h = map(float, t[1:5])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        if boxes:
            out.append((img, np.array(boxes, float)))
        if len(out) >= n:
            break
    return out


def load_visdrone(n, seed):
    from ultralytics.data.utils import check_det_dataset
    d = check_det_dataset("VisDrone.yaml")          # downloads + converts to YOLO
    return _yolo_items(d["val"], {0, 1}, n, seed)    # 0 pedestrian, 1 people


def load_coco128(n, seed):
    from ultralytics.data.utils import check_det_dataset
    d = check_det_dataset("coco128.yaml")
    src = d.get("val") or d.get("train")
    return _yolo_items(src, {0}, n, seed)            # 0 person


# ----------------------------- model adapters -----------------------------
def _person_id(cfg):
    for i, l in cfg.id2label.items():
        if str(l).lower() == "person":
            return int(i)
    return 0


class HFDetector:
    def __init__(self, mid):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(mid)
        self.model = AutoModelForObjectDetection.from_pretrained(mid).to(self.dev).eval()
        self.pid = _person_id(self.model.config)

    def __call__(self, img):
        import torch
        inp = self.proc(images=img, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_object_detection(
            out, target_sizes=[(img.height, img.width)], threshold=0.05)[0]
        lab = res["labels"].cpu().numpy()
        m = lab == self.pid
        return res["boxes"].cpu().numpy()[m], res["scores"].cpu().numpy()[m]


class YOLODetector:
    def __init__(self, weights):
        from ultralytics import YOLO
        import torch
        self.m = YOLO(weights)
        self.dev = 0 if torch.cuda.is_available() else "cpu"

    def __call__(self, img):
        r = self.m.predict(np.array(img), conf=0.05, classes=[0], verbose=False, device=self.dev)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


class OWLv2Detector:
    def __init__(self, mid="google/owlv2-base-patch16-ensemble"):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(self.dev).eval()
        self.texts = [["a photo of a person", "a pedestrian"]]

    def __call__(self, img):
        import torch
        inp = self.proc(text=self.texts, images=img, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            out = self.model(**inp)
        ts = torch.tensor([[img.height, img.width]]).to(self.dev)
        try:
            res = self.proc.post_process_grounded_object_detection(out, target_sizes=ts, threshold=0.05)[0]
        except Exception:
            res = self.proc.post_process_object_detection(out, target_sizes=ts, threshold=0.05)[0]
        return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


MODEL_FACTORY = {
    "rtdetr": lambda: HFDetector("PekingU/rtdetr_v2_r50vd"),
    "detr": lambda: HFDetector("facebook/detr-resnet-50"),
    "yolov8x": lambda: YOLODetector("yolov8x.pt"),
    "yolo11x": lambda: YOLODetector("yolo11x.pt"),
    "owlv2": lambda: OWLv2Detector(),
}


# ----------------------------- benchmark -----------------------------
def run(models, n_drone, n_cctv, seed=0):
    data = []
    try:
        data += [(im, gt, "drone") for im, gt in load_visdrone(n_drone, seed)]
    except Exception as e:
        print("[WARN] VisDrone load failed:", repr(e)[:160])
    try:
        data += [(im, gt, "cctv") for im, gt in load_coco128(n_cctv, seed)]
    except Exception as e:
        print("[WARN] COCO128 load failed:", repr(e)[:160])
    print(f"loaded {len(data)} GT images "
          f"({sum(d=='drone' for _,_,d in data)} drone / {sum(d=='cctv' for _,_,d in data)} cctv); "
          f"total GT persons: {sum(len(gt) for _,gt,_ in data)}")
    if not data:
        raise SystemExit("no GT images loaded")

    # cache raw predictions per model (boxes+scores at conf>=0.05) for ensemble + multi-thr
    raw = {m: [] for m in models}
    for name in models:
        try:
            det = MODEL_FACTORY[name]()
        except Exception as e:
            print(f"[WARN] model {name} failed to load -> skipped:", repr(e)[:160]);
            raw.pop(name); continue
        for im, _, _ in data:
            try:
                b, s = det(im)
            except Exception as e:
                b, s = np.zeros((0, 4)), np.zeros(0)
            raw[name].append((np.asarray(b, float).reshape(-1, 4), np.asarray(s, float).reshape(-1)))
        del det
        try:
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"  ran {name}")

    # ensemble = union of all models' boxes, NMS per image
    ens = []
    for i in range(len(data)):
        bb = [raw[m][i][0] for m in raw]; ss = [raw[m][i][1] for m in raw]
        B = np.concatenate(bb, 0) if bb else np.zeros((0, 4))
        S = np.concatenate(ss, 0) if ss else np.zeros(0)
        k = nms(B, S, 0.55)
        ens.append((B[k], S[k]))
    raw["ensemble"] = ens

    # evaluate at each threshold
    results = {}
    for name, preds in raw.items():
        results[name] = {}
        for thr in EVAL_THRS:
            meter = RecallMeter()
            for (im, gt, dom), (b, s) in zip(data, preds):
                keep = s >= thr
                meter.add(b[keep], s[keep], gt, IOU_THR, domain=dom)
            results[name][f"thr_{thr}"] = meter.summary()

    out = {"iou_thr": IOU_THR, "eval_thresholds": list(EVAL_THRS),
           "n_images": len(data), "models": list(raw.keys()), "results": results}
    with open(os.path.join(HERE, "results", "bench_recall.json"), "w") as f:
        json.dump(out, f, indent=2)

    # headline table @ thr 0.25
    print("\n=== Recall @ IoU0.5, score>=0.25 (micro) ===")
    print(f"{'model':<10} {'overall':>8} {'drone':>8} {'cctv':>8} {'small':>8} {'precision':>10}")
    for name in raw:
        r = results[name]["thr_0.25"]
        ov = r["overall"]["recall"]; pr = r["overall"]["precision"]
        dr = r["by_domain"].get("drone", {}).get("recall", float("nan"))
        cc = r["by_domain"].get("cctv", {}).get("recall", float("nan"))
        sm = r["by_size"].get("small", {}).get("recall", float("nan"))
        print(f"{name:<10} {ov:>8.3f} {dr:>8.3f} {cc:>8.3f} {sm:>8.3f} {pr:>10.3f}")
    print("\nsaved -> experiments/results/bench_recall.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rtdetr,detr,yolov8x,yolo11x,owlv2")
    ap.add_argument("--n_drone", type=int, default=50)
    ap.add_argument("--n_cctv", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run([m for m in a.models.split(",") if m], a.n_drone, a.n_cctv, a.seed)
