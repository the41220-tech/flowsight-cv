"""FT-0: build a person-only YOLO dataset from the ultralytics-converted VisDrone.

VisDrone YOLO labels use 0=pedestrian, 1=people, 2=bicycle, ... -> we merge
{0,1} into a single class 0 = person and drop the rest. Keeps only frames with
>=1 person. Train = VisDrone train, Val = VisDrone val (the 50-drone bench
images come from val and are NOT in train -> clean holdout).

Paths come from check_det_dataset (authoritative) and labels are derived from
the image path via images->labels (same method the benchmark loader uses), so
this is robust to the actual on-disk layout.

Run on Colab (VisDrone already downloaded by bench_recall):
  !PYTHONPATH=. python finetune/prepare_data.py
"""
from __future__ import annotations
import os
from pathlib import Path

OUT = Path("datasets/person_visdrone")
PERSON = {0, 1}                       # VisDrone: pedestrian, people


def _images(img_src):
    p = Path(img_src)
    if p.is_file():                   # a .txt listing image paths
        return [Path(x.strip()) for x in p.read_text().splitlines() if x.strip()]
    return sorted(q for q in p.rglob("*") if q.suffix.lower() in (".jpg", ".jpeg", ".png"))


def build(img_src, split):
    out_img = OUT / "images" / split
    out_lbl = OUT / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for ip in _images(img_src):
        lp = Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lp.exists():
            continue
        keep = []
        for ln in lp.read_text().splitlines():
            t = ln.split()
            if len(t) >= 5 and int(float(t[0])) in PERSON:
                keep.append("0 " + " ".join(t[1:5]))
        if not keep:
            continue
        (out_lbl / lp.name).write_text("\n".join(keep) + "\n")
        dst = out_img / ip.name
        if not dst.exists():
            os.symlink(Path(ip).resolve(), dst)
        n_img += 1
        n_box += len(keep)
    return n_img, n_box


if __name__ == "__main__":
    from ultralytics.data.utils import check_det_dataset
    d = check_det_dataset("VisDrone.yaml")
    ti, tb = build(d["train"], "train")
    vi, vb = build(d["val"], "val")
    (OUT / "person.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    print(f"train: {ti} imgs / {tb} person boxes")
    print(f"val:   {vi} imgs / {vb} person boxes")
    print("data yaml ->", (OUT / "person.yaml").resolve())
