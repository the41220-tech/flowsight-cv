"""FT-2 data: mixed person set = VisDrone (drone) + COCO val2017 (street/CCTV).

Why mix: FT-1 (VisDrone-only) collapsed CCTV recall 0.685->0.117 (forgetting).
Mixing street/person back in should hold drone gains AND recover CCTV.

No leakage: COCO val2017 is a separate split from coco128 (a train2017 subset),
so training on val2017-person does NOT touch the 100-img cctv holdout.

Builds datasets/person_mix/{images,labels}/{train,val}:
  train = VisDrone-person train  +  COCO val2017 person
  val   = VisDrone-person val (drone)         # cctv retention measured by the holdout eval
All class -> 0 (person). Run on Colab after FT-1's prepare (VisDrone-person exists).
  !PYTHONPATH=. python finetune/prepare_mix.py
"""
from __future__ import annotations
import json, os
from collections import defaultdict
from pathlib import Path

VIS = Path("datasets/person_visdrone")           # built by prepare_data.py (FT-1)
OUT = Path("datasets/person_mix")
COCO = Path("datasets/coco_raw")


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(Path(src).resolve(), dst)


def link_visdrone():
    n = 0
    for split in ("train", "val"):
        for lp in (VIS / "labels" / split).glob("*.txt"):
            ip = VIS / "images" / split / (lp.stem + ".jpg")
            if not ip.exists():
                continue
            _link(lp, OUT / "labels" / split / lp.name)
            _link(ip, OUT / "images" / split / ip.name)
            n += 1
    return n


def add_coco_person():
    from ultralytics.utils.downloads import download
    if not (COCO / "val2017").exists():
        download(["http://images.cocodataset.org/zips/val2017.zip",
                  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"],
                 dir=str(COCO), unzip=True, threads=2)
    J = json.load(open(COCO / "annotations" / "instances_val2017.json"))
    imgs = {im["id"]: im for im in J["images"]}
    boxes = defaultdict(list)
    for a in J["annotations"]:
        if a["category_id"] == 1 and not a.get("iscrowd", 0):     # person
            boxes[a["image_id"]].append(a["bbox"])                 # [x,y,w,h] abs
    n = 0
    ldir = OUT / "labels" / "train"; idir = OUT / "images" / "train"
    ldir.mkdir(parents=True, exist_ok=True); idir.mkdir(parents=True, exist_ok=True)
    for iid, bxs in boxes.items():
        im = imgs[iid]; W, H = im["width"], im["height"]
        lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                 for x, y, w, h in bxs if w > 1 and h > 1]
        if not lines:
            continue
        stem = Path(im["file_name"]).stem
        (ldir / f"coco_{stem}.txt").write_text("\n".join(lines) + "\n")
        src = COCO / "val2017" / im["file_name"]
        if src.exists():
            _link(src, idir / f"coco_{stem}.jpg")
            n += 1
    return n


if __name__ == "__main__":
    nv = link_visdrone()
    nc = add_coco_person()
    (OUT / "person.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    ntr = len(list((OUT / "images" / "train").glob("*")))
    nva = len(list((OUT / "images" / "val").glob("*")))
    print(f"VisDrone linked: {nv} | COCO-person added: {nc}")
    print(f"person_mix -> train {ntr} imgs / val {nva} imgs")
    print("data yaml ->", (OUT / "person.yaml").resolve())
