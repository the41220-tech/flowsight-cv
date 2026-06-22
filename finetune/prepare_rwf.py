"""Prepare RWF-2000 (DanJoshua/RWF-2000, webdataset) -> ImageFolder for YOLO-cls.

Extracts K evenly-spaced frames per clip into:
    rwf_cls/train/{Fight,NonFight}/*.jpg
    rwf_cls/val/{Fight,NonFight}/*.jpg

The dataset is a webdataset of videos; the `datasets` loader's column names and
the label location vary, so this script **inspects the schema at runtime** and
auto-detects (a) the video field (bytes / path / {bytes,path} / decoded object)
and (b) the Fight/NonFight label (from a label column or the __key__ path). Run
this ATTENDED once on Colab to confirm detection, then training runs unattended.

Run on Colab:
  !PYTHONPATH=. python -u finetune/prepare_rwf.py --out /content/rwf_cls \
      --frames-per-clip 3 --val-frac 0.15
"""
from __future__ import annotations

import argparse
import os
import random
import tempfile

import cv2
import numpy as np


def _find_video_bytes(ex: dict):
    """Return raw video bytes from a webdataset example, or None."""
    for k, v in ex.items():
        kl = k.lower()
        if isinstance(v, (bytes, bytearray)):
            if kl.endswith(("mp4", "avi", "webm", "mkv")) or "video" in kl or "mp4" in kl:
                return bytes(v)
        if isinstance(v, dict) and "bytes" in v and v["bytes"]:
            return bytes(v["bytes"])
        if isinstance(v, dict) and "path" in v and v["path"] and os.path.exists(v["path"]):
            return open(v["path"], "rb").read()
        if isinstance(v, str) and v.lower().endswith((".mp4", ".avi", ".webm")) and os.path.exists(v):
            return open(v, "rb").read()
    return None


def _find_label(ex: dict):
    """Return 'Fight' / 'NonFight' / None.

    Uses ONLY the __key__ path (e.g. 'RWF-2000/train/NonFight/xxx') and explicit
    non-bytes label fields — NEVER the video bytes (str(bytes) can randomly contain
    'fight'/'nonfight' and mislabel). Check 'nonfight' BEFORE 'fight' since the
    former contains the latter as a substring.
    """
    for k, v in ex.items():
        if k.lower() in ("cls", "label", "class") and not isinstance(v, (bytes, bytearray)):
            sv = str(v).lower()
            if "nonfight" in sv or "non_fight" in sv or "normal" in sv:
                return "NonFight"
            if "fight" in sv or "violen" in sv:
                return "Fight"
    key = str(ex.get("__key__", "")).lower()
    if "nonfight" in key or "non_fight" in key or "/normal" in key:
        return "NonFight"
    if "fight" in key:
        return "Fight"
    return None


def _save_frames(vbytes: bytes, k: int, dst_dir: str, stem: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(vbytes)
        tmp = f.name
    cap = cv2.VideoCapture(tmp)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved = 0
    if n > 0:
        idxs = np.linspace(n * 0.15, n * 0.85, k).astype(int)
        for j, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if ok:
                cv2.imwrite(os.path.join(dst_dir, "%s_%d.jpg" % (stem, j)), fr)
                saved += 1
    cap.release()
    os.unlink(tmp)
    return saved


def main(a) -> None:
    from datasets import load_dataset

    ds = load_dataset(a.dataset, split="train", streaming=a.streaming)
    it = iter(ds) if a.streaming else ds
    first = next(iter(ds)) if a.streaming else ds[0]
    print("[rwf] columns:", list(first.keys()))
    print("[rwf] types:", {k: type(v).__name__ for k, v in first.items()})
    print("[rwf] sample __key__:", str(first.get("__key__", ""))[:80])

    random.seed(0)
    counts = {"Fight": 0, "NonFight": 0, "skip": 0}
    n_clip = 0
    for ex in (it if a.streaming else ds):
        if a.max_clips and n_clip >= a.max_clips:
            break
        vb = _find_video_bytes(ex)
        lab = _find_label(ex)
        if vb is None or lab is None:
            counts["skip"] += 1
            continue
        split = "val" if random.random() < a.val_frac else "train"
        dst = os.path.join(a.out, split, lab)
        os.makedirs(dst, exist_ok=True)
        s = _save_frames(vb, a.frames_per_clip, dst, "%s_%05d" % (lab, n_clip))
        if s:
            counts[lab] += 1
        n_clip += 1
        if n_clip % 100 == 0:
            print("[rwf]   processed %d clips, counts=%s" % (n_clip, counts), flush=True)

    print("=== PREP_RWF_DONE: clips=%d, counts=%s -> %s ===" % (n_clip, counts, a.out),
          flush=True)
    if counts["Fight"] == 0 or counts["NonFight"] == 0:
        print("!! WARNING: a class is empty -> label auto-detect failed; inspect the "
              "printed columns/types above and adjust _find_label/_find_video_bytes.",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DanJoshua/RWF-2000")
    ap.add_argument("--out", default="/content/rwf_cls")
    ap.add_argument("--frames-per-clip", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all")
    ap.add_argument("--streaming", action="store_true",
                    help="stream the dataset instead of downloading fully")
    main(ap.parse_args())
