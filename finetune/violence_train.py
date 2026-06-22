"""Resilient violence classifier training (Phase B) — disconnect-safe, unattended.

YOLOv8-cls on the RWF-2000 ImageFolder (from prepare_rwf.py). Mirrors the proven
detector harness (auto_train.py): trains in epoch CHUNKS, copies each chunk's
best/last weights to `--ckpt` (point at Google Drive), AUTO-RESUMES from the
latest chunk on restart, and HALVES the batch on CUDA OOM. One JSON line per
event to `<ckpt>/run_log.jsonl`. Built for Colab Pro **background execution** so
it survives idle/disconnect over hours.

Run on Colab (mount Drive first):
  !PYTHONPATH=. python -u finetune/violence_train.py --data /content/rwf_cls \
      --total 30 --chunk 5 --imgsz 224 --batch 64 \
      --ckpt /content/drive/MyDrive/flowsight_violence_ckpt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path


def log(logpath: str, rec: dict) -> None:
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(logpath, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[LOG]", json.dumps(rec), flush=True)


def latest_ckpt(ckpt_dir: str):
    ws = sorted(glob.glob(os.path.join(ckpt_dir, "chunk*_last.pt")))
    return ws[-1] if ws else None


def train_chunk(model_path: str, data: str, epochs: int, imgsz: int, batch: int,
                name: str, project: str):
    from ultralytics import YOLO

    m = YOLO(model_path)
    m.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=0,
            project=project, name=name, exist_ok=True, patience=100, seed=0,
            cos_lr=True, plots=False, verbose=False)
    sd = Path(m.trainer.save_dir)
    top1 = float(getattr(getattr(m.trainer, "metrics", None), "top1", 0.0) or 0.0)
    return str(sd / "weights" / "best.pt"), str(sd / "weights" / "last.pt"), top1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ImageFolder dir (rwf_cls)")
    ap.add_argument("--total", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--base", default="yolov8s-cls.pt")
    ap.add_argument("--ckpt", default="/content/drive/MyDrive/flowsight_violence_ckpt")
    ap.add_argument("--project", default="runs_violence")
    a = ap.parse_args()

    os.makedirs(a.ckpt, exist_ok=True)
    logpath = os.path.join(a.ckpt, "run_log.jsonl")
    done = len(glob.glob(os.path.join(a.ckpt, "chunk*_last.pt")))
    model_path = latest_ckpt(a.ckpt) or a.base
    log(logpath, {"event": "start", "resume_from": model_path, "chunks_done": done,
                  "total": a.total, "chunk": a.chunk, "imgsz": a.imgsz, "batch": a.batch})

    chunk_idx, trained = done, done * a.chunk
    while trained < a.total:
        n = min(a.chunk, a.total - trained)
        batch = a.batch
        for _ in range(4):
            try:
                name = f"chunk{chunk_idx:02d}"
                best, last, top1 = train_chunk(model_path, a.data, n, a.imgsz, batch,
                                               name, a.project)
                shutil.copy(best, os.path.join(a.ckpt, f"chunk{chunk_idx:02d}_best.pt"))
                shutil.copy(last, os.path.join(a.ckpt, f"chunk{chunk_idx:02d}_last.pt"))
                model_path = os.path.join(a.ckpt, f"chunk{chunk_idx:02d}_last.pt")
                log(logpath, {"event": "chunk_done", "chunk": chunk_idx, "epochs": n,
                              "batch": batch, "top1": round(top1, 4), "weights": model_path})
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    try:
                        import gc

                        import torch
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    batch = max(8, batch // 2)
                    log(logpath, {"event": "oom_retry", "chunk": chunk_idx,
                                  "new_batch": batch, "err": str(e)[:120]})
                    continue
                log(logpath, {"event": "error", "chunk": chunk_idx, "err": str(e)[:200]})
                raise
        else:
            log(logpath, {"event": "abort", "chunk": chunk_idx, "reason": "OOM after retries"})
            break
        trained += n
        chunk_idx += 1

    log(logpath, {"event": "finished", "chunks": chunk_idx, "epochs_trained": trained,
                  "final_weights": model_path})
    print("FINAL:", model_path, flush=True)


if __name__ == "__main__":
    main()
