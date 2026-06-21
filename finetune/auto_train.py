"""Resilient chunked fine-tune harness (Colab) — autonomous, disconnect-safe.

Implements the autonomy requirements:
  - CHUNKS: trains in `--chunk` (default 5) epoch segments, each continuing from
    the previous chunk's weights -> a disconnect loses at most one chunk.
  - DISCONNECT MGMT: every chunk's best/last weights are copied to `--ckpt`
    (point this at Google Drive for true disconnect-safety). On restart the
    harness auto-resumes from the latest chunk checkpoint.
  - SELF ERROR-RECOVERY: on CUDA OutOfMemory it halves the batch and retries the
    chunk (up to 4x) instead of crashing.
  - STEP LOG: appends one JSON line per event to `<ckpt>/run_log.jsonl` so any
    regression is traceable to the exact chunk.

Usage (Colab; mount Drive first for safety):
  !PYTHONPATH=. python finetune/auto_train.py --data datasets/person_mix/person.yaml \
      --total 30 --chunk 5 --imgsz 768 --batch 12 \
      --ckpt /content/drive/MyDrive/flowsight_ckpt
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, time
from pathlib import Path


def log(logpath, rec):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(logpath, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[LOG]", json.dumps(rec))


def latest_ckpt(ckpt_dir):
    ws = sorted(glob.glob(os.path.join(ckpt_dir, "chunk*_last.pt")))
    return ws[-1] if ws else None


def train_chunk(model_path, data, epochs, imgsz, batch, name, project):
    from ultralytics import YOLO
    m = YOLO(model_path)
    m.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=0,
            project=project, name=name, exist_ok=True, patience=100, seed=0,
            cos_lr=True, mosaic=1.0, close_mosaic=0, plots=False, verbose=False)
    sd = Path(m.trainer.save_dir)
    fit = float(getattr(m.trainer, "best_fitness", 0.0) or 0.0)
    return str(sd / "weights" / "best.pt"), str(sd / "weights" / "last.pt"), fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--total", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--base", default="yolo11m.pt")
    ap.add_argument("--ckpt", default="runs_auto/ckpt")
    ap.add_argument("--project", default="runs_auto")
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
        for attempt in range(4):
            try:
                name = f"chunk{chunk_idx:02d}"
                best, last, fit = train_chunk(model_path, a.data, n, a.imgsz, batch, name, a.project)
                shutil.copy(best, os.path.join(a.ckpt, f"chunk{chunk_idx:02d}_best.pt"))
                shutil.copy(last, os.path.join(a.ckpt, f"chunk{chunk_idx:02d}_last.pt"))
                model_path = os.path.join(a.ckpt, f"chunk{chunk_idx:02d}_last.pt")
                log(logpath, {"event": "chunk_done", "chunk": chunk_idx, "epochs": n,
                              "batch": batch, "best_fitness": round(fit, 4), "weights": model_path})
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    try:
                        import torch, gc; gc.collect(); torch.cuda.empty_cache()
                    except Exception:
                        pass
                    batch = max(2, batch // 2)
                    log(logpath, {"event": "oom_retry", "chunk": chunk_idx, "new_batch": batch,
                                  "err": str(e)[:120]})
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
    print("FINAL:", model_path)


if __name__ == "__main__":
    main()
