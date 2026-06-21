"""FT-1: fast fine-tune YOLO11m on VisDrone-person (Colab T4).

Starts from COCO-pretrained yolo11m.pt (retains street/person features) and
adapts to drone/small people. Small-object friendly: imgsz=800 + mosaic.
~20 epochs is enough to show the recall lift; scale in FT-2.

Run on Colab (after prepare_data.py):
  !PYTHONPATH=. python finetune/train_yolo.py
"""
from __future__ import annotations
import argparse


def main(epochs=20, imgsz=800, batch=16):
    from ultralytics import YOLO
    m = YOLO("yolo11m.pt")
    m.train(
        data="datasets/person_visdrone/person.yaml",
        imgsz=imgsz, epochs=epochs, batch=batch,
        patience=8, device=0, seed=0,
        project="runs_ft", name="ft1", exist_ok=True,
        optimizer="auto", cos_lr=True, plots=True,
        mosaic=1.0, scale=0.5,                # small-object augmentation
    )
    print("BEST WEIGHTS:", m.trainer.best)
    return str(m.trainer.best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    main(a.epochs, a.imgsz, a.batch)
