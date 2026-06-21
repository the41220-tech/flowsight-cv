# FT-1 result — before/after (2026-06-21, Colab T4)

YOLO11m fine-tuned on VisDrone-person (5 epochs, imgsz800, interrupted at the
target checkpoint). Same 100-image holdout as the benchmark (50 VisDrone-val
drone + 50 COCO128 cctv; neither in training). Recall @IoU0.5, score≥0.25.

| model | overall | drone | cctv | small | precision |
|---|---|---|---|---|---|
| yolo11m baseline (COCO) | 0.241 | 0.154 | 0.685 | 0.126 | 0.842 |
| yolo11m FINE-TUNED (5ep) | 0.447 | **0.512** | **0.117** | **0.461** | 0.456 |

VisDrone-val (531 imgs) person mAP50 trajectory during training: 0.333 → 0.378
→ 0.411 → 0.451 → 0.490 (epochs 1–5). R 0.347 → 0.466.

## Findings (hypothesis loop)
- **H3 CONFIRMED (strongly):** drone fine-tuning lifts drone recall 0.154→0.512 (3.3×)
  and small-object 0.126→0.461 (3.7×) in just 5 epochs. The drone/small gap was
  real and is closable.
- **Catastrophic forgetting CONFIRMED:** training on VisDrone-only collapsed CCTV
  recall 0.685→0.117. Precision also dropped (0.842→0.456) as recall rose.
- **New hypothesis H3′:** mixed VisDrone + COCO-person training (FT-2) retains CCTV
  (≥0.88 target) while keeping the drone gains. This is the next iteration.

## Engineering notes
- batch16@imgsz800 sat at the T4's 14.9 GB edge → ultralytics fell back to CPU for
  the label-assigner on spike batches (graceful, slower). FT-2: batch12 or imgsz768.
- Subprocess stdout is block-buffered; use `python -u` to read live output via the browser.
- best.pt path: runs/detect/runs_ft/ft1/weights/best.pt
