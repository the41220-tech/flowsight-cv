# Recall benchmark — results (2026-06-21, Colab T4)

100 real GT images (50 VisDrone drone / 50 COCO128 street-CCTV), 1,303 GT persons.
Recall/precision @ IoU≥0.5, micro-averaged, person class. Headline at score≥0.25.

| model | overall | drone | cctv | small | precision |
|---|---|---|---|---|---|
| rtdetr (PekingU/rtdetr_v2_r50vd) | 0.573 | 0.506 | 0.915 | 0.468 | 0.379 |
| detr (facebook/detr-resnet-50) | 0.419 | 0.327 | 0.892 | 0.306 | 0.301 |
| yolov8x | 0.292 | 0.207 | 0.728 | 0.171 | 0.816 |
| yolo11x | 0.265 | 0.177 | 0.714 | 0.151 | 0.885 |
| owlv2 (open-vocab) | 0.432 | 0.501 | 0.080 | 0.435 | 0.510 |
| **ensemble (union+NMS)** | **0.642** | **0.588** | 0.915 | **0.551** | 0.271 |

## Read

1. **Drone / small-object gap is the real weakness.** Every COCO-pretrained model is far worse on drone (0.18–0.59) than CCTV (0.71–0.92); small-object recall is the floor (0.15–0.55). This is exactly hypothesis **H3** → motivates fine-tuning on the drone domain.
2. **CCTV/street is already solved** by COCO models (rtdetr 0.92, detr 0.89). So fine-tuning must target drone **without** wrecking CCTV (input-agnostic goal → mixed-domain training).
3. **Recall leader = RT-DETRv2** (0.573 overall, 0.506 drone, 0.468 small) — but low precision (0.38) at this threshold.
4. **Ensemble lifts recall** to 0.642 / 0.588 drone / 0.551 small ("다양한 모델로 정확도↑" realized) — at precision cost (0.27). Best recall-oriented baseline.
5. **Precision/recall split:** YOLO = high precision (0.82–0.89) / low recall; RT-DETR & ensemble = high recall / low precision. For crowd **safety, recall is paramount** (missing people is the dangerous error) → RT-DETR / ensemble is the right baseline to build on, then lift precision via fine-tuning.

## Caveats
- Recall is threshold-dependent; @0.10 is higher, @0.50 lower (full per-threshold data in `bench_recall.json` on the run).
- COCO128 is a street/person proxy, not surveillance-angle CCTV (CrowdHuman/MOT20 = heavier, next round).
- **OWLv2 cctv=0.08 is anomalous** (open-vocab score scale vs the 0.25 cut likely suppressed street boxes); treat as suspect, investigate before trusting OWLv2 numbers.
- Baselines are zero-shot COCO models on out-of-domain drone data — the gap is expected, not a bug.

Pre-finetune baseline to beat: **RT-DETRv2 drone recall 0.506, small 0.468** (and ensemble 0.588 / 0.551).
