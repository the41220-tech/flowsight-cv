# FT-2 결과 — 혼합학습으로 CCTV 망각 복구 (실측)

**날짜:** 2026-06-21 · **모델:** YOLO11m, person 1-class · **학습:** person_mix(VisDrone 6215 + COCO val2017 2693 = train 8377), 30 epoch, 6×5-epoch 청크, Colab T4, ~3.5h, Drive 무손실 체크포인트(chunk00~05 best/last 전부 저장).

## 전/후 recall (동일 100장 홀드아웃, thr=0.25, IoU=0.5)

| 모델 | overall | drone | cctv | small | precision |
|---|---|---|---|---|---|
| yolo11m baseline | 0.241 | 0.154 | **0.685** | 0.126 | 0.842 |
| FT-1 (VisDrone만, 5ep) | 0.447 | 0.512 | **0.117** | 0.461 | 0.456 |
| **FT-2 (혼합, 30ep)** | **0.552** | **0.571** | **0.455** | **0.517** | 0.612 |

## 판정 — H3′ 지지 (망각 복구 + 드론 향상)
- **망각 복구:** CCTV recall 0.117 → **0.455 (3.9배)**. FT-1의 치명적 망각을 혼합학습이 크게 되돌림.
- **드론 동시 향상:** 0.512 → **0.571**. 소형객체 0.461 → **0.517**. overall 0.447 → 0.552. precision도 0.456 → 0.612로 상승.
- **VisDrone-val mAP50:** FT-1 5ep 0.49 → FT-2 30ep **0.60**.

## 한계 (정직)
- CCTV 0.455는 baseline 0.685에 **아직 못 미침** — 완전 복구는 아니고 드론↔CCTV 트레이드오프가 남음. FT-2는 "균형형 제너럴리스트". 추가 개선책: COCO 비율↑(FT-3), 또는 SAHI(H7a, 재학습 없이 recall↑).
- 내적(홀드아웃) 평가 — 외적 타당성은 WILDTRACK/DroneCrowd 등 별도 데이터 필요.

## 재난 예측 의미 (실측 recall을 H7b 시뮬에 투입, #35)
실 FT-2 드론 recall 0.571을 비평면 crush 시뮬에 넣으면:

| 지표 | 값 |
|---|---|
| 검출 밀도 peak | 4.77 (GT 7.33, 과소계수 0.55) |
| 단순 계수 경보(임계 6.0/㎡) | **누락** (4.77 < 6.0) |
| **멀티모달(지형 위치에너지) 선행** | **5.8초** |

→ **계수만으로는 실 검출기로도 crush를 놓치지만, FlowSight 멀티모달 채널은 5.8초 조기경보를 유지.** 모트가 실 검출기에서도 성립. SAHI로 recall을 ~0.86까지 올리면 계수 경보까지 발화(H7).

## 산출물
- 최종 가중치(Drive): `/content/drive/MyDrive/flowsight_ckpt/chunk05_{best,last}.pt` + `run_log.jsonl`
- 평가: `finetune/eval_recall.py` (전/후), `finetune/eval_sahi.py` (H7a, 다음)
