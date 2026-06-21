# 파인튜닝 설계서 (검토용 — 승인 전 학습 미착수)

> 작성 2026-06-21 · 근거: `results/bench_recall_summary.md` · 컴퓨팅: Colab T4 · **이 문서는 검토/승인 게이트입니다. 승인 전엔 학습을 시작하지 않습니다.**

## 1. 문제 정의 (벤치마크가 말해주는 것)

COCO 사전학습 모델은 **CCTV/거리(0.71~0.92)는 이미 잘하지만 드론·소형 객체(0.15~0.59)에서 크게 무너진다.** 안전 시스템에서 치명적 오류는 "사람을 놓치는 것"(recall) → **드론 도메인 recall을 올리되 CCTV를 망치지 않는 것**이 목표.

## 2. 목표 지표 (사전 고정)

| 지표 | 베이스라인(현재) | 목표 |
|---|---|---|
| 드론 recall @IoU0.5, score0.25 | RT-DETRv2 0.506 / 앙상블 0.588 | **≥ 0.80** |
| 소형(small) recall | 0.468 / 0.551 | **≥ 0.70** |
| CCTV recall (퇴행 방지) | 0.915 | **≥ 0.88 유지** |
| precision (드론) | 0.38 | **≥ 0.60** (오탐 절반↓) |
| 실시간성 | — | T4에서 ≥ 15 FPS (배포 후보) |

검증 = **같은 `bench_recall.py` 100장 프로토콜**로 전/후 비교 + VisDrone val 표준 mAP50.

## 3. 모델 선택

- **1순위(정확도·recall): RT-DETRv2 파인튜닝.** recall 리더 + Apache-2.0. 작은 객체에 강한 고해상 입력 가능.
- **2순위(속도·엣지): YOLO11 파인튜닝.** 빠르고 파인튜닝·증강·검증 파이프라인이 ultralytics에 내장 → Colab에서 가장 빠르게 돌릴 수 있어 **1차 PoC에 적합**.
- 둘 다 학습 후 비교 → 정확도형/속도형 2개 배포 후보 확보. (앙상블은 recall 상한 참고용으로 유지.)

> 실용 결정: **1차는 ultralytics YOLO11로 빠르게 검증**(데이터·증강·val 내장), 효과 확인되면 RT-DETRv2로 확장. 이유: T4에서 반복 속도 + 소형객체 증강(mosaic/copy-paste)·타일 추론(SAHI) 기본 지원.

## 4. 데이터 설계

- **VisDrone-DET train (6,471장)** — 이미 다운로드/변환됨. `pedestrian`+`people` → 단일 **person** 클래스로 병합(우리는 "사람"만 필요).
- **혼합 학습(중요):** VisDrone(드론) + **COCO person 일부**(거리/CCTV)를 섞어 학습 → CCTV 성능 퇴행(catastrophic forgetting) 방지, "입력원 무관" 목표 부합.
- (선택) **DroneCrowd** 추가 → 초고밀도 드론 군중 보강(2차).
- split: VisDrone train→학습, VisDrone val(548장)→검증, + 우리 100장 벤치는 **건드리지 않는 홀드아웃**.

## 5. 학습 레시피 (Colab T4)

| 항목 | 1차(빠른 PoC) | 2차(확장) |
|---|---|---|
| 모델 | YOLO11m | YOLO11x + RT-DETRv2 |
| 입력 해상도 | 960 | 1024~1280 (소형객체↑) |
| 증강 | mosaic, copy-paste, scale jitter | + 타일/SAHI 추론 |
| epochs | ~30 (early stop) | ~80~100 |
| batch | T4 AMP에 맞춰 8~16 | grad-accum로 유효배치↑ |
| LR | cos, warmup | 동일 + EMA |
| 체크포인트 | **매 epoch → Google Drive**(세션 끊김 대비, resume) | 동일 |
| 소요(추정) | ~30~60분 | ~3~5시간(분할/resume) |

## 6. 검증·산출물

1. 학습 후 모델을 `bench_recall.py`에 1줄 추가(어댑터) → **전/후 드론·소형·CCTV recall 비교표**.
2. VisDrone val mAP50/recall 곡선, PR 커브.
3. 정성 오버레이(전/후 같은 프레임에서 검출 수 비교).
4. 결과를 `results/`에 저장 + 반복 로그(`hypotheses.md` H3) 기입.

## 7. 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| CCTV 퇴행 | 혼합 학습 + CCTV recall를 합격 조건(≥0.88)에 포함 |
| T4 세션 끊김 | Drive 체크포인트 + `resume=True` |
| 과적합 | 강한 증강 + val early-stop |
| 소형객체 한계 | 고해상 입력 + 타일 추론(SAHI) |
| 클래스 정의 | pedestrian+people→person 병합 일관 적용 |
| 시간 초과 | 1차는 작은 모델·subset으로 효과부터 확인(게이트) |

## 8. 단계별 실행안 (각 단계 후 결과 보고)

- **FT-0 데이터 준비:** VisDrone→person-only YOLO 라벨 + COCO person 혼합셋 구성, Drive에 캐시. (코드: `finetune/prepare_data.py`)
- **FT-1 빠른 PoC:** YOLO11m @960, ~30ep → 100장 벤치 재측정. *게이트: 드론 recall이 베이스라인 대비 유의하게 오르면 진행.*
- **FT-2 확장:** 해상도/에폭↑, DroneCrowd 추가, 타일 추론 → 목표(드론≥0.80) 도전.
- **FT-3 RT-DETRv2 파인튜닝:** recall 리더 버전 → 정확도형 배포 후보. 속도형(YOLO)과 비교.

## 9. 승인 요청 (당신 결정 필요)

1. **1차 모델**: YOLO11m로 빠르게 시작 OK? (아니면 처음부터 RT-DETRv2)
2. **혼합 학습**: VisDrone + COCO person 혼합으로 CCTV 보존 — 동의?
3. **목표치**(드론 recall ≥0.80, CCTV ≥0.88 유지) 적절한가?
4. **컴퓨팅**: T4로 1차 ~30~60분 진행 OK? (장시간 2차는 Drive 체크포인트로 분할)
5. 코드는 repo에 추가 → push 후 제가 Colab에서 학습 실행/모니터링.

> 승인(또는 수정 지시)을 주시면 FT-0 데이터 준비 코드부터 작성해 검토받고, FT-1 학습을 Colab에서 돌리겠습니다.
