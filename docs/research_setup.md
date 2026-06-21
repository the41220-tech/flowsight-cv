# FlowSight — 연구 세팅 & 모델 학습/데모 설계 (Colab GPU)

> 작성 2026-06-20 · 컴퓨팅 전제 = **Google Colab GPU** · 입력 = 단일 드론(부감/사선) · 우선순위 = 난제 검증.
> 비-GPU(geometry/physics/sim)는 로컬 CPU 재현, 모델 학습/추론은 Colab GPU.

## 1. 연구 루프 (운영 방식)

```
가설(5개) ─▶ 실험 스크립트 ─▶ 검증(지표/합격선) ─▶ results/*.json + figures/*
   ▲                                                        │
   └──────────────  hypotheses.md 반복 로그에 기록·개선  ◀──┘
```

- 모든 실험은 `experiments/run_*.py` 하나로 재현, 결과는 `results/*.json`·`figures/*.png`로 고정.
- 합격선/지표는 **실행 전에** 못 박는다(사후 정당화 금지). 합성→실데이터 순으로 외적 타당성 확장.
- 코드/가중치/결과는 Git + Drive에 체크포인트. 난수 seed 고정.

## 2. Colab 환경 제약과 대응

| 제약 | 대응 |
|---|---|
| 세션 ~12h·유휴 종료 | 매 epoch Drive 체크포인트, `resume_from` 지원 |
| GPU VRAM(T4 16GB / L4 24GB / A100 40GB) | 혼합정밀(bf16/fp16), grad accumulation, 입력 해상도/배치 조절 |
| 디스크 휘발 | 데이터셋·가중치는 Drive 마운트 캐시(`/content/drive`) |
| 대형 VLM(7B) | 4-bit(QInt4/AWQ) 로드, 또는 3B로 대체 |

권장 런타임: 검출 파인튜닝 = L4/A100, 추론·데모 = T4 가능.

## 3. 모델별 학습/추론 설계

### 3-1. 검출 — H3 (파인튜닝 핵심)
- 베이스: `PekingU/rtdetr_v2_r50vd`(Apache-2.0) 1순위, 정확도 비교군 RF-DETR.
- 데이터: `Voxel51/VisDrone2019-DET`(HF), DroneCrowd(드론 군중 점/박스).
- 절차: HF `Trainer` + `AutoImageProcessor`. 입력 800~1024px, bf16, lr 1e-4(backbone 1e-5), cosine, 30~50 epoch, EMA. 소형 객체 강화를 위해 타일링(슬라이스) 추론 옵션.
- 지표: person AP@0.5, 카운팅 MAE/RMSE, FPS. 베이스라인(zero-shot) 대비.

### 3-2. 깊이/측위 — H1 (실데이터 외적검증)
- 모델: `Depth-Anything-V2-Small-hf`(상대, transformers) → 앵커 메트릭화, 또는 `Depth-Anything-V2-Metric-VKITTI-Large`(네이티브 메트릭, Apache-2.0).
- 절차: 추론 중심(파인튜닝 선택). WILDTRACK 지면 GT로 측위 오차 평가, 합성 결과(H1)와 대조.

### 3-3. 장애물/식별 — H4
- 개방어휘: `IDEA-Research/grounding-dino-base` / `google/owlv2-base-patch16-ensemble`(zero-shot, 프롬프트).
- 파놉틱(선택): `facebook/mask2former-swin-base-coco-panoptic`.
- VLM: `Qwen/Qwen2.5-VL-7B-Instruct`(4-bit) — 애매 크롭 식별·경보 문구. 사람 신원엔 미사용(프라이버시).
- 절차: zero-shot 우선, 필요 시 LoRA 경량 파인튜닝.

### 3-4. 밀집/예측/이상 (v2)
- 밀집: CrowdDiff·MovingDroneCrowd SDNet 직접 학습(ShanghaiTech/DroneCrowd).
- 예측: MoFlow(ETH-UCY/SDD)로 10~30초 궤적.
- 이상·경보: EventVAD(멀티모달 LLM, training-free) + 규칙 → LLM→TTS 3단계.

## 4. 데이터 플랜

| 용도 | 데이터 | 비고 |
|---|---|---|
| 검출 파인튜닝(H3) | VisDrone(HF Voxel51), DroneCrowd | 드론 부감 |
| 측위 외적검증(H1) | WILDTRACK, MultiviewX | 지면 GT·캘리브 |
| 밀집(H5) | ShanghaiTech A/B, JHU-CROWD++ | 밀도 회귀 |
| 예측 | ETH-UCY, Stanford Drone | 궤적 |
| 비평면·압력(H1/H2) | 합성(Blender/CARLA 경사+군중) + 본 repo 시뮬 | 콜드스타트 대응 |

## 5. 데모 설계 (v0 → v1)

- **v0 데모:** 단일 드론 클립 → 검출 → (평면)호모그래피 2D 점 + BEV 밀도 히트맵. `pipeline/baseline_v0.py` + `render_bev`.
- **v1 데모(난제 시연):** 동일 클립에 ① 메트릭 깊이 3D 측위(경사부 오차 ↓ 시각화), ② 지형 위치에너지·압력 위험지도(전조 선행 타임라인), ③ 장애물→지오펜스 오버레이 + VLM 라벨.
- 형식: Colab 노트북 시각화 → 추후 Gradio/HF Space 대시보드. 발표는 BEV·히트맵·전조 타임라인 우선(탐지 박스보다).

## 6. 일정(제안)

| 주차 | 산출물 |
|---|---|
| W1 | v0 베이스라인 + H1/H2 합성 검증(완료) + 데이터 수배 |
| W2–3 | H3 검출 파인튜닝, H1 WILDTRACK 외적검증, H4 zero-shot |
| W4–5 | 결합 위험식 튜닝, v1 데모(3난제), 반복 로그 Iter-2 |
| W6+ | v2(예측·경보·다카메라), 엣지화 검토 |

## 7. 재현 방법

```bash
pip install -r requirements.txt          # CPU: numpy/opencv/matplotlib면 sim/geometry 재현
PYTHONPATH=. python experiments/run_h1_positioning.py   # results/h1.json + figures
PYTHONPATH=. python experiments/run_h2_pressure.py      # results/h2.json + figures
# 모델 파인튜닝/추론은 notebooks/00_baseline_and_setup.ipynb (Colab GPU)
```
