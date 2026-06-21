# FlowSight AI — research scaffold (v0)

입력원 무관 군중 안전 인텔리전스의 **난제 검증용 연구 스캐폴드**. 이번 단계 초점:
단일 사선/이동 드론에서 **① 비평면 3D 측위 → ② 지형 위치에너지·압력장 → ③ 장애물 인식·식별**.

> 컴퓨팅: Google Colab GPU(모델), 로컬 CPU(geometry/physics/sim). 방법론: 5가설 검증·개선 루프 (`experiments/hypotheses.md`).

## 구조

```
flowsight/
  geometry/   camera(핀홀·투영·역투영) · terrain(고도·경사·광선교차) · homography · metric_depth
  physics/    density(people/m²) · potential(U·∇U) · pressure(Helbing + 지형결합 위험식)
  sim/        social_force_terrain(비평면 Social-Force, H1/H2 합성 GT)
  perception/ detect(RT-DETRv2) · obstacles(GroundingDINO + Mask2Former + Qwen2.5-VL)
  pipeline/   baseline_v0(검출→지면맵→밀도·위험→BEV)
experiments/  run_h1_positioning.py · run_h2_pressure.py · hypotheses.md · results/ · figures/
docs/         research_setup.md (Colab 학습/데모 설계)
notebooks/    00_baseline_and_setup.ipynb (Colab)
```

## 빠른 시작 (CPU로 난제 검증 재현)

```bash
pip install numpy opencv-python-headless matplotlib
PYTHONPATH=. python experiments/run_h1_positioning.py   # 비평면 측위
PYTHONPATH=. python experiments/run_h2_pressure.py      # 지형 압력 전조
```

## 1차 검증 결과 (2026-06-20)

- **H1** 비평면 측위: 메트릭 깊이 역투영이 단일 호모그래피 대비 경사부 오차 **−78%** (광선교차 −73%). 단순 다중 호모그래피는 오히려 악화 → 깊이 채택.
- **H2** 지형 압력 전조: 위치에너지 전조가 압사를 **7.2s 선행**, Helbing 난류 신호를 18.8s 앞섬(상승기준 0.3–0.6 전 구간 robust).

자세한 가설·프로토콜·반복 로그: [`experiments/hypotheses.md`](experiments/hypotheses.md). 학습/데모 설계: [`docs/research_setup.md`](docs/research_setup.md).

## 모델 (HF, 라이선스)
RT-DETRv2 `PekingU/rtdetr_v2_r50vd`(Apache-2.0) · Depth-Anything-V2-Metric(Apache-2.0) · Grounding DINO/OWLv2(Apache-2.0) · Mask2Former · Qwen2.5-VL-7B(Apache-2.0). 상세 카탈로그·데이터셋은 상위 폴더 `FlowSight_구현로드맵_모델데이터셋_2026-06-20.md`.

> 프라이버시 바이 디자인: 얼굴인식 배제, 익명 흐름만. VLM은 장애물/물체 식별에만 사용.
