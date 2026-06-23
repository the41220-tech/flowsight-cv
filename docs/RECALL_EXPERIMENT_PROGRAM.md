# 사람검출 recall 개선 — 실험 프로그램 (가설→실험→검증→수정 loop)

> 문제: WILDTRACK 실데이터에서 엔드투엔드 recall ~0.1 (원인 = 검출기 도메인 불일치, geometry 아님).
> 설계: **fugu(Sakana)** 가 아키텍처/loop 설계, **Claude** 가 구현·검증. 코드 `flowsight/eval/{slice_metrics,nms_variants}.py` + `experiments/recall_lab.py`, 테스트 `tests/test_recall_lab.py` (8/8, 전체 37/37).

## 1. 아키텍처 (fugu 설계)

| 모듈 | 파일 | 역할 |
|---|---|---|
| **Registry** | `recall_lab.build_registry()` | 10개 가설 등록(필드: `tier`=cached/detector/train, `target_slices`, `params` sweep, `apply`/`spec`) |
| **Runner** | `recall_lab.run_cached_variant` | 고정 split, baseline 캐시, 변형 적용→추론/평가 |
| **Metrics** | `eval/slice_metrics.py` | recall@IoU0.5/0.75, AR@maxDets, **MR-2**, FPPI/FROC, **슬라이스별** recall |
| **Comparator** | `recall_lab.decide` | **동일 FPPI에서 ΔRecall** + 전체 recall 비하락 → ACCEPT/REJECT |
| **Loop** | `recall_lab.RecallLab` | propose(비용순)→run→verify→record, 사이클 단위 |

## 2. loop 단계와 채택 규칙

```
baseline 평가(현 파이프라인=hard-NMS)
  └ propose(): 미실행 중 가장 싼 tier 선택
      └ 실험: params sweep로 변형 적용
          └ 검증(Comparator): recall_at_fppi(변형) − recall_at_fppi(baseline) @ 동일 target_fppi
              ├ ΔRecall@FPPI > eps(0.005) AND 전체 recall ≥ baseline−0.02  → ACCEPT
              └ 아니면 REJECT → 다음 가설 제안
```

**핵심 원칙(절대 위반 금지):** 평균 recall 단독 비교 금지 — **동일 FPPI/precision에서** 비교. 평가 전 `full/visible/head/foot/ignore` 정의를 고정. 슬라이스(소형·가림·군집·절단·저조도)별 recall을 따로 본다.

## 3. 비용순 실행 (게이트)

1. **cached (재학습 0, 지금 실행 가능)** — 캐시된 검출 박스에만 작동:
   - **H3 NMS변형**(Soft/DIoU/WBF) — 군집 중복 억제 완화.
   - **H4 threshold** — 저confidence 소형·가림 생존(동일 FPPI 비교로 공정).
   - 게이트: 여기서 ΔRecall@FPPI가 유의하면 즉시 채택(무비용). 이번 합성 데모에선 +0.002~0.006로 REJECT → 검출기 문제임을 재확인.
2. **detector (재학습 0이나 라이브 모델 필요 → Colab)**:
   - **H5 고해상도·multi-scale·tiling + WBF**, **H10 head/pose cascade**.
3. **train (본학습 → Colab GPU)**:
   - **H1 머리/상체 aux head**, **H2 visible+amodal**, **H6 P2/P1·BiFPN**, **H7 anchor재설계/ATSS·SimOTA**, **H8 focal/varifocal·IoU-aware**, **H9 occlusion/copy-paste aug**.

## 4. 가설 레지스트리

| id | 변경점 | target slice | tier | 기대 | 리스크 |
|---|---|---|---|---|---|
| H3_nms | Soft/DIoU/WBF | crowd | cached | 군집 recall↑ | dup FP↑ |
| H4_threshold | 하향/적응 임계 | small,occluded | cached | 저score 생존 | precision↓ |
| H5_tiling | 고해상도·tiling | small | detector | 소형 feature 보존 | latency·경계 |
| H10_pose | head/pose 제안 | occluded | detector | 몸 가려도 머리 단서 | dup FP·latency |
| H1_head | 발→머리/상체 aux | occluded,trunc | train | 발 가림 recall↑ | head FP |
| H2_amodal | visible+full 이중 | occluded | train | occlusion recall↑ | 라벨비용 |
| H6_p2 | P2/P1·BiFPN | small | train | 8~32px 표현력 | bg 오검출 |
| H7_anchor | k-means·ATSS·SimOTA | trunc,small | train | positive↑ | loc 불안정 |
| H8_loss | focal/varifocal·IoU-aware | occluded,small | train | hard pos gradient↑ | FP·calib |
| H9_aug | occlusion/copy-paste | occluded,crowd | train | robustness↑ | 비현실 합성 |

## 5. 실행

```bash
# 지금(샌드박스, 데이터·GPU 불필요) — 지표/NMS/loop 자기검증 + cached-tier 데모
PYTHONPATH=. python tests/test_recall_lab.py
PYTHONPATH=. python experiments/recall_lab.py
```
detector/train tier는 실 검출기·데이터가 붙는 Colab에서 동일 Registry/Comparator로 실행(각 가설의 `spec`이 런 명세). 실데이터 평가 시 `synthetic_scene`을 실제 (preds, gts, gslices)로 교체하면 동일 harness가 그대로 작동.

## 6. 회귀 방지
- 모든 변경은 `tests/` 통과 동반(현재 37/37).
- baseline 결과를 캐시·고정, 변형은 항상 동일 split·동일 FPPI에서 비교.
- 슬라이스 정의·GT 정의를 사이클 간 고정(바꾸면 baseline 재측정).
