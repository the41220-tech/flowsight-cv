# FlowSight — 로드맵 반복 루프 (fugu 주도 · Claude 실행)

남은 로드맵(D-VLM, E 멀티카메라 하드닝, F 외적검증)을 **검증 → 수정 → 보고** 사이클로 전진시키는 절차.

## 역할
- **fugu (Sakana)** — 다음 단계 *제안* + 합격 기준 정의, 결과 *리뷰* + 사이클 보고서. (LLM, 도구 없음)
- **GLM-5.2** — (선택) 성과/문서 요약.
- **Claude** — *구현·수정·검증*(코드/테스트/진단 실행). **도구를 가진 유일한 실행 주체.**

## 한 사이클의 단계
1. **제안 (fugu)** — 백로그에서 다음 항목 1개 + 수용 기준(통과해야 할 테스트/지표) 제시.
2. **구현·검증 (Claude)** — 코드 작성/수정 → `tests/` 전체 통과 + 필요한 진단 실행. **테스트 실패 시 사이클 미완료.**
3. **리뷰·보고 (fugu)** — 결과를 합격 기준 대비 평가 + 사이클 보고서 작성.
4. **기록** — `docs/`에 사이클 로그, `MEMORY.md` 갱신, (코드 변경 시) 사용자 Mac에서 push.

## 트리거 (중요 — 자율 무한루프 아님)
`ask_model`(fugu/GLM)은 **명시적 사용 전용**(자율·자기검증 호출 금지)이 계약이다. 따라서 본 루프는 **사용자가 사이클을 트리거**할 때마다 1회 전진한다:
- 사용자: **"다음 사이클"** (또는 대상 지정, 예: "E의 교차뷰 연관") → Claude가 위 4단계를 1회 수행.
- 무인 백그라운드/스케줄로 fugu를 반복 호출하지 않는다(계약 위반 + 무감독 코드수정 위험 회피). 코드 수정은 항상 테스트 검증을 동반.

## 백로그 (우선순위)
1. **E — 멀티카메라 하드닝(차기 핵심 해자)**: ~~① 근지평선 클램프를 `wildtrack_validate.py`에 연결~~ ✅, ② 7뷰 전체 정량 평가(MODA/MODP), ③ 교차뷰 트랙 시간연관, ④ 학습형 멀티뷰 검출기(MVDet 계열)로 recall↑ — **실험 프로그램 준비됨**(`docs/RECALL_EXPERIMENT_PROGRAM.md`, `experiments/recall_lab.py`: 10가설 Registry + 동일 FPPI 비교 loop; cached tier 즉시, detector/train tier는 Colab).
2. **F — 외적검증·경보 정밀도**: WILDTRACK 정량(MODA/MODP), `HysteresisEventGate`를 anomaly/absolute/terror 러너에 연결, FT-3(CCTV recall).
3. **D — VLM 설명**: `narrate_vlm`(Qwen2.5-VL) 실연결.
4. 객체ID/개방어휘, 엣지 배포(v3).

## 사이클 로그
- **Cycle 1 (2026-06-23, 완료)** — *제안*: 실데이터에서 입증된 근지평선 투영 발산 → 클램프 필요. *수정*: `geometry/wildtrack.py`의 `to_ground(uv, bounds=...)`에 근지평선/경계 클램프 추가(기존 무인자 경로는 불변=하위호환). *검증*: `tests/test_wildtrack.py`에 클램프 테스트 추가 → **전체 27/27 통과**(moat2 12 + anomaly 10 + wildtrack 5). *다음*: 클램프를 `wildtrack_validate.py`에 연결(백로그 E①).
- **Cycle 2 (2026-06-23, 완료)** — *제안(fugu)*: bounds를 validation 경로 전체에 전달 + `GROUND_BOUNDS=(-3.0,-0.9,9.0,35.1)` 상수화. *수정*: `geometry/multicam.py` `CameraView.to_world(bounds=None)` + `MultiCameraFusion.fuse(bounds=None)` 시그니처 확장; `experiments/wildtrack_validate.py` `GROUND_BOUNDS` 상수 + 두 호출(single-cam, fuse)에 전달; `tests/test_moat2.py` `_IdentityCal` 하위호환 수정. *검증*: `tests/test_wildtrack.py` 신규 2개 추가(bounds 연동 + fuse bounds) → **전체 29/29 통과**(moat2 12 + anomaly 10 + wildtrack 7). *fugu 리뷰*: 모든 수용 기준 충족, 하위호환성 양호. *다음*: E② 7뷰 전체 정량 평가(MODA/MODP).
- **Cycle 3 (2026-06-23, 완료)** — *목표*: ~10% recall을 7캠+SAHI로 개선. *실험*: 4캠(C1,C2,C4,C5) + SAHI(slice=512, overlap=0.2, CONF=0.4), 20 frames, GT=637. *결과*: `[4cam+SAHI] prec=0.046 recall=0.089 fp=1195` vs `baseline 2cam=0.102` → **SAHI가 recall을 오히려 낮춤.** *근본원인*: (1) FT-2 드론→지상 도메인 불일치로 FP 폭발(21배), (2) CVLab1≈CVLab4 거의 동일 위치로 커버리지 무증가, (3) C3/C6/C7 zip 손상 복구 불가. *결론*: recall 문제는 geometry가 아닌 detector 도메인 문제. *필요 조건*: 완전한 7카메라 (WILDTRACK 재다운로드) + 도메인 적합 검출기(MVDet 계열) → E④ 학습형 멀티뷰 검출기 과제. *다음 트리거 시*: E② 7뷰 MODA/MODP (완전한 데이터 취득 선결).
- **Cycle 4 (2026-06-23, 완료)** — *목표*: recall ~0.1(=검출기 도메인 문제)을 체계적으로 푸는 **실험 프로그램** 설계·구축. *설계(fugu)*: Registry/Runner/Metrics/Comparator/Loop 아키텍처 + 동일 FPPI 비교 규칙(fugu는 긴 프롬프트 2회 타임아웃, 짧은 프롬프트로 설계 반환). *구현(Claude)*: `flowsight/eval/slice_metrics.py`(recall@IoU·AR·MR-2·FPPI/FROC·슬라이스별·**동일 FPPI 비교**), `flowsight/eval/nms_variants.py`(hard/Soft/DIoU/WBF, 재학습0), `experiments/recall_lab.py`(10가설 Registry + 비용순 loop + 합성 데모), 문서 `docs/RECALL_EXPERIMENT_PROGRAM.md`. *검증*: `tests/test_recall_lab.py` 8개 추가 → **전체 37/37 통과**(moat2 12 + anomaly 10 + wildtrack 7 + recall_lab 8). 데모: baseline recall 0.527; cached-tier(NMS/threshold) ΔRecall@FPPI +0.002~0.006 → REJECT(=값싼 추론옵션으론 도메인 문제 못 품을 재확인), detector/train tier는 Colab 런 스펙으로 연기. *다음 트리거 시*: detector-tier 실행(실 검출기로 H5 tiling/H10 pose) 또는 train-tier(H8 loss/H9 aug) — Colab + 실데이터.
