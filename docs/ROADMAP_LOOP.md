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
1. **E — 멀티카메라 하드닝(차기 핵심 해자)**: ~~① 근지평선 클램프를 `wildtrack_validate.py`에 연결~~ ✅, ② 7뷰 전체 정량 평가(MODA/MODP), ③ 교차뷰 트랙 시간연관, ④ 학습형 멀티뷰 검출기(MVDet 계열)로 recall↑.
2. **F — 외적검증·경보 정밀도**: WILDTRACK 정량(MODA/MODP), `HysteresisEventGate`를 anomaly/absolute/terror 러너에 연결, FT-3(CCTV recall).
3. **D — VLM 설명**: `narrate_vlm`(Qwen2.5-VL) 실연결.
4. 객체ID/개방어휘, 엣지 배포(v3).

## 사이클 로그
- **Cycle 1 (2026-06-23, 완료)** — *제안*: 실데이터에서 입증된 근지평선 투영 발산 → 클램프 필요. *수정*: `geometry/wildtrack.py`의 `to_ground(uv, bounds=...)`에 근지평선/경계 클램프 추가(기존 무인자 경로는 불변=하위호환). *검증*: `tests/test_wildtrack.py`에 클램프 테스트 추가 → **전체 27/27 통과**(moat2 12 + anomaly 10 + wildtrack 5). *다음*: 클램프를 `wildtrack_validate.py`에 연결(백로그 E①).
- **Cycle 2 (2026-06-23, 완료)** — *제안(fugu)*: bounds를 validation 경로 전체에 전달 + `GROUND_BOUNDS=(-3.0,-0.9,9.0,35.1)` 상수화. *수정*: `geometry/multicam.py` `CameraView.to_world(bounds=None)` + `MultiCameraFusion.fuse(bounds=None)` 시그니처 확장; `experiments/wildtrack_validate.py` `GROUND_BOUNDS` 상수 + 두 호출(single-cam, fuse)에 전달; `tests/test_moat2.py` `_IdentityCal` 하위호환 수정. *검증*: `tests/test_wildtrack.py` 신규 2개 추가(bounds 연동 + fuse bounds) → **전체 29/29 통과**(moat2 12 + anomaly 10 + wildtrack 7). *fugu 리뷰*: 모든 수용 기준 충족, 하위호환성 양호. *다음*: E② 7뷰 전체 정량 평가(MODA/MODP).
