# 실데이터 지도표시 PoC — Colab 실행 가이드

> 샌드박스(이 도구 환경)는 **외부 다운로드·GPU가 막혀** 있어 실모델/실데이터 실행이 불가합니다(PyPI 설치 실패, HF 미리보기는 이미지 URL만 반환·박스 없음을 확인). 그래서 **실footage 실행은 Colab**에서 합니다. 아래 셀을 그대로 붙여넣으면 실제 VisDrone 프레임에서 사람→2D 지도까지 나옵니다.

## 셀 1 — 설치 & 코드
```python
!pip -q install huggingface_hub datasets "transformers>=4.46" timm accelerate
import sys; sys.path.insert(0, "/content/flowsight")   # flowsight 폴더 위치
# !git clone <YOUR_REPO> /content/flowsight   # 또는 Drive에서 복사
```

## 셀 2 — 실데이터 지도표시 실행
```python
%cd /content/flowsight
import experiments.demo_map_colab as d
res = d.main(idx=0, score_thr=0.25)     # 실제 VisDrone 프레임 stream → RT-DETRv2 → 2D 지도
from IPython.display import Image; Image("experiments/figures/realmap_visdrone.png")
```

**볼 결과:** 좌측 = 실제 드론 프레임 + 검출된 사람 발끝점 + 지면 사각형, 우측 = 2D BEV 지도(사람 점 + 밀도 히트맵).

## 알아둘 점 (정직하게)
- **검출 약함은 예상된 것 = 가설 H3.** RT-DETRv2는 COCO 학습이라 부감의 작은 머리에 약합니다. `score_thr`를 낮추거나, 밀집 장면은 **타일링(SAHI)** 또는 **DroneCrowd/VisDrone 파인튜닝**(H3)으로 개선. 이게 다음 실험입니다.
- **지도 스케일은 근사.** 임의 웹 프레임이라 측량점이 없어 지면 사각형을 사다리꼴→직사각형으로 가정했습니다. 실제 현장은 **측량된 4점**을 넣으면 미터 단위 people/m²가 정확해지고, 경사·비평면은 H1의 메트릭 깊이/광선교차로 처리합니다.
- 즉 이 PoC는 **"실프레임 → 검출 → 2D 지도"의 엔드투엔드가 도는 것**을 보여줍니다. 정확도(검출/스케일)는 H3·H1에서 끌어올립니다.

## 다음
- H3: VisDrone 파인튜닝 전/후 카운팅 MAE 비교 → 같은 프레임에서 검출 수 급증 확인.
- 멀티소스(v2): WILDTRACK(다중 카메라 + 지면 정답)으로 "여러 소스를 한 지도에" 검증.
