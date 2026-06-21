# FT-2 무인 장기학습 — 안전 런북 (disconnect-safe)

원리: **코드=GitHub, 체크포인트=Drive.** VM이 리셋돼도 둘 다 살아있으므로, 아래 2셀을
**재실행만 하면 Drive의 최신 5-epoch 청크부터 자동 재개**(최대 1청크=5ep 손실).

## (전제) 1회: 새 코드 GitHub에 올리기 — 당신 터미널
```bash
cd ~/Desktop/magi/flowsight
git add -A && git commit -m "FT-2 autonomous: auto_train, prepare_mix, bootstrap" && git push
```

## Colab — 셀 2개 (fresh VM에서도 이대로)
**셀 A — Drive 마운트** (이미 했으면 그대로 통과):
```python
from google.colab import drive
drive.mount('/content/drive')
```
**셀 B — 부트스트랩(클론→설치→데이터→Drive 체크포인트 학습)**:
```python
!cd /content && (git clone https://github.com/the41220-tech/flowsight.git || true) && cd flowsight && git pull && bash finetune/bootstrap_ft2.sh
```

## 끊겼을 때
- **셀 A, 셀 B를 다시 실행**하면 됩니다. `auto_train.py`가 `/content/drive/MyDrive/flowsight_ckpt`의
  최신 `chunkNN_last.pt`를 찾아 그 다음 청크부터 이어서 학습합니다.
- 진행 상황·각 단계는 `flowsight_ckpt/run_log.jsonl`(Drive)에 기록 → 어디서 멈췄는지 추적.
- OOM 나면 배치 자동 축소 후 재시도(코드 내장).

## 완전 무인(탭 닫아도 지속)
- 무료 Colab은 유휴/사용량 한도로 끊길 수 있어, "무손실"은 되지만 "완전 방치"는 한계.
- 탭 닫아도 계속 = **Colab Pro 백그라운드 실행**. 그 경우 셀 B 한 번이면 30ep 완주 가능.

## 끝나면
- 최종 가중치: `/content/drive/MyDrive/flowsight_ckpt/chunk05_last.pt` (또는 마지막 청크)
- 전/후 재평가: `!PYTHONPATH=. python -u finetune/eval_recall.py --weights <그 가중치>`
  → 드론 recall 유지 + CCTV 0.117→≥0.88 회복 확인.
