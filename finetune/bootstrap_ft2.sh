#!/usr/bin/env bash
# Resilient FT-2 bootstrap for a FRESH Colab VM.
# Re-run after ANY disconnect -> resumes from Drive (loses at most one 5-epoch chunk).
# Assumes: cwd = /content/flowsight, and Google Drive is mounted at /content/drive.
set -e

echo "== install =="
pip -q install ultralytics

echo "== data: VisDrone-person (downloads VisDrone if missing) =="
PYTHONPATH=. python finetune/prepare_data.py

echo "== data: + COCO val2017 person -> person_mix =="
PYTHONPATH=. python finetune/prepare_mix.py

echo "== FT-2: chunked train, checkpoint to Drive, resume-aware =="
PYTHONPATH=. python -u finetune/auto_train.py \
  --data datasets/person_mix/person.yaml \
  --total 30 --chunk 5 --imgsz 768 --batch 12 \
  --ckpt /content/drive/MyDrive/flowsight_ckpt

echo "== done. weights + run_log.jsonl in /content/drive/MyDrive/flowsight_ckpt =="
