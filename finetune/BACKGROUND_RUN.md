# Unattended long runs on Colab Pro (background execution)

FlowSight long GPU jobs (violence training, FT-3) are **disconnect-safe**: code on
GitHub + checkpoints on Google Drive + chunked checkpoint/resume harness. On
**Colab Pro** they can run for hours with the laptop closed via **background
execution**.

## 0. One-time per session
1. Runtime → Change runtime type → **L4** (or A100) GPU. Connect.
2. Enable **background execution**: Pro setting — the run keeps going after you
   close the tab. (Colab Pro/Pro+; free tier stops on disconnect.)
3. Mount Drive (consent) + clone repo:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   ```
   ```bash
   %cd /content && rm -rf flowsight-cv && git clone -q https://github.com/the41220-tech/flowsight-cv.git
   %cd flowsight-cv && pip -q install ultralytics datasets
   ```

## 1. Phase B — violence classifier (RWF-2000)
**Prepare data ATTENDED once** (confirms the webdataset auto-detection prints
non-empty Fight/NonFight counts):
```bash
!PYTHONPATH=. python -u finetune/prepare_rwf.py --out /content/rwf_cls --frames-per-clip 3
# expect: PREP_RWF_DONE counts={'Fight': ~, 'NonFight': ~}. If a class is 0, the
# label auto-detect failed -> read the printed columns/types and adjust prepare_rwf.py.
```
**Then launch training UNATTENDED** (checkpoints to Drive; close the laptop):
```bash
!PYTHONPATH=. python -u finetune/violence_train.py --data /content/rwf_cls \
    --total 30 --chunk 5 --imgsz 224 --batch 64 \
    --ckpt /content/drive/MyDrive/flowsight_violence_ckpt
```

## 2. Resume after a VM recycle
Re-do step 0 (runtime + Drive + clone), then re-run the SAME `violence_train.py`
command — it auto-resumes from the latest `chunk*_last.pt` in the Drive ckpt dir.
Check progress any time:
```bash
!tail -5 /content/drive/MyDrive/flowsight_violence_ckpt/run_log.jsonl
```

## 3. FT-3 (detector recall, Phase F) — same pattern, existing harness
```bash
!bash finetune/bootstrap_ft2.sh   # (or auto_train.py with a higher COCO ratio mix)
!PYTHONPATH=. python -u finetune/auto_train.py --data datasets/person_mix/person.yaml \
    --total 40 --chunk 5 --imgsz 768 --batch 12 \
    --ckpt /content/drive/MyDrive/flowsight_ckpt
```

## Safety / cost notes
- Free Colab: idle ~90 min, no background → NOT for unattended. Pro required.
- Each chunk = a Drive checkpoint → a disconnect loses at most one chunk (~minutes).
- OOM is self-healed (batch halves and the chunk retries, up to 4×).
- GPU compute units are consumed even in background — stop the run when done.
