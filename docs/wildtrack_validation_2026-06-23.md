# WILDTRACK validation — code/geometry verified (synthetic), real-data run deferred

_2026-06-23_

## Situation

The real-dataset run (`experiments/wildtrack_validate.py`) was blocked on data + UI:

- the EPFL 6.3 GB `Wildtrack_dataset_full.zip` downloaded to Drive is **corrupt**
  (`4294967296 (4 GiB) extra bytes` — the classic >4 GB Zip64 central-directory
  bug). `unzip`, Python `zipfile`, and `7z` all fail to extract.
- the Colab notebook UI froze (scroll stuck on large cell outputs), so the run
  could not be driven there.

## Pivot — validate the moat-critical CODE without the dataset or a GPU

The integration risk for multi-camera + absolute-scale lives in the geometry /
loader / fusion code, **not** in the detector. That code can be proven on
synthetic calibration: build cameras in WILDTRACK's official world frame, write
their calibration in the **real WILDTRACK XML format**, and use OpenCV's own
`cv2.projectPoints` as the ground-truth world→image map, then check the pipeline
inverts it. See `experiments/wildtrack_selftest.py` + `tests/test_wildtrack.py`.

## Three integration bugs found and fixed (would have crashed / silently wrecked the first real run)

1. **positionID world origin.** `positionid_to_world` defaulted to MVDet's
   origin `(-300, -900)` cm. The dataset's shipped calibration is consistent with
   the **official toolkit** (`intersecting_area.py`): grid 1440×480, origin
   `(-300, -90)` cm, step 2.5 cm. The two frames differ by **8.1 m in Y** — GT
   would not have aligned with the calibration. Fixed default → `(-300, -90)`.

2. **Double projection in fusion.** `wildtrack_validate.py` pre-projected foot
   pixels to world with `to_ground`, then passed them to `MultiCameraFusion.fuse()`,
   which calls `to_ground` again internally. Fixed to pass **foot pixels** to
   `fuse()` (single-camera path projects once, directly).

3. **Extrinsic XML parsing.** `load_camera` read extrinsics with
   `cv2.FileStorage`, but real WILDTRACK extrinsics store `<rvec>`/`<tvec>` as
   **plain-text** elements (the official toolkit parses them with minidom), which
   FileStorage cannot read. Rewrote the loader on ElementTree; it now handles both
   the matrix-node intrinsics and the plain-text extrinsics (and matrix extrinsics).
   `intrinsic_zero` = undistorted images → zero distortion, so the no-distortion
   `to_ground` is exact for the provided frames.

## Self-test results (OpenCV projectPoints as ground truth)

| check | result |
|---|---|
| loader | 2 cams parsed from real-format XML (matrix intrinsic + text extrinsic) |
| round-trip | project → `to_ground` max recovery error **1.9e-08 m** (exact analytic inverse) |
| positionID convention | world span **X = 11.97 m, Y = 35.98 m** (official ~12 × 36 m) |
| multi-camera fusion | 15 people, 2 views → **15 fused, 15 multi-view, loc_err 0 m** |
| occlusion fill | 2nd cam sees 12/15 → still **15 fused, 12 multi-view** (single-view kept) |
| absolute scale | peak density sparse 0.16 → dense 2.69 /m² (physical, monotonic) |

Tests: **26/26** (`test_moat2` 12 + `test_anomaly` 10 + `test_wildtrack` 4).

## What is NOT yet validated (the deferred real-data run = Phase E recall)

Real-detector **recall vs GT on the real frames** (1-cam vs 2-cam) and absolute
density on the real crowd. Needs the frames + a GPU. Two clean paths:

- **(a) Kaggle mirror** `aryashah2k/large-scale-multicamera-detection-dataset`
  (original WILDTRACK layout, matches `wildtrack_validate.py`) — needs a Kaggle
  API token:
  ```bash
  pip install kaggle && mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  kaggle datasets download -d aryashah2k/large-scale-multicamera-detection-dataset -p /content/WT --unzip
  ```
- **(b) Recover from the existing (corrupt) Drive zip** after a Colab runtime
  restart (fixes the frozen UI). The small files (`calibrations/`,
  `annotations_positions/`) sit early in the archive and are almost certainly
  intact; extract them by scanning local headers, then use `Image_subsets/` frames.

Run (either path), 2 cameras, ~10 min:
```bash
PYTHONPATH=. python experiments/wildtrack_validate.py \
    --root /content/WT/Wildtrack_dataset --cams 2 --frames 20 \
    --weights /content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt
```
