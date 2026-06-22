# MOAT Layer 2 — Absolute crowd-pressure alarm + non-planar 3-D (2026-06-22)

**Owner:** 감경민 · **Compute:** Google Colab **Pro** (background execution, L4/A100)

## Why this is the moat

Competitors render a *relative* density heatmap: it shows where a scene is busiest,
but it cannot say "this is dangerous" in absolute terms, and it assumes a flat
ground plane. FlowSight's moat is to make the risk signal **physical and absolute**:

1. **Calibrate pixels → metres** so Helbing crowd pressure `P = ρ·Var(v)` comes out
   in its true unit **1/s²**. Then the peer-reviewed critical threshold
   **`P_CRIT = 0.02 /s²`** (Helbing, Johansson & Al-Abideen 2007 — crossed ~10 min
   before the Jamarat crush) becomes a **real early-warning alarm**, identical in
   meaning across every camera and venue.
2. **Drop the flat-plane assumption.** A monocular *metric* depth map
   (Depth-Anything-V2-Metric) + camera pose back-projects foot points to world
   `(X,Y,Z)`; the ground is gridded into a DEM, and gravity adds the terrain
   potential `U = ρ·m·g·h` and downhill force `∇U`. This is what makes the crush
   case work on sloped/funnel sites (Itaewon) where flat heatmaps fail.

## What was built (this session)

Key realization: `flowsight/physics/pressure.py` + `density.py` **already** compute
the metric Helbing field in 1/s² (used by the synthetic `DisasterMonitor`). So the
only missing pieces were the **pixel→metre bridge** and the **absolute alarm** — no
duplication of the verified physics.

| File | Role |
|---|---|
| `flowsight/geometry/calibration.py` | **NEW.** `HomographyCalibrator` (≥4 surveyed pts → image→ground homography + **Jacobian** velocity mapping, perspective-correct) and `PedestrianScaleCalibrator` (median person-box height ↔ 1.7 m → uniform m/px, calibration-free). `tracks_to_metric()` bridges a tracks JSON frame → metric arrays. |
| `flowsight/physics/crowd_pressure.py` | **+** `frame_pressure_metric()` (metric grid → P in **1/s²**), `alarm_level()` (absolute 3-tier at 0.5·CRIT / CRIT), constant `P_CRIT=0.02`. Pixel `frame_pressure()` kept for back-compat. |
| `flowsight/pipeline/moat_field.py` | **NEW.** `foot_points_to_world` / `dense_depth_to_terrain` / `build_dem_from_points` (metric depth → 3-D → DEM) and `MoatMonitor` (absolute Helbing alarm + terrain `U`/`∇U`, static terrain field cached once). |
| `tests/test_moat2.py` | **NEW.** 7 unit tests (run: `PYTHONPATH=. python tests/test_moat2.py`). |
| `experiments/moat2_synth_e2e.py` | **NEW.** Synthetic E2E on a sloped Social-Force crush, all signals in absolute units. |
| `experiments/absolute_alarm_run.py` | **NEW.** Demo: LEFT video \| RIGHT top-down **metric** pressure map with the 0.02 line fixed on an **absolute** colour scale + absolute KO labels + pressure timeline. |

## Results (CPU, validated)

Unit tests **7/7**. The decisive one — absolute units: a uniform crowd of ρ=4/m²
with per-axis velocity variance 0.09 → Var(v)=0.18 (m/s)² yields **P = 0.72 /s²**
exactly (= ρ·Var), i.e. the calibration→absolute-unit chain is correct.

Synthetic E2E (sloped funnel, 3 seeds, `experiments/results/moat2_e2e.json`):
the **absolute 0.02/s² alarm fires on average +10.47 s before crush onset**
(density ≥ 6/m²); peak pressure reaches 0.18–1.74/s² (9–87× threshold) and peak
density 7.3–10.1/m² (matches the Itaewon anchor 7.57). *Honest note:* the terrain
precursor's large lead was established earlier in H2/H6 (fused-risk vs density); in
this fast-crush E2E under a matched rise criterion it is coincident with Helbing
(−0.7 s) — not overclaimed.

## Run on Colab Pro (GPU — real metric depth, no surveyed points needed)

```bash
# Cell A (user clicks consent): mount Drive
from google.colab import drive; drive.mount('/content/drive')
```
```bash
# Cell B: clone + deps  (Runtime → change type → L4/A100; enable Background execution)
%cd /content && git clone -q https://github.com/the41220-tech/flowsight-cv.git
%cd flowsight-cv && pip -q install ultralytics sahi transformers timm
```
```python
# Cell C: REAL metric depth → terrain, then absolute alarm on real tracks
# 1) tracks: experiments/track_run.py (ByteTrack on FT-2 chunk05_best.pt)
# 2) metric depth on a reference frame:
from transformers import pipeline; import numpy as np, cv2
dep = pipeline("depth-estimation",
               model="depth-anything/Depth-Anything-V2-Metric-VKITTI-Large", device=0)
frame = cv2.cvtColor(cv2.imread("ref_frame.png"), cv2.COLOR_BGR2RGB)
depth_m = np.asarray(dep(Image.fromarray(frame))["predicted_depth"], float)
# 3) camera + dense_depth_to_terrain(...) → MoatMonitor(bounds, terrain) → step_metric per frame
# Quick path (no surveyed pts): absolute_alarm_run.py with pedestrian scale:
!PYTHONPATH=. python -u experiments/absolute_alarm_run.py \
    --video /content/umn_clip.mp4 --tracks /content/tracks_cctv.json \
    --type cctv --person-px 26 --out /content/drive/MyDrive/flowsight_demo
```

**Deploy tip (no git push):** for small edits, `gzip -c f | base64 -w0` then
`pathlib.Path(p).write_bytes(gzip.decompress(base64.b64decode("<b64>")))` —
base64 has no auto-close chars. Big/multiple files → push to GitHub and re-clone.

## Next
1. Colab GPU: real metric-depth → 3-D terrain absolute demo on UMN + a sloped/aerial clip.
2. Precision/hysteresis on the flow alarm (cached `rvt_arrays.npz`).
3. External validation (WILDTRACK ground GT) → metric accuracy of the calibration.
