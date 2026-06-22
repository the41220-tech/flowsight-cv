# FlowSight AI — Project Handoff & Continuation Log

**Last updated:** 2026-06-22 (MOAT layer 2: absolute 0.02/s² alarm + non-planar 3-D metric depth)
**Owner:** 감경민 (POSCO 청년 AI·Big Data 아카데미 project)
**Canonical repo:** https://github.com/the41220-tech/flowsight-cv
**Working folder:** `~/Desktop/magi/flowsight-cv` (this repo)

> ⚠️ **Repo note:** Use ONLY `flowsight-cv`. The older repo `the41220-tech/flowsight`
> was reused for a separate project ("트레이너ZIP / mytrainer MCP", TypeScript) and is
> cross-contaminated — do NOT push FlowSight there.

---

## 1. What FlowSight is

**Mission: real-time crowd-movement analysis & interpretation (군중 움직임 분석·해석).**
Input-source-agnostic CV platform. From drone / CCTV / tethered video it reconstructs **where every
person is on a 2D/3D map**, builds **crowd density + flow/movement fields**, and **detects and
interprets anomalous crowd dynamics in real time** so operators can monitor and respond to a *range*
of situations.

> ⚠️ **Scope (do not narrow this):** FlowSight is NOT a crush-only system. Crowd-crush prediction is
> **one high-stakes application** — not the mission. The core capability is *reading and interpreting
> how a crowd moves*, which generalizes across incident types. Earlier docs over-indexed on 압사
> (crush); that is the hardest validated case, not the boundary of the product.

**Applications (one engine, many events):**
1. **Safety / incident monitoring** — crush, stampede, panic/escape, surge, sudden dispersal,
   counterflow, falls, fights, blocked exits. *(Crush is the hardest, most-validated case → H1–H7.)*
2. **Operational awareness** — flow bottlenecks, queue build-up, capacity/occupancy, abnormal motion
   vs a learned "normal".
3. **Marketing / commercial (later)** — footfall, dwell time, path/flow patterns, hotspot analytics
   for venues, retail, events.

**The moat (what competitors' flat heatmaps can't do):** go beyond flat-plane homography to
**non-planar 3D positioning + terrain potential-energy / pressure**. Gravitational potential `U`
from elevation/slope is combined with Helbing crowd pressure `P = ρ·Var(v)`. This is what makes the
crush application work on sloped/funnel sites like Itaewon where a flat density map fails — and the
same 3D-positioned flow field is the substrate for every other application above.

**Four hard product requirements (original spec):**
1. Fuse multi-camera / multi-angle video → mark each person/object on a 2D map (even if not top-down).
2. Do NOT assume a flat plane — ingest terrain/geo info → compute position AND pressure-difference (potential energy).
3. Detect walls/obstacles when visible.
4. Identify objects via LLM / image recognition.

---

## 2. Roadmap — 7 stages (current position: end of Stage 6)

| # | Stage | Status |
|---|-------|--------|
| 1 | Model & dataset research + implementation roadmap | ✅ done |
| 2 | Core modules (geometry / physics / perception) + 5-hypothesis setup | ✅ done |
| 3 | Real-data map-display proof (single source: drone → people → 2D map) | ✅ done |
| 4 | Multi-model recall benchmark (found weakness = drone / small objects) | ✅ done |
| 5 | Detector fine-tuning — FT-1 → **FT-2** (fix CCTV forgetting, keep drone) | ✅ done |
| 6 | Multimodal slope → disaster prediction (H1·H2·H6·H7 + real-recall) + real-video anomaly detection | ✅ **DONE** (synthetic-GT physics + real FT-2 recall; **real-video DONE** on UMN panic, §7) |
| 7 | Multi-camera fusion + object ID (LLM) + external-data validation + edge deploy | ⬜ not started |

**Stage 6 caveat:** the physics (pressure/slope) is verified on a Social-Force **synthetic**
ground truth with the **real FT-2 detector recall** plugged in. Real-crowd-video solidification
(UMN panic dataset) is now **DONE** (2026-06-22): the real FT-2 detector + optical-flow pressure
channel ran on real footage and the flow-pressure alarm **led panic onset in all 7 episodes**
(see §7). External validity (WILDTRACK / DroneCrowd / real crush footage) belongs to Stage 7.

**Post-Stage-6 (2026-06-22):** built the per-person **tracker** (ByteTrack on FT-2), the user-spec **dashboard v2** (video + danger-coloured moving dots; panic demo delivered), **moat layer 1 = crowd-pressure heatmap** (Helbing P=ρ·Var(v), validated 137× erratic-vs-coherent), and **moat layer 2 = metric calibration → absolute 0.02/s² alarm + non-planar 3-D metric depth** (built, unit-tested 7/7, synth-E2E + Colab-Pro-L4 real-footage validated). See §10.

---

## 3. Hypothesis ledger (see `experiments/hypotheses.md`)

| ID | Hypothesis | Result |
|----|-----------|--------|
| H1 | Depth/terrain positioning beats single homography on non-planar ground | ✅ metric-depth −78% error, ray-cast −73% (naive multi-homography refuted) |
| H2 | Terrain potential-energy is an early crush precursor | ✅ leads crush by 7.2 s, leads Helbing turbulence by 18.8 s |
| H3 | Drone fine-tuning lifts drone/small recall | ✅ FT-1 drone 0.154→0.512 (3.3×) |
| H3′ | Mixed VisDrone+COCO training fixes CCTV forgetting while keeping drone | ✅ FT-2 (see §5) |
| H6 | Detection + terrain-3D + flow/potential predicts crush earlier & more robustly than density-only | ✅ multimodal lead 5.2–6.0 s vs density-only 4.2–4.4 s; holds at 30–51% recall |
| H7 | SAHI tiling recovers small/distant recall (no retrain); recall controls the density alarm; multimodal gives the early warning | ✅ H7b (CPU) verified; **H7a measured (2026-06-22): SAHI drone 0.571→0.712, small 0.517→0.673, but BELOW the 0.86 lit. assumption** |

**Itaewon empirical anchor (used for thresholds):** avg density 7.57 ppl/m² (max 9.95),
avg crowd pressure 1,063 N/m. Operational crush density `RHO_CRIT = 6.0`.

---

## 4. Environment & how the work is run

- **Compute = Google Colab** (free tier, T4 GPU). Claude drives the notebook via the
  Claude-in-Chrome MCP (user's browser, logged into Colab). User performs account actions
  (Google Drive OAuth, any GitHub push) — Claude cannot do auth.
- **Colab notebook:** https://colab.research.google.com/drive/1WRmxk075jom4KE_MiO8-OODZ0zIeGHuo
- **Code = GitHub**, **checkpoints/outputs = Google Drive** → survives VM recycles.
- **Drive paths:** weights `/content/drive/MyDrive/flowsight_ckpt/`, demo outputs `/content/drive/MyDrive/flowsight_demo/`.
- **Local sandbox** (Claude's): can run the numpy-only experiments (H1/H2/H6/H7b) on CPU, but
  **cannot download datasets or access Drive** — anything needing data/weights must run on Colab.

### Colab limits & unattended reality (important)
- Free Colab: idle disconnect ~90 min, max session ~12 h, plus dynamic usage caps. No exact countdown.
- **Long training IS unattended-safe** here via the chunked-checkpoint harness (survived 3 VM
  recycles over 3.5 h).
- **A single long CPU cell is NOT safe** (no intermediate save → a disconnect restarts it).
- **True hands-off (close laptop for hours) is not guaranteed on free Colab → use Colab Pro
  background execution** for that.
- **Every fresh VM needs:** (a) re-select T4 runtime [Claude], (b) re-mount Drive [user clicks
  consent; the mount often fails once → just re-run the cell], (c) re-clone repo + re-download data.

---

## 5. Detector fine-tuning results (Stage 5) — DONE

**FT-2:** YOLO11m, 1 class (person). Data `person_mix` = VisDrone-person (6215) + COCO val2017-person
(2693) = 8377 train imgs (no leakage: val2017 ≠ coco128 holdout). 30 epochs as 6×5-epoch chunks on
T4 (~3.5 h). All chunks saved to Drive.

**Before/after recall** (same 100-img holdout = 50 VisDrone + 50 COCO128, thr=0.25, IoU=0.5):

| model | overall | drone | cctv | small | precision |
|-------|---------|-------|------|-------|-----------|
| yolo11m baseline | 0.241 | 0.154 | **0.685** | 0.126 | 0.842 |
| FT-1 (VisDrone only, 5 ep) | 0.447 | 0.512 | **0.117** ← forgot CCTV | 0.461 | 0.456 |
| **FT-2 (mixed, 30 ep)** | **0.552** | **0.571** | **0.455** | **0.517** | 0.612 |

- Forgetting **recovered 3.9×** (cctv 0.117→0.455) AND drone improved (0.512→0.571). H3′ supported.
- VisDrone-val mAP50: FT-1 5 ep 0.49 → FT-2 30 ep **0.60**.
- **Honest limit:** cctv 0.455 still < baseline 0.685 → a drone↔CCTV tradeoff remains. FT-2 is a
  balanced generalist. Levers to push further: FT-3 with a higher COCO ratio; SAHI (H7a, no retrain).

**Final weights (Drive):** `/content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt` (and `chunk05_last.pt`),
plus `run_log.jsonl`. (All 6 chunks' best+last are there too.)

---

## 6. Disaster-prediction results (Stage 6) — verified on synthetic GT + real recall

- **H6** (`experiments/run_h6_disaster.py`): multimodal risk predicts crush 5.2–6.0 s ahead vs
  density-only 4.2–4.4 s; lead holds even at 30–51% detection recall.
- **H7b** (`experiments/run_h7_sahi_disaster.py`): density alarm (threshold = crush density 6.0/m²)
  is recall-sensitive — at FT-1 drone recall 0.51 the detected density under-counts to 4.54 < 6.0
  and **misses** the crush; at SAHI-level 0.86 it recovers to 7.1 and **fires**; the multimodal
  terrain-potential channel keeps ~5.2 s lead regardless. → SAHI makes the *density map*
  trustworthy; the *early warning* comes from the multimodal moat.
- **#35 real-recall E2E:** feeding the **measured FT-2 drone recall (0.571)** into the sim →
  detected density 4.77 (under-count 0.55), a pure counting alarm **misses** the crush, but the
  **multimodal channel gives a 5.8 s lead**. The moat holds with the real detector.

Results files: `experiments/results/{h6_disaster.json, h7_sahi_disaster.{json,md}, ft2_before_after.md}`
+ figures in `experiments/figures/`.

---

## 7. Real-video solidification (Stage 6) — ✅ DONE (2026-06-22)

**Goal:** run the real FT-2 detector on REAL crowd-panic video, compute the flow/pressure risk
signal over time, measure **anomaly-detection latency** (how fast our alarm fires vs panic onset),
and render the **3D human-point map** (dashboard preview) + risk timeline.

**Dataset:** UMN "Unusual Crowd Activity" (11 escape/panic clips, normal→panic).
`http://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi` — 7739 frames, 30 fps, 320×240.
Framing: this is a **general crowd-MOVEMENT anomaly** (panic/escape), aligning with the broadened
mission (§1) — NOT crush-specific. UMN panic is abrupt → it measures **detection latency/reliability**.
Scenes are flat → terrain-potential channel ≈ 0; the **flow/pressure channel** (`ρ·Var(v)` via
optical flow) is what fires. Detection = real FT-2 (`chunk05_best.pt`), `imgsz=320`.

**Run (Colab T4, 2026-06-22):** STEP=16 → 484 sampled frames, ~171 s. `experiments/real_video_threat.py`.

**Result — the flow-pressure alarm LEADS panic onset in all 7 episodes:**

| episode | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| onset (s) | 46.9 | 187.2 | 193.1 | 209.1 | 233.1 | 238.4 | 250.7 |
| alarm (s) | 42.7 | 185.6 | 188.8 | 204.8 | 228.8 | 234.1 | 246.4 |
| **lead (s)** | **+4.3** | **+1.6** | **+4.3** | **+4.3** | **+4.3** | **+4.3** | **+4.3** |

(onset = independent optical-flow mean-speed threshold; alarm = our `P=ρ·Var(v)` crossing 0.25;
lead = onset − alarm. In `summary.json` this is stored as negative latency = alarm fires first.)
Mean lead ≈ 3.9 s. Peak panic t=240.5 s, 19 people in the 3D map. Outputs on Drive
`/content/drive/MyDrive/flowsight_demo/`: risk_timeline.png, frame_3dmap.png, frame_annot.png, summary.json.

**Honest read:** on this real footage the flow-pressure channel is reliably *at or ahead of* the
visible panic onset — strong support that the movement/flow channel detects the anomaly early. The
+4.3 s lead is bounded by the 8-frame (=4.27 s) alarm look-back window, so the true lead may be larger.
**Precision — NOW MEASURED (2026-06-22, `experiments/rvt_precision.py`):** swept the alarm over
threshold {0.15…0.40} × duration-gate K {1,2,3} vs the speed-onset proxy GT (per-frame arrays cached to
`rvt_arrays.npz` so re-sweeps are instant). Result: at **every** setting **recall = 1.00** (all 7 panic
episodes caught) but **precision = 0.12–0.28** — the alarm fires 12–27 episodes vs 7 real. Best point
thr=0.40/K=3 → P=0.28 (n_alarm=18). So the alarm is **high-recall / low-precision = over-sensitive**, and
the duration gate helps only marginally. Honest caveats: precision is deflated by (a) episode
fragmentation (pressure re-crosses threshold within one event) and (b) a strict proxy GT. **Next (cheap,
uses cached arrays):** episode-merge/hysteresis + frame-level GT labels. Outputs:
`precision_sweep.json`, `precision_pr.png`.

### H7a — real SAHI recall on FT-2 (✅ measured 2026-06-22, `finetune/eval_sahi.py`, slice 512 / overlap 0.2)

Same 100-img holdout (50 VisDrone + 50 COCO128), thr 0.25, IoU 0.5:

| metric | FT-2 plain | FT-2 + SAHI | Δ |
|---|---|---|---|
| overall recall | 0.552 | **0.698** | +0.146 |
| drone recall | 0.571 | **0.712** | +24.7% |
| cctv recall | 0.455 | **0.629** | +38% |
| small recall | 0.517 | **0.673** | +30% |
| precision | 0.612 | 0.547 | −0.065 |

**Verdict: H7a supported** — SAHI sliced inference lifts recall across the board (drone +25%, small +30%,
even cctv) with NO retraining; precision dips modestly. **BUT the measured SAHI drone recall (0.712) is
well below the 0.86 literature figure H7b assumed** — so re-state H7b with 0.71, and re-run the H7b sim at
recall 0.71 to check whether the density-counting alarm now fires (the multimodal-moat conclusion — early
warning comes from the flow/terrain channel, not the count — holds regardless).

> ⚠️ **Repo note:** `real_video_threat.py` is NOT yet committed to the GitHub repo (it was deployed
> to Colab ad-hoc in past sessions). It lives in the local working folder — **commit + push it** so
> future Colab clones include it. The verified run used a compact equivalent of the local full file.

---

## 8. How to RESUME (exact Colab cells)

On a fresh Colab VM (T4 runtime selected by Claude):

```python
# Cell A — mount Drive (user clicks consent; if "mount failed", just re-run this cell)
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Cell B — clone clean repo + deps
%cd /content
!git clone -q https://github.com/the41220-tech/flowsight-cv.git
%cd /content/flowsight-cv
!pip -q install ultralytics sahi
```

```bash
# Cell C — confirm FT-2 weights survived on Drive
!ls -la /content/drive/MyDrive/flowsight_ckpt/
```

**To re-verify Stage 5 (before/after recall):**
```bash
!PYTHONPATH=. python -u finetune/eval_recall.py --weights /content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt
```

**To finish Stage 6 real-video (apply the speed fix first):**
```bash
# get the panic video
!wget -q --tries=2 --timeout=40 http://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi -O /content/umn_all.avi
# edit experiments/real_video_threat.py: STEP=16 and add imgsz=320 to model.predict(...), then:
!PYTHONPATH=. python -u experiments/real_video_threat.py
# outputs -> /content/drive/MyDrive/flowsight_demo/ (risk_timeline.png, frame_3dmap.png, summary.json)
```

**To run H7a (SAHI recall on real FT-2 weights):**
```bash
!PYTHONPATH=. python -u finetune/eval_sahi.py --weights /content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt --slice 512 --overlap 0.2
```

> Tip for editing files on Colab without a git push: write the file locally, `gzip -c file | base64 -w0`,
> then in a Colab cell `pathlib.Path(p).write_bytes(gzip.decompress(base64.b64decode("<b64>")))`.
> (base64 has no quotes/brackets so it survives the editor's auto-close; this is how rvt.py was deployed.)

---

## 9. Key files in this repo

```
flowsight/
  geometry/{camera,terrain,homography,metric_depth}.py   # pinhole cam, non-planar terrain, IPM, depth back-proj
  physics/{density,potential,pressure,crowd_pressure,flow_features}.py  # density, ∇U potential, Helbing pressure; crowd_pressure=P=ρ·Var(v) field from TRACKS (moat); flow_features=dense-flow divergence/curl/counterflow
  sim/social_force_terrain.py                            # sloped-terrain Social-Force crowd sim (synthetic GT)
  perception/{detect,obstacles}.py                       # RT-DETRv2 person detector, open-vocab + VLM identify
  eval/recall.py                                         # IoU match, NMS, RecallMeter (per-domain/size)
  pipeline/{baseline_v0,demo_map,disaster_v1}.py         # v0 BEV, map projection, DisasterMonitor (fusion+alerts)
experiments/
  hypotheses.md                                          # H1–H7 + iteration log
  bench_recall.py                                        # multi-model recall benchmark (loaders + detectors)
  run_h6_disaster.py / run_h7_sahi_disaster.py           # disaster verifications (CPU)
  real_video_threat.py                                   # Stage-6 real-video pipeline (speed-fixed; alarm leads onset 7/7)
  rvt_precision.py                                        # precision sweep (threshold × duration-gate), caches rvt_arrays.npz
  track_run.py                                           # ByteTrack on FT-2 → per-person tracks_<type>.json + overlay mp4
  dashboard2_build.py                                    # dashboard v2: LEFT video | RIGHT speed-coloured moving dots, KO labels
  pressure_run.py                                        # MOAT: LEFT video | RIGHT Helbing crowd-pressure heatmap (압착 위험 지도)
  results/ , figures/                                    # logged numbers + plots
finetune/
  prepare_data.py / prepare_mix.py                       # VisDrone-person ; + COCO val2017 -> person_mix
  train_yolo.py / auto_train.py                          # YOLO11m FT ; resilient chunked harness (checkpoint+resume+OOM-retry)
  eval_recall.py / eval_sahi.py                          # before/after recall ; SAHI (H7a)
  bootstrap_ft2.sh / BOOTSTRAP_FT2.md                    # one-shot FT-2 bootstrap + runbook
```

---

## 10. Pending / next steps

**Done since last handoff (2026-06-22, all committed + pushed to GitHub):**
- ✅ **Multi-object tracker** (`experiments/track_run.py`) — ByteTrack on FT-2/yolo11m → per-person track IDs + (x,y,vx,vy) trajectories + dwell (UMN: 117 unique IDs, ~13/frame). The shared data layer for dashboard + dwell/OD + pressure.
- ✅ **Dashboard v2** (`experiments/dashboard2_build.py`) — rebuilt to user spec: LEFT real video | RIGHT moving dots coloured by danger (green/amber/red by speed) + trails, plain KO labels, H.264. **Panic demo** delivered (auto-detected UMN peak-motion panic via frame-diff; 위험도 높음, 6 red dots).
- ✅ **MOAT layer 1 — crowd pressure** (`flowsight/physics/crowd_pressure.py` + `experiments/pressure_run.py`) — Helbing P=ρ·Var(v) field from tracks (vectorised Gaussian deposition); pressure HEATMAP demo (압착 위험 지도). Validated: erratic crowd **137×** pressure of equally-dense coherent crowd. Hotspot lands on the dense cluster; level 높음.

- ✅ **MOAT layer 2 — metric calibration → absolute 0.02/s² alarm + non-planar 3-D** (2026-06-22, committed + pushed; Colab-Pro-L4 validated). The metric Helbing path already existed (`physics/pressure.py`+`density.py`); the new pieces are the px→m **bridge** (`geometry/calibration.py`: homography+Jacobian / pedestrian-1.7m), the **absolute alarm** (`crowd_pressure.frame_pressure_metric`→1/s², `alarm_level` at P_CRIT=0.02), and **non-planar 3-D** (`pipeline/moat_field.py`: metric-depth→foot-3D→DEM→U=ρgh+∇U, `MoatMonitor`). Tests 7/7 (ρ4·Var0.18→P0.72/s² exact; backproject 0 m). Synth E2E: absolute alarm leads crush onset **+10.47 s** (3 seeds); peak P 9–87× threshold, density 7.3–10.1/m². **Real (Colab Pro L4):** FT-2 ByteTrack on UMN (433 IDs) → `absolute_alarm_run.py` → real-footage P in 1/s² (event spikes to ~20/s², 0.02 line); **Depth-Anything-V2-Metric-Outdoor-Large-hf** → real metric depth 4.9–59.4 m (median 12.3 m), people resolved at depth. Demo `experiments/absolute_alarm_run.py`; doc `docs/moat2_absolute_alarm_2026-06-22.md`; H8/Iter-6. Outputs on Drive `flowsight_demo/{absolute_cctv.mp4,_frame.png,_timeline.png,depth_ref.png}`.

**Next (priority order):**
1. **Wire metric-depth calibration into the alarm (TOP):** the demo used pedestrian-scale (`--person-px`) which INFLATES absolute magnitude on mixed/indoor scenes (peak ~1000× threshold). Add a `DepthCalibrator` (metric depth → per-track ground X,Y) so `absolute_alarm_run` uses TRUE metres (no surveyed points) — turns the 0.02/s² alarm physically accurate. Note: VKITTI metric ckpt fails in `transformers.pipeline` (no `model_type`); use the `-hf` metric checkpoints.
2. **Precision refinement:** lift the over-sensitive alarm (real run: 680/774 frames p_max≥0.02; P=0.12–0.28 on UMN) via duration-gate/hysteresis; cheap via cached `rvt_arrays.npz`.
3. **Drone tracking + dashboard:** FT-2 (Drive) on aerial (baseline=0 on aerial) — proves input-source-agnostic. Needs GPU.
4. **Broaden anomaly types (mission §1):** surge, counterflow, dispersal, falls, blocked exits — one flow/track engine, a detector+metric per event (datasets in user's `FlowSight_AnomalyPattern_Resources_2026-06-21.md`).
5. **Stage 7:** multi-camera fusion, object ID (VLM), external validation on real crush footage, edge deploy.

**Colab cost note (2026-06-22):** free **GPU quota can exhaust** → CPU works (tracking slow ~5min @ imgsz640 step3; pressure fast ~10s). Session cap may force ending other notebooks. After VM recycle re-pipeline: clone → UMN download → `find_panic`(b64) trim → `track_run` → `dashboard2_build` / `pressure_run` (pressure reuses tracks, no re-track).

## 11. Honest caveats to carry forward
- Physics/disaster results are **synthetic-GT mechanism proofs** + real detector recall — not yet
  validated on real labeled crush footage (Stage 7).
- The image→ground mapping in the demo 3D map is a rough inverse-perspective; **metric accuracy
  needs calibration** (surveyed points or metric depth, H1).
- CCTV recall not fully restored (drone↔CCTV tradeoff).
- Free Colab recycles; keep code on GitHub + checkpoints on Drive; re-run to resume.
