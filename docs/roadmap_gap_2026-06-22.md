# FlowSight — Progress vs Milestone & Gap to Multi-View + Full Anomaly Suite (2026-06-22)

**Perspective:** CV Engineer + Business Strategy Analyst

## 1. Where we are vs the current milestone

The current milestone = **crowd-movement analysis core with the crush application validated and a defensible physics moat.** With MOAT layer 2 (absolute 0.02/s² alarm + non-planar 3-D metric depth + accurate depth-ground calibration) this milestone is essentially **complete**.

| Capability block | % done | Status |
|---|---|---|
| Perception — detect + track (FT-2 YOLO11m + ByteTrack → per-person x,y,vx,vy) | ~85% | ✅ done; drone↔CCTV recall tradeoff remains (SAHI/FT-3 lever) |
| Geometry — BEV positioning, metric calibration, non-planar 3-D | ~80% | ✅ homography + pedestrian + **depth-ground (accurate, no survey)**; **multi-camera not yet** |
| Physics moat — Helbing pressure, absolute 0.02/s² alarm, terrain potential U=ρgh | ~90% | ✅ unit-tested + synth-E2E + real-footage (Colab L4); precision tuning + real-crush validation remain |
| **Anomaly-pattern suite (5 patterns)** | ~25% | ⚠️ divergence/curl primitives exist (`flow_features.py`); the 5 detectors not yet wired |
| Multi-camera fusion | ~10% | ⬜ single-cam done; **per-cam calibration now ready** (depth-ground), association not built |
| Object ID / VLM (open-vocab, Qwen2.5-VL) | ~5% | ⬜ designed (H4), not built |
| Productization (dashboard, edge) | ~30% | ✅ dashboard v2; edge deferred to v3 |

**Overall toward a full multi-view anomaly platform: ~55–60% of the hard tech.** The *hard, defensible* part (non-planar metric physics) is largely done; most of what remains for the anomaly suite is **integration of lightweight detectors that ride on the existing BEV tracker**, plus the genuinely-hard **multi-camera fusion**.

## 2. The key architectural leverage

Once the tracker emits per-person **(x, y, vx, vy) on the metric BEV map** — which we now have, in real metres — **4 of the 5 anomaly patterns reduce to lightweight signal detectors** (no heavy video models):

| Pattern | Detector | Lives where | Effort |
|---|---|---|---|
| Radial divergence (terror/explosion scatter) | `div(v_field) > θ` | BEV (we already compute divergence in `flow_features.py`) | **Low** |
| Fast directional approach (pre-attack) | speed z-score > 3σ + direction consistency | BEV tracks (history of vx,vy) | **Low** |
| Emergency / fall (void) | local density collapse Δρ → 0 | BEV density grid | **Low** |
| Geofence violation | point-in-polygon on map coords | BEV + H4 obstacle layer | **Low–Med** |
| Violence / fight | RWF-2000-trained classifier on the person patch | **raw-video cross-check** | **Med (GPU train)** |
| Terror (composite) | fast-approach → violence → divergence, time-windowed | rule over the above | **Low** |

Only **violence** needs a trained video model; **fall** optionally adds a skeletal/pose anomaly model for robustness. Everything else is numpy on data we already produce.

## 3. Gap → phases (effort, not calendar promises)

- **Phase A — BEV anomaly detectors (radial divergence, fast-approach, emergency-void, geofence).** Reuses tracker + `flow_features`. Mostly numpy; reference code exists. → ~1 focused week / a few sessions. *Covers 4 of 5 patterns.*
- **Phase B — Violence classifier.** RWF-2000 (`load_dataset`) → fine-tune YOLOv8-cls / Vi-SAFE on Colab Pro GPU; run as a patch cross-check. → ~2–3 days.
- **Phase C — Fall/pose robustness.** ViTPose → COSKAD / STG-NF skeletal anomaly, fused with the void detector. → ~3–4 days.
- **Phase D — Terror composite + explainability.** 3-stage composition rule + a VLM narrator (Hawk / Qwen2.5-VL: "crowd fleeing NW from a central disturbance"). → ~2 days.
- **Phase E — Multi-camera fusion (the next hard moat).** Per-camera depth-ground calibration (now feasible) → common world frame → cross-view track association (geometry + light re-ID) → unified BEV. → ~2–4 weeks (real research/integration).
- **Phase F — External validation + hardening.** WILDTRACK (multi-cam GT), real crush footage, precision/hysteresis on the alarm, FT-3 to lift CCTV recall. → ongoing.

**To a demo-able single-view full anomaly suite (all 5 + terror):** Phases A–D ≈ **~2–3 weeks**.
**To multi-view + full anomaly:** + Phase E ≈ **~5–7 weeks** total of focused work (Colab Pro accelerates the GPU-bound parts).

## 4. Strategic read

- The **moat is the physics** (non-planar 3-D metric pressure / potential), and it is ~90% done and now physically calibrated — this is the part competitors' flat heatmaps cannot replicate.
- The **anomaly suite is mostly commodity** detectors riding on our BEV tracker → fast to add, broad coverage, but lower defensibility. Build them for product breadth, not for the moat.
- The **next real differentiator after the physics moat is multi-camera fusion** (Phase E): it removes single-view blind spots and is hard to copy. Per-camera metric calibration — the piece that makes fusion tractable — is now in hand.
- Recommended sequence: **A (broad coverage, cheap) → B/C (the two model-backed patterns) → D (composite + explainability) → E (multi-cam moat) → F (validation)**, with FT-3/SAHI recall work folded in opportunistically.
