"""H7 - Detection recall (SAHI) -> disaster prediction quality.

Research-grounded thresholds (Itaewon/Seoul Halloween crush, empirical):
  average density 7.57 ppl/m^2 (max 9.95), i.e. dangerous compression sets in
  around ~6 ppl/m^2 -> we use RHO_CRIT = 6.0 as the operational crush density.

Hypothesis H7 (the part testable on CPU now):
  The density-threshold alarm (alarm when the *detected* density reaches the
  empirical crush density) is RECALL-SENSITIVE. A detector that misses people
  under-counts density by ~recall, so a low-recall detector (FT-1 drone ~0.51)
  reaches the alarm threshold late or never, while a SAHI-level detector
  (~0.86, tiling recovers small/distant people) recovers the true density and
  fires the alarm on time. The multimodal terrain-potential risk channel keeps
  its early lead regardless of recall (this re-confirms H6), so the safety net
  is multimodal -- but SAHI is what makes the *density map itself* trustworthy.

This motivates adding SAHI tiling on top of the FT-2 detector (no retraining).
The detector-side recall gain (H7a) is measured separately on Colab with the
real FT-2 weights (run_h7a_sahi_eval on Colab); this file is the CPU pipeline
study (H7b) on synthetic ground truth.

CPU-only (numpy). Writes results/h7_sahi_disaster.json + figures/h7_sahi.png.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowsight.geometry.camera import PinholeCamera
from flowsight.geometry.terrain import Terrain, ramp_basin_elevation
from flowsight.sim.social_force_terrain import CrowdSim
from flowsight.pipeline.disaster_v1 import DisasterMonitor

HERE = os.path.dirname(__file__)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

BOUNDS = (-7, 0, 7, 40)
RHO_CRIT = 6.0                      # operational crush density (Itaewon-grounded)
# Detector recall regimes:
#   1.00 = oracle (all people detected)
#   0.86 = SAHI-projected (sliced inference recovers small/distant; lit. 31.8->86.4%)
#   0.51 = FT-1/FT-2 drone recall (measured)
#   0.30 = weak baseline
RECALLS = [("oracle", 1.00), ("sahi", 0.86), ("ft1_drone", 0.51)]


def simulate_detection(cam, terrain, X, recall, rng, px_noise=2.0, depth_rel_noise=0.03):
    """GT world people -> detected map positions via drone cam + recall dropout
    + pixel noise + metric-depth 3D back-projection (H1)."""
    if len(X) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    Z = terrain.elevation(X[:, 0], X[:, 1])
    uv, depth = cam.project(np.column_stack([X[:, 0], X[:, 1], Z]))
    keep = rng.random(len(X)) < recall
    n = int(keep.sum())
    if n == 0:
        return np.zeros((0, 2)), keep
    uvk = uv[keep] + rng.normal(0, px_noise, (n, 2))
    dk = depth[keep] * (1 + rng.normal(0, depth_rel_noise, n))
    rec = cam.backproject_depth(uvk, dk)[:, :2]
    return rec, keep


def rise_time(t, sig, frac=0.5):
    sig = np.asarray(sig); mx = sig.max()
    return float(t[np.argmax(sig >= frac * mx)]) if mx > 0 else None


def first_cross(t, sig, thr):
    sig = np.asarray(sig)
    idx = np.argmax(sig >= thr)
    return float(t[idx]) if (sig >= thr).any() else None


def run_scenario(recall, seed=1, T=45):
    ter = Terrain(elevation_fn=ramp_basin_elevation())
    cam = PinholeCamera.look_at(C=(0, -8, 18), target=(0, 30, -6), f=900, width=1280, height=720)
    mon = DisasterMonitor(BOUNDS, terrain=ter, cell=0.5, sigma_m=0.75)
    sim = CrowdSim(ter, seed=seed, inflow_per_s=46.0, corridor_bottle_halfwidth=0.8,
                   bottleneck_y=33.0, y_exit=36.0, max_agents=1100)
    rng = np.random.default_rng(seed)
    crush_zone = (mon.Yg >= 29) & (mon.Yg <= 33)
    region = (mon.Yg >= 18) & (mon.Yg <= 33)
    ts, gt_rho, det_rho, s_multi = [], [], [], []
    for k in range(int(T / sim.dt)):
        sim.step()
        if k % 4:
            continue
        X, V = sim.state()
        gt = mon.step(X, V, use_terrain=True)
        gt_peak = float(np.percentile(gt["density"][crush_zone], 95)) if len(X) else 0.0
        det_xy, keep = simulate_detection(cam, ter, X, recall, rng)
        det_v = V[keep] if len(X) else np.zeros((0, 2))
        f = mon.step(det_xy, det_v, use_terrain=True)
        det_peak = float(np.percentile(f["density"][crush_zone], 95)) if len(det_xy) else 0.0
        ts.append(sim.time); gt_rho.append(gt_peak); det_rho.append(det_peak)
        s_multi.append(float(np.percentile(f["risk"][region], 95)))
    ts = np.array(ts); gt_rho = np.array(gt_rho); det_rho = np.array(det_rho)

    onset = first_cross(ts, gt_rho, RHO_CRIT)               # GT crush onset
    det_alarm = first_cross(ts, det_rho, RHO_CRIT)          # density-alarm on DETECTED density
    # under-count ratio over the active build-up window (GT density > 2)
    active = gt_rho > 2.0
    undercount = float(np.mean(det_rho[active] / np.clip(gt_rho[active], 1e-6, None))) if active.any() else None
    rm = rise_time(ts, s_multi)
    return {
        "recall": recall,
        "gt_peak_density": round(float(gt_rho.max()), 2),
        "det_peak_density": round(float(det_rho.max()), 2),
        "undercount_ratio": round(undercount, 3) if undercount is not None else None,
        "crush_onset_s": round(onset, 2) if onset is not None else None,
        "density_alarm_fired": det_alarm is not None,
        "density_alarm_lead_s": (round(onset - det_alarm, 2)
                                 if (onset is not None and det_alarm is not None) else None),
        "multimodal_lead_s": (round(onset - rm, 2)
                              if (onset is not None and rm is not None) else None),
        "_t": ts.tolist(), "_gt": gt_rho.tolist(), "_det": det_rho.tolist(), "_multi": s_multi,
    }


def main():
    res = {"rho_crit": RHO_CRIT, "note": "density-alarm threshold = crush density (Itaewon ~6-7.57/m2)",
           "scenarios": {}}
    keep = {}
    for name, rec in RECALLS:
        r = run_scenario(rec)
        res["scenarios"][name] = {k: v for k, v in r.items() if not k.startswith("_")}
        keep[name] = r

    with open(os.path.join(HERE, "results", "h7_sahi_disaster.json"), "w") as f:
        json.dump(res, f, indent=2)

    # figure: detected density vs GT, alarm threshold, per recall
    plt.figure(figsize=(10, 5))
    t = np.array(keep["oracle"]["_t"])
    plt.plot(t, keep["oracle"]["_gt"], "k", lw=2, label="GT density (truth)")
    for name, col in [("sahi", "tab:green"), ("ft1_drone", "tab:orange")]:
        plt.plot(t, keep[name]["_det"], col, label=f"detected density ({name}, r={dict(RECALLS)[name]})")
    plt.axhline(RHO_CRIT, c="purple", ls="--", lw=1.5, label=f"crush-density alarm = {RHO_CRIT}/m2")
    if keep["oracle"]["crush_onset_s"]:
        plt.axvline(keep["oracle"]["crush_onset_s"], c="gray", ls=":", label="GT crush onset")
    plt.title("H7: detection recall (SAHI) controls the density-alarm — low recall under-counts & fires late/never")
    plt.xlabel("time (s)"); plt.ylabel("density (ppl/m^2)"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "h7_sahi.png"), dpi=120)

    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
