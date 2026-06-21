"""H6 - Integrated multimodal slope -> disaster prediction.

Hypothesis: fusing detection + terrain-3D + flow/potential predicts a crowd
crush EARLIER and more ROBUSTLY (to missed detections) than density-only, on
non-planar terrain.

Full chain, end to end, on synthetic ground truth:
  sim crush -> SIMULATE DETECTION (project to drone cam, keep only ~FT-1 drone
  recall, add pixel noise) -> recover 3D map via terrain ray-cast (H1) ->
  DisasterMonitor risk (density / Helbing / terrain-potential, H2) -> compare
  multimodal vs density-only lead time before crush onset; sweep detection recall.

CPU-runnable. Writes results/h6_disaster.json + figure.
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
RHO_CRIT = 6.0
FT1_DRONE_RECALL = 0.51          # measured fine-tuned detector recall on drone


def simulate_detection(cam, terrain, X, recall, rng, px_noise=2.0, depth_rel_noise=0.03):
    """GT world people -> detected map positions via drone cam + recall dropout
    + pixel noise + metric-depth 3D back-projection (H1; vectorized/fast).
    Returns (map_xy, keep_mask)."""
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


def run_scenario(recall, seed=1, T=45):
    ter = Terrain(elevation_fn=ramp_basin_elevation())
    cam = PinholeCamera.look_at(C=(0, -8, 18), target=(0, 30, -6), f=900, width=1280, height=720)
    mon = DisasterMonitor(BOUNDS, terrain=ter, cell=0.5, sigma_m=0.75)
    sim = CrowdSim(ter, seed=seed, inflow_per_s=46.0, corridor_bottle_halfwidth=0.8,
                   bottleneck_y=33.0, y_exit=36.0, max_agents=1100)
    rng = np.random.default_rng(seed)
    crush_zone = (mon.Yg >= 29) & (mon.Yg <= 33)
    region = (mon.Yg >= 18) & (mon.Yg <= 33)
    ts, gt_rho, s_density, s_multi = [], [], [], []
    for k in range(int(T / sim.dt)):
        sim.step()
        if k % 4:
            continue
        X, V = sim.state()
        # ground-truth crush signal (full crowd)
        gt = mon.step(X, V, use_terrain=True)
        gt_peak = float(np.percentile(gt["density"][crush_zone], 95)) if len(X) else 0.0
        # detected (partial) -> recover velocities of kept agents
        det_xy, keep = simulate_detection(cam, ter, X, recall, rng)
        det_v = V[keep] if len(X) else np.zeros((0, 2))
        f = mon.step(det_xy, det_v, use_terrain=True)
        ts.append(sim.time); gt_rho.append(gt_peak)
        s_density.append(float(np.percentile(f["density"][crush_zone], 95)))
        s_multi.append(float(np.percentile(f["risk"][region], 95)))
    ts = np.array(ts)
    onset_i = np.argmax(np.array(gt_rho) >= RHO_CRIT)
    onset = float(ts[onset_i]) if max(gt_rho) >= RHO_CRIT else None
    rd, rm = rise_time(ts, s_density), rise_time(ts, s_multi)
    return {"recall": recall, "crush_onset_s": onset,
            "density_only_lead_s": round(onset - rd, 2) if (onset and rd is not None) else None,
            "multimodal_lead_s": round(onset - rm, 2) if (onset and rm is not None) else None,
            "_t": ts.tolist(), "_multi": s_multi, "_density": s_density, "_gt": gt_rho}


def main():
    res = {"rho_crit": RHO_CRIT, "scenarios": {}}
    for rec in (1.0, FT1_DRONE_RECALL, 0.30):
        r = run_scenario(rec)
        res["scenarios"][f"recall_{rec}"] = {k: v for k, v in r.items() if not k.startswith("_")}
        if rec == FT1_DRONE_RECALL:
            keep = r
    with open(os.path.join(HERE, "results", "h6_disaster.json"), "w") as f:
        json.dump(res, f, indent=2)

    plt.figure(figsize=(9, 5))
    t = np.array(keep["_t"])
    for sig, lab, col in [("_gt", "GT crush density", "k"), ("_density", "density-only (detected)", "tab:orange"),
                          ("_multi", "multimodal risk (detected)", "tab:blue")]:
        v = np.array(keep[sig]); v = v / (v.max() or 1)
        plt.plot(t, v, col, label=lab)
    if keep["crush_onset_s"]:
        plt.axvline(keep["crush_onset_s"], c="red", lw=1.5, label="crush onset")
    plt.title(f"H6: multimodal slope->disaster (detection recall={FT1_DRONE_RECALL})")
    plt.xlabel("time (s)"); plt.ylabel("signal / own max"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "h6_disaster.png"), dpi=120)

    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
