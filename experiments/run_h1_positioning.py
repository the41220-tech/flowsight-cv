"""H1 — Non-planar positioning.

Claim: on non-planar terrain, recovering ground position via terrain ray-cast
or metric-depth back-projection cuts horizontal error >=50% vs a single
flat-ground homography.

Synthetic, fully controlled: known camera + known terrain + known people.
Compares 4 methods against ground truth. Writes results/h1.json + a figure.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowsight.geometry.camera import PinholeCamera
from flowsight.geometry.terrain import Terrain, ramp_basin_elevation
from flowsight.geometry.homography import fit_homography, apply_homography, MultiHomography

HERE = os.path.dirname(__file__)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)


def main(seed=0, N=200, depth_rel_noise=0.03):
    rng = np.random.default_rng(seed)
    ter = Terrain(elevation_fn=ramp_basin_elevation())
    cam = PinholeCamera.look_at(C=(0, -8, 18), target=(0, 30, -6),
                                f=900, width=1280, height=720)

    # --- calibration points (operator clicks ground points w/ known map x,y) ---
    cal_plaza = np.array([[-4, 6], [4, 6], [-4, 12], [4, 12]], float)
    cal_ramp = np.array([[-3, 22], [3, 22], [-3, 30], [3, 30], [0, 34]], float)
    cal_xy = np.vstack([cal_plaza, cal_ramp])
    cal_z = ter.elevation(cal_xy[:, 0], cal_xy[:, 1])
    cal_uv, _ = cam.project(np.column_stack([cal_xy, cal_z]))

    H_single = fit_homography(cal_uv, cal_xy)             # flat assumption
    multi = MultiHomography()                              # piecewise fix
    pl = cal_xy[:, 1] < 15
    multi.fit("plaza", cal_uv[pl], cal_xy[pl])
    multi.fit("ramp", cal_uv[~pl], cal_xy[~pl])

    # --- test people across plaza+ramp+basin ---
    xs = rng.uniform(-4, 4, N); ys = rng.uniform(5, 36, N)
    zs = ter.elevation(xs, ys)
    P = np.column_stack([xs, ys, zs])
    uv, depthZ = cam.project(P)
    gt = P[:, :2]

    est = {}
    est["A_single_homography"] = apply_homography(H_single, uv)
    est["B_multi_homography"] = multi.apply(uv)
    rc = []
    for i in range(N):
        C, d = cam.ray(uv[i]); hit = ter.raycast(C, d[0])
        rc.append(hit[:2] if hit is not None else [np.nan, np.nan])
    est["C_terrain_raycast"] = np.array(rc)
    noisy = depthZ * (1 + rng.normal(0, depth_rel_noise, N))
    est["D_metric_depth"] = cam.backproject_depth(uv, noisy)[:, :2]

    elev_dev = np.abs(zs)                                  # deviation from flat
    on_slope = ys >= 15
    out = {"N": N, "depth_rel_noise": depth_rel_noise, "methods": {}}
    for k, e in est.items():
        err = np.linalg.norm(e - gt, axis=1)
        m = np.isfinite(err)
        out["methods"][k] = {
            "mean_m": float(np.mean(err[m])),
            "median_m": float(np.median(err[m])),
            "p90_m": float(np.percentile(err[m], 90)),
            "mean_on_slope_m": float(np.mean(err[m & on_slope])),
            "valid": int(m.sum()),
        }
    base = out["methods"]["A_single_homography"]["mean_on_slope_m"]
    for k in out["methods"]:
        out["methods"][k]["slope_err_reduction_vs_A_%"] = round(
            100 * (1 - out["methods"][k]["mean_on_slope_m"] / base), 1)

    with open(os.path.join(HERE, "results", "h1.json"), "w") as f:
        json.dump(out, f, indent=2)

    # figure: error vs elevation deviation
    plt.figure(figsize=(8, 5))
    for k, e in est.items():
        err = np.linalg.norm(e - gt, axis=1)
        order = np.argsort(elev_dev)
        plt.scatter(elev_dev, err, s=8, alpha=0.5, label=k.split("_", 1)[1])
    plt.xlabel("|elevation| deviation from flat plane (m)")
    plt.ylabel("horizontal positioning error (m)")
    plt.title("H1: positioning error vs terrain non-planarity")
    plt.legend(); plt.ylim(0, None); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "h1_positioning.png"), dpi=120)

    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
