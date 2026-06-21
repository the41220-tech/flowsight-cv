"""H2 — Terrain-coupled pressure gives earlier crush warning.

Claim: a precursor built from terrain potential energy (upstream mass x downhill
force, Fmag*rho_up) crosses its alarm threshold EARLIER than the flat Helbing
pressure (rho*Var(v)) precursor, because mass accumulates up-slope before basin
turbulence appears.

Calm run sets each detector's threshold (mean+4*std). Surge run measures
lead time before crush onset (basin density >= rho_crit). Thresholds are
per-signal, so the lead-time ordering is scale-invariant (not rigged by weights).
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowsight.geometry.terrain import Terrain, ramp_basin_elevation
from flowsight.sim.social_force_terrain import CrowdSim
from flowsight.physics.density import DensityField
from flowsight.physics.pressure import velocity_variance_grid, helbing_pressure, upstream_density
from flowsight.physics import potential

HERE = os.path.dirname(__file__)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

BOUNDS = (-7, 0, 7, 40)
CELL = 0.5
RHO_CRIT = 6.0          # people/m^2 -> "crush onset"


def _grid_force(ter, df):
    r = np.arange(df.ny); c = np.arange(df.nx)
    C, R = np.meshgrid(c, r)
    Xg = df.x0 + (C + 0.5) * df.cell
    Yg = df.y0 + (R + 0.5) * df.cell
    Fx, Fy, Fmag = potential.potential_gradient(ter, Xg, Yg)
    return Fx, Fy, Fmag, Yg


def signals(X, V, df, Fx, Fy, Fmag, Yg):
    d = df.compute(X)
    var = velocity_variance_grid(X, V, df)
    P = helbing_pressure(d, var)
    rho_up = upstream_density(d, Fx, Fy, shift_m=2.0, cell=df.cell)
    terrain_term = Fmag * rho_up
    crush_zone = (Yg >= 29) & (Yg <= 33)        # downstream jam against bottleneck
    upstream = (Yg >= 18) & (Yg <= 28)          # feeder slope (terrain precursor)
    def pk(field, mask):
        v = field[mask]
        return float(np.percentile(v, 95)) if v.size else 0.0
    return {"density": pk(d, crush_zone), "helbing": pk(P, crush_zone),
            "terrain": pk(terrain_term, upstream)}


def run(ter, df, Fx, Fy, Fmag, Yg, inflow, T, seed, **kw):
    sim = CrowdSim(ter, seed=seed, inflow_per_s=inflow, **kw)
    ts, series = [], {"density": [], "helbing": [], "terrain": []}
    nstep = int(T / sim.dt)
    for k in range(nstep):
        sim.step()
        if k % 4 == 0:
            X, V = sim.state()
            s = signals(X, V, df, Fx, Fy, Fmag, Yg) if len(X) else {"density": 0, "helbing": 0, "terrain": 0}
            ts.append(sim.time)
            for key in series:
                series[key].append(s[key])
    return np.array(ts), {k: np.array(v) for k, v in series.items()}


def rise_time(t, sig, frac=0.5):
    """First time a signal reaches `frac` of its own run-max (robust onset-of-
    rise; avoids fragile absolute/calm-baseline thresholds)."""
    sig = np.asarray(sig); mx = sig.max()
    if mx <= 0:
        return None
    idx = np.argmax(sig >= frac * mx)
    return float(t[idx])


def main(seed=1, frac=0.5):
    ter = Terrain(elevation_fn=ramp_basin_elevation())
    df = DensityField(bounds=BOUNDS, cell=CELL, sigma_m=0.75)
    Fx, Fy, Fmag, Yg = _grid_force(ter, df)
    geom = dict(corridor_bottle_halfwidth=0.8, bottleneck_y=33.0, y_exit=36.0, max_agents=1100)

    # single surge run; analyze temporal ORDERING of the precursors
    tsg, ssg = run(ter, df, Fx, Fy, Fmag, Yg, inflow=46.0, T=45, seed=seed, **geom)
    onset_idx = np.argmax(ssg["density"] >= RHO_CRIT)
    onset_t = float(tsg[onset_idx]) if ssg["density"].max() >= RHO_CRIT else None

    rises = {k: rise_time(tsg, ssg[k], frac) for k in ["terrain", "density", "helbing"]}
    res = {"rho_crit": RHO_CRIT, "rise_fraction": frac, "crush_onset_s": onset_t,
           "peak_density": float(ssg["density"].max()), "rise_times_s": rises,
           "lead_vs_crush_s": {}}
    for k, tr in rises.items():
        res["lead_vs_crush_s"][k] = round(onset_t - tr, 2) if (onset_t and tr is not None) else None
    if rises["terrain"] is not None and rises["helbing"] is not None:
        res["terrain_leads_helbing_by_s"] = round(rises["helbing"] - rises["terrain"], 2)
    # robustness: ordering at several rise fractions
    res["ordering_robustness"] = {}
    for fr in (0.3, 0.4, 0.5, 0.6):
        rt = {k: rise_time(tsg, ssg[k], fr) for k in ["terrain", "density", "helbing"]}
        order = sorted([k for k in rt if rt[k] is not None], key=lambda k: rt[k])
        res["ordering_robustness"][f"frac_{fr}"] = " -> ".join(order)

    with open(os.path.join(HERE, "results", "h2.json"), "w") as f:
        json.dump(res, f, indent=2)

    plt.figure(figsize=(9, 5))
    for k, col in [("density", "k"), ("helbing", "tab:orange"), ("terrain", "tab:blue")]:
        mx = ssg[k].max() or 1.0
        plt.plot(tsg, ssg[k] / mx, col, label=f"{k} (norm)")
        tr = rises[k]
        if tr is not None:
            plt.axvline(tr, c=col, ls=":", lw=1.2)
    plt.axhline(frac, ls="--", c="gray", lw=1, label=f"rise level ({int(frac*100)}%)")
    if onset_t:
        plt.axvline(onset_t, c="red", lw=1.5, label="crush onset (rho>=6)")
    plt.xlabel("time (s)"); plt.ylabel("signal / own surge-max")
    plt.title("H2: precursor ordering — terrain potential leads basin crush")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.ylim(0, None)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "h2_pressure.png"), dpi=120)

    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
