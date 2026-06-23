"""H4 (Cycle 10): does a density-FREE disorder channel lead the Helbing product alarm?

Helbing P = rho * Var(v) is density-GATED, so it fires late. Two density-free
channels are compared on the SAME social-force crush sim (non-circular: one sim,
several readouts), using the codebase's own honest rise-to-X%-of-max criterion:

  * Var(v) magnitude disorder (the non-density factor of P) — leads in coherent flow.
  * direction disorder (1-phi) / velocity entropy — needs MULTIDIRECTIONAL turbulence
    (counterflow / shockwaves); the single-exit funnel sim is coherent-flow, so these
    stay ~0 here. That is an honest limitation: the direction channel needs Itaewon-
    type counterflow or real footage to exercise (PLOS One 2024 entropy term).

Run:  PYTHONPATH=. python experiments/coherence_lead.py
"""
from __future__ import annotations

import numpy as np

from flowsight.geometry.terrain import Terrain, ramp_basin_elevation
from flowsight.physics.coherence import disorder_index, velocity_entropy
from flowsight.physics.crowd_pressure import P_CRIT
from flowsight.pipeline.moat_field import MoatMonitor
from flowsight.sim.social_force_terrain import CrowdSim

BOUNDS = (-7, 0, 7, 40)
CELL, SIGMA, RHO_CRIT = 0.5, 0.75, 6.0


def _rise(t, sig, frac):
    sig = np.asarray(sig)
    mx = sig.max()
    return float(t[np.argmax(sig >= frac * mx)]) if mx > 0 else None


def _cross(t, sig, thr):
    sig = np.asarray(sig)
    return float(t[np.argmax(sig >= thr)]) if (sig >= thr).any() else None


def run(seed=1, T=38):
    terr = Terrain(elevation_fn=ramp_basin_elevation(theta_deg=22.0, basin_depth=2.0))
    mon = MoatMonitor(BOUNDS, terrain=terr, cell_m=CELL, sigma_m=SIGMA)
    x0, y0, x1, y1 = BOUNDS
    gw = int(np.ceil((x1 - x0) / CELL)); gh = int(np.ceil((y1 - y0) / CELL))
    ys = y0 + (np.arange(gh) + 0.5) * CELL
    Y = np.repeat(ys[:, None], gw, axis=1)
    crush = (Y >= 29) & (Y <= 33); region = (Y >= 18) & (Y <= 33)
    sim = CrowdSim(terr, seed=seed, inflow_per_s=46.0, corridor_bottle_halfwidth=0.8,
                   bottleneck_y=33.0, y_exit=36.0, max_agents=1100)
    ts, rho, P, VV, Ddir, Hent = [], [], [], [], [], []
    for k in range(int(T / sim.dt)):
        sim.step()
        if k % 4:
            continue
        X, V = sim.state()
        r = mon.step_metric(X, V)
        ts.append(round(sim.time, 2))
        has = len(X) > 0
        rho.append(float(np.percentile(r["density"][crush], 95)) if has else 0.0)
        P.append(float(np.percentile(r["pressure"][region], 95)) if has else 0.0)
        VV.append(float(np.percentile(r["var_v"][region], 95)) if has else 0.0)
        m = (X[:, 1] >= 18) & (X[:, 1] <= 33) if has else np.zeros(0, bool)
        Vr = V[m] if has else np.zeros((0, 2))
        Ddir.append(disorder_index(Vr)); Hent.append(velocity_entropy(Vr))
    ts = np.array(ts)
    onset = _cross(ts, rho, RHO_CRIT); t_abs = _cross(ts, P, P_CRIT)
    t_hel = _rise(ts, P, 0.25); t_var = _rise(ts, VV, 0.25)
    out = {
        "onset_s": onset, "abs_alarm_s": t_abs,
        "helbing_rise25_s": t_hel, "varv_rise25_s": t_var,
        "varv_lead_over_helbing_s": round(t_hel - t_var, 2) if (t_hel and t_var) else None,
        "varv_lead_over_abs_alarm_s": round(t_abs - t_var, 2) if (t_abs and t_var) else None,
        "varv_lead_over_onset_s": round(onset - t_var, 2) if (onset and t_var) else None,
        "peak_pressure": round(max(P), 3), "peak_varv": round(max(VV), 3),
        "peak_dir_disorder": round(max(Ddir), 3), "peak_entropy": round(max(Hent), 3),
    }
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
