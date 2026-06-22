"""MOAT layer 2 — synthetic end-to-end validation IN ABSOLUTE UNITS.

Builds an Itaewon-like sloped funnel, drives a Social-Force crush, and runs the
new ``MoatMonitor`` so every signal is physical:

  * Helbing crowd pressure in true ``1/s^2`` (calibrated), with the literature
    alarm at ``P_CRIT = 0.02 /s^2``.
  * The non-planar terrain-potential precursor (downhill push x upstream density).

Claims checked (printed + logged), nested early-warning chain:

  t_precursor   <   t_abs_alarm(0.02/s^2)   <   t_crush_onset(rho>=6/m^2)

i.e. (1) the ABSOLUTE Helbing alarm fires before the crowd reaches crush density,
and (2) the gravitational precursor leads even the absolute alarm — the moat's
extra margin. A FLAT control (no slope) removes the precursor channel, so the
extra lead disappears: that gap is what non-planar 3-D buys you.

Pure-numpy / CPU. Writes results/moat2_e2e.json + figures/moat2_e2e.png.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flowsight.geometry.terrain import Terrain, ramp_basin_elevation  # noqa: E402
from flowsight.physics.crowd_pressure import P_CRIT  # noqa: E402
from flowsight.pipeline.moat_field import MoatMonitor  # noqa: E402
from flowsight.sim.social_force_terrain import CrowdSim  # noqa: E402

HERE = os.path.dirname(__file__)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

BOUNDS = (-7, 0, 7, 40)
RHO_CRIT = 6.0  # operational crush density (persons/m^2), Itaewon anchor
CELL = 0.5
SIGMA = 0.75


def _masks(mon: MoatMonitor):
    """Grid-cell Y centres -> crush-zone and analysis-region boolean masks."""
    x0, y0, x1, y1 = mon.bounds
    gw = max(1, int(np.ceil((x1 - x0) / mon.cell)))
    gh = max(1, int(np.ceil((y1 - y0) / mon.cell)))
    ys = y0 + (np.arange(gh) + 0.5) * mon.cell
    Y = np.repeat(ys[:, None], gw, axis=1)
    crush = (Y >= 29) & (Y <= 33)
    region = (Y >= 18) & (Y <= 33)
    return crush, region


def _rise_time(t, sig, frac=0.5):
    sig = np.asarray(sig)
    mx = sig.max()
    return float(t[np.argmax(sig >= frac * mx)]) if mx > 0 else None


def _first_cross(t, sig, thr):
    sig = np.asarray(sig)
    return float(t[np.argmax(sig >= thr)]) if (sig >= thr).any() else None


def run_scenario(slope: bool, seed: int = 1, T: int = 38) -> dict:
    theta = 22.0 if slope else 0.0
    depth = 2.0 if slope else 0.0
    terr = Terrain(elevation_fn=ramp_basin_elevation(theta_deg=theta, basin_depth=depth))
    mon = MoatMonitor(BOUNDS, terrain=terr, cell_m=CELL, sigma_m=SIGMA)
    crush, region = _masks(mon)
    sim = CrowdSim(
        terr, seed=seed, inflow_per_s=46.0, corridor_bottle_halfwidth=0.8,
        bottleneck_y=33.0, y_exit=36.0, max_agents=1100,
    )
    ts, rho_peak, p_abs, precursor = [], [], [], []
    for k in range(int(T / sim.dt)):
        sim.step()
        if k % 4:
            continue
        X, V = sim.state()
        r = mon.step_metric(X, V)
        ts.append(round(sim.time, 2))
        rho_peak.append(float(np.percentile(r["density"][crush], 95)) if len(X) else 0.0)
        p_abs.append(float(np.percentile(r["pressure"][region], 95)) if len(X) else 0.0)
        precursor.append(
            float(np.percentile(r["terrain_push"][region], 95)) if len(X) else 0.0
        )
    ts = np.array(ts)
    onset = _first_cross(ts, rho_peak, RHO_CRIT)
    t_abs = _first_cross(ts, p_abs, P_CRIT)
    # honest precursor vs Helbing comparison: SAME rise criterion (25% of own max)
    t_pre = _rise_time(ts, precursor, 0.25) if slope else None
    t_hel = _rise_time(ts, p_abs, 0.25)
    return {
        "slope": slope,
        "crush_onset_s": onset,
        "abs_alarm_s": t_abs,  # P crosses 0.02 /s^2 (the product alarm)
        "abs_alarm_lead_before_onset_s": round(onset - t_abs, 2)
        if (onset and t_abs is not None) else None,
        "helbing_rise25_s": t_hel,
        "precursor_rise25_s": t_pre,
        "precursor_lead_over_helbing_s": round(t_hel - t_pre, 2)
        if (slope and t_hel is not None and t_pre is not None) else None,
        "peak_pressure_1_s2": round(float(np.max(p_abs)), 3),
        "peak_density_m2": round(float(np.max(rho_peak)), 2),
        "_t": ts.tolist(), "_rho": rho_peak, "_p": p_abs, "_pre": precursor,
    }


def main() -> dict:
    # multi-seed the headline (absolute-alarm lead) for robustness
    seeds = [1, 2, 3]
    slope_runs = [run_scenario(slope=True, seed=s) for s in seeds]
    sl = slope_runs[0]  # representative run for the figure
    fl = run_scenario(slope=False, seed=1)
    leads = [r["abs_alarm_lead_before_onset_s"] for r in slope_runs
             if r["abs_alarm_lead_before_onset_s"] is not None]
    pre_leads = [r["precursor_lead_over_helbing_s"] for r in slope_runs
                 if r["precursor_lead_over_helbing_s"] is not None]
    res = {
        "P_CRIT_1_s2": P_CRIT, "rho_crit_m2": RHO_CRIT, "seeds": seeds,
        "abs_alarm_lead_before_onset_mean_s": round(float(np.mean(leads)), 2)
        if leads else None,
        "precursor_lead_over_helbing_mean_s": round(float(np.mean(pre_leads)), 2)
        if pre_leads else None,
        "slope_representative": {k: v for k, v in sl.items() if not k.startswith("_")},
        "flat": {k: v for k, v in fl.items() if not k.startswith("_")},
    }
    with open(os.path.join(HERE, "results", "moat2_e2e.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    # figure: absolute pressure (with 0.02 line) + precursor + density, sloped scene
    t = np.array(sl["_t"])
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t, sl["_p"], "b-", label="Helbing pressure P (1/s²)")
    ax1.axhline(P_CRIT, color="b", ls=":", lw=1, label="P_CRIT = 0.02 /s² (Helbing)")
    pre = np.array(sl["_pre"])
    pre_n = pre / (pre.max() or 1) * (max(sl["_p"]) or 1)  # scaled for overlay
    ax1.plot(t, pre_n, color="green", ls="--", label="terrain precursor (scaled)")
    ax1.set_yscale("log")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("pressure (1/s², log)")
    ax2 = ax1.twinx()
    ax2.plot(t, sl["_rho"], color="orange", alpha=0.6, label="crush density (/m²)")
    ax2.axhline(RHO_CRIT, color="orange", ls=":", lw=1)
    ax2.set_ylabel("density (persons/m²)")
    for tt, c, lab in [
        (sl["precursor_rise25_s"], "green", "precursor"),
        (sl["abs_alarm_s"], "blue", "abs alarm 0.02"),
        (sl["crush_onset_s"], "red", "crush onset"),
    ]:
        if tt is not None:
            ax1.axvline(tt, color=c, lw=1.2, alpha=0.7)
    ax1.set_title("MOAT-2: absolute alarm + terrain precursor (sloped funnel)")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "moat2_e2e.png"), dpi=120)

    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
