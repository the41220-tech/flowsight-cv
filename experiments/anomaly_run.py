"""Phase A anomaly run — apply the BEV anomaly detectors to a tracks_<type>.json
on the metric BEV map, emit a per-frame events JSON + a timeline plot.

Calibration (px->m), pick one (same as absolute_alarm_run):
  --depth-npz depth.npy [--fov-deg 65]   accurate metric scale (no survey)
  --person-px 22                         pedestrian-height approx
  --mpp 0.05                             uniform metres-per-pixel

Detectors (FlowSight_AnomalyPattern_Resources, Phase A):
  radial divergence (flee), fast approach (pre-attack), emergency void (fall),
  geofence (optional polygon), terror composite (fast -> [violence] -> divergence).

Run on Colab:
  !PYTHONPATH=. python -u experiments/anomaly_run.py --tracks /content/tracks_cctv.json \
      --type cctv --depth-npz /content/depth_ref.npy --out /content/drive/MyDrive/flowsight_demo
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from flowsight.anomaly import AnomalyMonitor, TerrorComposite
from flowsight.geometry.calibration import (
    DepthGroundCalibrator,
    PedestrianScaleCalibrator,
    metric_bounds,
    tracks_to_metric,
)


def build_calibrator(a):
    if a.depth_npz:
        cal = DepthGroundCalibrator(np.load(a.depth_npz), fov_deg=float(a.fov_deg))
        return cal, "metric-depth (fov=%.0f, d=%.2fm)" % (a.fov_deg, cal.d)
    if a.person_px:
        cal = PedestrianScaleCalibrator(1.7 / float(a.person_px))
        return cal, "pedestrian %.0fpx (%.4f m/px)" % (a.person_px, cal.s)
    cal = PedestrianScaleCalibrator(float(a.mpp or 0.05))
    return cal, "uniform %.4f m/px" % (a.mpp or 0.05)


def main(a):
    data = json.load(open(a.tracks))
    per = data["per_frame"]
    cal, desc = build_calibrator(a)
    # metric tracks per frame + global bounds
    metric = []
    for r in per:
        xy, vel = tracks_to_metric(cal, r["tracks"])
        metric.append((xy, vel, [t["id"] for t in r["tracks"]]))
    bounds = metric_bounds([m[0] for m in metric], pad_m=2.0)
    mon = AnomalyMonitor(bounds, cell=a.cell_m)
    terror = TerrorComposite(window_s=a.terror_window_s)
    print("[anom] %s: %d frames bounds=%s cal=%s" %
          (a.type, len(per), tuple(round(b, 1) for b in bounds), desc), flush=True)

    rows, counts = [], {"divergence": 0, "fast_approach": 0, "void": 0,
                        "geofence": 0, "terror": 0}
    for i, r in enumerate(per):
        xy, vel, ids = metric[i]
        tracks_m = [{"id": ids[k], "x": float(xy[k, 0]), "y": float(xy[k, 1]),
                     "vx": float(vel[k, 0]), "vy": float(vel[k, 1])}
                    for k in range(len(xy))]
        res = mon.step(tracks_m)
        div_alert = res["divergence"]["alert"]
        n_fast = len(res["fast_approach"])
        n_void = len(res["void"])
        n_geo = len(res["geofence"])
        terr = terror.update(r["t"], fast=n_fast > 0, violence=False,
                             divergence=div_alert)
        counts["divergence"] += int(div_alert)
        counts["fast_approach"] += int(n_fast > 0)
        counts["void"] += int(n_void > 0)
        counts["geofence"] += int(n_geo > 0)
        counts["terror"] += int(terr)
        rows.append({"t": r["t"], "max_div": round(res["divergence"]["max_div"], 3),
                     "div_alert": bool(div_alert), "n_fast": n_fast,
                     "n_void": n_void, "n_geofence": n_geo, "terror": bool(terr)})
        if i % 50 == 0:
            print("[anom]   %d t=%.1fs max_div=%.3f fast=%d void=%d"
                  % (i, r["t"], res["divergence"]["max_div"], n_fast, n_void), flush=True)

    out_json = a.out + "/anomaly_%s.json" % a.type
    json.dump({"source_type": a.type, "calibration": desc, "bounds": bounds,
               "frames": len(per), "event_frame_counts": counts, "per_frame": rows},
              open(out_json, "w"), default=float)
    _timeline(rows, a.out + "/anomaly_%s_timeline.png" % a.type, a.type)
    print("=== ANOM_DONE %s: %d frames; event-frames %s -> anomaly_%s.json ==="
          % (a.type, len(per), counts, a.type), flush=True)


def _timeline(rows, path, source_type):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    t = np.array([r["t"] for r in rows])
    md = np.array([r["max_div"] for r in rows])
    nf = np.array([r["n_fast"] for r in rows])
    fig, ax1 = plt.subplots(figsize=(9, 3.4))
    ax1.plot(t, md, "b-", lw=1.1, label="radial divergence max (1/s)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("divergence (1/s)")
    ax2 = ax1.twinx()
    ax2.plot(t, nf, color="orange", alpha=0.7, label="fast-approach count")
    ax2.set_ylabel("fast-approach count")
    for r in rows:
        if r["terror"]:
            ax1.axvline(r["t"], color="red", lw=0.8, alpha=0.5)
    ax1.set_title("BEV anomaly timeline - %s (red = terror composite)" % source_type)
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--out", default="/content")
    ap.add_argument("--cell-m", type=float, default=1.0)
    ap.add_argument("--terror-window-s", type=float, default=8.0)
    ap.add_argument("--depth-npz", type=str, default=None)
    ap.add_argument("--fov-deg", type=float, default=65.0)
    ap.add_argument("--person-px", type=float, default=None)
    ap.add_argument("--mpp", type=float, default=None)
    main(ap.parse_args())
