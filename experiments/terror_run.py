"""Terror-composite pipeline (Phase D) — fuses the BEV anomaly signals with the
trained violence classifier and narrates the result.

Per sampled frame:
  BEV (from tracks)     -> radial divergence + fast-approach + emergency-void
  frame (video)         -> violence classifier (finetune/violence_train weights)
  TerrorComposite       -> fast-approach -> [violence] -> divergence, time-windowed
  narrate()             -> plain-Korean event line for the operator

Violence is OPTIONAL: omit --weights to run BEV-only (violence stage off).

Run on Colab (L4):
  !PYTHONPATH=. python -u experiments/terror_run.py --video /content/umn_all.avi \
      --tracks /content/tracks_cctv.json --type cctv --person-px 22 \
      --weights /content/drive/MyDrive/flowsight_violence_ckpt/chunk05_best.pt \
      --out /content/drive/MyDrive/flowsight_demo
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from experiments.anomaly_run import build_calibrator
from flowsight.anomaly import AnomalyMonitor, TerrorComposite, narrate
from flowsight.geometry.calibration import metric_bounds, tracks_to_metric


def main(a) -> None:
    data = json.load(open(a.tracks))
    per = data["per_frame"]
    step = int(data["step"])
    cal, desc = build_calibrator(a)
    metric = []
    for r in per:
        xy, vel = tracks_to_metric(cal, r["tracks"])
        metric.append((xy, vel, [t["id"] for t in r["tracks"]]))
    bounds = metric_bounds([m[0] for m in metric], pad_m=2.0)
    mon = AnomalyMonitor(bounds, cell=a.cell_m)
    terror = TerrorComposite(window_s=a.terror_window_s)

    vd = None
    if a.weights:
        from flowsight.anomaly import ViolenceDetector
        vd = ViolenceDetector(a.weights, conf_thresh=a.viol_thresh)
    cap = cv2.VideoCapture(a.video) if a.video else None
    print("[terror] %s: %d frames cal=%s violence=%s" %
          (a.type, len(per), desc, "on" if vd else "off"), flush=True)

    events, counts = [], {"divergence": 0, "fast": 0, "void": 0,
                          "violence": 0, "terror": 0}
    series_t, series_viol, series_div = [], [], []
    i = fi = 0
    while fi < len(per):
        frame = None
        if cap is not None:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step:
                i += 1
                continue
            frame = fr
        r = per[fi]
        xy, vel, ids = metric[fi]
        tracks_m = [{"id": ids[k], "x": float(xy[k, 0]), "y": float(xy[k, 1]),
                     "vx": float(vel[k, 0]), "vy": float(vel[k, 1])}
                    for k in range(len(xy))]
        res = mon.step(tracks_m)
        div = res["divergence"]
        n_fast, n_void = len(res["fast_approach"]), len(res["void"])
        viol = vd.predict_frame(frame) if (vd and frame is not None) else \
            {"fight_prob": 0.0, "violence": False}
        terr = terror.update(r["t"], fast=n_fast > 0,
                             violence=viol["violence"], divergence=div["alert"])
        counts["divergence"] += int(div["alert"])
        counts["fast"] += int(n_fast > 0)
        counts["void"] += int(n_void > 0)
        counts["violence"] += int(viol["violence"])
        counts["terror"] += int(terr)
        series_t.append(r["t"])
        series_viol.append(viol["fight_prob"])
        series_div.append(div["max_div"])
        state = {"terror": terr, "violence": viol["violence"],
                 "fight_prob": viol["fight_prob"], "divergence": div["alert"],
                 "div_center": div["center_xy"], "n_fast": n_fast, "n_void": n_void}
        line = narrate(r["t"], state)
        if line:
            events.append({"t": r["t"], "line": line, "fight_prob": viol["fight_prob"],
                           "max_div": round(div["max_div"], 3), "n_fast": n_fast,
                           "terror": bool(terr)})
        if fi % 50 == 0:
            print("[terror]   %d t=%.1fs div=%.2f fast=%d fight=%.2f terror=%s"
                  % (fi, r["t"], div["max_div"], n_fast, viol["fight_prob"], terr),
                  flush=True)
        fi += 1
        i += 1
    if cap is not None:
        cap.release()

    json.dump({"source_type": a.type, "calibration": desc, "violence": bool(vd),
               "event_frame_counts": counts, "events": events},
              open(a.out + "/terror_%s.json" % a.type, "w"), default=float)
    _timeline(series_t, series_viol, series_div, a.viol_thresh,
              a.out + "/terror_%s_timeline.png" % a.type, a.type)
    print("=== TERROR_DONE %s: counts=%s, %d narrated events -> terror_%s.json ==="
          % (a.type, counts, len(events), a.type), flush=True)


def _timeline(t, viol, div, vth, path, source_type):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    t = np.array(t)
    fig, ax1 = plt.subplots(figsize=(9, 3.4))
    ax1.plot(t, viol, "r-", lw=1.1, label="violence prob (frame cls)")
    ax1.axhline(vth, color="r", ls="--", lw=1, label="violence thresh")
    ax1.set_ylabel("violence prob")
    ax1.set_xlabel("time (s)")
    ax2 = ax1.twinx()
    ax2.plot(t, div, color="b", alpha=0.6, label="radial divergence (1/s)")
    ax2.set_ylabel("divergence (1/s)")
    ax1.set_title("Terror pipeline - %s (violence x divergence x fast-approach)"
                  % source_type)
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--weights", default=None, help="violence cls weights (optional)")
    ap.add_argument("--out", default="/content")
    ap.add_argument("--cell-m", type=float, default=1.0)
    ap.add_argument("--terror-window-s", type=float, default=8.0)
    ap.add_argument("--viol-thresh", type=float, default=0.5)
    ap.add_argument("--depth-npz", type=str, default=None)
    ap.add_argument("--fov-deg", type=float, default=65.0)
    ap.add_argument("--person-px", type=float, default=None)
    ap.add_argument("--mpp", type=float, default=None)
    main(ap.parse_args())
