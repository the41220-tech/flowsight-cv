"""Absolute crowd-pressure ALARM demo — FlowSight moat layer 2.

Unlike ``pressure_run.py`` (relative pixel heatmap), this renders the pressure in
TRUE physical units (1/s^2) on a top-down METRIC ground map, with the Helbing
critical line ``P_CRIT = 0.02 /s^2`` fixed on the colour scale. Same colour =>
same physical risk in every video — so the alarm (안전 / 주의 / 위험) is absolute,
not within-scene.

Pipeline:  tracks_<type>.json  -- calibrate (px->m) -->  metric foot points
           -- frame_pressure_metric -->  P (1/s^2)  -- alarm_level -->  안전/주의/위험

Calibration (pick one):
  --mpp 0.05            uniform metres-per-pixel
  --person-px 26        median pedestrian height in px (-> mpp = 1.7 / 26)
  --homography "u1,v1,X1,Y1; u2,v2,X2,Y2; u3,v3,X3,Y3; u4,v4,X4,Y4"
                        >=4 surveyed image<->ground(m) correspondences (accurate)

Outputs (to --out):
  absolute_<type>.mp4        LEFT video | RIGHT top-down metric pressure map
  absolute_<type>_frame.png  peak-pressure still
  absolute_<type>_timeline.png  P_max(t) in 1/s^2 with the 0.02 alarm line

Run on Colab (reuses existing tracks, no re-tracking):
  !PYTHONPATH=. python -u experiments/absolute_alarm_run.py \
      --video /content/umn_clip.mp4 --tracks /content/tracks_cctv.json \
      --type cctv --person-px 26 --out /content
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flowsight.geometry.calibration import (
    DepthGroundCalibrator,
    HomographyCalibrator,
    PedestrianScaleCalibrator,
    metric_bounds,
    tracks_to_metric,
)
from flowsight.physics.crowd_pressure import (
    CAUTION_FRAC,
    P_CRIT,
    alarm_level,
    frame_pressure_metric,
)

CMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
WHITE = (235, 235, 235)
GREY = (150, 150, 150)
GREEN = (90, 210, 90)
AMBER = (245, 200, 40)
RED = (235, 60, 60)
SEV_COLOR = {"safe": GREEN, "caution": AMBER, "danger": RED}


def load_font(sz: int):
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def build_calibrator(args):
    if getattr(args, "depth_npz", None):
        dm = np.load(args.depth_npz)
        cal = DepthGroundCalibrator(dm, fov_deg=float(args.fov_deg))
        return cal, "metric-depth ground (fov=%.0f, plane d=%.2fm)" % (
            args.fov_deg, cal.d)
    if args.homography:
        rows = [r for r in args.homography.replace("\n", ";").split(";") if r.strip()]
        img, wld = [], []
        for r in rows:
            u, v, X, Y = (float(x) for x in r.split(","))
            img.append([u, v])
            wld.append([X, Y])
        return HomographyCalibrator.from_points(img, wld), "homography(%d pts)" % len(img)
    if args.person_px:
        cal = PedestrianScaleCalibrator(1.7 / float(args.person_px))
        return cal, "pedestrian %.0fpx->1.7m (%.4f m/px)" % (args.person_px, cal.s)
    mpp = float(args.mpp or 0.05)
    return PedestrianScaleCalibrator(mpp), "uniform %.4f m/px" % mpp


def heatmap_panel(P, pmax, W, H):
    """Absolute heatmap: full-scale = pmax (1/s^2). Same colour == same risk."""
    norm = np.clip(P / (pmax + 1e-12), 0.0, 1.0).astype(np.float32)
    big = cv2.resize(norm, (W, H), interpolation=cv2.INTER_CUBIC)
    big = np.clip(big, 0.0, 1.0)
    return cv2.applyColorMap((big * 255).astype(np.uint8), CMAP)


def draw_strip(canvas, W, strip_h, al, pmax):
    Himg = canvas.shape[0] - strip_h
    im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    f_t, f_k, f_s = load_font(16), load_font(20), load_font(13)
    sev_rgb = SEV_COLOR[al["severity"]]
    d.text((12, 7), "● 실시간 영상", fill=WHITE, font=f_t)
    d.text((12, 35), "압착 위험", fill=WHITE, font=f_k)
    d.text((110, 41), "현재", fill=GREY, font=f_s)
    d.text((150, 35), al["label"], fill=sev_rgb, font=f_k)
    pct = int(round(al["frac"] * 100))
    d.text((205, 41), "안전기준 대비 %d%%" % pct, fill=GREY, font=f_s)
    d.text((W + 12, 7), "● 압착 위험 지도 (실측 미터)", fill=WHITE, font=f_t)
    # absolute legend bar with the 0.02 위험 line marked at its true position
    bx, by, bw, bh = W + 12, 40, 170, 12
    grad = cv2.applyColorMap(
        np.tile(np.linspace(0, 255, bw, dtype=np.uint8), (bh, 1)), CMAP)
    grad = cv2.cvtColor(grad, cv2.COLOR_BGR2RGB)
    im.paste(Image.fromarray(grad), (bx, by))
    x_crit = bx + int(bw * min(1.0, P_CRIT / pmax))
    x_caut = bx + int(bw * min(1.0, CAUTION_FRAC * P_CRIT / pmax))
    d.line([(x_caut, by - 2), (x_caut, by + bh + 2)], fill=AMBER, width=1)
    d.line([(x_crit, by - 2), (x_crit, by + bh + 2)], fill=(255, 255, 255), width=2)
    d.text((bx - 2, by + bh + 1), "안전", fill=GREY, font=f_s)
    d.text((x_crit - 10, by + bh + 1), "위험", fill=WHITE, font=f_s)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def metric_to_panel(xy_m, geo, W, H):
    gx = (xy_m[:, 0] - geo["x0"]) / geo["cell"]
    gy = (xy_m[:, 1] - geo["y0"]) / geo["cell"]
    px = (gx / geo["gw"] * W).astype(int)
    py = (gy / geo["gh"] * H).astype(int)
    return np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)


def main(video, tracks_json, out, source_type, cal, cal_desc,
         cell_m=0.5, sigma_m=1.0, pmax=None):
    data = json.load(open(tracks_json))
    per = data["per_frame"]
    step = int(data["step"])
    fps = float(data["fps"])
    W, H = int(data["W"]), int(data["H"])

    # calibrate all frames -> metric, fixed bounds + grid for a stable map
    metric = [tracks_to_metric(cal, r["tracks"]) for r in per]
    bounds = metric_bounds([m[0] for m in metric], pad_m=2.0)
    fields = [frame_pressure_metric(xy, vel, bounds, cell_m, sigma_m)
              for (xy, vel) in metric]
    pvals = [f["p_max"] for f in fields]
    if pmax is None:  # absolute full-scale = max(3*CRIT, observed peak) so the
        pmax = max(3.0 * P_CRIT, float(np.percentile(pvals, 99)) if pvals else 0.0)
    print("[abs] %s: %d frames bounds=%s cell=%.2fm  cal=%s  pmax=%.4g /s^2"
          % (source_type, len(per), tuple(round(b, 1) for b in bounds), cell_m,
             cal_desc, pmax), flush=True)

    strip_h = 64
    out_w, out_h = W * 2, H + strip_h
    raw = out + "/_abs_%s_raw.mp4" % source_type
    final = out + "/absolute_%s.mp4" % source_type
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"),
                         max(1.0, fps / step), (out_w, out_h))
    cap = cv2.VideoCapture(video)
    best = (-1.0, None)
    times, series = [], []
    i = si = fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        if fi >= len(per):
            break
        rec, fld = per[fi], fields[fi]
        xy_m = metric[fi][0]
        fi += 1
        if fr.shape[1] != W or fr.shape[0] != H:
            fr = cv2.resize(fr, (W, H))

        hm = heatmap_panel(fld["pressure"], pmax, W, H)
        if len(xy_m):
            px, py = metric_to_panel(xy_m, fld["georef"], W, H)
            for x, y in zip(px, py):
                cv2.circle(hm, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        P = fld["pressure"]
        gy, gx = np.unravel_index(int(np.argmax(P)), P.shape)
        if P[gy, gx] > 0:
            cx = int((gx + 0.5) * W / P.shape[1])
            cy = int((gy + 0.5) * H / P.shape[0])
            cv2.circle(hm, (cx, cy), 16, (255, 255, 255), 2, cv2.LINE_AA)

        al = alarm_level(fld["p_max"])
        canvas = np.full((out_h, out_w, 3), 18, np.uint8)
        canvas[strip_h:strip_h + H, 0:W] = fr
        canvas[strip_h:strip_h + H, W:out_w] = hm
        cv2.line(canvas, (W, strip_h), (W, out_h), (60, 60, 60), 1)
        canvas = draw_strip(canvas, W, strip_h, al, pmax)
        vw.write(canvas)

        times.append(rec["t"])
        series.append(fld["p_max"])
        if fld["p_max"] > best[0]:
            best = (fld["p_max"], canvas.copy())
        if si % 50 == 0:
            print("[abs]   %d t=%.1fs P=%.4g /s^2 (%s)"
                  % (si, rec["t"], fld["p_max"], al["label"]), flush=True)
        i += 1
        si += 1
    cap.release()
    vw.release()

    if best[1] is not None:
        cv2.imwrite(out + "/absolute_%s_frame.png" % source_type, best[1])
    _timeline(times, series, out + "/absolute_%s_timeline.png" % source_type, source_type)

    ok = False
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", final], check=True)
        ok = os.path.exists(final)
    except Exception as e:  # noqa: BLE001
        print("[abs] ffmpeg failed (%s); keeping raw" % e, flush=True)
    if not ok:
        final = raw
    n_danger = int(np.sum(np.array(series) >= P_CRIT))
    print("=== ABS_DONE %s: %d frames, %d danger(>=0.02/s^2), peakP=%.4g -> %s ==="
          % (source_type, si, n_danger, max(series) if series else 0.0,
             os.path.basename(final)), flush=True)
    return final


def _timeline(times, series, path, source_type):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    t = np.array(times)
    s = np.array(series)
    plt.figure(figsize=(9, 3.2))
    # English labels here: matplotlib's default font lacks Hangul glyphs (the
    # mp4 overlay uses PIL+NanumGothic for Korean). Avoids tofu boxes.
    plt.plot(t, s, "b-", lw=1.2, label="crowd pressure P (1/s^2)")
    plt.axhline(P_CRIT, color="r", ls="--", lw=1, label="danger 0.02 /s^2 (Helbing)")
    plt.axhline(CAUTION_FRAC * P_CRIT, color="orange", ls=":", lw=1, label="caution 0.01")
    plt.fill_between(t, P_CRIT, s, where=(s >= P_CRIT), color="red", alpha=0.2)
    plt.xlabel("time (s)")
    plt.ylabel("P (1/s^2)")
    plt.title("Absolute crowd-pressure alarm — %s" % source_type)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--out", default="/content")
    ap.add_argument("--cell-m", type=float, default=0.5)
    ap.add_argument("--sigma-m", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=None, help="absolute full-scale (1/s^2)")
    ap.add_argument("--mpp", type=float, default=None, help="metres per pixel (uniform)")
    ap.add_argument("--person-px", type=float, default=None, help="median person height in px")
    ap.add_argument("--depth-npz", type=str, default=None,
                    help="metric depth map .npy -> ACCURATE ground scale (no survey)")
    ap.add_argument("--fov-deg", type=float, default=65.0,
                    help="horizontal FOV (deg) for depth-ground intrinsics")
    ap.add_argument("--homography", type=str, default=None,
                    help="'u,v,X,Y; ...' >=4 image<->ground(m) points")
    a = ap.parse_args()
    cal, desc = build_calibrator(a)
    main(a.video, a.tracks, a.out, a.type, cal, desc,
         cell_m=a.cell_m, sigma_m=a.sigma_m, pmax=a.pmax)
