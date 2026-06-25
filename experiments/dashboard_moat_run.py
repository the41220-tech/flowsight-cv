"""Non-expert dashboard, MOAT-DRIVEN (Cycle 17d).

Replaces dashboard2's per-person SPEED colouring with the validated crush-pressure
moat: each tracked person is coloured by the absolute Helbing pressure P=rho*Var(v)
[1/s^2] at their location (0.02/s^2 critical -> 안전/주의/위험), NOT raw speed. So the
"위험" colour is physics, not the (still-maturing) detector's confidence.

Pipeline: tracks_<type>.json (px foot points + px/s vel, from track_run.py)
  -> WILDTRACK calibration: foot px -> world metres; per-id world delta -> m/s
  -> flowsight.physics.moat_dashboard.frame_risk -> per-person colour + KO banner
  -> render LEFT real video | RIGHT risk-coloured moving dots + plain-Korean banner.

The metric conversion core (`metric_tracks`) is numpy-only and unit-tested; the
cv2/PIL rendering in main() runs on Colab (GPU not required for render).

Run (Colab):
  python experiments/dashboard_moat_run.py --video clip.mp4 --tracks tracks_cctv.json \
      --root /content/WTx/Wildtrack_dataset --cam CVLab1 --out /content/drive/MyDrive/flowsight_demo
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from flowsight.geometry.wildtrack import load_camera
from flowsight.physics.moat_dashboard import frame_risk, ko_banner

BOUNDS_M = (-3.0, -0.9, 9.0, 35.1)   # WILDTRACK ground extent (metres)


def _cam(root, cam):
    import os
    return load_camera(os.path.join(root, "calibrations", "intrinsic_zero", "intr_%s.xml" % cam),
                       os.path.join(root, "calibrations", "extrinsic", "extr_%s.xml" % cam))


def metric_tracks(per_frame, cam, bounds, dt):
    """Project per-frame pixel foot tracks to METRIC world + per-id velocity (m/s).

    per_frame: [{t, tracks:[{id,x,y,...}]}] with x,y = foot pixels.
    cam      : WildtrackCamera (foot px -> world via to_ground).
    Returns  : [{ids, xy_m (N,2), vel_m (N,2)}] per frame (vel = world delta / dt).
    """
    prev = {}                       # id -> world (x,y) from previous frame
    out = []
    for rec in per_frame:
        ids, xy = [], []
        for tk in rec["tracks"]:
            ids.append(tk["id"]); xy.append([tk["x"], tk["y"]])
        if xy:
            w = cam.to_ground(np.asarray(xy, float), bounds=bounds)   # (N,2) metres, clamped
        else:
            w = np.zeros((0, 2))
        vel = np.zeros((len(ids), 2))
        for i, tid in enumerate(ids):
            if tid in prev:
                vel[i] = (w[i] - prev[tid]) / dt
        prev = {tid: w[i] for i, tid in enumerate(ids)}
        out.append({"ids": ids, "xy_m": w, "vel_m": vel})
    return out


def main(a):
    import os
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    data = json.load(open(a.tracks))
    per_frame = data["per_frame"]
    W, H = data["W"], data["H"]
    fps = data.get("fps", 30.0); step = data.get("step", 1)
    dt = step / max(1e-6, fps)
    cam = _cam(a.root, a.cam)
    metr = metric_tracks(per_frame, cam, BOUNDS_M, dt)

    def font(sz):
        for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                  "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"):
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()

    os.makedirs(a.out, exist_ok=True)
    raw = a.out + "/dashboard_moat_%s_raw.mp4" % a.cam
    strip_h = 64
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps / step), (W * 2, H + strip_h))
    cap = cv2.VideoCapture(a.video)
    trails = {}
    best = (-1, None)
    for fi, (rec, m) in enumerate(zip(per_frame, metr)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["t"] * fps if "t" in rec else fi * step)
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H))
        right = np.full((H, W, 3), 18, np.uint8)
        r = frame_risk(m["xy_m"], m["vel_m"], BOUNDS_M)
        for (tk, a_) in zip(rec["tracks"], r["per_person"]):
            c = a_["color"][::-1]                    # RGB -> BGR for cv2
            x, y = int(tk["x"]), int(tk["y"])
            tr = trails.setdefault(tk["id"], [])
            tr.append((x, y)); tr[:] = tr[-a.trail:]
            for j in range(1, len(tr)):
                cv2.line(right, tr[j - 1], tr[j], c, 1, cv2.LINE_AA)
            cv2.circle(right, (x, y), 5, c, -1, cv2.LINE_AA)
        canvas = np.hstack([fr, right])
        strip = np.full((strip_h, W * 2, 3), 28, np.uint8)
        canvas = np.vstack([canvas, strip])
        im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im)
        col = {"safe": (0, 190, 0), "caution": (240, 170, 0), "danger": (220, 30, 30)}[r["frame"]["severity"]]
        d.text((12, H + 16), ko_banner(r), font=font(26), fill=col)
        d.text((12, 10), "실시간 영상", font=font(20), fill=(235, 235, 235))
        d.text((W + 12, 10), "위험 지도 (압사압력)", font=font(20), fill=(235, 235, 235))
        canvas = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        vw.write(canvas)
        score = r["frame"]["n_danger"] * 10 + r["frame"]["frac"]
        if score > best[0]:
            best = (score, canvas.copy())
    cap.release(); vw.release()
    if best[1] is not None:
        cv2.imwrite(a.out + "/dashboard_moat_%s_frame.png" % a.cam, best[1])
    h264 = a.out + "/dashboard_moat_%s.mp4" % a.cam
    os.system("ffmpeg -y -loglevel error -i %s -vcodec libx264 -pix_fmt yuv420p %s" % (raw, h264))
    print("=== DASH_MOAT_DONE %s frames=%d -> %s ===" % (a.cam, len(metr), h264), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--root", default="/content/WTx/Wildtrack_dataset")
    ap.add_argument("--cam", default="CVLab1")
    ap.add_argument("--out", default="/content/drive/MyDrive/flowsight_demo")
    ap.add_argument("--trail", type=int, default=8)
    main(ap.parse_args())
