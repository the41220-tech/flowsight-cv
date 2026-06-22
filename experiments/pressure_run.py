"""Pressure-heatmap demo — FlowSight moat layer 1 (Helbing crowd pressure).

LEFT  = real video.
RIGHT = the crowd-PRESSURE field from flowsight/physics/crowd_pressure.py
        (cool = safe -> hot = crush risk), with people as dots and the peak-risk
        spot ringed. Pressure = local density x velocity variance (Helbing 2007):
        a DENSE crowd that moves ERRATICALLY is the crush signature — density
        alone misses it.

Scale is RELATIVE (uncalibrated pixels): it shows WHERE risk concentrates, not
absolute /s^2. Metric calibration (-> the 0.02/s^2 threshold) and non-planar
3-D terrain are the next moat layers.

Reads tracks_<type>.json (track_run.py) + the clip. Writes pressure_<type>.mp4
(H.264) + a peak-pressure still PNG.

Run on Colab (reuses existing tracks, no re-tracking):
  !PYTHONPATH=. python -u experiments/pressure_run.py \
      --video /content/umn_clip.mp4 --tracks /content/tracks_cctv.json \
      --type cctv --out /content
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flowsight.physics.crowd_pressure import clip_pressure_scale, frame_pressure

CMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
WHITE = (235, 235, 235)
GREY = (150, 150, 150)
GREEN = (90, 210, 90)
AMBER = (245, 200, 40)
RED = (235, 60, 60)


def load_font(sz):
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def level(norm):
    if norm >= 0.67:
        return "높음", RED
    if norm >= 0.34:
        return "보통", AMBER
    return "낮음", GREEN


def heatmap_panel(P, scale, W, H):
    norm = np.clip(P / (scale + 1e-12), 0.0, 1.0).astype(np.float32)
    big = cv2.resize(norm, (W, H), interpolation=cv2.INTER_CUBIC)
    big = np.clip(big, 0.0, 1.0)
    hm = cv2.applyColorMap((big * 255).astype(np.uint8), CMAP)
    return hm


def draw_strip(canvas, W, strip_h, lv, lv_c, t):
    Himg = canvas.shape[0] - strip_h
    im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    f_t = load_font(16)
    f_k = load_font(20)
    f_s = load_font(13)
    d.text((12, 7), "● 실시간 영상", fill=WHITE, font=f_t)
    d.text((12, 35), "압착 위험", fill=WHITE, font=f_k)
    d.text((110, 41), "현재", fill=GREY, font=f_s)
    d.text((150, 35), lv, fill=lv_c, font=f_k)
    d.text((W + 12, 7), "● 압착 위험 지도", fill=WHITE, font=f_t)
    d.text((W + 150, 11), "(밀도 × 움직임의 무질서)", fill=GREY, font=f_s)
    # gradient legend bar (cool -> hot) with 안전 / 위험
    bx, by, bw, bh = W + 12, 40, 150, 12
    grad = cv2.applyColorMap(
        np.tile(np.linspace(0, 255, bw, dtype=np.uint8), (bh, 1)), CMAP)
    grad = cv2.cvtColor(grad, cv2.COLOR_BGR2RGB)
    im.paste(Image.fromarray(grad), (bx, by))
    d.text((bx - 1, by + bh + 1), "안전", fill=GREY, font=f_s)
    d.text((bx + bw - 26, by + bh + 1), "위험", fill=GREY, font=f_s)
    d.text((10, strip_h + Himg - 22), "t = %.1fs" % t, fill=(220, 220, 220), font=f_s)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def main(video, tracks_json, out, source_type, gh=24, gw=32):
    data = json.load(open(tracks_json))
    per = data["per_frame"]
    step = int(data["step"])
    fps = float(data["fps"])
    W = int(data["W"])
    H = int(data["H"])
    fields = [frame_pressure(r["tracks"], W, H, gh=gh, gw=gw) for r in per]
    scale = clip_pressure_scale(fields, 95.0)
    print("[press] %s: %d frames %dx%d grid=%dx%d clip_scale(p95)=%.4g"
          % (source_type, len(per), W, H, gh, gw, scale), flush=True)

    strip_h = 64
    out_w, out_h = W * 2, H + strip_h
    raw = out + "/_press_%s_raw.mp4" % source_type
    final = out + "/pressure_%s.mp4" % source_type
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"),
                         max(1.0, fps / step), (out_w, out_h))
    cap = cv2.VideoCapture(video)
    best = (-1.0, None)
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
        rec = per[fi]
        fld = fields[fi]
        fi += 1
        if fr.shape[1] != W or fr.shape[0] != H:
            fr = cv2.resize(fr, (W, H))

        hm = heatmap_panel(fld["pressure"], scale, W, H)
        for tk in rec["tracks"]:
            cv2.circle(hm, (int(tk["x"]), int(tk["y"])), 3, (255, 255, 255), -1, cv2.LINE_AA)
        # ring the peak-pressure cell
        P = fld["pressure"]
        gy, gx = np.unravel_index(int(np.argmax(P)), P.shape)
        if P[gy, gx] > 0:
            px = int((gx + 0.5) * W / P.shape[1])
            py = int((gy + 0.5) * H / P.shape[0])
            cv2.circle(hm, (px, py), 16, (255, 255, 255), 2, cv2.LINE_AA)

        nmax = float(fld["p_max"] / (scale + 1e-12))
        lv, lv_c = level(nmax)
        canvas = np.full((out_h, out_w, 3), 18, np.uint8)
        canvas[strip_h:strip_h + H, 0:W] = fr
        canvas[strip_h:strip_h + H, W:out_w] = hm
        cv2.line(canvas, (W, strip_h), (W, out_h), (60, 60, 60), 1)
        canvas = draw_strip(canvas, W, strip_h, lv, (lv_c[2], lv_c[1], lv_c[0]), rec["t"])
        vw.write(canvas)

        if nmax > best[0]:
            best = (nmax, canvas.copy())
        if si % 50 == 0:
            print("[press]   %d t=%.1fs p_max=%.4g level=%s"
                  % (si, rec["t"], fld["p_max"], lv), flush=True)
        i += 1
        si += 1
    cap.release()
    vw.release()

    if best[1] is not None:
        cv2.imwrite(out + "/pressure_%s_frame.png" % source_type, best[1])
    ok = False
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", final], check=True)
        ok = os.path.exists(final)
    except Exception as e:
        print("[press] ffmpeg failed (%s); keeping raw" % e, flush=True)
    if not ok:
        final = raw
    print("=== PRESS_DONE %s: %d frames -> %s (%d bytes) + _frame.png ==="
          % (source_type, si, os.path.basename(final),
             os.path.getsize(final) if os.path.exists(final) else 0), flush=True)
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--out", default="/content")
    ap.add_argument("--gh", type=int, default=24)
    ap.add_argument("--gw", type=int, default=32)
    a = ap.parse_args()
    main(a.video, a.tracks, a.out, a.type, gh=a.gh, gw=a.gw)
