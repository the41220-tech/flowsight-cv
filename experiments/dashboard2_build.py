"""FlowSight dashboard v2 — side-by-side: LEFT real video | RIGHT moving coloured dots.

Rebuilt to user spec (2026-06-22 feedback):
  - LEFT  = the real footage, untouched (showable to real stakeholders).
  - RIGHT = each tracked person as a DOT; DANGER is shown by COLOUR
            (green = normal, amber = caution, red = danger), with a short motion
            trail so movement is legible.
  - Plain-language labels only: 사람 수 / 위험도 / 빠르게 이동.  No jargon
    (no divergence / curl / flow-efficiency).

Danger model (v1, intentionally simple + explainable): per-person speed
|v| = hypot(vx, vy) from the tracker, coloured against the clip's own speed
distribution (p70 / p90). Fast movement = the surge / panic / fast-approach
signal. Scene risk grade = how many people are moving fast.

Inputs (both produced by track_run.py / the clip trim, already on Colab):
  --video        clean source clip  (e.g. /content/umn_clip.mp4)
  --tracks       tracks_<type>.json (per-frame tracks + summary)
Outputs to --out:
  dashboard2_<type>.mp4        H.264, plays anywhere (browser / QuickTime)
  dashboard2_<type>_frame.png  one busy frame, for quick inline verification

Run on Colab:
  !apt-get -qq install -y fonts-nanum >/dev/null
  !PYTHONPATH=. python -u experiments/dashboard2_build.py \
      --video /content/umn_clip.mp4 --tracks /content/tracks_cctv.json \
      --type cctv --out /content
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# colours are BGR for OpenCV drawing, RGB for PIL text
GREEN_BGR = (90, 210, 90)
AMBER_BGR = (40, 200, 245)
RED_BGR = (60, 60, 235)
GREEN_RGB = (90, 210, 90)
AMBER_RGB = (245, 200, 40)
RED_RGB = (235, 60, 60)
WHITE_RGB = (235, 235, 235)
GREY_RGB = (150, 150, 150)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def speed_thresholds(per_frame: list) -> tuple[float, float]:
    sp = [math.hypot(t["vx"], t["vy"]) for f in per_frame for t in f["tracks"]]
    a = np.array(sp) if sp else np.array([0.0])
    return float(np.percentile(a, 70)), float(np.percentile(a, 90))


def dot_colour(s: float, p70: float, p90: float) -> tuple:
    if s >= p90:
        return RED_BGR
    if s >= p70:
        return AMBER_BGR
    return GREEN_BGR


def scene_grade(tracks: list, p90: float) -> tuple:
    """Return (label, rgb, n_fast). Plain Korean grade from how many move fast."""
    n = len(tracks)
    nfast = sum(1 for t in tracks if math.hypot(t["vx"], t["vy"]) >= p90)
    frac = (nfast / n) if n else 0.0
    if frac >= 0.25 or nfast >= 4:
        return "높음", RED_RGB, nfast
    if frac >= 0.10 or nfast >= 1:
        return "보통", AMBER_RGB, nfast
    return "낮음", GREEN_RGB, nfast


def draw_strip(canvas_bgr, W, strip_h, n, grade, gc, nfast, t):
    """Draw the top label/KPI strip + legend with PIL (Korean-capable).

    Layout: LEFT half (video) = title + 사람/위험도 KPI; RIGHT half (dots) =
    title + colour legend + 빠른 이동 count. Kept within each half so labels
    never overlap. Clock burned bottom-left of the video panel.
    """
    Himg = canvas_bgr.shape[0] - strip_h
    im = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    f_title = load_font(16)
    f_kpi = load_font(20)
    f_small = load_font(13)
    # LEFT half (video): title + KPI
    d.text((12, 7), "● 실시간 영상", fill=WHITE_RGB, font=f_title)
    d.text((12, 35), f"사람 {n}명", fill=WHITE_RGB, font=f_kpi)
    d.text((120, 41), "위험도", fill=GREY_RGB, font=f_small)
    d.text((172, 35), grade, fill=gc, font=f_kpi)
    # RIGHT half (dots): title + legend + fast count
    d.text((W + 12, 7), "● 이동 분석 · 점 = 사람", fill=WHITE_RGB, font=f_title)
    lx = W + 12
    for label, rgb in (("정상", GREEN_RGB), ("주의", AMBER_RGB), ("위험", RED_RGB)):
        d.ellipse((lx, 40, lx + 11, 51), fill=rgb)
        d.text((lx + 15, 37), label, fill=GREY_RGB, font=f_small)
        lx += 62
    d.text((lx + 6, 37), f"빠른 이동 {nfast}명", fill=(235, 170, 90), font=f_small)
    # clock on the video panel (bottom-left)
    d.text((10, strip_h + Himg - 22), f"t = {t:0.1f}s", fill=(220, 220, 220), font=f_small)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def main(video, tracks_json, out, source_type, trail=8):
    data = json.load(open(tracks_json))
    per_frame = data["per_frame"]
    step = int(data["step"])
    fps = float(data["fps"])
    W = int(data["W"])
    H = int(data["H"])
    p70, p90 = speed_thresholds(per_frame)
    print("[dash2] %s: %d rec frames %dx%d step=%d fps=%.1f  speed p70=%.1f p90=%.1f"
          % (source_type, len(per_frame), W, H, step, fps, p70, p90), flush=True)

    strip_h = 64
    out_w, out_h = W * 2, H + strip_h
    raw = out + "/_dash2_%s_raw.mp4" % source_type
    final = out + "/dashboard2_%s.mp4" % source_type
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"),
                         max(1.0, fps / step), (out_w, out_h))

    cap = cv2.VideoCapture(video)
    trails: dict[int, list] = {}
    i = si = fi = 0
    best = (-1, None)  # (n_tracks, frame) -> save busiest frame as png
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        if fi >= len(per_frame):
            break
        rec = per_frame[fi]
        fi += 1
        if fr.shape[1] != W or fr.shape[0] != H:
            fr = cv2.resize(fr, (W, H))

        # RIGHT panel: dark bg + faint grid
        right = np.full((H, W, 3), 28, np.uint8)
        for gx in range(0, W, 40):
            cv2.line(right, (gx, 0), (gx, H), (40, 40, 40), 1)
        for gy in range(0, H, 40):
            cv2.line(right, (0, gy), (W, gy), (40, 40, 40), 1)

        for tk in rec["tracks"]:
            x, y = int(tk["x"]), int(tk["y"])
            s = math.hypot(tk["vx"], tk["vy"])
            c = dot_colour(s, p70, p90)
            tr = trails.setdefault(tk["id"], [])
            tr.append((x, y))
            if len(tr) > trail:
                tr.pop(0)
            for j in range(1, len(tr)):
                cv2.line(right, tr[j - 1], tr[j], c, 1, cv2.LINE_AA)
            cv2.circle(right, (x, y), 5, c, -1, cv2.LINE_AA)

        grade, gc, nfast = scene_grade(rec["tracks"], p90)
        canvas = np.full((out_h, out_w, 3), 18, np.uint8)
        canvas[strip_h:strip_h + H, 0:W] = fr
        canvas[strip_h:strip_h + H, W:out_w] = right
        cv2.line(canvas, (W, strip_h), (W, out_h), (60, 60, 60), 1)
        canvas = draw_strip(canvas, W, strip_h, len(rec["tracks"]), grade, gc, nfast, rec["t"])
        vw.write(canvas)

        score = nfast * 10 + len(rec["tracks"])  # busy + some danger = best demo still
        if score > best[0]:
            best = (score, canvas.copy())
        if si % 50 == 0:
            print("[dash2]   %d t=%.1fs n=%d grade=%s fast=%d"
                  % (si, rec["t"], len(rec["tracks"]), grade, nfast), flush=True)
        i += 1
        si += 1
    cap.release()
    vw.release()

    if best[1] is not None:
        cv2.imwrite(out + "/dashboard2_%s_frame.png" % source_type, best[1])

    # re-encode to H.264 for universal playback (browser-friendly)
    ok = False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", final],
            check=True)
        ok = os.path.exists(final)
    except Exception as e:
        print("[dash2] ffmpeg re-encode failed (%s); keeping mp4v raw" % e, flush=True)
    if not ok:
        final = raw
    size = os.path.getsize(final) if os.path.exists(final) else 0
    print("=== DASH2_DONE %s: %d frames -> %s (%d bytes) + _frame.png ==="
          % (source_type, si, os.path.basename(final), size), flush=True)
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--out", default="/content")
    ap.add_argument("--trail", type=int, default=8)
    a = ap.parse_args()
    main(a.video, a.tracks, a.out, a.type, trail=a.trail)
