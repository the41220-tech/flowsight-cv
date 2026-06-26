"""Synthetic moat-driven dashboard demo (self-contained, no Colab/dataset).

Demonstrates the product: LEFT a crowd scene, RIGHT the SAME people coloured by the
validated crush-pressure moat (Helbing P=rho*Var(v), absolute 0.02/s^2 -> 안전/주의/위험)
+ plain-Korean banner. A calm coherent flow stays green; a cluster that compresses and
turns erratic crosses into 주의 then 위험. Uses the exact frame_risk path the real
(WILDTRACK) renderer uses -- only the input (synthetic metric tracks) differs.

  python experiments/dashboard_moat_synth.py --out /path/out
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from flowsight.physics.moat_dashboard import frame_risk, ko_banner

BOUNDS = (0.0, 0.0, 20.0, 20.0)   # 20x20 m plaza
T = 130                            # frames
FPS = 10


def synth_frames():
    """Build T frames of metric (xy, vel). Returns list of (xy(N,2), vel(N,2))."""
    rng = np.random.default_rng(7)
    n_flow, n_clu = 24, 30
    # calm coherent flow: left->right band, low velocity variance
    flow0 = np.column_stack([rng.uniform(1, 5, n_flow), rng.uniform(2, 18, n_flow)])
    # cluster: starts loose around (14,10), converges + grows erratic over time
    clu0 = np.array([14.0, 10.0]) + rng.normal(0, 2.5, (n_clu, 2))
    frames = []
    for t in range(T):
        a = t / (T - 1)                                   # 0..1 progression
        # flow drifts right at ~1.2 m/s, tiny jitter
        flow = flow0 + np.array([1.2 * t / FPS % 6, 0]) + rng.normal(0, 0.05, (n_flow, 2))
        flow_v = np.tile([1.2, 0.0], (n_flow, 1)) + rng.normal(0, 0.1, (n_flow, 2))
        # cluster contracts toward centre (density up) + jitter grows (Var(v) up)
        centre = np.array([14.0, 10.0])
        contract = 1.0 - 0.6 * a                          # 1 -> 0.4 (tighter)
        jit = 0.3 + 3.0 * a                               # erratic grows with a
        clu = centre + (clu0 - centre) * contract + rng.normal(0, 0.15, (n_clu, 2))
        clu_v = rng.normal(0, jit, (n_clu, 2))            # incoherent, faster over time
        xy = np.vstack([flow, clu])
        vel = np.vstack([flow_v, clu_v])
        frames.append((xy, vel))
    return frames


def main(a):
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    S = 560                                               # panel px (square plaza)
    x0, y0, x1, y1 = BOUNDS
    def to_px(p):
        gx = int((p[0] - x0) / (x1 - x0) * (S - 20)) + 10
        gy = int((p[1] - y0) / (y1 - y0) * (S - 20)) + 10
        return gx, gy

    fontp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    def font(sz):
        try:
            return ImageFont.truetype(fontp, sz)
        except Exception:
            return ImageFont.load_default()

    os.makedirs(a.out, exist_ok=True)
    strip = 70
    raw = a.out + "/dashboard_moat_demo_raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (S * 2, S + strip))
    frames = synth_frames()
    trails = {i: [] for i in range(54)}
    best = (-1, None)
    for (xy, vel) in frames:
        r = frame_risk(xy, vel, BOUNDS)
        left = np.full((S, S, 3), 30, np.uint8)
        right = np.full((S, S, 3), 16, np.uint8)
        for gv in range(0, S, S // 10):
            cv2.line(right, (gv, 0), (gv, S), (38, 38, 38), 1)
            cv2.line(right, (0, gv), (S, gv), (38, 38, 38), 1)
        for i, (p, ap) in enumerate(zip(xy, r["per_person"])):
            gx, gy = to_px(p)
            cv2.circle(left, (gx, gy), 5, (180, 180, 180), -1, cv2.LINE_AA)   # scene: neutral
            c = ap["color"][::-1]                                             # RGB->BGR
            tr = trails[i]; tr.append((gx, gy)); tr[:] = tr[-8:]
            for j in range(1, len(tr)):
                cv2.line(right, tr[j - 1], tr[j], c, 1, cv2.LINE_AA)
            cv2.circle(right, (gx, gy), 6, c, -1, cv2.LINE_AA)
        canvas = np.vstack([np.hstack([left, right]), np.full((strip, S * 2, 3), 26, np.uint8)])
        im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im)
        d.text((12, 10), "현장", font=font(22), fill=(235, 235, 235))
        d.text((S + 12, 10), "위험 지도 (압사압력)", font=font(22), fill=(235, 235, 235))
        col = {"safe": (0, 200, 0), "caution": (245, 175, 0), "danger": (235, 45, 45)}[r["frame"]["severity"]]
        d.text((14, S + 18), ko_banner(r) + ("  · 압력 %.3f/s²" % r["frame"]["p"]), font=font(28), fill=col)
        canvas = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        vw.write(canvas)
        sc = r["frame"]["n_danger"] * 10 + r["frame"]["frac"]
        if sc > best[0]:
            best = (sc, canvas.copy())
    vw.release()
    if best[1] is not None:
        cv2.imwrite(a.out + "/dashboard_moat_demo_frame.png", best[1])
    h264 = a.out + "/dashboard_moat_demo.mp4"
    rc = os.system("ffmpeg -y -loglevel error -i %s -vcodec libx264 -pix_fmt yuv420p %s" % (raw, h264))
    out = h264 if rc == 0 and os.path.exists(h264) else raw
    print("DEMO_OK ->", out, "| frames", len(frames), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    main(ap.parse_args())
