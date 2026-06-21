"""Local render check for the map-display PoC (no model / no network).

Paints a controlled oblique 'drone' frame, places people at known ground
positions, projects them to image pixels (stand-in for a detector), then runs
the REAL pipeline (flowsight.pipeline.demo_map.frame_to_map) to recover the 2D
map. Verifies the projection round-trips and renders the output format that the
real-footage Colab demo (demo_map_colab.py) produces.

NOTE: this is a render/format check on a synthetic frame — NOT a real-data
result. Real footage runs in demo_map_colab.py on Colab.
"""
from __future__ import annotations
import os
import numpy as np
import cv2

from flowsight.geometry.homography import fit_homography, apply_homography
from flowsight.pipeline.demo_map import frame_to_map

HERE = os.path.dirname(__file__)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
W, H = 1280, 720

# ground calibration: image trapezoid (oblique) <-> metric rectangle 20m x 30m
img_quad = np.float32([[230, 250], [1050, 250], [1240, 690], [40, 690]])
map_quad = np.float32([[0, 30], [20, 30], [20, 0], [0, 0]])
bounds = (0, 0, 20, 30)
H_map2img = fit_homography(map_quad, img_quad)

# paint a plausible oblique ground frame
rng = np.random.default_rng(7)
frame = np.full((H, W, 3), (70, 80, 95), np.uint8)
cv2.fillConvexPoly(frame, img_quad.astype(np.int32), (120, 125, 120))      # ground
for _ in range(450):                                                       # texture
    p = (int(rng.uniform(0, W)), int(rng.uniform(250, H)))
    cv2.circle(frame, p, 1, (135, 140, 135), -1)

# people: two crowds (one dense near a corner) -> visible density hot-spot
gt = np.vstack([rng.normal([6, 8], [3.0, 4.0], (70, 2)),
                rng.normal([15, 22], [1.2, 1.2], (90, 2))])               # dense cluster
gt = gt[(gt[:, 0] > 0.3) & (gt[:, 0] < 19.7) & (gt[:, 1] > 0.3) & (gt[:, 1] < 29.7)]
foot_px = apply_homography(H_map2img, gt)
for (u, v) in foot_px:                                                     # draw bodies
    cv2.line(frame, (int(u), int(v)), (int(u), int(v - 11)), (40, 40, 60), 3)
    cv2.circle(frame, (int(u), int(v - 13)), 3, (30, 30, 45), -1)

res = frame_to_map(frame[:, :, ::-1], foot_px, img_quad, map_quad, bounds,
                   cell=0.5, sigma_m=0.8,
                   out_path=os.path.join(HERE, "figures", "map_display_format.png"),
                   title="Map-display PoC (output format; synthetic frame stand-in)")

recovered = apply_homography(fit_homography(img_quad, map_quad), foot_px)
err = np.linalg.norm(recovered - gt, axis=1).max()
print(f"people placed: {len(gt)} | detected(stand-in): {res['n_detected']} | on map: {res['n_on_map']}")
print(f"projection round-trip max error: {err:.2e} m  (flat ground -> exact)")
print(f"BEV peak density: {res['max_density']:.1f}/m^2  -> figures/map_display_format.png")
