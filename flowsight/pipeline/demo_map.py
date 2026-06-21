"""Map-display PoC: a single frame -> people on a 2D ground map.

frame_to_map() is detector-agnostic: pass foot pixels from ANY source (real
RT-DETR on Colab, GT boxes, or a stand-in). It does the ground projection +
density and renders a 2-panel figure (frame+detections | BEV map).

This is the shared core for both the real-footage Colab demo
(experiments/demo_map_colab.py) and the in-sandbox render check.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from ..geometry.homography import fit_homography, apply_homography
from ..physics.density import DensityField


def detect_foot_points(image_rgb, detector):
    """Colab: detector = HeadPersonDetector (RT-DETRv2). Returns (N,2) pixels."""
    return detector.foot_points(image_rgb)


def frame_to_map(image_rgb, foot_pixels, img_quad, map_quad, bounds,
                 cell=0.5, sigma_m=0.8, out_path=None, title="FlowSight map-display PoC"):
    """
    image_rgb   : HxWx3 array (for display); may be None.
    foot_pixels : (N,2) ground-contact pixels of detected people.
    img_quad    : (4,2) pixel coords of a ground quad (calibration).
    map_quad    : (4,2) metric coords (metres) of that same quad.
    bounds      : (x0,y0,x1,y1) metres of the BEV canvas.
    """
    foot_pixels = np.asarray(foot_pixels, float).reshape(-1, 2)
    H = fit_homography(img_quad, map_quad)
    map_xy = apply_homography(H, foot_pixels) if len(foot_pixels) else np.zeros((0, 2))

    x0, y0, x1, y1 = bounds
    inb = (map_xy[:, 0] >= x0) & (map_xy[:, 0] <= x1) & (map_xy[:, 1] >= y0) & (map_xy[:, 1] <= y1)
    map_in = map_xy[inb]
    df = DensityField(bounds, cell=cell, sigma_m=sigma_m)
    dens = df.compute(map_in)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    # Left: source frame + detections + calibration quad
    if image_rgb is not None:
        ax[0].imshow(image_rgb)
    ax[0].add_patch(Polygon(img_quad, closed=True, fill=False, ec="cyan", lw=1.5, ls="--"))
    if len(foot_pixels):
        ax[0].scatter(foot_pixels[:, 0], foot_pixels[:, 1], s=14, c="red", marker="o",
                      edgecolors="white", linewidths=0.4, label=f"people: {len(foot_pixels)}")
        ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("source frame: detections (foot points) + ground quad")
    ax[0].set_xlabel("px"); ax[0].set_ylabel("px")

    # Right: BEV ground map + density
    im = ax[1].imshow(dens, origin="lower", extent=[x0, x1, y0, y1], aspect="equal",
                      cmap="inferno", alpha=0.9)
    if len(map_in):
        ax[1].scatter(map_in[:, 0], map_in[:, 1], s=18, c="cyan", edgecolors="k", linewidths=0.3)
    fig.colorbar(im, ax=ax[1], label="people/m^2", fraction=0.046)
    ax[1].set_title(f"2D ground map (BEV) — peak {dens.max():.1f}/m^2")
    ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("y (m)")

    fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {"n_detected": int(len(foot_pixels)), "n_on_map": int(len(map_in)),
            "max_density": float(dens.max()), "map_xy": map_in}
