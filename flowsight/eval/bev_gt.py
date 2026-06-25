"""BEV projection grid + ground-truth occupancy heatmap for the learned multi-view
detector (H2, MVDet/3DROM-style). Pure numpy — the geometry foundation the trained net
consumes; sandbox-verifiable without torch.

* ``bev_projection_grid``  precomputes, for every BEV ground cell (world X,Y,Z=0), its
  normalised pixel coordinate in a camera -> the (Hg,Wg,2) sampling grid that warps a
  per-view feature map onto the shared BEV plane (torch.grid_sample at train time). Built
  from the real calibration via ``WildtrackCamera.project_world``.
* ``bev_gt_heatmap``  rasterises GT world positions (positionID->world) to a BEV Gaussian
  occupancy target.
"""
from __future__ import annotations

import numpy as np


def bev_grid_centres(bounds, cell: float):
    """(Hg,Wg,2) world XY (m) of BEV cell centres, plus (Hg,Wg)."""
    x0, y0, x1, y1 = bounds
    gw = max(1, int(np.ceil((x1 - x0) / cell)))
    gh = max(1, int(np.ceil((y1 - y0) / cell)))
    xs = x0 + (np.arange(gw) + 0.5) * cell
    ys = y0 + (np.arange(gh) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)            # (gh,gw)
    return np.stack([gx, gy], axis=-1), gh, gw


def bev_projection_grid(cam, bounds, cell: float, img_wh):
    """For each BEV ground cell -> its pixel in `cam`, normalised to [-1,1] for grid_sample.

    Returns (Hg,Wg,2) float32 (gx,gy in [-1,1]) and a (Hg,Wg) bool `valid` mask (cell in
    front of the camera AND inside the image). Out-of-view cells get sampling coords that
    grid_sample reads as zero-padding; the mask lets the model ignore them."""
    W, H = img_wh
    centres, gh, gw = bev_grid_centres(bounds, cell)
    flat = centres.reshape(-1, 2)
    uv = cam.project_world(flat)                       # (Hg*Wg,2) pixels
    # in-front check: camera-frame Z>0
    Pw = np.column_stack([flat / cam.s, np.zeros(len(flat))])
    Pcz = (cam.R @ Pw.T).T[:, 2] + cam.t[2]
    gx = (2.0 * uv[:, 0] / W - 1.0)
    gy = (2.0 * uv[:, 1] / H - 1.0)
    valid = (Pcz > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    grid = np.stack([gx, gy], axis=-1).reshape(gh, gw, 2).astype(np.float32)
    return grid, valid.reshape(gh, gw)


def bev_gt_heatmap(world_pts, bounds, cell: float, sigma_m: float = 0.5):
    """GT world positions (N,2) m -> (Hg,Wg) Gaussian occupancy target in [0,1]
    (peak 1.0 per person)."""
    x0, y0, x1, y1 = bounds
    gw = max(1, int(np.ceil((x1 - x0) / cell)))
    gh = max(1, int(np.ceil((y1 - y0) / cell)))
    hm = np.zeros((gh, gw), np.float32)
    pts = np.atleast_2d(np.asarray(world_pts, float)) if len(world_pts) else np.zeros((0, 2))
    if not len(pts):
        return hm
    s = sigma_m / cell
    r = int(np.ceil(3 * s))
    for x, y in pts:
        cx = (x - x0) / cell
        cy = (y - y0) / cell
        ix, iy = int(round(cx)), int(round(cy))
        for jy in range(max(0, iy - r), min(gh, iy + r + 1)):
            for jx in range(max(0, ix - r), min(gw, ix + r + 1)):
                d2 = (jx - cx) ** 2 + (jy - cy) ** 2
                hm[jy, jx] = max(hm[jy, jx], np.exp(-d2 / (2 * s * s)))
    return hm
