"""Directional crowd-flow features from a dense optical-flow field (H,W,2).

Beyond magnitude/variance (which only feed the crush/Helbing channel), this layer
extracts the DIRECTIONAL structure of crowd motion — the analysis substrate the
FlowSight dashboard's Flow Analytics panel needs, and the generalization point for
non-crush anomalies (counterflow, dispersal, swirl, surge):

  - divergence  div(v) = dVx/dx + dVy/dy   (+ = 확산/이탈, - = 쏠림/수렴)
  - curl        curl(v) = dVy/dx - dVx/dy  (와류 / rotation)
  - counterflow ratio = fraction of moving pixels opposing the dominant flow (역류)
  - flow efficiency = |mean velocity| / mean speed  (1 = aligned, 0 = chaotic)

Input-source-agnostic: works on CCTV or drone footage alike.
"""
from __future__ import annotations
import numpy as np


def _cell_means(flow, gh, gw, moving_thr=0.5):
    """Mean (vx, vy, speed) over a gh x gw grid, using moving pixels only."""
    H, W = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    spd = np.sqrt(fx * fx + fy * fy)
    VX = np.zeros((gh, gw)); VY = np.zeros((gh, gw)); SP = np.zeros((gh, gw))
    ys = np.linspace(0, H, gh + 1).astype(int)
    xs = np.linspace(0, W, gw + 1).astype(int)
    for r in range(gh):
        for c in range(gw):
            sl = (slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1]))
            cs = spd[sl]; m = cs > moving_thr
            if m.any():
                VX[r, c] = fx[sl][m].mean()
                VY[r, c] = fy[sl][m].mean()
                SP[r, c] = cs[m].mean()
    return VX, VY, SP


def frame_flow_features(flow, gh=12, gw=16, moving_thr=0.5):
    """Global directional scalars + small grids for one frame's dense flow."""
    fx, fy = flow[..., 0], flow[..., 1]
    spd = np.sqrt(fx * fx + fy * fy)
    mov = spd > moving_thr
    z = [[0.0] * gw for _ in range(gh)]
    if int(mov.sum()) < 10:
        return {"speed_mean": 0.0, "flow_efficiency": 0.0, "counterflow": 0.0,
                "divergence_abs": 0.0, "curl_abs": 0.0,
                "grids": {"vx": z, "vy": z, "speed": z, "div": z, "curl": z}}
    mvx, mvy = float(fx[mov].mean()), float(fy[mov].mean())     # dominant flow vector
    speed_mean = float(spd[mov].mean())
    flow_eff = float(np.hypot(mvx, mvy) / (speed_mean + 1e-9))  # alignment 0..1
    dotp = fx[mov] * mvx + fy[mov] * mvy                        # vs dominant direction
    counter = float((dotp < 0).mean())                         # opposing fraction
    VX, VY, SP = _cell_means(flow, gh, gw, moving_thr)
    div = np.gradient(VX, axis=1) + np.gradient(VY, axis=0)
    curl = np.gradient(VY, axis=1) - np.gradient(VX, axis=0)
    return {"speed_mean": speed_mean, "flow_efficiency": flow_eff, "counterflow": counter,
            "divergence_abs": float(np.abs(div).mean()), "curl_abs": float(np.abs(curl).mean()),
            "grids": {"vx": VX.round(2).tolist(), "vy": VY.round(2).tolist(),
                      "speed": SP.round(2).tolist(), "div": div.round(3).tolist(),
                      "curl": curl.round(3).tolist()}}
