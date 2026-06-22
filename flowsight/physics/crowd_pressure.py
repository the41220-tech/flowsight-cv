"""Crowd pressure (Helbing) from tracks — the physics core of FlowSight's moat.

Helbing, Johansson & Al-Abideen (2007), "The Dynamics of Crowd Disasters: An
Empirical Study" (Phys. Rev. E 75, 046109; arXiv:physics/0701203) define the
*crowd pressure* at a place as

        P(x) = rho(x) * Var_x(v)

i.e. local crowd DENSITY times the local VELOCITY VARIANCE. Empirically the
Jamarat Bridge crush (2006-01-12) began ~10 min after P exceeded ~0.02 /s^2,
and the same metric flagged the high-risk zones of the Love Parade and Mina
disasters. It is the peer-reviewed crush-risk indicator — far better than speed
or density alone, because crushes happen where a DENSE crowd ALSO moves
ERRATICALLY (high velocity variance): the signature of turbulent, multi-
directional shock waves. Helbing notes such conditions can be detected on-line
by automated video analysis and used for ADVANCE WARNING — which is exactly
FlowSight's thesis.

This module computes the pressure FIELD from per-frame tracks {id,x,y,vx,vy}.
Coordinates are pixels, so P is in *relative* (pixel) units — good for showing
WHERE risk concentrates within a scene. Absolute /s^2 values (and the 0.02
threshold) need metric calibration (pixels->metres); that, plus non-planar 3-D
terrain and gravitational potential U=rho*g*h, is the next moat layer.

Method (vectorised Gaussian deposition): deposit each person's 1, v and |v|^2
onto a grid, Gaussian-smooth each, then
        mean_v  = S_v  / S0 ;  mean_v2 = S_v2 / S0
        Var(v)  = mean_v2 - |mean_v|^2
        rho     = S0 / cell_area
        P       = rho * Var(v)
"""
from __future__ import annotations

import numpy as np

try:  # prefer scipy if present
    from scipy.ndimage import gaussian_filter

    def _smooth(a, s):
        return gaussian_filter(a.astype(np.float32), s, mode="constant")
except Exception:  # scipy-free fallback via OpenCV
    import cv2

    def _smooth(a, s):
        k = int(max(1, round(s * 3)) * 2 + 1)
        return cv2.GaussianBlur(a.astype(np.float32), (k, k), s)


def frame_pressure(tracks, W, H, gh=24, gw=32, sigma_cells=2.0, eps=1e-6):
    """Crowd-pressure field for one frame.

    tracks: list of dicts with x, y (px) and vx, vy (px/s).
    Returns grids (gh, gw): pressure, density, var_v, speed; plus per-person
    pressure (sampled at each person's cell) and scalar p_max / p_mean.
    """
    cw = W / float(gw)
    ch = H / float(gh)
    S0 = np.zeros((gh, gw), np.float32)  # count
    Sx = np.zeros((gh, gw), np.float32)  # sum vx
    Sy = np.zeros((gh, gw), np.float32)  # sum vy
    S2 = np.zeros((gh, gw), np.float32)  # sum |v|^2
    cells = []
    for t in tracks:
        gx = int(min(gw - 1, max(0, int(t["x"] // cw))))
        gy = int(min(gh - 1, max(0, int(t["y"] // ch))))
        vx = float(t["vx"])
        vy = float(t["vy"])
        S0[gy, gx] += 1.0
        Sx[gy, gx] += vx
        Sy[gy, gx] += vy
        S2[gy, gx] += vx * vx + vy * vy
        cells.append((gy, gx))

    S0s = _smooth(S0, sigma_cells)
    Sxs = _smooth(Sx, sigma_cells)
    Sys = _smooth(Sy, sigma_cells)
    S2s = _smooth(S2, sigma_cells)
    inv = 1.0 / (S0s + eps)
    mvx = Sxs * inv
    mvy = Sys * inv
    mv2 = S2s * inv
    var_v = np.clip(mv2 - (mvx * mvx + mvy * mvy), 0.0, None)  # >= 0
    rho = S0s / (cw * ch)
    P = rho * var_v
    speed = np.sqrt(mvx * mvx + mvy * mvy)
    per_person = [float(P[gy, gx]) for (gy, gx) in cells]
    return {
        "pressure": P,
        "density": rho,
        "var_v": var_v,
        "speed": speed,
        "per_person": per_person,
        "p_max": float(P.max()) if P.size else 0.0,
        "p_mean": float(P.mean()) if P.size else 0.0,
    }


def clip_pressure_scale(per_frame_fields, pct=95.0, floor=1e-9):
    """Robust display scale = percentile of all per-frame p_max over the clip."""
    mx = [f["p_max"] for f in per_frame_fields if f["p_max"] > 0]
    if not mx:
        return floor
    return max(floor, float(np.percentile(mx, pct)))
