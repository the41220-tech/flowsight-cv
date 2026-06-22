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
by automated video analysis and used for ADVANCE WARNING — exactly FlowSight's
thesis.

Two ways to read the field:

* ``frame_pressure``        — PIXEL coordinates -> P in *relative* pixel units.
  Good for showing WHERE risk concentrates within a scene (heatmap), but the
  absolute 0.02 threshold does NOT apply (units are arbitrary).

* ``frame_pressure_metric`` — METRIC coordinates (metres, m/s) -> P in true
  ``1/s^2``. Feed it tracks that were calibrated pixels->metres (see
  ``flowsight.geometry.calibration``). Now ``alarm_level`` can apply the Helbing
  critical threshold ``P_CRIT = 0.02 /s^2`` as a real early-warning alarm. This,
  with non-planar 3-D terrain and gravitational potential U=rho*g*h, is
  MOAT layer 2.

Method (vectorised Gaussian deposition): deposit each person's 1, v and |v|^2
onto a grid, Gaussian-smooth each, then
        mean_v  = S_v  / S0 ;  mean_v2 = S_v2 / S0
        Var(v)  = mean_v2 - |mean_v|^2
        rho     = S0 / cell_area
        P       = rho * Var(v)
"""
from __future__ import annotations

import numpy as np

# Helbing critical crowd-pressure (1/s^2). Above this, crowd turbulence / crush
# conditions emerge (Helbing et al. 2007; crossed ~10 min before the Jamarat
# crush). Only meaningful on a METRIC field (frame_pressure_metric).
P_CRIT: float = 0.02
CAUTION_FRAC: float = 0.5  # caution band starts at 0.5 * P_CRIT (= 0.01 /s^2)

try:  # prefer scipy if present
    from scipy.ndimage import gaussian_filter

    def _smooth(a: np.ndarray, s: float) -> np.ndarray:
        return gaussian_filter(a.astype(np.float32), s, mode="constant")
except Exception:  # scipy-free fallback via OpenCV
    import cv2

    def _smooth(a: np.ndarray, s: float) -> np.ndarray:
        k = int(max(1, round(s * 3)) * 2 + 1)
        return cv2.GaussianBlur(a.astype(np.float32), (k, k), s)


def _pressure_from_cells(
    gy: np.ndarray,
    gx: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    gh: int,
    gw: int,
    cell_area: float,
    sigma_cells: float,
    eps: float = 1e-6,
) -> dict:
    """Shared core: per-person cell indices + velocities -> pressure grids.

    cell_area is in the SAME area unit as you want density in (px^2 -> relative,
    m^2 -> persons/m^2). Velocities carry their own unit (px/s -> relative,
    m/s -> the variance is (m/s)^2 so P is 1/s^2).
    """
    S0 = np.zeros((gh, gw), np.float32)  # count
    Sx = np.zeros((gh, gw), np.float32)  # sum vx
    Sy = np.zeros((gh, gw), np.float32)  # sum vy
    S2 = np.zeros((gh, gw), np.float32)  # sum |v|^2
    for k in range(len(gy)):
        S0[gy[k], gx[k]] += 1.0
        Sx[gy[k], gx[k]] += vx[k]
        Sy[gy[k], gx[k]] += vy[k]
        S2[gy[k], gx[k]] += vx[k] * vx[k] + vy[k] * vy[k]

    S0s = _smooth(S0, sigma_cells)
    Sxs = _smooth(Sx, sigma_cells)
    Sys = _smooth(Sy, sigma_cells)
    S2s = _smooth(S2, sigma_cells)
    inv = 1.0 / (S0s + eps)
    mvx = Sxs * inv
    mvy = Sys * inv
    mv2 = S2s * inv
    var_v = np.clip(mv2 - (mvx * mvx + mvy * mvy), 0.0, None)  # (vel unit)^2
    rho = S0s / float(cell_area)
    P = rho * var_v
    speed = np.sqrt(mvx * mvx + mvy * mvy)
    per_person = [float(P[gy[k], gx[k]]) for k in range(len(gy))]
    return {
        "pressure": P,
        "density": rho,
        "var_v": var_v,
        "speed": speed,
        "per_person": per_person,
        "p_max": float(P.max()) if P.size else 0.0,
        "p_mean": float(P.mean()) if P.size else 0.0,
    }


def frame_pressure(tracks, W, H, gh=24, gw=32, sigma_cells=2.0, eps=1e-6) -> dict:
    """Crowd-pressure field for one frame in PIXEL units (relative scale).

    tracks: list of dicts with x, y (px) and vx, vy (px/s).
    Returns grids (gh, gw): pressure, density, var_v, speed; per-person pressure
    and scalar p_max / p_mean. P is in arbitrary pixel units — for the absolute
    0.02 /s^2 alarm use ``frame_pressure_metric``.
    """
    cw = W / float(gw)
    ch = H / float(gh)
    if tracks:
        gx = np.clip(np.array([int(t["x"] // cw) for t in tracks]), 0, gw - 1)
        gy = np.clip(np.array([int(t["y"] // ch) for t in tracks]), 0, gh - 1)
        vx = np.array([float(t["vx"]) for t in tracks])
        vy = np.array([float(t["vy"]) for t in tracks])
    else:
        gx = gy = np.zeros(0, int)
        vx = vy = np.zeros(0)
    return _pressure_from_cells(gy, gx, vx, vy, gh, gw, cw * ch, sigma_cells, eps)


def frame_pressure_metric(
    xy_m: np.ndarray,
    vel_m: np.ndarray,
    bounds_m: tuple,
    cell_m: float = 0.5,
    sigma_m: float = 1.0,
    eps: float = 1e-6,
) -> dict:
    """Crowd-pressure field for one frame in METRIC units -> P in true 1/s^2.

    xy_m  (N,2): ground positions in metres (calibrated foot points).
    vel_m (N,2): velocities in m/s.
    bounds_m   : (x0,y0,x1,y1) ground extent in metres.
    Returns the same grids as ``frame_pressure`` (density in persons/m^2,
    var_v in (m/s)^2, pressure in 1/s^2) PLUS georef {x0,y0,cell,gh,gw} so cells
    map back to metres. Apply ``alarm_level`` to p_max for the absolute alarm.
    """
    x0, y0, x1, y1 = bounds_m
    gw = max(1, int(np.ceil((x1 - x0) / cell_m)))
    gh = max(1, int(np.ceil((y1 - y0) / cell_m)))
    xy_m = np.atleast_2d(np.asarray(xy_m, float)) if len(xy_m) else np.zeros((0, 2))
    vel_m = np.atleast_2d(np.asarray(vel_m, float)) if len(vel_m) else np.zeros((0, 2))
    if len(xy_m):
        gx = np.clip(((xy_m[:, 0] - x0) / cell_m).astype(int), 0, gw - 1)
        gy = np.clip(((xy_m[:, 1] - y0) / cell_m).astype(int), 0, gh - 1)
        vx, vy = vel_m[:, 0], vel_m[:, 1]
    else:
        gx = gy = np.zeros(0, int)
        vx = vy = np.zeros(0)
    out = _pressure_from_cells(
        gy, gx, vx, vy, gh, gw, cell_m * cell_m, sigma_m / cell_m, eps
    )
    out["georef"] = {"x0": float(x0), "y0": float(y0), "cell": float(cell_m),
                     "gh": gh, "gw": gw}
    return out


def alarm_level(p_value: float, crit: float = P_CRIT) -> dict:
    """Absolute 3-tier alarm from a metric pressure value (1/s^2).

    안전 (safe)    : P < 0.5*crit   (< 0.01 /s^2)
    주의 (caution) : 0.5*crit <= P < crit
    위험 (danger)  : P >= crit      (>= 0.02 /s^2, Helbing critical)
    Returns label/severity/fraction(=P/crit). Color is left to the renderer.
    """
    p = float(p_value)
    frac = p / crit if crit > 0 else 0.0
    if p >= crit:
        return {"label": "위험", "severity": "danger", "frac": frac, "p": p}
    if p >= CAUTION_FRAC * crit:
        return {"label": "주의", "severity": "caution", "frac": frac, "p": p}
    return {"label": "안전", "severity": "safe", "frac": frac, "p": p}


def clip_pressure_scale(per_frame_fields, pct=95.0, floor=1e-9) -> float:
    """Robust display scale = percentile of all per-frame p_max over the clip."""
    mx = [f["p_max"] for f in per_frame_fields if f["p_max"] > 0]
    if not mx:
        return floor
    return max(floor, float(np.percentile(mx, pct)))
