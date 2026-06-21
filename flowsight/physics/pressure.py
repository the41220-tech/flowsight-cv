"""Crowd pressure and the terrain-coupled risk field.

Helbing crowd pressure: P = rho * Var(v)  (density x velocity variance), the
established precursor to crowd turbulence / crush.

FlowSight extension (non-planar): add a terrain term — upstream mass on a slope
exerts a downhill push proportional to the potential-gradient force times the
upstream density. Risk = w1*rho + w2*P + w3*(Fmag * rho_up).
"""
from __future__ import annotations
import numpy as np


def helbing_pressure(density, vel_variance):
    return np.asarray(density, float) * np.asarray(vel_variance, float)


def velocity_variance_grid(points_xy, vel_xy, density_field):
    """Per-cell variance of velocity vectors: mean(||v - <v>||^2)."""
    nx, ny, cell = density_field.nx, density_field.ny, density_field.cell
    sx = np.zeros((ny, nx)); sy = np.zeros((ny, nx))
    sxx = np.zeros((ny, nx)); n = np.zeros((ny, nx))
    cx, cy, ok = density_field._idx(points_xy)
    vel_xy = np.atleast_2d(np.asarray(vel_xy, float))
    for i in np.where(ok)[0]:
        n[cy[i], cx[i]] += 1
        sx[cy[i], cx[i]] += vel_xy[i, 0]; sy[cy[i], cx[i]] += vel_xy[i, 1]
        sxx[cy[i], cx[i]] += vel_xy[i, 0] ** 2 + vel_xy[i, 1] ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.where(n > 0, sx / n, 0); my = np.where(n > 0, sy / n, 0)
        var = np.where(n > 0, sxx / n - (mx ** 2 + my ** 2), 0)
    return np.clip(var, 0, None)


def upstream_density(density, Fx, Fy, shift_m=2.0, cell=0.5):
    """Density sampled one step UPHILL (opposite the downhill force), i.e. the
    mass that is about to press down into this cell."""
    sx = -Fx / (np.hypot(Fx, Fy) + 1e-9) * (shift_m / cell)
    sy = -Fy / (np.hypot(Fx, Fy) + 1e-9) * (shift_m / cell)
    ny, nx = density.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    rs = np.clip((ys + sy).astype(int), 0, ny - 1)
    cs = np.clip((xs + sx).astype(int), 0, nx - 1)
    return density[rs, cs]


def risk_field(density, vel_variance, Fmag=None, rho_up=None,
               w=(0.15, 1.0, 0.0)):
    """w = (w_density, w_helbing, w_terrain). Set w_terrain>0 to enable the
    non-planar potential-energy coupling."""
    P = helbing_pressure(density, vel_variance)
    R = w[0] * density + w[1] * P
    if Fmag is not None and rho_up is not None and w[2] > 0:
        R = R + w[2] * (Fmag * rho_up)
    return R, P
