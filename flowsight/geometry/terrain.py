"""Terrain model: elevation, slope/gradient, and ray-casting.

Supports an analytic elevation callable (used in synthetic experiments) or a
gridded DEM (bilinear). Ray-cast is numerical (sample + bisection) so it works
for ANY non-planar terrain — this is the core of dropping the flat-plane
assumption (requirement: "평면을 가정하는 게 아니라 지리 정보를 받아서").
"""
from __future__ import annotations
import numpy as np


class Terrain:
    def __init__(self, elevation_fn=None, dem=None, origin=(0, 0), cell=1.0):
        """elevation_fn: f(x, y) -> z (vectorized). dem: 2D array of z, with
        world (x,y) = origin + (col,row)*cell."""
        self._fn = elevation_fn
        self._dem = None if dem is None else np.asarray(dem, float)
        self._origin = np.asarray(origin, float)
        self._cell = float(cell)

    # --- elevation ---
    def elevation(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if self._fn is not None:
            return self._fn(x, y)
        col = (x - self._origin[0]) / self._cell
        row = (y - self._origin[1]) / self._cell
        c0 = np.clip(np.floor(col).astype(int), 0, self._dem.shape[1] - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, self._dem.shape[0] - 2)
        fc = np.clip(col - c0, 0, 1); fr = np.clip(row - r0, 0, 1)
        z = (self._dem[r0, c0] * (1 - fc) * (1 - fr) + self._dem[r0, c0 + 1] * fc * (1 - fr)
             + self._dem[r0 + 1, c0] * (1 - fc) * fr + self._dem[r0 + 1, c0 + 1] * fc * fr)
        return z

    # --- slope (dz/dx, dz/dy) via central differences ---
    def gradient(self, x, y, h=0.25):
        x = np.asarray(x, float); y = np.asarray(y, float)
        gx = (self.elevation(x + h, y) - self.elevation(x - h, y)) / (2 * h)
        gy = (self.elevation(x, y + h) - self.elevation(x, y - h)) / (2 * h)
        return gx, gy

    def slope_angle(self, x, y):
        gx, gy = self.gradient(x, y)
        return np.arctan(np.hypot(gx, gy))

    # --- ray-cast: first intersection of ray C + t*d (t>0) with the surface ---
    def raycast(self, C, d, t_near=0.1, t_far=500.0, n=2000, iters=40):
        C = np.asarray(C, float); d = np.asarray(d, float)
        ts = np.linspace(t_near, t_far, n)
        P = C[None, :] + ts[:, None] * d[None, :]
        g = P[:, 2] - self.elevation(P[:, 0], P[:, 1])
        sign = np.sign(g)
        cross = np.where(sign[:-1] != sign[1:])[0]
        if len(cross) == 0:
            return None
        i = cross[0]
        t0, t1 = ts[i], ts[i + 1]

        def gof(t):
            p = C + t * d
            return p[2] - float(self.elevation(p[0], p[1]))

        g0 = gof(t0)
        for _ in range(iters):
            tm = 0.5 * (t0 + t1)
            gm = gof(tm)
            if np.sign(gm) == np.sign(g0):
                t0, g0 = tm, gm
            else:
                t1 = tm
        return C + 0.5 * (t0 + t1) * d


def ramp_basin_elevation(y0=15.0, theta_deg=22.0, basin_y=35.0, basin_depth=2.0):
    """Itaewon-like non-planar profile: flat plaza (y<y0) -> downhill ramp ->
    shallow basin. Non-planar (piecewise + curved) so a single homography
    cannot fit it. Returns a vectorized f(x,y)."""
    t = np.tan(np.deg2rad(theta_deg))

    def f(x, y):
        y = np.asarray(y, float)
        z = np.zeros_like(y, dtype=float)
        ramp = (y >= y0)
        z = np.where(ramp, -t * (y - y0), z)
        # smooth concave basin near the bottom (adds genuine curvature)
        bowl = -basin_depth * np.exp(-((y - basin_y) ** 2) / (2 * 4.0 ** 2))
        return z + bowl
    return f
