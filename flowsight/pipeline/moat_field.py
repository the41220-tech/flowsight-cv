"""MOAT layer 2 — non-planar 3-D positioning + ABSOLUTE crowd-pressure alarm.

What competitors' flat heatmaps cannot do, in one place:

1. **Absolute alarm.** Calibrated metric tracks -> Helbing pressure in true
   ``1/s^2`` (``physics.crowd_pressure.frame_pressure_metric``), so the literature
   critical threshold ``P_CRIT = 0.02 /s^2`` becomes a real early-warning alarm —
   not a within-scene relative heatmap.

2. **Non-planar 3-D.** A metric monocular depth map (Depth-Anything-V2-Metric on
   GPU) + camera pose back-projects foot points to world ``(X,Y,Z)``. The ground
   surface is gridded into a DEM (``dense_depth_to_terrain``) so we DROP the
   flat-plane assumption — the original hard requirement that the system "must not
   assume a plane; ingest terrain and compute position AND pressure-difference".

3. **Gravitational potential.** On that terrain, upstream mass on a slope stores
   potential energy ``U = rho*m*g*h`` and exerts a downhill body force
   ``F = -m*g*grad(z)`` (``physics.potential``). The terrain push (force x upstream
   density) is the EARLY precursor: it builds before basin turbulence shows up in
   the Helbing channel (H2 / H6).

``MoatMonitor`` fuses these into a per-frame report with a two-channel alarm:
imminent (absolute Helbing >= 0.02 /s^2) and precursor (terrain potential push).

Heavy deps (torch / depth model) are only needed to PRODUCE the depth map; this
module is numpy-only and CPU-importable. On a flat scene pass ``terrain=None`` and
only the absolute Helbing channel is used.
"""
from __future__ import annotations

import numpy as np

from ..geometry.terrain import Terrain
from ..physics import potential
from ..physics.crowd_pressure import P_CRIT, alarm_level, frame_pressure_metric
from ..physics.pressure import upstream_density

G = 9.81
MASS = 70.0  # kg per person (population average)


# --------------------------------------------------------------------------- #
# 3-D from metric depth
# --------------------------------------------------------------------------- #
def foot_points_to_world(camera, foot_uv, metric_depth_map) -> np.ndarray:
    """Foot pixels (N,2) + metric depth map -> world (N,3) via the camera pose."""
    from ..geometry.metric_depth import MetricDepth  # staticmethod, no torch

    return MetricDepth.backproject(camera, foot_uv, metric_depth_map)


def _fill_nan(grid: np.ndarray, iters: int = 64) -> np.ndarray:
    """Fill NaN cells from 4-neighbour means (iterative), then global mean."""
    g = grid.astype(float).copy()
    for _ in range(iters):
        nan = np.isnan(g)
        if not nan.any():
            break
        acc = np.zeros_like(g)
        cnt = np.zeros_like(g)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.full_like(g, np.nan)
            ys = slice(max(0, dy), g.shape[0] + min(0, dy))
            xs = slice(max(0, dx), g.shape[1] + min(0, dx))
            yt = slice(max(0, -dy), g.shape[0] + min(0, -dy))
            xt = slice(max(0, -dx), g.shape[1] + min(0, -dx))
            sh[yt, xt] = g[ys, xs]
            ok = ~np.isnan(sh)
            acc[ok] += sh[ok]
            cnt[ok] += 1
        upd = nan & (cnt > 0)
        g[upd] = acc[upd] / cnt[upd]
    if np.isnan(g).any():
        g[np.isnan(g)] = np.nanmean(g) if np.isfinite(np.nanmean(g)) else 0.0
    return g


def build_dem_from_points(points3d, bounds, cell: float = 0.5) -> Terrain:
    """Grid world (X,Y,Z) points -> DEM elevation; return a Terrain(dem=...)."""
    pts = np.asarray(points3d, float).reshape(-1, 3)
    x0, y0, x1, y1 = bounds
    nx = max(1, int(np.ceil((x1 - x0) / cell)))
    ny = max(1, int(np.ceil((y1 - y0) / cell)))
    zsum = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx))
    if len(pts):
        cx = np.clip(((pts[:, 0] - x0) / cell).astype(int), 0, nx - 1)
        cy = np.clip(((pts[:, 1] - y0) / cell).astype(int), 0, ny - 1)
        for i in range(len(pts)):
            zsum[cy[i], cx[i]] += pts[i, 2]
            cnt[cy[i], cx[i]] += 1
    dem = np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.nan)
    return Terrain(dem=_fill_nan(dem), origin=(x0, y0), cell=cell)


def dense_depth_to_terrain(
    camera, metric_depth_map, bounds, cell: float = 0.5, stride: int = 8
) -> Terrain:
    """Back-project a (sub-sampled) metric depth map to world points -> DEM."""
    dm = np.asarray(metric_depth_map, float)
    ys, xs = np.mgrid[0 : dm.shape[0] : stride, 0 : dm.shape[1] : stride]
    uv = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    z = dm[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    world = camera.backproject_depth(uv, z)
    return build_dem_from_points(world, bounds, cell)


# --------------------------------------------------------------------------- #
# The integrated monitor
# --------------------------------------------------------------------------- #
class MoatMonitor:
    """Absolute Helbing alarm + non-planar terrain potential, on one metric grid.

    bounds_m : (x0,y0,x1,y1) ground extent in metres.
    terrain  : Terrain (DEM or analytic). None -> flat (Helbing channel only).
    """

    def __init__(
        self,
        bounds_m: tuple,
        terrain: Terrain | None = None,
        cell_m: float = 0.5,
        sigma_m: float = 1.0,
        p_crit: float = P_CRIT,
        mass: float = MASS,
        g: float = G,
    ) -> None:
        self.bounds = bounds_m
        self.terrain = terrain
        self.cell = float(cell_m)
        self.sigma = float(sigma_m)
        self.p_crit = float(p_crit)
        self.mass = float(mass)
        self.g = float(g)

        # Static grid geometry + terrain force field — precomputed ONCE (the
        # terrain does not change frame to frame), matching DisasterMonitor.
        x0, y0, x1, y1 = bounds_m
        self.gw = max(1, int(np.ceil((x1 - x0) / self.cell)))
        self.gh = max(1, int(np.ceil((y1 - y0) / self.cell)))
        xs = x0 + (np.arange(self.gw) + 0.5) * self.cell
        ys = y0 + (np.arange(self.gh) + 0.5) * self.cell
        self.Xg, self.Yg = np.meshgrid(xs, ys)
        self._h = self._Fx = self._Fy = self._Fmag = None
        if terrain is not None:
            self._h = np.asarray(terrain.elevation(self.Xg, self.Yg), float)
            self._Fx, self._Fy, self._Fmag = potential.potential_gradient(
                terrain, self.Xg, self.Yg, self.mass, self.g
            )

    def step_metric(self, xy_m: np.ndarray, vel_m: np.ndarray) -> dict:
        """Calibrated metric foot points + velocities -> fused risk report."""
        fld = frame_pressure_metric(xy_m, vel_m, self.bounds, self.cell, self.sigma)
        geo = fld["georef"]
        gh, gw = geo["gh"], geo["gw"]

        U = np.zeros((gh, gw))
        Fmag = np.zeros((gh, gw))
        terrain_push = np.zeros((gh, gw))
        if self.terrain is not None:
            U = self.mass * self.g * self._h * fld["density"]  # potential-energy density
            Fmag = self._Fmag
            rho_up = upstream_density(
                fld["density"], self._Fx, self._Fy, shift_m=2.0, cell=self.cell
            )
            terrain_push = Fmag * rho_up

        al = alarm_level(fld["p_max"], self.p_crit)
        P = fld["pressure"]
        gy, gx = np.unravel_index(int(np.argmax(P)), P.shape) if P.size else (0, 0)
        peak_xy = (
            float(geo["x0"] + (gx + 0.5) * geo["cell"]),
            float(geo["y0"] + (gy + 0.5) * geo["cell"]),
        )
        return {
            **fld,
            "U": U,
            "Fmag": Fmag,
            "terrain_push": terrain_push,
            "terrain_push_max": float(terrain_push.max()) if terrain_push.size else 0.0,
            "alarm": al,
            "peak_xy": peak_xy,
        }

    def step_pixels(self, foot_uv, vel_px, calibrator) -> dict:
        """Pixel foot points + pixel velocities (+ a Calibrator) -> report."""
        xy_m = calibrator.to_ground(foot_uv)
        vel_m = calibrator.velocity_to_metric(foot_uv, vel_px)
        return self.step_metric(xy_m, vel_m)
