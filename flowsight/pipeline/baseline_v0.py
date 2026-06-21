"""FlowSight v0 baseline: single drone -> 2D ground map -> density + risk.

Works with real detections (Colab) OR precomputed map points (CPU test). The
ground projection can be flat homography (v0) or terrain-aware (v1: ray-cast /
metric depth) by passing a different `img2map` callable.
"""
from __future__ import annotations
import numpy as np

from ..physics.density import DensityField
from ..physics.pressure import (helbing_pressure, velocity_variance_grid,
                                 upstream_density, risk_field)
from ..physics import potential


class FlowSightV0:
    def __init__(self, bounds, cell=0.5, sigma_m=1.0, terrain=None,
                 risk_weights=(0.15, 1.0, 0.0)):
        self.df = DensityField(bounds, cell=cell, sigma_m=sigma_m)
        self.terrain = terrain
        self.w = risk_weights
        self._Fx = self._Fy = self._Fmag = None
        if terrain is not None:
            r = np.arange(self.df.ny); c = np.arange(self.df.nx)
            C, R = np.meshgrid(c, r)
            Xg = self.df.x0 + (C + 0.5) * cell; Yg = self.df.y0 + (R + 0.5) * cell
            self._Fx, self._Fy, self._Fmag = potential.potential_gradient(terrain, Xg, Yg)

    def process_points(self, map_xy, vel_xy=None):
        d = self.df.compute(map_xy)
        out = {"density": d, "n": len(map_xy), "max_density": float(d.max())}
        if vel_xy is not None:
            var = velocity_variance_grid(map_xy, vel_xy, self.df)
            terrain_on = self.terrain is not None and self.w[2] > 0
            rho_up = upstream_density(d, self._Fx, self._Fy, cell=self.df.cell) if terrain_on else None
            R, P = risk_field(d, var, self._Fmag if terrain_on else None, rho_up, w=self.w)
            out.update({"helbing": P, "risk": R, "max_helbing": float(P.max()),
                        "max_risk": float(R.max())})
        return out

    def process_frame(self, image, detector, img2map, velocities=None):
        """detector.foot_points(image)->pixels; img2map(pixels)->metres."""
        foot = detector.foot_points(image)
        map_xy = img2map(foot) if len(foot) else np.zeros((0, 2))
        return self.process_points(map_xy, velocities), map_xy

    @staticmethod
    def render_bev(field, bounds, path, title="BEV density (people/m^2)"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x0, y0, x1, y1 = bounds
        plt.figure(figsize=(5, 7))
        plt.imshow(field, origin="lower", extent=[x0, x1, y0, y1], aspect="equal", cmap="inferno")
        plt.colorbar(label="people/m^2"); plt.title(title)
        plt.xlabel("x (m)"); plt.ylabel("y (m)")
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
