"""Multimodal slope -> disaster (crowd-crush) prediction — integration layer.

Fuses three modalities into one risk field on the ground map:
  1) DETECTION  -> where people are (foot points from the fine-tuned detector)
  2) TERRAIN/3D -> non-planar positioning + slope (H1: metric-depth / ray-cast)
  3) FLOW+POTENTIAL -> Helbing pressure (rho * Var(v)) + terrain potential-energy
     push (Fmag * upstream-density)  (H2)

Risk = w0*rho + w1*Helbing + w2*(Fmag * rho_up). Crush is predicted where Risk
crosses an alarm threshold, with 3 severity tiers (info/warn/critical). This is
the engine behind "어떤 영상이든, 사고가 나기 전에".
"""
from __future__ import annotations
import numpy as np

from ..physics.density import DensityField
from ..physics.pressure import velocity_variance_grid, helbing_pressure, upstream_density
from ..physics import potential


class DisasterMonitor:
    def __init__(self, bounds, terrain=None, cell=0.5, sigma_m=1.0,
                 weights=(0.15, 1.0, 2.0e-4), rho_crit=6.0):
        self.df = DensityField(bounds, cell=cell, sigma_m=sigma_m)
        self.terrain = terrain
        self.w = weights                      # (density, helbing, terrain-potential)
        self.rho_crit = rho_crit
        self.Fx = self.Fy = self.Fmag = self.Yg = None
        if terrain is not None:
            r = np.arange(self.df.ny); c = np.arange(self.df.nx)
            C, R = np.meshgrid(c, r)
            Xg = self.df.x0 + (C + 0.5) * cell
            Yg = self.df.y0 + (R + 0.5) * cell
            self.Fx, self.Fy, self.Fmag = potential.potential_gradient(terrain, Xg, Yg)
            self.Yg = Yg

    def step(self, map_xy, vel_xy, use_terrain=True):
        """map_xy (N,2) m, vel_xy (N,2) m/s -> fields dict."""
        map_xy = np.asarray(map_xy, float).reshape(-1, 2)
        d = self.df.compute(map_xy)
        var = velocity_variance_grid(map_xy, vel_xy, self.df) if len(map_xy) else d * 0
        P = helbing_pressure(d, var)
        terrain_term = np.zeros_like(d)
        if use_terrain and self.terrain is not None:
            rho_up = upstream_density(d, self.Fx, self.Fy, shift_m=2.0, cell=self.df.cell)
            terrain_term = self.Fmag * rho_up
        R = self.w[0] * d + self.w[1] * P + (self.w[2] * terrain_term if use_terrain else 0.0)
        return {"density": d, "helbing": P, "terrain_term": terrain_term, "risk": R,
                "max_density": float(d.max()), "max_risk": float(R.max())}

    def alerts(self, fields, thresholds):
        """thresholds=(info,warn,critical) on the risk field. Returns zones
        (x,y,severity,risk) for cells above 'info', deduped to local maxima."""
        R = fields["risk"]; cell = self.df.cell
        info, warn, crit = thresholds
        ys, xs = np.where(R >= info)
        out = []
        for r, c in zip(ys, xs):
            v = R[r, c]
            # local maximum only (3x3) to avoid duplicate adjacent alerts
            sub = R[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            if v < sub.max():
                continue
            sev = "critical" if v >= crit else ("warn" if v >= warn else "info")
            x = self.df.x0 + (c + 0.5) * cell; y = self.df.y0 + (r + 0.5) * cell
            out.append({"x": round(float(x), 2), "y": round(float(y), 2),
                        "severity": sev, "risk": round(float(v), 3)})
        return out
