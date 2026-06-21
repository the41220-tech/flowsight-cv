"""Density field on the ground plane: people/m^2 from map-coordinate points.

Counts on a grid, Gaussian-smooths (mass-preserving), divides by cell area.
Crowd-dynamics anchors (req. §6-3): 4-5/m^2 motion constrained, 6 risky,
8-10 Itaewon-level.
"""
from __future__ import annotations
import numpy as np
import cv2


class DensityField:
    def __init__(self, bounds, cell=0.5, sigma_m=1.0):
        self.x0, self.y0, self.x1, self.y1 = bounds
        self.cell = float(cell)
        self.sigma_m = float(sigma_m)
        self.nx = int(np.ceil((self.x1 - self.x0) / self.cell))
        self.ny = int(np.ceil((self.y1 - self.y0) / self.cell))

    def _idx(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        cx = ((xy[:, 0] - self.x0) / self.cell).astype(int)
        cy = ((xy[:, 1] - self.y0) / self.cell).astype(int)
        ok = (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)
        return cx, cy, ok

    def compute(self, points_xy):
        g = np.zeros((self.ny, self.nx), np.float32)
        cx, cy, ok = self._idx(points_xy)
        for i in np.where(ok)[0]:
            g[cy[i], cx[i]] += 1.0
        sig_cells = self.sigma_m / self.cell
        if sig_cells > 0:
            g = cv2.GaussianBlur(g, (0, 0), sig_cells)
        return g / (self.cell ** 2)            # people / m^2

    def sample(self, field, xy):
        cx, cy, ok = self._idx(xy)
        out = np.zeros(len(cx))
        out[ok] = field[cy[ok], cx[ok]]
        return out
