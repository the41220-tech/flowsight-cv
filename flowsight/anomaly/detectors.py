"""BEV anomaly-pattern detectors (Phase A) — lightweight signal detectors on the
metric BEV tracker output (per-person x, y, vx, vy in metres / m·s).

Patterns (FlowSight_AnomalyPattern_Resources):
  A. Fast directional approach  -> FastApproachDetector  (speed z-score + direction)
  B. Radial divergence          -> RadialDivergenceDetector (div(v) > θ)
  C. Emergency / fall (void)    -> EmergencyVoidDetector (local density collapse)
  D. Geofence violation         -> GeofenceDetector (point-in-polygon)
  (E. Violence is a separate raw-video model, not here.)
  Terror proxy = fast-approach -> [violence] -> divergence, time-windowed.

All numpy; cv2 for Gaussian smoothing (scipy-free), matplotlib.path for polygons.
"""
from __future__ import annotations

from collections import deque

import numpy as np

try:
    import cv2

    def _smooth(a, s):
        return cv2.GaussianBlur(a.astype(np.float32), (0, 0), max(1e-3, s))
except Exception:  # pragma: no cover
    def _smooth(a, s):
        return a


def _grid(xy, vel, bounds, cell, sigma_cells):
    """Deposit count + velocity onto a metric grid; return smoothed (rho, mvx, mvy)."""
    x0, y0, x1, y1 = bounds
    gw = max(1, int(np.ceil((x1 - x0) / cell)))
    gh = max(1, int(np.ceil((y1 - y0) / cell)))
    S0 = np.zeros((gh, gw), np.float32)
    Sx = np.zeros((gh, gw), np.float32)
    Sy = np.zeros((gh, gw), np.float32)
    if len(xy):
        xy = np.atleast_2d(xy)
        vel = np.atleast_2d(vel)
        gx = np.clip(((xy[:, 0] - x0) / cell).astype(int), 0, gw - 1)
        gy = np.clip(((xy[:, 1] - y0) / cell).astype(int), 0, gh - 1)
        for k in range(len(xy)):
            S0[gy[k], gx[k]] += 1.0
            Sx[gy[k], gx[k]] += vel[k, 0]
            Sy[gy[k], gx[k]] += vel[k, 1]
    S0s = _smooth(S0, sigma_cells)
    inv = 1.0 / (S0s + 1e-6)
    return S0s / (cell * cell), _smooth(Sx, sigma_cells) * inv, _smooth(Sy, sigma_cells) * inv


# --------------------------------------------------------------------------- #
# B. Radial divergence — crowd fleeing a central disturbance
# --------------------------------------------------------------------------- #
class RadialDivergenceDetector:
    def __init__(self, bounds, cell: float = 1.0, sigma_m: float = 1.5,
                 div_thresh: float = 0.30) -> None:
        self.bounds = bounds
        self.cell = float(cell)
        self.sigma_cells = sigma_m / cell
        self.div_thresh = float(div_thresh)

    def step(self, xy, vel) -> dict:
        rho, mvx, mvy = _grid(xy, vel, self.bounds, self.cell, self.sigma_cells)
        dvx_dx = np.gradient(mvx, self.cell, axis=1)
        dvy_dy = np.gradient(mvy, self.cell, axis=0)
        div = dvx_dx + dvy_dy            # 1/s; positive = expansion (fleeing)
        div_w = div * (rho > 0.02)        # only where there are people
        gy, gx = np.unravel_index(int(np.argmax(div_w)), div_w.shape) if div_w.size else (0, 0)
        x0, y0 = self.bounds[0], self.bounds[1]
        return {"divergence": div, "max_div": float(div_w.max()) if div_w.size else 0.0,
                "center_xy": (float(x0 + (gx + 0.5) * self.cell),
                              float(y0 + (gy + 0.5) * self.cell)),
                "alert": bool(div_w.max() >= self.div_thresh) if div_w.size else False}


# --------------------------------------------------------------------------- #
# A. Fast directional approach — pre-attack signal
# --------------------------------------------------------------------------- #
class FastApproachDetector:
    def __init__(self, z_thresh: float = 3.0, hist_n: int = 5,
                 consistency: float = 0.85, baseline_buf: int = 400) -> None:
        self.z_thresh = z_thresh
        self.n = hist_n
        self.consistency = consistency
        self.mu = self.sigma = None
        self._buf = deque(maxlen=baseline_buf)
        self._hist = {}  # id -> deque of unit velocity vectors

    def fit_baseline(self, speeds) -> None:
        s = np.asarray(list(speeds), float)
        self.mu, self.sigma = float(np.mean(s)), float(np.std(s) + 1e-6)

    def step(self, tracks) -> list[dict]:
        alerts = []
        for t in tracks:
            sp = float(np.hypot(t["vx"], t["vy"]))
            self._buf.append(sp)
            v = np.array([t["vx"], t["vy"]], float)
            nv = v / (np.linalg.norm(v) + 1e-6)
            h = self._hist.setdefault(t["id"], deque(maxlen=self.n))
            h.append(nv)
        if self.mu is None and len(self._buf) >= 30:
            self.fit_baseline(self._buf)
        if self.mu is None:
            return alerts
        for t in tracks:
            sp = float(np.hypot(t["vx"], t["vy"]))
            z = (sp - self.mu) / self.sigma
            if z < self.z_thresh:
                continue
            h = self._hist.get(t["id"])
            if h is None or len(h) < self.n:
                continue
            u = np.stack(h)
            cons = float(np.mean(u @ u[-1]))  # cosine sim to latest
            if cons > self.consistency:
                alerts.append({"id": int(t["id"]), "speed_z": round(z, 2),
                               "direction_consistency": round(cons, 3)})
        return alerts


# --------------------------------------------------------------------------- #
# C. Emergency / fall — sudden local density collapse (void)
# --------------------------------------------------------------------------- #
class EmergencyVoidDetector:
    def __init__(self, bounds, cell: float = 1.0, sigma_m: float = 1.5,
                 void_thresh: float = 0.15, delta_thresh: float = -0.3,
                 window: int = 5) -> None:
        self.bounds = bounds
        self.cell = float(cell)
        self.sigma_cells = sigma_m / cell
        self.void_thresh = void_thresh
        self.delta_thresh = delta_thresh
        self.window = window
        self.history = deque(maxlen=window * 2 + 1)

    def update(self, xy) -> list[dict]:
        rho, _, _ = _grid(xy, np.zeros((len(xy), 2)) if len(xy) else np.zeros((0, 2)),
                          self.bounds, self.cell, self.sigma_cells)
        alerts = []
        if len(self.history) >= self.window:
            prev = self.history[-self.window]
            delta = rho - prev
            void = (prev > self.void_thresh) & (delta < self.delta_thresh)
            if void.any():
                ys, xs = np.where(void)
                x0, y0 = self.bounds[0], self.bounds[1]
                cx = x0 + (xs.mean() + 0.5) * self.cell
                cy = y0 + (ys.mean() + 0.5) * self.cell
                alerts.append({"center_xy": (float(cx), float(cy)),
                               "delta_mean": float(delta[void].mean()),
                               "severity": float(-delta[void].min())})
        self.history.append(rho)
        return alerts


# --------------------------------------------------------------------------- #
# D. Geofence violation — point in a forbidden polygon (metric coords)
# --------------------------------------------------------------------------- #
class GeofenceDetector:
    def __init__(self, polygons) -> None:
        from matplotlib.path import Path
        self.paths = [Path(np.asarray(p, float)) for p in polygons]

    def check(self, xy, ids=None) -> list[dict]:
        xy = np.atleast_2d(np.asarray(xy, float)) if len(xy) else np.zeros((0, 2))
        out = []
        for k in range(len(xy)):
            for zi, path in enumerate(self.paths):
                if path.contains_point(xy[k]):
                    out.append({"id": int(ids[k]) if ids is not None else k,
                                "zone": zi, "xy": (float(xy[k, 0]), float(xy[k, 1]))})
                    break
        return out


# --------------------------------------------------------------------------- #
# Unified monitor + terror composite
# --------------------------------------------------------------------------- #
class AnomalyMonitor:
    """Runs the BEV detectors each frame and aggregates alerts."""

    def __init__(self, bounds, cell: float = 1.0, geofences=None) -> None:
        self.div = RadialDivergenceDetector(bounds, cell=cell)
        self.fast = FastApproachDetector()
        self.void = EmergencyVoidDetector(bounds, cell=cell)
        self.geo = GeofenceDetector(geofences) if geofences else None

    def step(self, tracks) -> dict:
        xy = np.array([[t["x"], t["y"]] for t in tracks], float) if tracks else np.zeros((0, 2))
        vel = np.array([[t["vx"], t["vy"]] for t in tracks], float) if tracks else np.zeros((0, 2))
        ids = [t["id"] for t in tracks] if tracks else []
        return {
            "divergence": self.div.step(xy, vel),
            "fast_approach": self.fast.step(tracks),
            "void": self.void.update(xy),
            "geofence": self.geo.check(xy, ids) if self.geo else [],
        }


class TerrorComposite:
    """Terror proxy: fast-approach -> [violence] -> radial divergence, time-windowed.

    Feed per-frame booleans (violence from the external video model is optional).
    Fires when the three stages occur in order within ``window_s`` of each other.
    """

    def __init__(self, window_s: float = 8.0) -> None:
        self.window = window_s
        self.t_fast = self.t_violence = None

    def update(self, t: float, fast: bool, violence: bool, divergence: bool) -> bool:
        if fast:
            self.t_fast = t
        if violence and self.t_fast is not None and t - self.t_fast <= self.window:
            self.t_violence = t
        ref = self.t_violence if self.t_violence is not None else self.t_fast
        if divergence and ref is not None and t - ref <= self.window:
            return True
        return False
