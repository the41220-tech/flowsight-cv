"""Social-Force crowd simulation on NON-planar terrain.

Helbing-Molnar social force + a gravity-along-slope term, inside a funnel
corridor that narrows toward an exit at the basin bottom (Itaewon-like).
Produces ground-truth trajectories used to validate H1 (positioning) and
H2 (terrain-coupled pressure lead-time) WITHOUT real labelled data.

Pure numpy. O(N^2) pairwise repulsion — keep N <= ~400.
"""
from __future__ import annotations
import numpy as np

G = 9.81


class CrowdSim:
    def __init__(self, terrain, *, seed=0, dt=0.05,
                 corridor_top_halfwidth=6.0, corridor_bottle_halfwidth=1.6,
                 y_spawn=2.0, y_exit=38.0, bottleneck_y=33.0,
                 v0=1.3, v_max=2.5, tau=0.5, radius=0.22,
                 A=2.1, B=0.30, Aw=8.0, Bw=0.20, grav_gain=1.0,
                 inflow_per_s=14.0, max_agents=380):
        self.t = terrain
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.wt, self.wb = corridor_top_halfwidth, corridor_bottle_halfwidth
        self.y_spawn, self.y_exit, self.by = y_spawn, y_exit, bottleneck_y
        self.v0, self.v_max, self.tau, self.r = v0, v_max, tau, radius
        self.A, self.B, self.Aw, self.Bw = A, B, Aw, Bw
        self.grav_gain = grav_gain
        self.inflow = inflow_per_s
        self.max_agents = max_agents
        self.X = np.zeros((0, 2)); self.V = np.zeros((0, 2))
        self._spawn_credit = 0.0
        self.time = 0.0

    def halfwidth(self, y):
        s = np.clip((y - self.y_spawn) / (self.by - self.y_spawn), 0, 1)
        s = s * s * (3 - 2 * s)                 # smoothstep
        return self.wt + (self.wb - self.wt) * s

    def _spawn(self):
        self._spawn_credit += self.inflow * self.dt
        k = int(self._spawn_credit)
        if k <= 0 or len(self.X) >= self.max_agents:
            return
        self._spawn_credit -= k
        k = min(k, self.max_agents - len(self.X))
        hw = self.halfwidth(self.y_spawn) * 0.8
        nx = self.rng.uniform(-hw, hw, k)
        ny = self.y_spawn + self.rng.uniform(-0.5, 0.5, k)
        self.X = np.vstack([self.X, np.column_stack([nx, ny])])
        self.V = np.vstack([self.V, np.zeros((k, 2))])

    def step(self):
        self._spawn()
        X, V = self.X, self.V
        n = len(X)
        if n == 0:
            self.time += self.dt
            return
        exit_xy = np.array([0.0, self.y_exit])
        e = exit_xy[None, :] - X
        e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        a = (self.v0 * e - V) / self.tau         # driving

        # gravity along slope (downhill horizontal push)
        gx, gy = self.t.gradient(X[:, 0], X[:, 1])
        slope = np.hypot(gx, gy)
        theta = np.arctan(slope)
        downhill = -np.column_stack([gx, gy])
        downhill /= (np.linalg.norm(downhill, axis=1, keepdims=True) + 1e-9)
        a += self.grav_gain * (G * np.sin(theta) * np.cos(theta))[:, None] * downhill

        # pairwise repulsion (vectorized)
        if n > 1:
            diff = X[:, None, :] - X[None, :, :]
            dist = np.linalg.norm(diff, axis=2) + 1e-9
            np.fill_diagonal(dist, 1e9)
            nij = diff / dist[:, :, None]
            mag = self.A * np.exp((2 * self.r - dist) / self.B)
            a += np.sum(mag[:, :, None] * nij, axis=1)

        # wall repulsion (funnel)
        hw = self.halfwidth(X[:, 1])
        dl = (X[:, 0] + hw); dr = (hw - X[:, 0])      # dist to left/right wall
        a[:, 0] += self.Aw * np.exp((self.r - dl) / self.Bw)
        a[:, 0] -= self.Aw * np.exp((self.r - dr) / self.Bw)

        V += a * self.dt
        sp = np.linalg.norm(V, axis=1)
        too = sp > self.v_max
        V[too] *= (self.v_max / sp[too])[:, None]
        X += V * self.dt

        keep = X[:, 1] < self.y_exit             # egress past exit
        self.X, self.V = X[keep], V[keep]
        self.time += self.dt

    def state(self):
        return self.X.copy(), self.V.copy()
