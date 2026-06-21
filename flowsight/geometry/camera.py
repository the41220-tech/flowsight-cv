"""Pinhole camera: look-at construction, projection, ray, depth back-projection.

Convention: camera frame +Z forward (into scene), +X right, +Y down (standard
computer-vision). World frame +Z up. R maps world->camera: Pc = R @ (Pw - C).
"""
from __future__ import annotations
import numpy as np


def look_at_R(C, target, world_up=(0, 0, 1)) -> np.ndarray:
    C = np.asarray(C, float); target = np.asarray(target, float)
    up = np.asarray(world_up, float)
    z = target - C
    z = z / np.linalg.norm(z)                 # forward (+Z cam)
    x = np.cross(z, up)                        # right (+X cam)
    if np.linalg.norm(x) < 1e-8:              # looking straight down -> fallback up
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)                         # down (+Y cam)
    return np.stack([x, y, z], axis=0)        # rows -> Pc = R @ (Pw - C)


class PinholeCamera:
    def __init__(self, K, R, C):
        self.K = np.asarray(K, float)
        self.R = np.asarray(R, float)
        self.C = np.asarray(C, float).reshape(3)
        self._Kinv = np.linalg.inv(self.K)

    @classmethod
    def look_at(cls, C, target, f, width, height, world_up=(0, 0, 1)):
        K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
        return cls(K, look_at_R(C, target, world_up), C)

    def project(self, Pw):
        """World points (N,3) -> (pixel uv (N,2), camera-Z depth (N,))."""
        Pw = np.atleast_2d(np.asarray(Pw, float))
        Pc = (self.R @ (Pw - self.C).T).T
        uvw = (self.K @ Pc.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        return uv, Pc[:, 2]

    def ray(self, uv):
        """Pixels (N,2) -> (origin C (3,), unit world directions (N,3))."""
        uv = np.atleast_2d(np.asarray(uv, float))
        pix = np.hstack([uv, np.ones((len(uv), 1))])
        d_cam = (self._Kinv @ pix.T).T
        d_world = (self.R.T @ d_cam.T).T
        d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
        return self.C, d_world

    def backproject_depth(self, uv, depth_camZ):
        """Pixels + per-pixel camera-Z depth -> world points (N,3).

        Mirrors what a metric monocular depth model (e.g. Depth-Anything-V2-Metric)
        plus known pose gives you. depth_camZ is the optical-axis (Z) distance.
        """
        uv = np.atleast_2d(np.asarray(uv, float))
        z = np.asarray(depth_camZ, float).reshape(-1, 1)
        pix = np.hstack([uv, np.ones((len(uv), 1))])
        d_cam = (self._Kinv @ pix.T).T          # [X/Z, Y/Z, 1]
        Pc = d_cam * z
        return (self.R.T @ Pc.T).T + self.C
