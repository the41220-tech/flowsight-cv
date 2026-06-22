"""Metric calibration (pixels -> metres) — MOAT layer 2.

This is the bridge that turns FlowSight's RELATIVE pixel pressure map into an
ABSOLUTE physical field in 1/s^2, so the Helbing crowd-pressure critical
threshold (~0.02 /s^2; Helbing, Johansson & Al-Abideen 2007) can be applied as
a real early-warning alarm instead of a within-scene heatmap.

Two calibrators, one interface:

  HomographyCalibrator       — accurate, perspective-correct. From >=4 surveyed
      ground correspondences (image px <-> world metres) it fits the image->ground
      homography H. Foot pixels map to ground metres; pixel velocities map to m/s
      via the LOCAL JACOBIAN of H. A homography is non-linear, so the metric scale
      varies across the image under perspective — a single scalar would be wrong.

  PedestrianScaleCalibrator  — approximate, calibration-free. When no surveyed
      points exist (UMN, ad-hoc CCTV) it estimates a uniform metres-per-pixel from
      pedestrian bounding-box heights: the median person-box height in px maps to
      ~1.7 m. First-order only (ignores perspective foreshortening) but it gets a
      scene into physical units so the 0.02/s^2 alarm becomes meaningful.

Common interface:
    to_ground(uv)            -> (N,2) ground metres
    velocity_to_metric(uv,v) -> (N,2) m/s
    area_scale(uv)           -> m^2 per px^2 (cell-area conversion / sanity)

Plus tracks_to_metric(cal, tracks): a tracks_<type>.json frame -> (xy_m, vel_m).
"""
from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

try:  # cv2 present in the FlowSight env; DLT fallback keeps this importable
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

PERSON_HEIGHT_M: float = 1.7  # population-average standing height


class Calibrator(Protocol):
    """Anything that maps image pixels + pixel velocities to ground metres."""

    def to_ground(self, uv: np.ndarray) -> np.ndarray: ...
    def velocity_to_metric(self, uv: np.ndarray, v_px: np.ndarray) -> np.ndarray: ...
    def area_scale(self, uv: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Uniform pedestrian-height scale (no surveyed points required)
# --------------------------------------------------------------------------- #
class PedestrianScaleCalibrator:
    """Uniform metres-per-pixel. Perspective-naive but calibration-free."""

    def __init__(self, m_per_px: float) -> None:
        if not (m_per_px > 0):
            raise ValueError("m_per_px must be > 0")
        self.s = float(m_per_px)

    @classmethod
    def from_bbox_heights(
        cls, heights_px: Iterable[float], person_h: float = PERSON_HEIGHT_M
    ) -> "PedestrianScaleCalibrator":
        """Scale from a set of person bbox heights: median px <-> person_h metres."""
        h = np.asarray(list(heights_px), float)
        h = h[np.isfinite(h) & (h > 0)]
        if h.size == 0:
            raise ValueError("no positive bbox heights")
        return cls(person_h / float(np.median(h)))

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(uv, float)) * self.s

    def velocity_to_metric(self, uv: np.ndarray, v_px: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(v_px, float)) * self.s

    def area_scale(self, uv: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.s * self.s, float)


# --------------------------------------------------------------------------- #
# Perspective-correct homography (>=4 surveyed ground points)
# --------------------------------------------------------------------------- #
def _dlt_homography(img: np.ndarray, wld: np.ndarray) -> np.ndarray:
    """Direct Linear Transform homography (cv2-free fallback)."""
    img = np.asarray(img, float)
    wld = np.asarray(wld, float)
    A = []
    for (u, v), (X, Y) in zip(img, wld):
        A.append([-u, -v, -1, 0, 0, 0, u * X, v * X, X])
        A.append([0, 0, 0, -u, -v, -1, u * Y, v * Y, Y])
    _, _, Vt = np.linalg.svd(np.asarray(A, float))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


class HomographyCalibrator:
    """Perspective-correct image->ground (metres) via homography + its Jacobian."""

    def __init__(self, H: np.ndarray) -> None:
        self.H = np.asarray(H, float).reshape(3, 3)

    @classmethod
    def from_points(
        cls, img_pts: np.ndarray, world_pts_m: np.ndarray
    ) -> "HomographyCalibrator":
        img = np.asarray(img_pts, np.float32)
        wld = np.asarray(world_pts_m, np.float32)
        if len(img) < 4 or len(img) != len(wld):
            raise ValueError(">=4 matched (image, world) correspondences required")
        if _HAS_CV2:
            if len(img) == 4:
                H = cv2.getPerspectiveTransform(img, wld)
            else:
                H, _ = cv2.findHomography(img, wld, cv2.RANSAC, 3.0)
        else:  # pragma: no cover
            H = _dlt_homography(img, wld)
        return cls(H)

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        hom = np.hstack([uv, np.ones((len(uv), 1))]) @ self.H.T
        return hom[:, :2] / hom[:, 2:3]

    def _jacobian(self, uv: np.ndarray) -> np.ndarray:
        """Per-point 2x2 d(X,Y)/d(u,v) of the homography map -> (N,2,2)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        H = self.H
        u, v = uv[:, 0], uv[:, 1]
        Xp = H[0, 0] * u + H[0, 1] * v + H[0, 2]
        Yp = H[1, 0] * u + H[1, 1] * v + H[1, 2]
        Wp = H[2, 0] * u + H[2, 1] * v + H[2, 2]
        Wp2 = Wp * Wp
        J = np.empty((len(uv), 2, 2), float)
        J[:, 0, 0] = (H[0, 0] * Wp - Xp * H[2, 0]) / Wp2  # dX/du
        J[:, 0, 1] = (H[0, 1] * Wp - Xp * H[2, 1]) / Wp2  # dX/dv
        J[:, 1, 0] = (H[1, 0] * Wp - Yp * H[2, 0]) / Wp2  # dY/du
        J[:, 1, 1] = (H[1, 1] * Wp - Yp * H[2, 1]) / Wp2  # dY/dv
        return J

    def velocity_to_metric(self, uv: np.ndarray, v_px: np.ndarray) -> np.ndarray:
        J = self._jacobian(uv)
        v_px = np.atleast_2d(np.asarray(v_px, float))
        return np.einsum("nij,nj->ni", J, v_px)  # v_m = J @ v_px, per point

    def area_scale(self, uv: np.ndarray) -> np.ndarray:
        return np.abs(np.linalg.det(self._jacobian(uv)))


# --------------------------------------------------------------------------- #
# Metric-depth ground calibrator (accurate, no surveyed points)
# --------------------------------------------------------------------------- #
def intrinsics_from_fov(W: int, H: int, fov_deg: float = 65.0) -> tuple:
    """Pinhole intrinsics from image size + horizontal FOV. Returns (fx,fy,cx,cy)."""
    fx = (W / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return fx, fx, W / 2.0, H / 2.0


def _fit_plane_ransac(pts, iters=300, thresh=0.15, seed=0):
    """RANSAC plane n·P = d (|n|=1) on (N,3) points -> (n, d, inlier_mask)."""
    pts = np.asarray(pts, float)
    rng = np.random.default_rng(seed)
    best = (None, None, None, -1)
    n_pts = len(pts)
    if n_pts < 3:
        raise ValueError("need >=3 points to fit a plane")
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts[idx]
        nrm = np.cross(p2 - p1, p3 - p1)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = float(nrm @ p1)
        dist = np.abs(pts @ nrm - d)
        inl = dist < thresh
        c = int(inl.sum())
        if c > best[3]:
            best = (nrm, d, inl, c)
    n, d, inl, _ = best
    # least-squares refit on inliers, orient normal toward the camera (origin)
    q = pts[inl]
    c = q.mean(0)
    _, _, Vt = np.linalg.svd(q - c)
    n = Vt[-1]
    n = n / np.linalg.norm(n)
    d = float(n @ c)
    if d < 0:  # make d>0 so the plane is in front (n points from camera to plane)
        n, d = -n, -d
    return n, d, inl


class DepthGroundCalibrator:
    """Pixels -> ACCURATE ground metres from a metric depth map (no survey, no
    pedestrian-height assumption).

    Works entirely in CAMERA space (no extrinsics needed): a metric monocular
    depth map (Depth-Anything-V2-Metric) gives each pixel's metric camera-space
    3-D point; a ground plane is RANSAC-fit to those points; a foot pixel's camera
    ray is intersected with that plane to get its metric (X,Y) on the ground, and
    in-plane axes give a 2-D metric map. Scale accuracy depends on the depth
    model's metric accuracy and the FOV estimate (a wrong FOV scales the map
    uniformly). This replaces the perspective-naive PedestrianScaleCalibrator.
    """

    def __init__(self, depth_map, fov_deg: float = 65.0, subsample: int = 8,
                 ransac_thresh_m: float = 0.15) -> None:
        dm = np.asarray(depth_map, float)
        H, W = dm.shape[:2]
        self.fx, self.fy, self.cx, self.cy = intrinsics_from_fov(W, H, fov_deg)
        # backproject a subsample of pixels to camera-space metric 3-D
        ys, xs = np.mgrid[0:H:subsample, 0:W:subsample]
        u, v = xs.ravel().astype(float), ys.ravel().astype(float)
        z = dm[ys.ravel(), xs.ravel()]
        ok = np.isfinite(z) & (z > 0)
        u, v, z = u[ok], v[ok], z[ok]
        Pc = np.column_stack([(u - self.cx) * z / self.fx,
                              (v - self.cy) * z / self.fy, z])
        self.n, self.d, _ = _fit_plane_ransac(Pc, thresh=ransac_thresh_m)
        # in-plane orthonormal basis + origin (plane point nearest the camera)
        a = np.array([1.0, 0.0, 0.0])
        if abs(self.n @ a) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        self.u_axis = np.cross(self.n, a)
        self.u_axis /= np.linalg.norm(self.u_axis)
        self.v_axis = np.cross(self.n, self.u_axis)
        self.origin = self.d * self.n

    def _ray_dirs(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        return np.column_stack([(uv[:, 0] - self.cx) / self.fx,
                                (uv[:, 1] - self.cy) / self.fy,
                                np.ones(len(uv))])

    def _ground_3d(self, uv: np.ndarray) -> np.ndarray:
        d_cam = self._ray_dirs(uv)
        denom = d_cam @ self.n
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        t = self.d / denom
        return d_cam * t[:, None]  # camera-space 3-D on the ground plane

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        P = self._ground_3d(uv) - self.origin
        return np.column_stack([P @ self.u_axis, P @ self.v_axis])

    def velocity_to_metric(self, uv: np.ndarray, v_px: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        v_px = np.atleast_2d(np.asarray(v_px, float))
        return self.to_ground(uv + v_px) - self.to_ground(uv)  # per second (dt=1)

    def area_scale(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        g0 = self.to_ground(uv)
        gx = self.to_ground(uv + [1.0, 0.0]) - g0
        gy = self.to_ground(uv + [0.0, 1.0]) - g0
        return np.abs(gx[:, 0] * gy[:, 1] - gx[:, 1] * gy[:, 0])


# --------------------------------------------------------------------------- #
# tracks_<type>.json frame -> metric arrays
# --------------------------------------------------------------------------- #
def tracks_to_metric(
    cal: Calibrator, tracks: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's track list [{x,y,vx,vy}] (pixels, px/s) -> (xy_m, vel_m)."""
    if not tracks:
        return np.zeros((0, 2)), np.zeros((0, 2))
    uv = np.array([[t["x"], t["y"]] for t in tracks], float)
    v_px = np.array([[t.get("vx", 0.0), t.get("vy", 0.0)] for t in tracks], float)
    return cal.to_ground(uv), cal.velocity_to_metric(uv, v_px)


def metric_bounds(xy_m_frames: Iterable[np.ndarray], pad_m: float = 2.0) -> tuple:
    """(x0,y0,x1,y1) bounds covering all metric points across frames, padded."""
    pts = [p for p in xy_m_frames if len(p)]
    if not pts:
        return (0.0, 0.0, 1.0, 1.0)
    allp = np.vstack(pts)
    x0, y0 = allp.min(0) - pad_m
    x1, y1 = allp.max(0) + pad_m
    return (float(x0), float(y0), float(x1), float(y1))
