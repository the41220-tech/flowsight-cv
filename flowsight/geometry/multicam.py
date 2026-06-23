"""Multi-camera fusion (Phase E foundation) — the next moat.

Single-view has blind spots (occlusion, limited FOV). Fusion places every
camera's detections onto ONE common-world BEV and associates detections of the
SAME person across views, so the system sees through occlusions and across a
wider area. Per-camera metric calibration (DepthGroundCalibrator / homography)
is what makes this tractable — each view maps its foot pixels to metres, then a
rigid 2-D transform takes that view's ground frame into the common world frame.

PoC scope: given each view's calibrator + its (R, t) to the common frame, fuse
per-frame foot detections by greedy world-space clustering within
``assoc_radius_m`` → one fused person per cluster (deduped across overlapping
views; people seen by only one view are still recovered = occlusion fill).
Cross-view track association over time and automatic (R, t) estimation are the
next steps; here (R, t) is provided (from survey or a shared calibration target).
"""
from __future__ import annotations

import numpy as np


class CameraView:
    def __init__(self, name: str, calibrator, world_R=None, world_t=None) -> None:
        self.name = name
        self.cal = calibrator
        self.R = np.eye(2) if world_R is None else np.asarray(world_R, float).reshape(2, 2)
        self.t = np.zeros(2) if world_t is None else np.asarray(world_t, float).reshape(2)

    def to_world(self, foot_uv: np.ndarray, bounds=None) -> np.ndarray:
        """Foot pixels (N,2) -> common-world metres (N,2).

        bounds=None  -> exact analytic intersection for every pixel (unchanged).
        bounds=(x0,y0,x1,y1) [m] -> passed through to cal.to_ground for
        near-horizon clamping (drops out-of-bounds / upward-ray pixels)."""
        if foot_uv is None or len(foot_uv) == 0:
            return np.zeros((0, 2))
        g = self.cal.to_ground(np.atleast_2d(np.asarray(foot_uv, float)), bounds=bounds)
        return g @ self.R.T + self.t


class MultiCameraFusion:
    def __init__(self, views, assoc_radius_m: float = 1.5) -> None:
        self.views = {v.name: v for v in views}
        self.r = float(assoc_radius_m)

    def fuse(self, dets_by_view: dict, bounds=None) -> dict:
        """dets_by_view: {view_name: foot_uv (N,2)} -> fused world people.

        bounds=None  -> exact analytic intersection for every pixel (unchanged).
        bounds=(x0,y0,x1,y1) [m] -> passed through to each view's to_world for
        near-horizon clamping (drops out-of-bounds / upward-ray pixels).

        Returns fused centroids (M,2), per-cluster contributing views, and counts
        (n_fused unique people; multi_view = people confirmed by >1 camera).
        """
        pts = []  # (view_name, world_xy)
        for name, d in dets_by_view.items():
            v = self.views.get(name)
            if v is None:
                continue
            for p in v.to_world(d, bounds=bounds):
                pts.append((name, p))

        clusters = []  # {pts:[xy], views:set, centroid:xy}
        for name, p in pts:
            best, bestd = None, self.r
            for c in clusters:
                dist = float(np.linalg.norm(c["centroid"] - p))
                # don't merge two dets from the SAME view into one person
                if dist <= bestd and name not in c["views"]:
                    best, bestd = c, dist
            if best is None:
                clusters.append({"pts": [p], "views": {name}, "centroid": p.copy()})
            else:
                best["pts"].append(p)
                best["views"].add(name)
                best["centroid"] = np.mean(best["pts"], axis=0)

        fused = (np.array([c["centroid"] for c in clusters]) if clusters
                 else np.zeros((0, 2)))
        return {
            "fused": fused,
            "n_fused": len(clusters),
            "multi_view": sum(1 for c in clusters if len(c["views"]) > 1),
            "views_per_person": [sorted(c["views"]) for c in clusters],
        }
