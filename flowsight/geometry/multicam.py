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

    def project_with_sigma(self, foot_uv, sigma_px=2.0, bounds=None):
        """(N,2) foot pixels -> (world (N,2), sigma_world (N,), valid (N,) bool).

        sigma_world = ground displacement from a `sigma_px` VERTICAL pixel
        perturbation; it blows up near the horizon (grazing rays), so it is a
        per-detection localisation-uncertainty proxy (cf. MonoLoco's distance-bound).
        Unlike ``to_ground(bounds=...)`` it does NOT drop rows — it flags `valid`
        instead, so arrays stay ALIGNED across views for weighted fusion."""
        uv = np.atleast_2d(np.asarray(foot_uv, float))
        if not len(uv):
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool)
        g = self.cal.to_ground(uv)                       # analytic, no drop
        w = g @ self.R.T + self.t
        gp = self.cal.to_ground(uv + np.array([0.0, float(sigma_px)]))
        sig = np.linalg.norm(gp - g, axis=1)             # metres per sigma_px (rotation-invariant)
        valid = np.isfinite(w).all(axis=1) & np.isfinite(sig)
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            valid &= (w[:, 0] > x0) & (w[:, 0] < x1) & (w[:, 1] > y0) & (w[:, 1] < y1)
        return w, sig, valid


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

    def fuse_weighted(self, dets_by_view, sigma_px=2.0, bounds=None,
                      sigma_gate=None, base_r=None):
        """Cycle10/H3: UNCERTAINTY-AWARE fusion. For each detection compute the
        near-horizon localisation sigma (``project_with_sigma``); (1) GATE out dets
        with sigma > `sigma_gate` (horizon noise that would become FPs), (2) associate
        within a sigma-SCALED radius, and (3) use an INVERSE-VARIANCE weighted centroid
        so a noisy far detection cannot corrupt a confident near one. Contrast with
        ``fuse`` (unweighted mean + fixed radius), which lets a noisy added camera drop
        precision / pull centroids. Same return shape as ``fuse``."""
        r0 = self.r if base_r is None else float(base_r)
        items = []  # (name, xy, sigma)
        for name, d in dets_by_view.items():
            v = self.views.get(name)
            if v is None:
                continue
            w, sig, valid = v.project_with_sigma(d, sigma_px=sigma_px, bounds=bounds)
            for p, s, ok in zip(w, sig, valid):
                if not ok or (sigma_gate is not None and s > sigma_gate):
                    continue
                items.append((name, p, float(s)))
        clusters = []  # {sumw, sumwp, views, centroid}
        for name, p, s in items:
            wgt = 1.0 / (s * s + 1e-9)
            rad = min(r0 + s, 3.0 * r0)                 # sigma-scaled association radius
            best, bestd = None, rad
            for c in clusters:
                dist = float(np.linalg.norm(c["centroid"] - p))
                if dist <= bestd and name not in c["views"]:
                    best, bestd = c, dist
            if best is None:
                clusters.append({"sumw": wgt, "sumwp": wgt * p, "views": {name},
                                 "centroid": p.copy()})
            else:
                best["sumw"] += wgt
                best["sumwp"] += wgt * p
                best["views"].add(name)
                best["centroid"] = best["sumwp"] / best["sumw"]
        fused = (np.array([c["centroid"] for c in clusters]) if clusters
                 else np.zeros((0, 2)))
        return {
            "fused": fused,
            "n_fused": len(clusters),
            "multi_view": sum(1 for c in clusters if len(c["views"]) > 1),
            "views_per_person": [sorted(c["views"]) for c in clusters],
        }
