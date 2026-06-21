"""Image<->ground-plane homography (the flat-plane baseline) and a
piecewise/multi-homography that approximates a non-planar surface by regions.

Single homography is exact only for ONE plane (req. limitation §5 of the tech
reference). multi_homography is the cheap fix; metric-depth / terrain ray-cast
(see camera.py / terrain.py) is the robust fix.
"""
from __future__ import annotations
import numpy as np
import cv2


def fit_homography(img_pts, map_pts):
    img_pts = np.asarray(img_pts, np.float32)
    map_pts = np.asarray(map_pts, np.float32)
    if len(img_pts) == 4:
        return cv2.getPerspectiveTransform(img_pts, map_pts)
    H, _ = cv2.findHomography(img_pts, map_pts, cv2.RANSAC, 3.0)
    return H


def apply_homography(H, img_xy):
    img_xy = np.asarray(img_xy, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(img_xy, H).reshape(-1, 2)


class MultiHomography:
    """One homography per (planar) region. region_fn(map_xy)->region id is used
    at calibration; at inference we assign a test point to the region whose
    homography we trust (here: nearest calibration centroid in image space)."""

    def __init__(self):
        self.Hs = {}
        self._img_centroids = {}

    def fit(self, region_id, img_pts, map_pts):
        self.Hs[region_id] = fit_homography(img_pts, map_pts)
        self._img_centroids[region_id] = np.mean(np.asarray(img_pts, float), axis=0)

    def apply(self, img_xy):
        img_xy = np.asarray(img_xy, float)
        ids = list(self.Hs.keys())
        cents = np.array([self._img_centroids[i] for i in ids])
        out = np.zeros((len(img_xy), 2))
        for k, p in enumerate(img_xy):
            j = int(np.argmin(np.linalg.norm(cents - p, axis=1)))
            out[k] = apply_homography(self.Hs[ids[j]], p[None, :])[0]
        return out
