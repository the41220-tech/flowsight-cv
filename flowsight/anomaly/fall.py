"""Fall / collapse detector (Pattern C) — lightweight, training-free.

Two image-space signals per person, fused with the BEV density-void detector for
dual confirmation:
  1. LYING posture  — bbox aspect ratio (w/h) OR torso angle from vertical
     (shoulders->hips vector) indicates a horizontal body.
  2. FALL EVENT     — a sudden fractional collapse of bbox height over a short
     window (standing -> ground).

Consumes per-person {id, bbox:[x1,y1,x2,y2], keypoints:(17,3) optional} from a
pose estimator (YOLO-pose on Colab). COCO-17 keypoint indices: 5/6 shoulders,
11/12 hips. Numpy-only; the posture logic is unit-testable without a model.
"""
from __future__ import annotations

from collections import deque

import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def torso_angle_deg(keypoints, kp_conf: float = 0.3) -> float | None:
    """Angle (deg) of the shoulders->hips torso vector FROM VERTICAL.
    0 = upright, 90 = horizontal (lying). None if keypoints unreliable."""
    k = np.asarray(keypoints, float)
    if k.shape[0] <= R_HIP:
        return None
    if k.shape[1] >= 3 and min(k[[L_SH, R_SH, L_HIP, R_HIP], 2]) < kp_conf:
        return None
    sh = (k[L_SH, :2] + k[R_SH, :2]) / 2.0
    hp = (k[L_HIP, :2] + k[R_HIP, :2]) / 2.0
    v = hp - sh
    if np.linalg.norm(v) < 1e-6:
        return None
    # image y points down; vertical body -> v≈(0,+). angle from vertical:
    return float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-9)))


class FallDetector:
    def __init__(self, aspect_thresh: float = 1.1, torso_horiz_deg: float = 50.0,
                 drop_frac: float = 0.35, window: int = 4) -> None:
        self.aspect_thresh = aspect_thresh
        self.torso_horiz_deg = torso_horiz_deg
        self.drop_frac = drop_frac
        self.window = window
        self._h: dict = {}  # id -> deque of bbox heights

    def _lying(self, p: dict):
        reasons = []
        bbox = p.get("bbox")
        if bbox is not None:
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if h > 0 and w / h >= self.aspect_thresh:
                reasons.append("aspect")
        ang = torso_angle_deg(p["keypoints"]) if p.get("keypoints") is not None else None
        if ang is not None and ang >= self.torso_horiz_deg:
            reasons.append("torso")
        return (len(reasons) > 0), reasons

    def step(self, persons) -> dict:
        """persons: [{id, bbox, keypoints?}] -> {falls:[...], n_fall}."""
        alerts = []
        for p in persons:
            lying, reasons = self._lying(p)
            fall_event = False
            bbox = p.get("bbox")
            if bbox is not None:
                h = bbox[3] - bbox[1]
                hh = self._h.setdefault(p["id"], deque(maxlen=self.window))
                hh.append(h)
                if len(hh) >= self.window and hh[0] > 0 and (hh[0] - h) / hh[0] >= self.drop_frac:
                    fall_event = True
                    reasons = reasons + ["drop"]
                foot = (float((bbox[0] + bbox[2]) / 2), float(bbox[3]))
            else:
                foot = None
            if lying or fall_event:
                alerts.append({"id": int(p["id"]), "lying": bool(lying),
                               "fall_event": bool(fall_event), "reasons": reasons,
                               "foot": foot})
        return {"falls": alerts, "n_fall": len(alerts)}


def confirm_with_void(fall_alerts, void_alerts, radius_m: float = 3.0,
                      calibrator=None) -> int:
    """Dual confirmation: count falls whose foot maps near a BEV density-void
    centre. If no calibrator, returns min(n_fall, n_void) as a loose proxy."""
    if not fall_alerts or not void_alerts:
        return 0
    if calibrator is None:
        return min(len(fall_alerts), len(void_alerts))
    voids = np.array([v["center_xy"] for v in void_alerts], float)
    n = 0
    for a in fall_alerts:
        if a.get("foot") is None:
            continue
        fm = calibrator.to_ground(np.array([a["foot"]], float))[0]
        if np.min(np.linalg.norm(voids - fm, axis=1)) <= radius_m:
            n += 1
    return n
