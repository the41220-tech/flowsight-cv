"""WILDTRACK calibration loader + ground projection (Phase E/F validation).

WILDTRACK (EPFL, 7 synchronized HD cameras over a ~12x36 m plaza) ships full
OpenCV-pinhole calibration (intrinsic K + distortion; extrinsic Rodrigues rvec +
tvec, world->camera) and ground-truth pedestrian positions on a discretized
ground grid. This gives us REAL metric calibration to validate:
  * multi-camera fusion accuracy (fused positions vs GT), and
  * absolute scale (density/pressure on a true metric ground plane).

A `WildtrackCamera` maps a foot pixel to the world GROUND plane (Z=0) by
intersecting the back-projected camera ray with the ground — exactly the
image->ground map our `multicam.MultiCameraFusion` consumes (it exposes
``to_ground`` so it is a drop-in Calibrator). All geometry is OpenCV-convention:
    Pc = R * Pw + t  (R = Rodrigues(rvec)),  camera centre  C = -R^T t.

World units: WILDTRACK's world frame is in CENTIMETRES; pass unit_scale=0.01 to
get metres (default). Verify against GT once the dataset is extracted.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return R


class WildtrackCamera:
    def __init__(self, K, rvec=None, tvec=None, R=None, unit_scale: float = 0.01) -> None:
        self.K = np.asarray(K, float).reshape(3, 3)
        self.Kinv = np.linalg.inv(self.K)
        self.R = _rodrigues(rvec) if R is None else np.asarray(R, float).reshape(3, 3)
        self.t = np.asarray(tvec, float).reshape(3)
        self.C = -self.R.T @ self.t          # camera centre in world coords
        self.s = float(unit_scale)            # world units -> metres (cm -> 0.01)

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        """Foot pixels (N,2) -> world ground (Z=0) (X,Y) in METRES."""
        uv = np.atleast_2d(np.asarray(uv, float))
        pix = np.hstack([uv, np.ones((len(uv), 1))])
        d_cam = (self.Kinv @ pix.T).T          # ray dir in camera frame
        d_world = (self.R.T @ d_cam.T).T       # rotate to world
        lam = -self.C[2] / d_world[:, 2]       # intersect Z=0
        P = self.C[None, :] + lam[:, None] * d_world
        return P[:, :2] * self.s

    # multicam.MultiCameraFusion only needs to_ground; the world frame IS common
    # across WILDTRACK cameras, so CameraView(R=I, t=0) wraps this directly.
    def velocity_to_metric(self, uv, v_px):
        g0 = self.to_ground(uv)
        return self.to_ground(np.atleast_2d(uv) + np.atleast_2d(v_px)) - g0


def _read_opencv_xml(path: str, keys):
    """Read named matrices from an OpenCV XML/YAML calibration file."""
    import cv2

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    out = {}
    for k in keys:
        node = fs.getNode(k)
        out[k] = node.mat() if not node.empty() else None
    fs.release()
    return out


def load_camera(intr_path: str, extr_path: str, unit_scale: float = 0.01) -> WildtrackCamera:
    """Load one camera from its intrinsic + extrinsic OpenCV XML files.

    Tries the common WILDTRACK key names; the actual keys are verified against the
    extracted files (see load_all's autodetect).
    """
    intr = _read_opencv_xml(intr_path, ["camera_matrix", "cameraMatrix", "K"])
    K = next(v for v in intr.values() if v is not None)
    extr = _read_opencv_xml(extr_path, ["rvec", "tvec", "R", "T", "rotation", "translation"])
    rvec = extr.get("rvec") if extr.get("rvec") is not None else extr.get("rotation")
    tvec = extr.get("tvec") if extr.get("tvec") is not None else (
        extr.get("translation") if extr.get("translation") is not None else extr.get("T"))
    R = extr.get("R")
    if rvec is not None:
        return WildtrackCamera(K, rvec=rvec, tvec=tvec, unit_scale=unit_scale)
    return WildtrackCamera(K, R=R, tvec=tvec, unit_scale=unit_scale)


def positionid_to_world(pid, grid_w: int = 480, step_cm: float = 2.5,
                        origin_cm=(-300.0, -900.0), unit_scale: float = 0.01):
    """WILDTRACK positionID -> world (X,Y) metres. grid index: x=pid%W, y=pid//W.
    (Defaults follow the standard MVDet grid; verify origin/step against the data.)"""
    pid = np.asarray(pid)
    gx = pid % grid_w
    gy = pid // grid_w
    X = (origin_cm[0] + gx * step_cm) * unit_scale
    Y = (origin_cm[1] + gy * step_cm) * unit_scale
    return np.stack([X, Y], axis=-1).astype(float)


def load_gt_positions(json_path: str, **grid_kw) -> np.ndarray:
    """Annotation JSON -> (N,2) GT world positions in metres."""
    data = json.load(open(json_path))
    pids = [d["positionID"] for d in data if "positionID" in d]
    if not pids:
        return np.zeros((0, 2))
    return positionid_to_world(np.array(pids), **grid_kw)


def match_to_gt(pred_xy, gt_xy, radius_m: float = 1.0) -> dict:
    """Greedy nearest-neighbour match of predicted vs GT world positions.

    Returns precision / recall / mean localisation error (m) — the standard
    multi-camera detection metrics (a simple MODA/MODP-style scorer)."""
    pred = np.atleast_2d(np.asarray(pred_xy, float)) if len(pred_xy) else np.zeros((0, 2))
    gt = np.atleast_2d(np.asarray(gt_xy, float)) if len(gt_xy) else np.zeros((0, 2))
    used = set()
    tp, errs = 0, []
    for p in pred:
        if len(gt) == 0:
            break
        d = np.linalg.norm(gt - p, axis=1)
        order = np.argsort(d)
        for j in order:
            if d[j] > radius_m:
                break
            if j not in used:
                used.add(int(j))
                tp += 1
                errs.append(float(d[j]))
                break
    fp = len(pred) - tp
    fn = len(gt) - tp
    return {"precision": tp / (tp + fp + 1e-9), "recall": tp / (tp + fn + 1e-9),
            "mean_loc_err_m": float(np.mean(errs)) if errs else None,
            "tp": tp, "fp": fp, "fn": fn}
