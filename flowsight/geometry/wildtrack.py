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

    def to_plane(self, uv: np.ndarray, height_m: float = 0.0, bounds=None) -> np.ndarray:
        """Back-project pixels (N,2) onto the world plane Z=height_m -> (X,Y) METRES.

        height_m=0 is the ground plane (``to_ground``). Projecting a HEAD pixel onto
        Z~=1.7 m recovers the SAME (X,Y) as the person's ground contact while staying
        well-posed when the FEET ARE OCCLUDED (the head is near-always visible). This
        is the height-prior vertical-segment anchor (Niu 2021; head>ankle, Zhang & Ye
        2024) and, unlike the data-fitted bbox fraction alpha, needs NO per-camera fit.
        `bounds` applies the same near-horizon clamp as ``to_ground``."""
        uv = np.atleast_2d(np.asarray(uv, float))
        if not len(uv):
            return np.zeros((0, 2))
        Zc = float(height_m) / self.s          # plane height in world units (cm)
        pix = np.hstack([uv, np.ones((len(uv), 1))])
        d_cam = (self.Kinv @ pix.T).T          # ray dir in camera frame
        d_world = (self.R.T @ d_cam.T).T       # rotate to world
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (Zc - self.C[2]) / d_world[:, 2]   # intersect Z=Zc
        XY = (self.C[None, :] + lam[:, None] * d_world)[:, :2] * self.s
        if bounds is None:
            return XY
        x0, y0, x1, y1 = bounds
        ok = ((d_world[:, 2] < -1e-9) & (lam > 0) & np.isfinite(XY).all(axis=1)
              & (XY[:, 0] > x0) & (XY[:, 0] < x1) & (XY[:, 1] > y0) & (XY[:, 1] < y1))
        return XY[ok]

    def to_ground(self, uv: np.ndarray, bounds=None) -> np.ndarray:
        """Foot pixels (N,2) -> world ground (Z=0) (X,Y) in METRES.

        bounds=None  -> exact analytic intersection for every pixel (unchanged).
        bounds=(x0,y0,x1,y1) [m] -> NEAR-HORIZON CLAMP: drop pixels whose ray is
        not pointing toward the ground (d_world_z >= 0, i.e. parallel/upward, which
        makes lam diverge) or that land outside the plaza bounds. Validated as
        necessary on real WILDTRACK data (far/near-horizon foot points otherwise
        project to thousands of metres). Returns only the surviving (M,2) points."""
        return self.to_plane(uv, 0.0, bounds)

    # multicam.MultiCameraFusion only needs to_ground; the world frame IS common
    # across WILDTRACK cameras, so CameraView(R=I, t=0) wraps this directly.
    def velocity_to_metric(self, uv, v_px):
        g0 = self.to_ground(uv)
        return self.to_ground(np.atleast_2d(uv) + np.atleast_2d(v_px)) - g0


def _vec_from_node(node):
    """Flat float list from an XML calibration node.

    Handles BOTH formats WILDTRACK ships:
      * OpenCV matrix nodes  -> ``<x type_id="opencv-matrix"><data>...</data></x>``
        (used by the intrinsic files: camera_matrix, distortion_coefficients), and
      * plain-text vectors   -> ``<rvec>r0 r1 r2</rvec>`` (used by the extrinsic
        files; the official toolkit reads these with minidom, NOT FileStorage).
    """
    if node is None:
        return None
    data = node.find("data")
    text = data.text if data is not None else node.text
    if not text or not text.strip():
        return None
    return [float(x) for x in text.replace(",", " ").split()]


def _read_calibration_xml(path: str, names):
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    return {n: _vec_from_node(root.find(n)) for n in names}


def load_camera(intr_path: str, extr_path: str, unit_scale: float = 0.01) -> WildtrackCamera:
    """Load one camera from its intrinsic + extrinsic WILDTRACK XML files.

    Robust to the real WILDTRACK layout (intrinsic = OpenCV matrices, extrinsic =
    plain-text rvec/tvec) as well as plain OpenCV-matrix extrinsics, parsed with
    ElementTree (matches the official ``intersecting_area.py``).
    """
    intr = _read_calibration_xml(intr_path, ["camera_matrix", "cameraMatrix", "K"])
    Kvals = intr.get("camera_matrix") or intr.get("cameraMatrix") or intr.get("K")
    if not Kvals or len(Kvals) < 9:
        raise ValueError("camera_matrix (>=9 values) not found in %s" % intr_path)
    K = np.asarray(Kvals[:9], float).reshape(3, 3)

    extr = _read_calibration_xml(
        extr_path, ["rvec", "tvec", "rotation", "translation", "R", "T"])
    rvec = extr.get("rvec") or extr.get("rotation")
    tvec = extr.get("tvec") or extr.get("translation") or extr.get("T")
    Rvals = extr.get("R")
    if not tvec or len(tvec) < 3:
        raise ValueError("tvec (>=3 values) not found in %s" % extr_path)
    if rvec and len(rvec) >= 3:
        return WildtrackCamera(K, rvec=np.asarray(rvec[:3], float),
                               tvec=np.asarray(tvec[:3], float), unit_scale=unit_scale)
    if Rvals and len(Rvals) >= 9:
        return WildtrackCamera(K, R=np.asarray(Rvals[:9], float).reshape(3, 3),
                               tvec=np.asarray(tvec[:3], float), unit_scale=unit_scale)
    raise ValueError("neither rvec nor R found in %s" % extr_path)


def positionid_to_world(pid, grid_w: int = 480, step_cm: float = 2.5,
                        origin_cm=(-300.0, -90.0), unit_scale: float = 0.01):
    """WILDTRACK positionID -> world (X,Y) metres. grid index: x=pid%W, y=pid//W.

    Defaults follow the OFFICIAL WILDTRACK toolkit (``intersecting_area.py``):
    grid 1440x480, origin (-300, -90) cm, step 2.5 cm -> X span ~12 m (480 cells),
    Y span ~36 m (1440 cells). NOTE: MVDet uses origin_y=-900, but the calibration
    files shipped WITH the dataset are consistent with -90, so that is the default
    here (the two frames differ by 8.1 m in Y)."""
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
