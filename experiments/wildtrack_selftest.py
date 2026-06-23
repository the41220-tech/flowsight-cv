"""WILDTRACK pipeline self-test with SYNTHETIC calibration (no dataset needed).

Why this exists: the real WILDTRACK validation (recall vs ground truth on the real
frames) needs the 6 GB dataset + a GPU. But the *geometry / loader / fusion code*
that run does not — and that code is where the integration bugs live. This test
builds synthetic cameras in WILDTRACK's OFFICIAL world frame (cm; grid origin
(-300, -90), step 2.5 cm — from the official ``intersecting_area.py``), writes
their calibration in the REAL WILDTRACK XML format (intrinsic = OpenCV matrices,
extrinsic = plain-text rvec/tvec), and uses OpenCV's own ``cv2.projectPoints`` as
the ground-truth world->image map. It then checks that our pipeline inverts it:

  1. loader      — load_camera parses the real-format XMLs,
  2. round-trip  — to_ground recovers projected ground points to < 1 cm,
  3. convention  — positionID -> world spans the real ~12 x 36 m plaza,
  4. fusion      — two views of the same people dedup to the right count, and a
                   person seen by only one view is still recovered (occlusion fill),
  5. abs-scale   — density (persons/m^2) on the true metric ground is physical.

What it does NOT cover: real-detector recall on real frames (that is the deferred
GPU run, ``wildtrack_validate.py``). This isolates and proves the math/code.

Run:  PYTHONPATH=. python experiments/wildtrack_selftest.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from flowsight.geometry.multicam import CameraView, MultiCameraFusion
from flowsight.geometry.wildtrack import load_camera, positionid_to_world
from flowsight.physics.density import DensityField

# Official WILDTRACK ground grid (intersecting_area.py): origin (-300,-90) cm.
ORIGIN_CM = (-300.0, -90.0)
STEP_CM = 2.5
GRID_W = 480     # X cells -> ~12 m
GRID_H = 1440    # Y cells -> ~36 m
IMG_W, IMG_H = 1920, 1080


def _look_at(C, target, world_up=(0.0, 0.0, 1.0)):
    """World->camera rotation (rows = cam axes in world) + tvec, OpenCV convention
    (x right, y down, z forward).  Pc = R @ Pw + t,  t = -R @ C."""
    C = np.asarray(C, float)
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    up = np.asarray(world_up, float)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:                  # looking straight down
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=0)
    return R_wc, -R_wc @ C


def _K(fx=1100.0, fy=1100.0):
    return np.array([[fx, 0, IMG_W / 2.0], [0, fy, IMG_H / 2.0], [0, 0, 1]], float)


def _write_wildtrack_xml(dirpath, name, K, rvec, tvec):
    """Write intrinsic + extrinsic XMLs in the REAL WILDTRACK format."""
    intr = os.path.join(dirpath, "intr_%s.xml" % name)
    extr = os.path.join(dirpath, "extr_%s.xml" % name)
    kdata = " ".join("%.10g" % v for v in np.asarray(K).reshape(-1))
    with open(intr, "w") as f:
        f.write('<?xml version="1.0"?>\n<opencv_storage>\n'
                '<camera_matrix type_id="opencv-matrix">\n'
                '<rows>3</rows><cols>3</cols><dt>d</dt>\n<data>%s</data></camera_matrix>\n'
                '<distortion_coefficients type_id="opencv-matrix">\n'
                '<rows>5</rows><cols>1</cols><dt>d</dt>\n<data>0 0 0 0 0</data>'
                '</distortion_coefficients>\n</opencv_storage>\n' % kdata)
    with open(extr, "w") as f:                     # plain-text rvec/tvec (real format)
        f.write('<?xml version="1.0"?>\n<opencv_storage>\n<rvec>%s</rvec>\n<tvec>%s</tvec>\n'
                '</opencv_storage>\n' % (
                    " ".join("%.10g" % v for v in np.asarray(rvec).reshape(-1)),
                    " ".join("%.10g" % v for v in np.asarray(tvec).reshape(-1))))
    return intr, extr


def build_scene(tmpdir):
    """Two synthetic cameras over the official plaza, loaded via the real loader.
    Returns (cams: {name: WildtrackCamera}, raw: {name: (K, rvec, tvec)})."""
    import cv2

    centre = (3.0, 17.0, 0.0)                       # plaza centre-ish (m)
    specs = {"CVLab1": (-4.0, -3.0, 6.0), "CVLab2": (10.0, 38.0, 6.0)}  # cam centres (m)
    cams, raw = {}, {}
    for nm, C_m in specs.items():
        R_wc, t = _look_at(np.array(C_m) * 100.0, np.array(centre) * 100.0)  # cm
        rvec = cv2.Rodrigues(R_wc)[0].reshape(-1)
        K = _K()
        intr, extr = _write_wildtrack_xml(tmpdir, nm, K, rvec, t)
        cams[nm] = load_camera(intr, extr, unit_scale=0.01)               # cm -> m
        raw[nm] = (K, rvec, t)
    return cams, raw


def project(K, rvec, tvec, world_xy_m):
    """Ground points (m) -> pixels via OpenCV (the ground-truth forward map)."""
    import cv2

    xy = np.atleast_2d(np.asarray(world_xy_m, float))
    P = np.column_stack([xy * 100.0, np.zeros(len(xy))]).astype(np.float64)  # m -> cm
    uv, _ = cv2.projectPoints(P, np.asarray(rvec, float), np.asarray(tvec, float),
                              np.asarray(K, float), np.zeros(5))
    return uv.reshape(-1, 2)


def _spaced_people():
    """15 people on a >=3 m grid (no accidental fusion merges)."""
    return np.array([[-1.0 + i * 3.0, 2.0 + j * 6.0] for i in range(3) for j in range(5)],
                    float)


def _plaza_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-2.5, 8.5, n), rng.uniform(0.0, 34.0, n)])


# ---- checks (return numbers so tests can assert on them) --------------------

def check_roundtrip(cams, raw):
    pts = _plaza_points(60, seed=1)
    return max(np.linalg.norm(cams[nm].to_ground(project(*raw[nm], pts)) - pts, axis=1).max()
               for nm in cams)


def check_convention():
    corners = np.array([0, GRID_W - 1, GRID_W * (GRID_H - 1), GRID_W * GRID_H - 1])
    w = positionid_to_world(corners)
    return float(np.ptp(w[:, 0])), float(np.ptp(w[:, 1])), w


def check_fusion(cams, raw):
    people = _spaced_people()
    fusion = MultiCameraFusion([CameraView(nm, cams[nm]) for nm in cams], assoc_radius_m=1.0)
    dets = {nm: project(*raw[nm], people) for nm in cams}
    full = fusion.fuse(dets)
    occ = dict(dets)
    names = list(cams)
    occ[names[1]] = project(*raw[names[1]], people[:12])      # 2nd cam: occlusion
    occ = fusion.fuse(occ)
    loc_err = np.mean([np.linalg.norm(full["fused"] - p, axis=1).min() for p in people])
    return full, occ, float(loc_err), len(people)


def check_density():
    rng = np.random.default_rng(7)
    df = DensityField((-3.0, -1.0, 9.0, 35.0), cell=1.0, sigma_m=1.0)
    sparse = df.compute(_spaced_people()).max()
    dense = df.compute(np.column_stack([rng.uniform(2, 5, 30), rng.uniform(15, 18, 30)])).max()
    return float(sparse), float(dense)


def main():
    with tempfile.TemporaryDirectory() as d:
        cams, raw = build_scene(d)
        err = check_roundtrip(cams, raw)
        xspan, yspan, _ = check_convention()
        full, occ, lerr, n = check_fusion(cams, raw)
        sparse, dense = check_density()
    print("=== WILDTRACK SELF-TEST (synthetic calibration, OpenCV ground truth) ===")
    print("[loader]      2 cams parsed from REAL-format XML (matrix intr + text extr): OK")
    print("[round-trip]  project->to_ground max recovery error: %.3e m" % err)
    print("[convention]  positionID->world span: X=%.2f m  Y=%.2f m  (official ~12 x 36)"
          % (xspan, yspan))
    print("[fusion]      %d people, 2 views -> n_fused=%d  multi_view=%d  loc_err=%.4f m"
          % (n, full["n_fused"], full["multi_view"], lerr))
    print("[occlusion]   2nd cam sees 12/%d -> n_fused=%d  multi_view=%d (single-view kept)"
          % (n, occ["n_fused"], occ["multi_view"]))
    print("[abs-scale]   peak density  sparse=%.2f /m^2   dense(30 in ~9 m^2)=%.2f /m^2"
          % (sparse, dense))
    print("NOTE: validates geometry/loader/fusion/convention CODE. Real-detector recall")
    print("      on real frames is the deferred GPU run (experiments/wildtrack_validate.py).")


if __name__ == "__main__":
    main()
