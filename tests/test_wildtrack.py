"""WILDTRACK loader + geometry + fusion tests (synthetic calibration).

Proves the code that the real-dataset run exercises, without needing the 6 GB
dataset or a GPU. Uses OpenCV's own projectPoints as ground truth.

    PYTHONPATH=. python tests/test_wildtrack.py
"""
from __future__ import annotations

import tempfile

from experiments.wildtrack_selftest import (
    build_scene,
    check_convention,
    check_density,
    check_fusion,
    check_roundtrip,
)


def test_loader_and_roundtrip():
    """Real-format XML loads; to_ground is the exact inverse of projectPoints."""
    with tempfile.TemporaryDirectory() as d:
        cams, raw = build_scene(d)
        assert set(cams) == {"CVLab1", "CVLab2"}
        err = check_roundtrip(cams, raw)
        assert err < 0.01, "recovery error %.3e m exceeds 1 cm" % err


def test_positionid_convention_matches_official_plaza():
    xspan, yspan, _ = check_convention()
    assert 11.5 < xspan < 12.5, xspan      # 480 cells * 2.5 cm ~ 12 m
    assert 35.0 < yspan < 36.5, yspan      # 1440 cells * 2.5 cm ~ 36 m


def test_multicam_fusion_dedup_and_occlusion_fill():
    with tempfile.TemporaryDirectory() as d:
        cams, raw = build_scene(d)
        full, occ, lerr, n = check_fusion(cams, raw)
        assert full["n_fused"] == n, (full["n_fused"], n)     # dedup across views
        assert full["multi_view"] == n                        # all seen by both
        assert lerr < 0.05, lerr                              # fused within 5 cm
        assert occ["n_fused"] == n                            # occlusion fill works
        assert occ["multi_view"] == 12                        # only 12 confirmed by 2


def test_absolute_density_is_physical():
    sparse_peak, dense_peak = check_density()
    assert dense_peak > sparse_peak                           # monotonic in crowding
    assert 0.0 < dense_peak < 12.0                            # plausible persons/m^2


def test_to_ground_near_horizon_clamp():
    """Production clamp keeps in-bounds plaza points, drops near-horizon rays
    (validated as necessary on real WILDTRACK data)."""
    import tempfile
    import numpy as np
    from experiments.wildtrack_selftest import build_scene, project, _spaced_people
    bounds = (-10.0, -12.0, 20.0, 45.0)
    with tempfile.TemporaryDirectory() as d:
        cams, raw = build_scene(d)
        cam = cams["CVLab1"]
        gt = _spaced_people()
        kept = cam.to_ground(project(*raw["CVLab1"], gt), bounds=bounds)
        assert len(kept) == len(gt)                          # valid plaza points survive
        assert (kept[:, 0] > bounds[0]).all() and (kept[:, 1] < bounds[3]).all()
        top = np.array([[960.0, 2.0]])                       # top-row pixel ~ horizon
        assert len(cam.to_ground(top)) == 1                  # unclamped returns it
        assert len(cam.to_ground(top, bounds=bounds)) == 0   # clamp drops it


def test_cameraview_to_world_bounds_drops_out_of_range():
    """CameraView.to_world(foot_uv, bounds=...) must drop pixels whose ground
    projection falls outside the given bounds — relies on WildtrackCamera.to_ground."""
    import tempfile
    import numpy as np
    from experiments.wildtrack_selftest import build_scene, project, _spaced_people
    from flowsight.geometry.multicam import CameraView
    # narrow bounds that exclude the top-row near-horizon pixel but keep plaza points
    bounds = (-10.0, -12.0, 20.0, 45.0)
    with tempfile.TemporaryDirectory() as d:
        cams, raw = build_scene(d)
        cam = cams["CVLab1"]
        view = CameraView("CVLab1", cam)
        gt = _spaced_people()
        px = project(*raw["CVLab1"], gt)
        # All valid plaza pixels should survive
        world_all = view.to_world(px, bounds=bounds)
        assert len(world_all) == len(gt), (
            "expected %d points, got %d" % (len(gt), len(world_all)))
        # A near-horizon pixel (top row) should be dropped when bounds provided
        top = np.array([[960.0, 2.0]])
        world_no_bounds = view.to_world(top)
        world_with_bounds = view.to_world(top, bounds=bounds)
        assert len(world_no_bounds) == 1, "unclamped should return 1 point"
        assert len(world_with_bounds) == 0, "clamped should drop near-horizon pixel"


def test_multicamfusion_fuse_bounds_drops_out_of_range():
    """MultiCameraFusion.fuse(dets, bounds=...) must exclude detections that
    project outside the given bounds from the fused result."""
    import tempfile
    import numpy as np
    from experiments.wildtrack_selftest import build_scene, project, _spaced_people
    from flowsight.geometry.multicam import CameraView, MultiCameraFusion
    # tight bounds that contain all _spaced_people but exclude near-horizon pixels
    bounds = (-5.0, -5.0, 15.0, 40.0)
    with tempfile.TemporaryDirectory() as d:
        cams, raw = build_scene(d)
        views = [CameraView(nm, cams[nm]) for nm in cams]
        fusion = MultiCameraFusion(views, assoc_radius_m=1.0)
        gt = _spaced_people()
        # Normal in-bounds detections: all should survive fusion
        dets_normal = {nm: project(*raw[nm], gt) for nm in cams}
        result_normal = fusion.fuse(dets_normal, bounds=bounds)
        assert result_normal["n_fused"] == len(gt), (
            "expected %d fused, got %d" % (len(gt), result_normal["n_fused"]))
        # Add a near-horizon pixel (top row) to one camera: should NOT appear in fusion
        top_px = np.array([[960.0, 2.0]])
        dets_with_horizon = dict(dets_normal)
        cam0 = list(cams.keys())[0]
        dets_with_horizon[cam0] = np.vstack([dets_normal[cam0], top_px])
        result_with_horizon = fusion.fuse(dets_with_horizon, bounds=bounds)
        # Fused count should still equal len(gt): the extra horizon pixel is clamped out
        assert result_with_horizon["n_fused"] == len(gt), (
            "horizon pixel leaked into fusion: expected %d fused, got %d"
            % (len(gt), result_with_horizon["n_fused"]))
        # Without bounds, the extra pixel adds one spurious detection
        result_unclamped = fusion.fuse(dets_with_horizon)
        assert result_unclamped["n_fused"] > len(gt), (
            "expected extra spurious detection without bounds clamping")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
            passed += 1
        except Exception as e:  # noqa: BLE001
            print("FAIL", fn.__name__, "->", repr(e))
    print("\n%d/%d passed" % (passed, len(fns)))
    raise SystemExit(0 if passed == len(fns) else 1)
