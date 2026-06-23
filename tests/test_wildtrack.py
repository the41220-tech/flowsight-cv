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
