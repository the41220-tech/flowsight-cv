"""Phase A anomaly-detector unit tests.

    PYTHONPATH=. python tests/test_anomaly.py   # plain-python fallback runner
"""
from __future__ import annotations

import numpy as np

from flowsight.anomaly import (
    EmergencyVoidDetector,
    FastApproachDetector,
    GeofenceDetector,
    RadialDivergenceDetector,
    TerrorComposite,
)


def test_radial_divergence_sign():
    bounds = (-10, -10, 10, 10)
    det = RadialDivergenceDetector(bounds, cell=1.0, div_thresh=0.2)
    rng = np.random.default_rng(0)
    xy = rng.uniform(-7, 7, (80, 2))
    r = np.linalg.norm(xy, axis=1, keepdims=True) + 1e-6
    vout = xy / r * 1.5            # fleeing outward
    out = det.step(xy, vout)
    inn = det.step(xy, -vout)      # converging inward
    assert out["max_div"] > inn["max_div"]   # divergence separates flee vs converge
    assert out["max_div"] > 0
    assert np.hypot(*out["center_xy"]) < 6    # disturbance center near origin


def test_fast_approach():
    det = FastApproachDetector(z_thresh=3.0, hist_n=3, consistency=0.8)
    rng = np.random.default_rng(1)
    for _ in range(40):  # calm baseline
        det.step([{"id": i, "x": 0, "y": 0,
                   "vx": rng.normal(0, 0.1), "vy": rng.normal(0, 0.1)}
                  for i in range(5)])
    al = []
    for _ in range(4):  # one fast, consistently +x track
        al = det.step([{"id": 99, "x": 0, "y": 0, "vx": 3.0, "vy": 0.0}])
    assert any(a["id"] == 99 for a in al)


def test_emergency_void():
    bounds = (-5, -5, 5, 5)
    det = EmergencyVoidDetector(bounds, cell=1.0, window=2,
                                void_thresh=0.1, delta_thresh=-0.2)
    cluster = np.random.default_rng(2).normal(0, 0.5, (20, 2))
    for _ in range(3):
        det.update(cluster)
    ev = det.update(np.zeros((0, 2)))  # crowd suddenly gone
    assert len(ev) >= 1


def test_geofence():
    det = GeofenceDetector([[(0, 0), (4, 0), (4, 4), (0, 4)]])
    v = det.check(np.array([[2.0, 2.0], [10.0, 10.0]]), ids=[1, 2])
    assert len(v) == 1 and v[0]["id"] == 1 and v[0]["zone"] == 0


def test_terror_composite():
    tc = TerrorComposite(window_s=5.0)
    assert not tc.update(0.0, fast=True, violence=False, divergence=False)
    assert not tc.update(2.0, fast=False, violence=True, divergence=False)
    assert tc.update(4.0, fast=False, violence=False, divergence=True)
    tc2 = TerrorComposite(window_s=2.0)
    tc2.update(0.0, True, False, False)
    assert not tc2.update(10.0, False, False, True)  # outside the window


def test_fight_index():
    from flowsight.anomaly import _fight_index
    assert _fight_index({0: "Fight", 1: "NonFight"}) == 0
    assert _fight_index({0: "NonFight", 1: "Fight"}) == 1
    assert _fight_index(["NonFight", "Fight"]) == 1


def test_narrate():
    from flowsight.anomaly import narrate
    assert narrate(1.0, {}) == ""
    s = narrate(5.0, {"terror": True, "violence": True, "fight_prob": 0.9,
                      "divergence": True, "div_center": (3, 4), "n_fast": 2})
    assert "테러" in s and "폭력" in s and "5.0" in s


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
