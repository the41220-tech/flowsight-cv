"""Moat -> dashboard colour bridge: per-person 위험 driven by absolute pressure."""
import numpy as np

from flowsight.physics.moat_dashboard import RISK_COLOR, frame_risk, ko_banner


def test_dense_erratic_cluster_flags_danger_calm_isolated_safe():
    bounds = (0.0, 0.0, 12.0, 12.0)
    rng = np.random.default_rng(0)
    # dense erratic cluster around (3,3): high density + high Var(v) -> high pressure
    cl = np.array([[3.0, 3.0]]) + rng.normal(0, 0.4, (40, 2))
    cl_v = rng.normal(0, 2.0, (40, 2))          # erratic, fast, incoherent
    # one calm isolated person far away
    iso = np.array([[10.0, 10.0]])
    iso_v = np.array([[0.05, 0.0]])             # nearly still
    xy = np.vstack([cl, iso])
    vel = np.vstack([cl_v, iso_v])

    r = frame_risk(xy, vel, bounds)
    sev = [a["severity"] for a in r["per_person"]]
    # the cluster members should reach danger; the isolated calm person must be safe
    assert sev[-1] == "safe"
    assert any(s == "danger" for s in sev[:40])
    assert r["frame"]["severity"] == "danger"
    assert r["frame"]["color"] == RISK_COLOR["danger"]
    assert r["frame"]["n"] == 41 and r["frame"]["n_danger"] >= 1
    assert "위험도 위험" in ko_banner(r)


def test_empty_frame_is_safe():
    r = frame_risk(np.zeros((0, 2)), np.zeros((0, 2)), (0.0, 0.0, 5.0, 5.0))
    assert r["frame"]["severity"] == "safe"
    assert r["per_person"] == []
