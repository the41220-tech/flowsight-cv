"""Cycle 10 physics early-warning channel tests — H4 (velocity disorder) + H5 (terrain).

    PYTHONPATH=. python tests/test_coherence.py
"""
from __future__ import annotations

import numpy as np

from flowsight.physics.coherence import disorder_index, order_parameter, velocity_entropy


def test_coherence_metrics_detect_disorder():
    """Direction-disorder metrics: ~0 for coherent flow, high for multidirectional."""
    coherent = np.tile([1.0, 0.0], (50, 1))                       # all moving +x
    assert disorder_index(coherent) < 0.02
    assert velocity_entropy(coherent) < 0.05
    assert order_parameter(coherent) > 0.98

    ang = np.linspace(-np.pi, np.pi, 50, endpoint=False)          # uniform headings
    multi = np.column_stack([np.cos(ang), np.sin(ang)])
    assert disorder_index(multi) > 0.9
    assert velocity_entropy(multi) > 0.9

    opp = np.vstack([np.tile([1.0, 0.0], (25, 1)),
                     np.tile([-1.0, 0.0], (25, 1))])              # counterflow (2 streams)
    assert disorder_index(opp) > 0.9          # phi ~ 0 (opposing cancels) — disorder catches it
    assert velocity_entropy(opp) < 0.4        # but only 2 directions -> entropy modest (nuance)


def test_varv_channel_leads_density_gated_pressure():
    """H4 mechanism (deterministic): the density-FREE disorder factor Var(v) rises
    before the density-gated product P = rho * Var(v), so it gives earlier warning."""
    t = np.arange(0.0, 40.0, 0.5)
    varv = np.clip((t - 6.0) / 6.0, 0, 1) * 0.06     # turbulence (Var) builds EARLY then plateaus
    rho = np.clip((t - 10.0) / 8.0, 0, 1) * 7.0      # density ramps LATER
    P = rho * varv                                    # Helbing product

    def rise(tt, sig, frac):
        sig = np.asarray(sig); mx = sig.max()
        return float(tt[np.argmax(sig >= frac * mx)]) if mx > 0 else None

    t_var = rise(t, varv, 0.25)
    t_p = rise(t, P, 0.25)
    assert t_var < t_p                                # density-free disorder leads the product
    assert (t_p - t_var) >= 1.0                       # by a meaningful margin


def test_terrain_channel_active_on_slope_zero_on_flat():
    """H5 (Cycle10): the terrain-potential (gravitational) channel is a NON-PLANAR
    DIFFERENTIATOR -- exactly ~0 on flat ground (where competitors' flat-homography
    assumption holds), large on a slope. Same static crowd, two terrains. (This is the
    structural moat; the early-warning LEAD over Helbing is REGIME-DEPENDENT -0.8..+0.6s
    in sim and needs real sloped footage to settle -- see docs/ROADMAP_LOOP.md H5.)"""
    from flowsight.geometry.terrain import Terrain, ramp_basin_elevation
    from flowsight.pipeline.moat_field import MoatMonitor
    bounds = (-7, 0, 7, 40)
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(28, 33, 200)])   # dense cluster
    V = rng.normal(0, 0.3, (200, 2))
    flat = MoatMonitor(bounds, terrain=Terrain(elevation_fn=ramp_basin_elevation(theta_deg=0, basin_depth=0)),
                       cell_m=0.5, sigma_m=0.75)
    slope = MoatMonitor(bounds, terrain=Terrain(elevation_fn=ramp_basin_elevation(theta_deg=22, basin_depth=2.0)),
                        cell_m=0.5, sigma_m=0.75)
    rf = flat.step_metric(X, V); rs = slope.step_metric(X, V)
    assert float(rf["terrain_push"].max()) < 1e-6     # flat: terrain channel contributes nothing
    assert float(rs["terrain_push"].max()) > 1.0      # slope: differentiator is active


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
