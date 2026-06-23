"""Velocity-disorder / collective-motion coherence channel (Cycle 10 / H4).

Helbing crowd pressure P = rho * Var(v) needs BOTH high density AND high velocity
variance, so as a *product* it fires late (density-gated). The Itaewon disaster
reconstruction (Unravelling the causes of the Seoul Halloween crowd crush, PLOS
One 2024) characterised the lethal regime with density + pressure + a velocity
*disorder* (entropy) term, and that disorder rises BEFORE the pressure peak.

This module adds that density-FREE leading channel:

* ``order_parameter``  phi = |mean(v_hat)| in [0,1] (Vicsek); 1 = coherent flow,
  0 = fully multidirectional.
* ``disorder_index``   D = 1 - phi.
* ``velocity_entropy`` normalised Shannon entropy of heading directions in [0,1];
  0 = single direction, 1 = uniform over all directions.

Because they ignore density, they flag the LOSS OF COLLECTIVE MOTION that precedes
a density-driven crush — an earlier warning than the density-gated product alarm.
Pure numpy.
"""
from __future__ import annotations

import numpy as np


def _moving(vel, speed_eps):
    v = np.atleast_2d(np.asarray(vel, float))
    if not len(v):
        return np.zeros((0, 2))
    sp = np.linalg.norm(v, axis=1)
    return v[sp > speed_eps]


def order_parameter(vel, speed_eps: float = 0.05) -> float:
    """Vicsek order parameter phi = |mean unit-velocity| in [0,1].

    1 = all moving the same way (coherent flow), ~0 = multidirectional/turbulent.
    Fewer than 2 moving agents -> 1.0 (no disorder to report)."""
    v = _moving(vel, speed_eps)
    if len(v) < 2:
        return 1.0
    u = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return float(np.linalg.norm(u.mean(axis=0)))


def disorder_index(vel, speed_eps: float = 0.05) -> float:
    """1 - order_parameter: 0 = coherent flow, ->1 = multidirectional/turbulent."""
    return 1.0 - order_parameter(vel, speed_eps)


def velocity_entropy(vel, bins: int = 12, speed_eps: float = 0.05) -> float:
    """Normalised Shannon entropy [0,1] of the heading-angle histogram.

    0 = a single heading, 1 = uniform over all directions. Density-free, so it
    rises with directional turbulence regardless of how many people there are."""
    v = _moving(vel, speed_eps)
    if len(v) < 2:
        return 0.0
    ang = np.arctan2(v[:, 1], v[:, 0])
    h, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    s = h.sum()
    if s <= 0:
        return 0.0
    p = h / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))
