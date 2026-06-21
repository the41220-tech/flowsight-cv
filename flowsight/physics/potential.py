"""Terrain gravitational potential energy and its gradient (downhill force).

This is the FlowSight moat: beyond a flat heatmap, the terrain's potential
energy U = m*g*h adds a directional, gravity-driven pressure bias. On a
downhill funnel (Itaewon), upstream mass converts potential energy into a
push on the people below — measurable BEFORE basin turbulence shows up.
"""
from __future__ import annotations
import numpy as np

G = 9.81
MASS = 70.0   # kg per person (population average)


def potential_energy(elevation, mass=MASS, g=G):
    return mass * g * np.asarray(elevation, float)


def potential_gradient(terrain, x, y, mass=MASS, g=G):
    """Returns (Fx, Fy, |F|): horizontal gravity force components from slope.
    |F| ~ m*g*sin(theta) -> the downhill body force per person."""
    gx, gy = terrain.gradient(x, y)            # dz/dx, dz/dy
    Fx, Fy = -mass * g * gx, -mass * g * gy     # force points downhill (-grad z)
    return Fx, Fy, np.hypot(Fx, Fy)


def potential_field(terrain, bounds, cell=0.5, mass=MASS, g=G):
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0, x1, cell); ys = np.arange(y0, y1, cell)
    X, Y = np.meshgrid(xs, ys)
    U = potential_energy(terrain.elevation(X, Y), mass, g)
    Fx, Fy, Fmag = potential_gradient(terrain, X, Y, mass, g)
    return {"U": U, "Fx": Fx, "Fy": Fy, "Fmag": Fmag, "xs": xs, "ys": ys}
