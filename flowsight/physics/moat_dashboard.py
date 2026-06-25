"""Moat -> non-expert dashboard bridge.

Drives the dashboard's per-person 위험(danger) colour from the VALIDATED
crush-pressure moat -- Helbing P = rho * Var(v) in absolute 1/s^2 with the
0.02/s^2 critical threshold -- instead of raw per-person speed. Each tracked
person is coloured by the metric pressure AT THEIR LOCATION; the frame banner
uses the field maximum. This is the product-facing layer fugu flagged as the
real moat (physics), kept independent of the still-maturing learned detector.

Usage (renderer calls this per frame):
    r = frame_risk(xy_m, vel_m, bounds_m)
    for person, a in zip(people, r["per_person"]):
        draw_dot(person, color=a["color"])      # green/amber/red
    draw_banner(r["frame"]["label"], r["frame"]["n_danger"])
"""
from __future__ import annotations

import numpy as np

from flowsight.physics.crowd_pressure import P_CRIT, alarm_level, frame_pressure_metric

# RGB colours for the three absolute-alarm tiers (renderer-agnostic).
RISK_COLOR = {
    "safe": (0, 190, 0),       # 안전  green
    "caution": (240, 170, 0),  # 주의  amber
    "danger": (220, 30, 30),   # 위험  red
}


def frame_risk(xy_m, vel_m, bounds_m, cell_m: float = 0.5, sigma_m: float = 1.0,
               crit: float = P_CRIT) -> dict:
    """Per-person + frame crush-risk from the moat pressure field.

    xy_m (N,2) metric ground positions, vel_m (N,2) m/s, bounds_m (x0,y0,x1,y1) m.
    Returns {per_person:[{label,severity,frac,p,color}], frame:{...,n,n_danger}, field}.
    """
    field = frame_pressure_metric(xy_m, vel_m, bounds_m, cell_m, sigma_m)
    per_person = []
    for p in field["per_person"]:
        a = alarm_level(p, crit)
        a["color"] = RISK_COLOR[a["severity"]]
        per_person.append(a)
    frame = alarm_level(field["p_max"], crit)
    frame["color"] = RISK_COLOR[frame["severity"]]
    frame["n"] = len(field["per_person"])
    frame["n_danger"] = sum(1 for a in per_person if a["severity"] == "danger")
    return {"per_person": per_person, "frame": frame, "field": field}


def ko_banner(frame_risk_result: dict) -> str:
    """Plain-Korean one-line banner for non-experts (no jargon)."""
    fr = frame_risk_result["frame"]
    return "사람 %d명 · 위험도 %s · 위험 인원 %d명" % (fr["n"], fr["label"], fr["n_danger"])
