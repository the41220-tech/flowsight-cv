"""Hysteresis event gate — fixes the over-firing alarms (precision).

The raw per-frame alarms over-fire: on UMN the divergence flag was high in
545/774 frames and the absolute-pressure flag in 680/774, because a noisy signal
crossing a single threshold is counted every frame. This gate turns a per-frame
signal into a few DISCRETE EVENTS:

  * debounce  — rise to ON only after ``k_on`` consecutive samples >= ``t_high``
  * hysteresis — fall to OFF only when the signal drops below ``t_low`` (< t_high),
    so brief dips inside one episode don't end it
  * merge      — firings within ``merge_gap_s`` of the last OFF count as the same
    event (no fragmentation)

Result: one event per real episode instead of hundreds of frame-flags → precision
goes from ~0.1–0.3 (frame-level) to event-level. Wire it on top of any alarm
series (divergence max_div, absolute pressure p_max, fast-approach count, ...).
"""
from __future__ import annotations


class HysteresisEventGate:
    def __init__(self, t_high: float, t_low: float | None = None,
                 k_on: int = 3, merge_gap_s: float = 2.0) -> None:
        if t_low is None:
            t_low = 0.6 * t_high
        if not (t_low <= t_high):
            raise ValueError("t_low must be <= t_high")
        self.t_high = float(t_high)
        self.t_low = float(t_low)
        self.k_on = int(k_on)
        self.merge_gap = float(merge_gap_s)
        self.state = False
        self._above = 0
        self._last_off_t = -1e18
        self.n_events = 0

    def update(self, t: float, value: float) -> dict:
        """Feed one (time, value) sample -> {state, event_start, n_events}."""
        event_start = False
        if not self.state:
            self._above = self._above + 1 if value >= self.t_high else 0
            if self._above >= self.k_on:
                self.state = True
                if t - self._last_off_t > self.merge_gap:
                    event_start = True
                    self.n_events += 1
        else:
            if value < self.t_low:
                self.state = False
                self._above = 0
                self._last_off_t = t
        return {"state": self.state, "event_start": event_start,
                "n_events": self.n_events}

    def run(self, ts, values) -> list[float]:
        """Whole series -> list of event-START times."""
        out = []
        for t, v in zip(ts, values):
            if self.update(t, v)["event_start"]:
                out.append(float(t))
        return out


def naive_event_frames(values, thresh) -> int:
    """How many frames a single-threshold alarm would fire (the over-firing count)."""
    return int(sum(1 for v in values if v >= thresh))
