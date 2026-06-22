"""FlowSight anomaly-pattern detectors (Phase A).

Lightweight, numpy-only detectors that ride on the metric BEV tracker output
(per-person x, y, vx, vy in metres / m·s). Four of the five anomaly patterns
reduce to signal detectors here; violence (Pattern E) is a separate video model.
"""
from .detectors import (  # noqa: F401
    AnomalyMonitor,
    EmergencyVoidDetector,
    FastApproachDetector,
    GeofenceDetector,
    RadialDivergenceDetector,
    TerrorComposite,
)
from .narrate import narrate  # noqa: F401
from .violence import ViolenceDetector, _fight_index  # noqa: F401
