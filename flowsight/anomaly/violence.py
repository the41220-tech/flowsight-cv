"""Frame-level violence detector (Pattern E) — the one anomaly pattern that needs
a raw-video model. Wraps the YOLOv8-cls classifier fine-tuned on RWF-2000
(finetune/violence_train.py) and returns a per-frame Fight probability that feeds
the TerrorComposite's violence stage.

The classifier was trained on WHOLE RWF-2000 frames (prepare_rwf.py extracts full
frames, not person crops), so inference is frame-level — no bounding boxes needed.

Heavy dep (ultralytics) is lazy-imported, so this module stays CPU-importable; the
`_fight_index` name mapping is unit-testable without weights.
"""
from __future__ import annotations

import numpy as np


def _fight_index(names) -> int:
    """ImageFolder classes sort alphabetically -> {0:'Fight', 1:'NonFight'}.
    Return the index of the FIGHT class robustly from a names dict/list."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for i, n in items:
        nl = str(n).lower()
        if "fight" in nl and "non" not in nl:
            return int(i)
    return 0


class ViolenceDetector:
    def __init__(self, weights: str, conf_thresh: float = 0.5,
                 imgsz: int = 224) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.thresh = float(conf_thresh)
        self.imgsz = int(imgsz)
        self._fi = _fight_index(self.model.names)

    def predict_frame(self, frame) -> dict:
        """BGR/RGB frame -> {'fight_prob', 'violence'}."""
        r = self.model.predict(frame, imgsz=self.imgsz, verbose=False)[0]
        probs = r.probs.data.detach().cpu().numpy()
        fight_p = float(probs[self._fi]) if self._fi < len(probs) else 0.0
        return {"fight_prob": round(fight_p, 4), "violence": fight_p >= self.thresh}

    def predict_clip(self, frames) -> dict:
        """Aggregate over a list of frames (max fight prob = clip violence)."""
        ps = [self.predict_frame(f)["fight_prob"] for f in frames]
        mx = float(np.max(ps)) if ps else 0.0
        return {"fight_prob": round(mx, 4), "violence": mx >= self.thresh,
                "per_frame": ps}
