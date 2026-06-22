"""Explainability narrator — turns detector states into a plain-language event
description for the operator ("무엇이 일어났는가"). Rule-based by default (no
heavy deps); an optional VLM hook (Qwen2.5-VL etc.) can enrich it on GPU.
"""
from __future__ import annotations

_DIRS_KO = ["동", "북동", "북", "북서", "서", "남서", "남", "남동"]


def _dir_ko(dx: float, dy: float) -> str:
    """8-way Korean compass for a vector (image y points down -> +dy = 남)."""
    import math

    ang = math.degrees(math.atan2(-dy, dx)) % 360  # 0=동, CCW
    return _DIRS_KO[int((ang + 22.5) % 360 // 45)]


def narrate(t: float, state: dict, lang: str = "ko") -> str:
    """state keys: terror(bool), violence(bool/float), divergence(bool),
    div_center(x,y), n_fast(int), n_void(int), geofence(int). Returns one line."""
    parts = []
    if state.get("terror"):
        parts.append("⚠ 테러 의심: 빠른 접근 → 폭력 → 군중 분산 연쇄 감지")
    if state.get("violence"):
        fp = state.get("fight_prob")
        parts.append("폭력 행동 감지" + (f"(확률 {fp:.0%})" if fp is not None else ""))
    if state.get("divergence"):
        c = state.get("div_center")
        seg = "중심부에서 군중 방사형 이탈(분산)"
        if c is not None:
            seg += f" — 중심 ({c[0]:.0f}, {c[1]:.0f})m"
        parts.append(seg)
    if state.get("n_fast"):
        parts.append("빠르게 접근하는 대상 %d명" % state["n_fast"])
    if state.get("n_void"):
        parts.append("국소 군중 공백(쓰러짐 의심) %d곳" % state["n_void"])
    if state.get("geofence"):
        parts.append("금지구역 침입 %d건" % state["geofence"])
    if not parts:
        return ""
    return "[%.1fs] " % t + " · ".join(parts)


def narrate_vlm(frame, prompt: str = "What is happening in this crowd scene?",
                model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> str:
    """OPTIONAL GPU narrator: caption the alert frame with a VLM. Lazy import; only
    call at alert timestamps (expensive). Falls back to '' on any failure."""
    try:
        from PIL import Image
        from transformers import pipeline

        if not isinstance(frame, Image.Image):
            import numpy as np

            frame = Image.fromarray(np.asarray(frame)[:, :, ::-1])
        pipe = pipeline("image-text-to-text", model=model_id, device_map="auto")
        out = pipe(images=frame, text=prompt, max_new_tokens=60)
        return str(out)[:300]
    except Exception as e:  # noqa: BLE001
        return "[vlm unavailable: %s]" % str(e)[:80]
