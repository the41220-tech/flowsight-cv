"""Recall-lab tests — slice metrics + NMS variants + experiment loop (synthetic).

    PYTHONPATH=. python tests/test_recall_lab.py
"""
from __future__ import annotations

import numpy as np

from flowsight.eval import nms_variants as nv
from flowsight.eval import slice_metrics as sm


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    assert abs(sm.iou_matrix(a, a)[0, 0] - 1.0) < 1e-9
    assert sm.iou_matrix(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0.0


def test_match_recall_precision_fppi():
    gt = np.array([[0, 0, 10, 20], [100, 0, 110, 20]])
    pred = np.array([[0, 0, 10, 20, 0.9], [100, 0, 110, 20, 0.8]])
    a = sm.aggregate([pred], [gt], 0.5)
    assert a["recall"] > 0.999 and a["fp"] == 0 and a["fppi"] == 0.0
    pred2 = np.vstack([pred, [500, 500, 520, 520, 0.7]])      # +1 FP
    a2 = sm.aggregate([pred2], [gt], 0.5)
    assert a2["recall"] > 0.999 and a2["fp"] == 1 and a2["fppi"] == 1.0
    assert a2["precision"] < 1.0


def test_slice_recall():
    gt = np.array([[0, 0, 10, 20], [100, 0, 110, 20]])
    tags = np.array(["small", "normal"])
    pred = np.array([[100, 0, 110, 20, 0.9]])                  # only the 'normal' one detected
    r_small, n_small = sm.slice_recall([pred], [gt], [tags], "small")
    r_norm, _ = sm.slice_recall([pred], [gt], [tags], "normal")
    assert n_small == 1 and r_small == 0.0 and r_norm > 0.999


def test_full_report_has_slices():
    gt = np.array([[0, 0, 10, 20]]); tags = np.array(["small"])
    pred = np.array([[0, 0, 10, 20, 0.9]])
    rep = sm.full_report([pred], [gt], [tags])
    assert "slice_recall" in rep and "small" in rep["slice_recall"]
    assert 0.0 <= rep["MR-2"] <= 1.0 and rep["AR@100"] >= 0.0


def test_softnms_recovers_crowd_neighbour():
    # two DIFFERENT people, boxes overlap IoU=1/3 -> hard-NMS@0.3 deletes one
    boxes = np.array([[0, 0, 10, 20], [5, 0, 15, 20]], float)
    scores = np.array([0.9, 0.8])
    keep_hard = nv.hard_nms(boxes, scores, 0.3)
    keep_soft, _ = nv.soft_nms(boxes, scores, 0.3, score_thr=0.01)
    assert len(keep_hard) == 1                                  # neighbour suppressed
    assert len(keep_soft) == 2                                  # neighbour recovered
    keep_diou = nv.diou_nms(boxes, scores, 0.3)
    assert len(keep_diou) == 2                                  # centre-distance spares it


def test_wbf_fuses():
    fb, fs = nv.weighted_boxes_fusion(
        [np.array([[0, 0, 10, 10]]), np.array([[1, 1, 11, 11]])],
        [np.array([0.9]), np.array([0.8])], iou_thr=0.5)
    assert len(fb) == 1 and 0.8 <= fs[0] <= 0.9                 # overlapping -> 1 fused box


def test_compare_at_matched_fppi():
    gt = [np.array([[0, 0, 10, 20], [50, 0, 60, 20]])]
    base = [np.array([[0, 0, 10, 20, 0.9]])]                    # finds 1/2
    var = [np.array([[0, 0, 10, 20, 0.9], [50, 0, 60, 20, 0.6]])]  # finds 2/2
    c = sm.compare_at_matched_fppi(base, var, gt, target_fppi=1.0)
    assert c["delta_recall"] > 0 and c["recall_var"] >= c["recall_base"]


def test_lab_registry_and_loop():
    from experiments.recall_lab import RecallLab, build_registry
    reg = build_registry()
    assert len(reg) == 10
    raw, gts, gsl = __import__("experiments.recall_lab", fromlist=["synthetic_scene"]).synthetic_scene(
        n_img=12, seed=1)
    lab = RecallLab(raw, gts, gsl, target_fppi=1.0)
    assert lab.propose().tier == "cached"                       # cheapest tier first
    cached_log = lab.run_all_cached()
    assert cached_log and all(r["tier"] == "cached" for r in cached_log)
    assert all("verdict" in r for r in cached_log)
    rec = lab.run_cycle()                                       # next = detector/train tier
    assert rec["tier"] in ("detector", "train") and rec["executed"] is False
    assert "DEFERRED" in rec["verdict"] and rec["spec"]


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
