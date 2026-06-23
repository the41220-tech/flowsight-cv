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


def _mock_detect(gboxes, score=0.9):
    """detect(region)->local boxes for GT fully inside the region (Cycle-5 mock)."""
    def detect(region):
        x0, y0, x1, y1 = region
        out = []
        for b in np.atleast_2d(gboxes):
            if b[0] >= x0 and b[1] >= y0 and b[2] <= x1 and b[3] <= y1:
                out.append([b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0, score])
        return np.array(out).reshape(-1, 5)
    return detect


def test_tiling_remap_accuracy():
    """Acceptance #1: tile->detect->remap->WBF recovers global coords (IoU>=0.99)."""
    from flowsight.eval.tiling import run_tiled
    gt = np.array([100.0, 100.0, 140.0, 160.0])
    out = run_tiled(_mock_detect(gt[None, :]), (400, 400), slice=256, overlap=0.2)
    assert len(out) >= 1
    assert sm.iou_matrix(out[:, :4], gt[None, :])[:, 0].max() >= 0.99


def test_tiling_wbf_merges_crosstile():
    """Acceptance #2: a box seen in two overlapping tiles fuses to ONE box."""
    from flowsight.eval.tiling import run_tiled
    gt = np.array([220.0, 100.0, 250.0, 160.0])               # lies in x-tile overlap
    out = run_tiled(_mock_detect(gt[None, :]), (400, 400), slice=256, overlap=0.2)
    assert len(out) == 1
    assert sm.iou_matrix(out[:, :4], gt[None, :])[:, 0].max() >= 0.99


def test_tiling_recovers_small_via_lab():
    """Acceptance #3: whole-image (downscaled) misses small; tiled recovers ->
    ΔRecall@matched-FPPI > 0 through the lab Comparator."""
    from flowsight.eval.tiling import run_tiled
    rng = np.random.default_rng(0)
    W = H = 800
    gts, whole, tiled = [], [], []
    for _ in range(6):
        cx = rng.uniform(80, 720, 4); cy = rng.uniform(80, 720, 4); s = 16.0
        g = np.column_stack([cx - s/2, cy - s/2, cx + s/2, cy + s/2])
        gts.append(g)
        whole.append(np.zeros((0, 5)))                         # whole-frame misses small
        tiled.append(run_tiled(_mock_detect(g), (W, H), slice=256, overlap=0.2))
    c = sm.compare_at_matched_fppi(whole, tiled, gts, target_fppi=1.0)
    assert c["recall_base"] == 0.0 and c["recall_var"] > 0.0 and c["delta_recall"] > 0.0


def test_bodyprior_geometry_and_monotonic():
    """Acceptance #1: head->body recovers the full body (IoU>=0.9); foot monotonic in k."""
    from flowsight.eval.body_prior import head_to_body, head_to_foot
    head = np.array([[110.0, 100.0, 130.0, 140.0]])           # h=40, w=20
    body = head_to_body(head, k=7.5, w_ratio=2.0)
    gt_body = np.array([[100.0, 100.0, 140.0, 400.0]])         # 7.5 head-heights tall
    assert sm.iou_matrix(body, gt_body)[0, 0] >= 0.9
    assert head_to_foot(head, 8.0)[0, 1] > head_to_foot(head, 7.0)[0, 1]


def test_bodyprior_recovers_occluded_via_lab():
    """Acceptance #2: heads recover foot-occluded people -> ΔRecall@matched-FPPI > 0."""
    from flowsight.eval.body_prior import merge_head_proposals
    rng = np.random.default_rng(0)
    gts, person, merged = [], [], []
    for _ in range(8):
        n = 6
        cx = rng.uniform(120, 680, n); top = rng.uniform(60, 280, n); bh, bw = 200.0, 40.0
        body = np.column_stack([cx - bw/2, top, cx + bw/2, top + bh]); gts.append(body)
        occluded = rng.random(n) < 0.5
        pd, hd = [], []
        for i in range(n):
            hd.append([cx[i] - 10, top[i], cx[i] + 10, top[i] + bh/7.5, 0.8])   # head visible
            if not occluded[i]:
                pd.append([*body[i], 0.9])                     # body detector misses occluded
        pd = np.array(pd).reshape(-1, 5); hd = np.array(hd).reshape(-1, 5)
        person.append(pd)
        merged.append(merge_head_proposals(pd, hd, k=7.5, w_ratio=2.0))
    c = sm.compare_at_matched_fppi(person, merged, gts, target_fppi=1.0)
    assert c["delta_recall"] > 0 and c["recall_var"] > c["recall_base"]


def test_bodyprior_contract_and_dedup():
    """Acceptance #3: callable/empty robustness + no double-count on existing person."""
    from flowsight.eval.body_prior import merge_head_proposals
    person = np.array([[100.0, 100.0, 140.0, 400.0, 0.9]])
    assert len(merge_head_proposals(person, np.zeros((0, 5)))) == 1          # empty heads -> unchanged
    assert len(merge_head_proposals(np.zeros((0, 5)),
                                    np.array([[10.0, 0.0, 30.0, 27.0, 0.8]]))) == 1  # heads only
    head_over = np.array([[110.0, 100.0, 130.0, 140.0, 0.8]])                # body prop == person
    assert len(merge_head_proposals(person, head_over, dedup_iou=0.4)) == 1  # deduped


def test_realdata_parser_and_eval():
    """Cycle7 prep: WILDTRACK 2D-GT parse + slice tags + eval_view wiring (mock detector)."""
    import json
    import os
    import tempfile
    from experiments.recall_realdata import (eval_view, gt_boxes_for_view, load_view,
                                             slice_tags)
    anno = [   # small boxes (h=40) so tiling can fully contain them (tiling = small-obj lever)
        {"personID": 1, "positionID": 10, "views": [
            {"viewNum": 0, "xmin": 100, "ymin": 100, "xmax": 140, "ymax": 140},
            {"viewNum": 1, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}]},
        {"personID": 2, "positionID": 11, "views": [
            {"viewNum": 0, "xmin": 120, "ymin": 100, "xmax": 160, "ymax": 138},   # overlaps p1
            {"viewNum": 1, "xmin": 200, "ymin": 50, "xmax": 230, "ymax": 90}]},   # small in view1
        {"personID": 3, "positionID": 12, "views": [
            {"viewNum": 0, "xmin": 500, "ymin": 300, "xmax": 540, "ymax": 340},
            {"viewNum": 1, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}]},
    ]
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "annotations_positions"))
    json.dump(anno, open(os.path.join(d, "annotations_positions", "00000000.json"), "w"))
    g0 = gt_boxes_for_view(os.path.join(d, "annotations_positions", "00000000.json"), 0)
    g1 = gt_boxes_for_view(os.path.join(d, "annotations_positions", "00000000.json"), 1)
    assert len(g0) == 3 and len(g1) == 1                       # view-1: only p2 visible
    assert "occluded" in slice_tags(g0)                        # p1/p2 overlap
    fids, gts, gsl, imgs = load_view(d, "C1", 10)
    assert len(fids) == 1 and len(gts[0]) == 3

    def whole_for(i):
        return np.array([[100, 100, 140, 140, 0.9]])           # whole-frame finds only p1

    def tiled_detect_for(i):
        def detect(region):
            x0, y0, x1, y1 = region
            out = [[b[0]-x0, b[1]-y0, b[2]-x0, b[3]-y0, 0.8] for b in gts[0]
                   if b[0] >= x0 and b[1] >= y0 and b[2] <= x1 and b[3] <= y1]
            return np.array(out).reshape(-1, 5)
        return detect

    rep, _, _ = eval_view(whole_for, tiled_detect_for, [(800, 600)], [gts[0]], [gsl[0]],
                          target_fppis=(1.0,), slice_kw={"slice": 256, "overlap": 0.2})
    c = rep["matched_fppi"][1.0]
    assert c["recall_var"] >= c["recall_base"] and "slice_recall" in rep["whole"]


def _proj3d(K, rvec, tvec, xyz_m):
    import cv2
    P = (np.atleast_2d(np.asarray(xyz_m, float)) * 100.0)
    uv, _ = cv2.projectPoints(P, np.asarray(rvec, float), np.asarray(tvec, float),
                              np.asarray(K, float), np.zeros(5))
    return uv.reshape(-1, 2)


def test_anchor_foot_recovers_ground_visible():
    """Cycle8: with feet VISIBLE, foot anchor (alpha=1) recovers the ground pos."""
    import tempfile
    from experiments.wildtrack_selftest import build_scene
    from flowsight.eval.anchor_proj import median_err
    with tempfile.TemporaryDirectory() as d:
        cams, raw = build_scene(d)
        cam = cams["CVLab1"]; K, rvec, tvec = raw["CVLab1"]
        W = np.array([[3.0, 17.0]])
        feet = _proj3d(K, rvec, tvec, [[3, 17, 0]])[0]
        head = _proj3d(K, rvec, tvec, [[3, 17, 1.7]])[0]
        cx = (feet[0] + head[0]) / 2
        box = np.array([[cx - 10, head[1], cx + 10, feet[1]]])
        assert median_err(cam, box, W, 1.0) < 0.2          # foot anchor ~ground


def test_anchor_calibrated_beats_foot_when_occluded():
    """Cycle8: feet OCCLUDED (bbox stops at knee) -> calibrated alpha beats foot."""
    import tempfile
    from experiments.wildtrack_selftest import build_scene
    from flowsight.eval.anchor_proj import calibrate_alpha, median_err
    with tempfile.TemporaryDirectory() as d:
        cams, raw = build_scene(d)
        cam = cams["CVLab1"]; K, rvec, tvec = raw["CVLab1"]
        W = np.array([[3.0, 17.0]])
        head = _proj3d(K, rvec, tvec, [[3, 17, 1.7]])[0]
        knee = _proj3d(K, rvec, tvec, [[3, 17, 0.5]])[0]
        cx = head[0]
        box_occ = np.array([[cx - 10, head[1], cx + 10, knee[1]]])   # bottom = knee, not feet
        foot_err = median_err(cam, box_occ, W, 1.0)
        _, best_err = calibrate_alpha(cam, box_occ, W, grid=np.linspace(0.5, 2.5, 81))
        assert best_err < foot_err and foot_err > 0.2      # calibrated recovers, foot errs


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
