"""Fall / collapse run (Phase C) — YOLO-pose keypoints -> FallDetector + BEV
density-void (dual confirmation) -> narrated fall events + timeline.

Per sampled frame:
  YOLO-pose (model.track)   -> per-person bbox + 17 keypoints + stable id
  FallDetector              -> lying posture (aspect / torso angle) + height-drop
  EmergencyVoidDetector     -> local density collapse (pixel BEV)
  confirm_with_void         -> falls co-located with a void (dual confirmation)

Run on Colab (L4):
  !PYTHONPATH=. python -u experiments/fall_run.py --video /content/umn_all.avi \
      --type cctv --weights yolo11s-pose.pt --step 10 \
      --out /content/drive/MyDrive/flowsight_demo
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from flowsight.anomaly import EmergencyVoidDetector, FallDetector, confirm_with_void


def main(a) -> None:
    from ultralytics import YOLO

    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fall = FallDetector()
    void = EmergencyVoidDetector((0, 0, W, H), cell=max(1.0, W / 40.0),
                                 sigma_m=max(1.0, W / 40.0), window=a.void_window)
    print("[fall] %s: %dx%d @%.1ffps step=%d weights=%s"
          % (a.type, W, H, fps, a.step, a.weights), flush=True)

    events, counts = [], {"fall": 0, "void": 0, "confirmed": 0}
    series_t, series_nfall = [], []
    i = si = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % a.step:
            i += 1
            continue
        t = round(si * a.step / fps, 2)
        r = model.track(fr, persist=True, classes=[0], imgsz=a.imgsz,
                        conf=a.conf, verbose=False)[0]
        persons, foots = [], []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            kdata = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for j, (b, tid) in enumerate(zip(xyxy, ids)):
                persons.append({"id": int(tid), "bbox": [float(b[0]), float(b[1]),
                                float(b[2]), float(b[3])],
                                "keypoints": kdata[j] if kdata is not None else None})
                foots.append([(b[0] + b[2]) / 2.0, b[3]])
        fres = fall.step(persons)
        vres = void.update(np.array(foots, float) if foots else np.zeros((0, 2)))
        conf = confirm_with_void(fres["falls"], vres)
        counts["fall"] += int(fres["n_fall"] > 0)
        counts["void"] += int(len(vres) > 0)
        counts["confirmed"] += int(conf > 0)
        series_t.append(t)
        series_nfall.append(fres["n_fall"])
        if fres["n_fall"]:
            ids_f = [f["id"] for f in fres["falls"]]
            line = "[%.1fs] 쓰러짐/이상자세 감지 %d명 (id %s)%s" % (
                t, fres["n_fall"], ids_f,
                " · 군중 공백 동시발생(확정)" if conf > 0 else "")
            events.append({"t": t, "n_fall": fres["n_fall"], "confirmed": int(conf),
                           "ids": ids_f, "line": line})
        if si % 50 == 0:
            print("[fall]   %d t=%.1fs persons=%d falls=%d void=%d"
                  % (si, t, len(persons), fres["n_fall"], len(vres)), flush=True)
        i += 1
        si += 1
    cap.release()

    json.dump({"source_type": a.type, "event_frame_counts": counts, "events": events},
              open(a.out + "/fall_%s.json" % a.type, "w"), default=float)
    _timeline(series_t, series_nfall, a.out + "/fall_%s_timeline.png" % a.type, a.type)
    print("=== FALL_DONE %s: %d frames, counts=%s, %d fall events -> fall_%s.json ==="
          % (a.type, si, counts, len(events), a.type), flush=True)


def _timeline(t, nfall, path, source_type):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    plt.figure(figsize=(9, 3))
    plt.plot(np.array(t), np.array(nfall), "m-", lw=1.1)
    plt.xlabel("time (s)")
    plt.ylabel("# fallen / abnormal-pose")
    plt.title("Fall / collapse timeline - %s" % source_type)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--out", default="/content")
    ap.add_argument("--weights", default="yolo11s-pose.pt")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--void-window", type=int, default=5)
    main(ap.parse_args())
