"""FlowSight multi-object tracker — ByteTrack on FT-2.

Turns per-frame detections into PERSISTENT per-person tracks: each person gets a
stable track_id and a trajectory of (x, y, vx, vy) over time. This is the shared
data layer that unlocks:
  - the live dashboard's "moving dots" view (person = dot, danger = colour)
  - movement / dwell / origin-destination analysis
  - the fast-approach detector (per-track velocity z-score)

Method: ultralytics `model.track(persist=True, tracker="bytetrack.yaml")` — no
extra training. Foot point (box bottom-centre) approximates ground (x, y);
metric calibration (H1) comes later.

Outputs to <out>:
  tracks_<type>.json  — per-frame tracks + summary (unique ids, mean dwell)
  track_<type>.mp4    — overlay of dots + ids on the video (validation + preview)

Run on Colab:
  !PYTHONPATH=. python -u experiments/track_run.py --avi /content/umn_all.avi \
      --type cctv --weights /content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt \
      --imgsz 960 --step 2
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import cv2


def main(avi, source_type, weights, out, imgsz=960, step=2, conf=0.25):
    from ultralytics import YOLO
    model = YOLO(weights)
    cap = cv2.VideoCapture(avi)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dt = step / fps
    os.makedirs(out, exist_ok=True)
    print("[track] %s (%s): %d frames @%.1ffps %dx%d step=%d imgsz=%d" %
          (os.path.basename(avi), source_type, nfr, fps, W, H, step, imgsz), flush=True)

    vw = cv2.VideoWriter(out + "/track_%s.mp4" % source_type,
                         cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / step), (W, H))
    prev = {}            # id -> (x, y) previous foot point
    traj = {}            # id -> [(t, x, y)]
    per_frame = []       # [{t, n, tracks:[{id, x, y, vx, vy}]}]
    i = si = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        r = model.track(fr, persist=True, tracker="bytetrack.yaml",
                        classes=[0], imgsz=imgsz, conf=conf, verbose=False)[0]
        t = round(si * dt, 2)
        ft = []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                tid = int(tid)
                cx = float((x1 + x2) / 2.0); cy = float(y2)      # foot point
                px, py = prev.get(tid, (cx, cy))
                vx = (cx - px) / dt; vy = (cy - py) / dt
                prev[tid] = (cx, cy)
                ft.append({"id": tid, "x": round(cx, 1), "y": round(cy, 1),
                           "vx": round(vx, 1), "vy": round(vy, 1)})
                traj.setdefault(tid, []).append((t, round(cx, 1), round(cy, 1)))
                cv2.circle(fr, (int(cx), int(cy)), 5, (0, 230, 0), -1)
                cv2.putText(fr, str(tid), (int(cx) + 5, int(cy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(fr, "%s t=%.1fs  tracks=%d" % (source_type, t, len(ft)),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        per_frame.append({"t": t, "n": len(ft), "tracks": ft})
        vw.write(fr)
        if si % 25 == 0:
            print("[track]   %d t=%.1fs ids_now=%d total_ids=%d" % (si, t, len(ft), len(traj)), flush=True)
        i += 1; si += 1
    cap.release(); vw.release()

    dwell = {tid: xy[-1][0] - xy[0][0] for tid, xy in traj.items() if len(xy) > 1}
    summary = {"video": os.path.basename(avi), "source_type": source_type, "fps": round(fps, 1),
               "step": step, "imgsz": imgsz, "sampled_frames": int(si), "W": W, "H": H,
               "n_unique_tracks": len(traj),
               "mean_dwell_s": round(float(np.mean(list(dwell.values()))), 1) if dwell else 0.0,
               "max_dwell_s": round(float(np.max(list(dwell.values()))), 1) if dwell else 0.0,
               "mean_tracks_per_frame": round(float(np.mean([f["n"] for f in per_frame])), 1) if per_frame else 0.0}
    out_d = dict(summary); out_d["per_frame"] = per_frame
    json.dump(out_d, open(out + "/tracks_%s.json" % source_type, "w"), default=float)
    print("=== TRACK_DONE %s: unique_ids=%d mean_dwell=%.1fs mean/frame=%.1f frames=%d -> tracks_%s.json + track_%s.mp4 ===" %
          (source_type, summary["n_unique_tracks"], summary["mean_dwell_s"],
           summary["mean_tracks_per_frame"], si, source_type, source_type), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--avi", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--out", default="/content")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    main(a.avi, a.type, a.weights, a.out, imgsz=a.imgsz, step=a.step)
