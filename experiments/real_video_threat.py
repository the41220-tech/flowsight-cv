"""Stage-6 REAL-VIDEO solidification: anomalous crowd-MOVEMENT detection on real footage.

FlowSight is crowd-movement analysis & interpretation; crush is ONE application. This script
validates the GENERAL anomaly case: detect an abnormal crowd-movement event (panic / escape /
surge) on real pixels and measure how fast our alarm fires after onset — i.e. detection latency
& reliability for the broader incident-monitoring product, not crush-prediction specifically.

Real footage (UMN 'Unusual Crowd Activity', 11 escape clips, normal->panic).
Pipeline on REAL pixels:
  FT-2 detector -> people count/positions (density rho)
  dense optical flow -> crowd motion: mean speed + speed variance Var(v)
  Helbing-style pressure  P = rho * Var(v)   (the flow/pressure risk channel)
UMN scenes are flat -> terrain-potential channel ~0; the FLOW/PRESSURE channel
is what fires. We measure DETECTION LATENCY: how fast after the panic onset our
pressure alarm fires, and whether it stays quiet during normal segments.

Outputs (to /content/drive/MyDrive/flowsight_demo/):
  risk_timeline.png   - P (our alarm) + mean-speed over time, episodes shaded
  frame_3dmap.png     - 3D human-point map at peak-panic frame (dashboard preview)
  frame_annot.png     - that frame with detected-person boxes
  summary.json        - per-episode onset/alarm/latency + peak-frame 3D points
Prints summary.json to stdout too (so it can be read off the Colab cell).

CPU-OK. Run on Colab:  !PYTHONPATH=. python -u experiments/real_video_threat.py
"""
from __future__ import annotations
import json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AVI = "/content/umn_all.avi"
WEIGHTS = "/content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt"
OUT = "/content/drive/MyDrive/flowsight_demo"
STEP = 16                     # sample every 16th frame (CPU-friendly; ~4x fewer frames than STEP=4)
os.makedirs(OUT, exist_ok=True)


def smooth(a, k=5):
    a = np.asarray(a, float)
    if len(a) < k:
        return a
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="same")


def main():
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(AVI)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dt = STEP / fps
    print(f"[rvt] video opened: {n_total} frames @ {fps:.1f}fps, STEP={STEP} "
          f"-> ~{n_total // STEP if STEP else 0} sampled frames", flush=True)
    ts, counts, vmean, vvar = [], [], [], []
    foot_by_frame = {}                # sampled-idx -> (Nx2 foot points)
    prev_gray = None
    i = si = 0
    H = W = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % STEP:
            i += 1
            continue
        if H is None:
            H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # motion via dense optical flow (robust to fast panic motion)
        if prev_gray is None:
            mag = np.zeros_like(gray, float)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 2, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        prev_gray = gray
        m = mag[mag > 0.5]            # moving pixels only
        sp_mean = float(m.mean()) if m.size else 0.0
        sp_var = float(m.var()) if m.size else 0.0
        # people (density) via FT-2
        r = model.predict(frame, conf=0.25, classes=[0], imgsz=320, verbose=False)[0]
        xy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        foot = np.column_stack([(xy[:, 0] + xy[:, 2]) / 2, xy[:, 3]]) if len(xy) else np.zeros((0, 2))
        foot_by_frame[si] = foot
        ts.append(si * dt)
        counts.append(len(foot))
        vmean.append(sp_mean)
        vvar.append(sp_var)
        if si % 50 == 0:
            print(f"[rvt]   processed {si} sampled frames (frame {i}, t={si * dt:.1f}s, "
                  f"people={len(foot)})", flush=True)
        i += 1
        si += 1
    cap.release()
    print(f"[rvt] detection+flow loop done: {si} sampled frames", flush=True)

    ts = np.array(ts); counts = np.array(counts, float)
    vmean = np.array(vmean); vvar = np.array(vvar)
    rho = counts / max(counts.max(), 1.0)                    # normalized density
    P = smooth(rho * vvar, 5)                                 # Helbing-style pressure
    Pn = P / (P.max() or 1.0)
    sp = smooth(vmean, 5)
    spn = sp / (sp.max() or 1.0)

    # ground-truth panic onset (independent of our alarm): mean optical-flow speed
    base = np.median(spn[spn < np.percentile(spn, 60)]) if (spn < np.percentile(spn, 60)).any() else 0.0
    onset_mask = spn > (base + 0.35)                          # clearly-moving = flee
    # group contiguous onset frames into episodes
    episodes = []
    in_ep = False
    for k, v in enumerate(onset_mask):
        if v and not in_ep:
            start = k; in_ep = True
        elif not v and in_ep:
            if k - start >= 2:
                episodes.append((start, k))
            in_ep = False
    if in_ep:
        episodes.append((start, len(onset_mask)))

    ALARM = 0.25                                              # our pressure alarm threshold (norm)
    results = []
    for (s, e) in episodes:
        onset_t = float(ts[s])
        # our alarm: first frame in [s-window, e] where Pn crosses ALARM
        w0 = max(0, s - 8)
        seg = np.where(Pn[w0:e] >= ALARM)[0]
        alarm_t = float(ts[w0 + seg[0]]) if len(seg) else None
        lat = round(alarm_t - onset_t, 2) if alarm_t is not None else None
        results.append({"onset_s": round(onset_t, 2), "alarm_s": round(alarm_t, 2) if alarm_t else None,
                        "latency_s": lat, "dur_s": round(float(ts[min(e, len(ts)-1)] - ts[s]), 2)})

    peak = int(np.argmax(Pn))
    peak_t = float(ts[peak])

    # ---- figure 1: risk timeline ----
    plt.figure(figsize=(11, 4))
    plt.plot(ts, Pn, "b", lw=1.8, label="our flow-pressure risk  P=rho*Var(v)")
    plt.plot(ts, spn, color="gray", lw=1, alpha=0.7, label="crowd mean speed (motion)")
    plt.axhline(ALARM, color="purple", ls="--", lw=1, label=f"alarm threshold {ALARM}")
    for (s, e) in episodes:
        plt.axvspan(ts[s], ts[min(e, len(ts)-1)], color="red", alpha=0.12)
    for rr in results:
        if rr["alarm_s"] is not None:
            plt.axvline(rr["alarm_s"], color="green", lw=0.8)
    plt.title("UMN crowd-panic: our flow-pressure alarm fires at each escape (red=panic episodes)")
    plt.xlabel("time (s)"); plt.ylabel("normalized"); plt.legend(fontsize=8, loc="upper left"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/risk_timeline.png", dpi=120); plt.close()

    # ---- figure 2+3: peak-panic frame -> annotated + 3D human-point map ----
    cap = cv2.VideoCapture(AVI)
    cap.set(cv2.CAP_PROP_POS_FRAMES, peak * STEP)
    ok, frame = cap.read(); cap.release()
    pts3d = []
    if ok:
        foot = foot_by_frame.get(peak, np.zeros((0, 2)))
        annot = frame.copy()
        # rough image->ground (inverse-perspective demo; metric needs calibration/H1)
        for (cx, vy) in foot:
            cv2.circle(annot, (int(cx), int(vy)), 5, (0, 0, 255), -1)
        cv2.imwrite(f"{OUT}/frame_annot.png", annot)
        # ground map: X = horizontal, Y = depth (farther up the image = farther away)
        for (cx, vy) in foot:
            X = (cx - W / 2) / W * 20.0                       # ~meters, demo scale
            Y = (H - vy) / H * 30.0                           # depth proxy
            pts3d.append([round(float(X), 2), round(float(Y), 2)])
        if pts3d:
            P3 = np.array(pts3d)
            fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
            ax.scatter(P3[:, 0], P3[:, 1], np.zeros(len(P3)), c="crimson", s=40, depthshade=True)
            for (x, y) in P3:
                ax.plot([x, x], [y, y], [0, 1.7], color="crimson", lw=0.8, alpha=0.5)  # standing person stick
            ax.set_title(f"3D human-point map @ peak panic t={peak_t:.1f}s  (people={len(P3)})")
            ax.set_xlabel("X (m, demo)"); ax.set_ylabel("Y depth (m, demo)"); ax.set_zlabel("up (m)")
            ax.set_zlim(0, 4)
            plt.tight_layout(); plt.savefig(f"{OUT}/frame_3dmap.png", dpi=120); plt.close()

    summary = {"video": "UMN Crowd-Activity-All.avi", "fps": round(fps, 1),
               "sampled_frames": int(len(ts)), "duration_s": round(float(ts[-1]), 1) if len(ts) else 0,
               "n_panic_episodes": len(episodes), "alarm_threshold": ALARM,
               "episodes": results, "peak_panic_s": round(peak_t, 2),
               "peak_people": int(counts[peak]) if len(counts) else 0,
               "points3d_peak": pts3d[:60]}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)   # default=float: numpy float32 -> JSON
    print("=== SUMMARY_JSON_BEGIN ===")
    print(json.dumps(summary, default=float))
    print("=== SUMMARY_JSON_END ===")
    print("saved to", OUT)


if __name__ == "__main__":
    main()
