"""Stage-6 precision: characterize the flow-pressure alarm (threshold x duration-gate)
against the optical-flow speed-onset as a PROXY ground truth for panic windows.
Caches per-frame arrays to .npz so re-sweeps are instant (no re-detection).

Honest caveat: GT (mean-speed onset) and the detector (pressure rho*Var(v)) are
correlated motion signals, so this is a cross-check of alarm selectivity, not a
fully independent eval. It answers: 'how often does the alarm fire OUTSIDE a panic
window, and how does a sustained-duration gate trade precision for recall?'.
"""
import json, os
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

AVI = "/content/umn_all.avi"
WEIGHTS = "/content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt"
OUT = "/content/drive/MyDrive/flowsight_demo"
STEP, IMGSZ = 16, 320
os.makedirs(OUT, exist_ok=True)


def smooth(a, k=5):
    a = np.asarray(a, float)
    return a if len(a) < k else np.convolve(a, np.ones(k) / k, mode="same")


def episodes(mask, minlen=1):
    eps = []; ie = False; st = 0
    for k, v in enumerate(mask):
        if v and not ie:
            st, ie = k, True
        elif not v and ie:
            if k - st >= minlen:
                eps.append((st, k))
            ie = False
    if ie:
        eps.append((st, len(mask)))
    return eps


def detect_flow():
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(AVI)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = STEP / fps
    ts = []; counts = []; vmean = []; vvar = []
    prev = None; i = si = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % STEP:
            i += 1
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev is None:
            mag = np.zeros_like(g, float)
        else:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
        prev = g
        m = mag[mag > 0.5]
        r = model.predict(fr, conf=0.25, classes=[0], imgsz=IMGSZ, verbose=False)[0]
        n = len(r.boxes) if r.boxes is not None else 0
        ts.append(si * dt); counts.append(n)
        vmean.append(float(m.mean()) if m.size else 0.0)
        vvar.append(float(m.var()) if m.size else 0.0)
        if si % 25 == 0:
            print("[prec] %d t=%.1fs ppl=%d" % (si, si * dt, n), flush=True)
        i += 1; si += 1
    cap.release()
    return (np.array(ts), np.array(counts, float), np.array(vmean), np.array(vvar))


def main():
    arrp = OUT + "/rvt_arrays.npz"
    if os.path.exists(arrp):
        z = np.load(arrp); ts, counts, vmean, vvar = z["ts"], z["counts"], z["vmean"], z["vvar"]
        print("[prec] loaded cached arrays:", len(ts), flush=True)
    else:
        ts, counts, vmean, vvar = detect_flow()
        np.savez(arrp, ts=ts, counts=counts, vmean=vmean, vvar=vvar)
        print("[prec] cached arrays ->", arrp, flush=True)

    dt = float(ts[1] - ts[0]) if len(ts) > 1 else 0.533
    rho = counts / max(counts.max(), 1.0)
    Pn = smooth(rho * vvar, 5); Pn = Pn / (Pn.max() or 1.0)
    spn = smooth(vmean, 5); spn = spn / (spn.max() or 1.0)
    lo = spn < np.percentile(spn, 60)
    base = float(np.median(spn[lo])) if lo.any() else 0.0
    gt = spn > (base + 0.35)
    gt_eps = episodes(gt, 2)
    gt_frames = np.zeros(len(ts), bool)
    for s, e in gt_eps:
        gt_frames[s:e] = True

    rows = []
    for thr in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for K in [1, 2, 3]:
            fire = Pn >= thr
            al = np.zeros(len(fire), bool); run = 0
            for k, v in enumerate(fire):
                run = run + 1 if v else 0
                if run >= K:
                    al[k - K + 1:k + 1] = True
            al_eps = episodes(al, 1)
            tp = sum(1 for s, e in al_eps if gt_frames[s:e].any())
            prec = tp / len(al_eps) if al_eps else 0.0
            rec = (sum(1 for s, e in gt_eps if al[s:e].any()) / len(gt_eps)) if gt_eps else 0.0
            rows.append({"thr": thr, "K": K, "n_alarm_ep": len(al_eps),
                         "precision": round(prec, 2), "recall": round(rec, 2)})

    res = {"n_gt_episodes": len(gt_eps), "sampled_frames": int(len(ts)),
           "dt_s": round(dt, 3), "gt_def": "mean optical-flow speed > p60-baseline + 0.35",
           "sweep": rows}
    with open(OUT + "/precision_sweep.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("=== PREC_JSON_BEGIN ==="); print(json.dumps(res, default=float)); print("=== PREC_JSON_END ===")

    plt.figure(figsize=(7, 5))
    for K, c in [(1, "red"), (2, "orange"), (3, "green")]:
        pts = sorted([(r["recall"], r["precision"]) for r in rows if r["K"] == K])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        plt.plot(xs, ys, "o-", color=c, alpha=0.8, label="duration-gate K=%d" % K)
    plt.xlabel("recall (GT panic episodes caught)")
    plt.ylabel("precision (alarm episodes that are real)")
    plt.title("Flow-pressure alarm precision-recall (thr 0.15-0.4, n_GT=%d)" % len(gt_eps))
    plt.grid(alpha=0.3); plt.legend(); plt.xlim(0, 1.05); plt.ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(OUT + "/precision_pr.png", dpi=120); plt.close()
    print("saved", OUT + "/precision_sweep.json", "and", OUT + "/precision_pr.png")


if __name__ == "__main__":
    main()
