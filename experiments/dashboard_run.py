"""FlowSight dashboard build: run the analysis pipeline on multiple crowd clips
(CCTV + drone), then emit ONE self-contained HTML dashboard with the real data
embedded. Analysis-first (flow + density), safety as one layer.

Per frame: FT-2 detect (count) + dense optical flow -> directional flow features
(divergence/curl/counterflow/efficiency) + flow-pressure safety risk.

Usage on Colab:
  !PYTHONPATH=. python -u experiments/dashboard_run.py \
     --clip cctv:/content/umn_all.avi --clip drone:/content/drone.mp4 \
     --weights /content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt \
     --out /content/drive/MyDrive/flowsight_demo
Writes <out>/flowsight_dashboard.html (and tries files.download in Colab).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import cv2
sys.path.insert(0, ".")
from flowsight.physics.flow_features import frame_flow_features


def smooth(a, k=5):
    a = np.asarray(a, float)
    return a if len(a) < k else np.convolve(a, np.ones(k) / k, mode="same")


def episodes(mask, minlen=2):
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


def run_clip(avi, source_type, weights, step=16, imgsz=320):
    from ultralytics import YOLO
    model = YOLO(weights)
    cap = cv2.VideoCapture(avi)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dt = step / fps
    print("[dash] %s (%s): %d frames @%.1ffps step=%d" % (os.path.basename(avi), source_type, nfr, fps, step), flush=True)
    ts = []; count = []; feff = []; cflow = []; divg = []; curl = []; spd = []; vvar = []; grids = []
    prev = None; i = si = 0; H = W = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        if H is None:
            H, W = fr.shape[:2]
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        flow = np.zeros((H, W, 2), np.float32) if prev is None else \
            cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 15, 3, 5, 1.2, 0)
        prev = g
        ff = frame_flow_features(flow)
        r = model.predict(fr, conf=0.25, classes=[0], imgsz=imgsz, verbose=False)[0]
        n = len(r.boxes) if r.boxes is not None else 0
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2); mv = mag[mag > 0.5]
        ts.append(round(si * dt, 2)); count.append(n)
        feff.append(round(ff["flow_efficiency"], 3)); cflow.append(round(ff["counterflow"], 3))
        divg.append(round(ff["divergence_abs"], 3)); curl.append(round(ff["curl_abs"], 3))
        spd.append(round(ff["speed_mean"], 3)); vvar.append(float(mv.var()) if mv.size else 0.0)
        grids.append(ff["grids"])
        if si % 25 == 0:
            print("[dash]   %d t=%.1fs ppl=%d feff=%.2f cflow=%.2f" % (si, si * dt, n, ff["flow_efficiency"], ff["counterflow"]), flush=True)
        i += 1; si += 1
    cap.release()
    ts = np.array(ts); count = np.array(count, float); vvar = np.array(vvar)
    rho = count / max(count.max(), 1.0)
    risk = smooth(rho * vvar, 5); risk = risk / (risk.max() or 1.0)
    spn = smooth(spd, 5); spn = spn / (spn.max() or 1.0)
    base = np.median(spn[spn < np.percentile(spn, 60)]) if (spn < np.percentile(spn, 60)).any() else 0.0
    gt = spn > (base + 0.35); eps = episodes(gt, 2)
    res = []
    for s, e in eps:
        ot = float(ts[s]); w0 = max(0, s - 8); seg = np.where(risk[w0:e] >= 0.25)[0]
        at = float(ts[w0 + seg[0]]) if len(seg) else None
        res.append({"onset": round(ot, 1), "alarm": round(at, 1) if at is not None else None,
                    "lead": round(ot - at, 1) if at is not None else None})
    peak = int(np.argmax(risk)) if len(risk) else 0
    return {"video": os.path.basename(avi), "source_type": source_type, "fps": round(fps, 1),
            "sampled_frames": int(len(ts)), "step": step, "duration_s": round(float(ts[-1]), 1) if len(ts) else 0,
            "ts": ts.tolist(), "count": count.astype(int).tolist(),
            "flow_efficiency": feff, "counterflow": cflow, "divergence": divg, "curl": curl,
            "speed": spd, "risk": [round(float(x), 3) for x in risk],
            "episodes": res, "peak_t": round(float(ts[peak]), 1) if len(ts) else 0,
            "peak_people": int(count[peak]) if len(count) else 0,
            "peak_grids": grids[peak] if grids else {},
            "summary": {"peak_people": int(count.max()) if len(count) else 0,
                        "mean_count": round(float(count.mean()), 1) if len(count) else 0,
                        "mean_flow_eff": round(float(np.mean(feff)), 3) if feff else 0,
                        "max_counterflow": round(float(np.max(cflow)), 3) if cflow else 0,
                        "max_curl": round(float(np.max(curl)), 3) if curl else 0,
                        "n_episodes": len(res)}}


def build_html(data_by_type):
    return HTML_TEMPLATE.replace("__DATA__", json.dumps(data_by_type, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True, help="type:path (e.g. cctv:/content/umn_all.avi)")
    ap.add_argument("--weights", default="/content/drive/MyDrive/flowsight_ckpt/chunk05_best.pt")
    ap.add_argument("--out", default="/content/drive/MyDrive/flowsight_demo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {}
    for spec in a.clip:
        st, path = spec.split(":", 1)
        data[st] = run_clip(path, st, a.weights)
        json.dump(data[st], open("%s/dash_%s.json" % (a.out, st), "w"), default=float)
        s = data[st]["summary"]
        print("[dash] DONE %s: peak=%dppl mean_feff=%.2f max_cflow=%.2f eps=%d" %
              (st, s["peak_people"], s["mean_flow_eff"], s["max_counterflow"], s["n_episodes"]), flush=True)
    html = build_html(data)
    hp = "%s/flowsight_dashboard.html" % a.out
    open(hp, "w").write(html)
    print("=== DASHBOARD_HTML -> %s (%d bytes) ===" % (hp, len(html)), flush=True)
    try:
        from google.colab import files
        files.download(hp)
        print("[dash] files.download triggered (check your Downloads)", flush=True)
    except Exception as e:
        print("[dash] (no colab download: %s)" % e, flush=True)


HTML_TEMPLATE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlowSight — Crowd-Movement Analysis</title>
<style>
:root{--bg:#0b0f17;--panel:#131a26;--panel2:#0f1623;--ink:#e6edf6;--mut:#8aa0b8;--line:#22304a;
--cyan:#22d3ee;--teal:#2dd4bf;--amber:#f59e0b;--red:#ef4444;--grn:#34d399;--vio:#a78bfa;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:22px}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px}
h1{font-size:20px;margin:0;letter-spacing:.2px}
.sub{color:var(--mut);font-size:12.5px}
.badge{margin-left:auto;display:flex;gap:8px;align-items:center}
.tg{display:inline-flex;background:var(--panel2);border:1px solid var(--line);border-radius:999px;overflow:hidden}
.tg button{background:transparent;color:var(--mut);border:0;padding:7px 16px;font-weight:600;cursor:pointer}
.tg button.on{background:var(--cyan);color:#06222a}
.note{color:var(--mut);font-size:12px;margin:10px 0 16px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:700}.kpi .l{color:var(--mut);font-size:12px;margin-top:2px}
.kpi .s{font-size:11px;color:var(--mut);margin-top:3px}
.tabs{display:flex;gap:6px;margin:18px 0 12px;border-bottom:1px solid var(--line)}
.tabs button{background:transparent;border:0;color:var(--mut);padding:9px 14px;font-weight:600;cursor:pointer;border-bottom:2px solid transparent}
.tabs button.on{color:var(--ink);border-bottom-color:var(--cyan)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.card h3{margin:0 0 4px;font-size:14px}.card .cap{color:var(--mut);font-size:12px;margin-bottom:10px}
.row{display:grid;gap:14px}.row.g3{grid-template-columns:repeat(3,1fr)}.row.g2{grid-template-columns:1fr 1fr}
svg{display:block;width:100%;height:auto}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--mut);font-size:12px;margin-top:6px}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.hl{color:var(--amber)}.warn{background:#1c1407;border:1px solid #3a2a08;border-radius:10px;padding:10px 12px;color:#f1c97a;font-size:12.5px}
footer{color:var(--mut);font-size:11.5px;margin-top:18px;border-top:1px solid var(--line);padding-top:10px}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}.row.g3,.row.g2{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<header>
  <div><h1>FlowSight · 군중 움직임 분석 <span class="sub">Crowd-Movement Analysis &amp; Interpretation</span></h1>
  <div class="sub">입력원 무관 분석 — 안전은 한 렌즈. <span id="meta"></span></div></div>
  <div class="badge"><span class="sub">소스</span><div class="tg" id="srcTg"></div></div>
</header>
<div class="kpis" id="kpis"></div>
<div class="tabs" id="tabs">
  <button data-t="flow" class="on">흐름 분석 Flow</button>
  <button data-t="density">밀도·점유 Density</button>
  <button data-t="safety">안전·이상 Safety</button>
</div>
<div id="view"></div>
<footer>FlowSight 연구 프리뷰 · 검출 FT-2(+SAHI) · dense optical-flow 흐름특징 · 비평면 3D 측위(H1).
밀도·속도는 정규화된 상대값(메트릭 캘리브 전). 안전 경보는 고recall·저정밀(정제 예정).</footer>
</div>
<script>
const DATA = __DATA__;
const SRC = Object.keys(DATA);
let cur = SRC[0], tab = "flow";
const $ = s => document.querySelector(s);
const C = {cyan:"#22d3ee",teal:"#2dd4bf",amber:"#f59e0b",red:"#ef4444",grn:"#34d399",vio:"#a78bfa",mut:"#8aa0b8"};
const pct = x => Math.round(x*100);

function svgLine(series, ts, o){o=o||{};const W=o.w||720,H=o.h||190,P=34;
  const xs=ts,xmin=xs[0]||0,xmax=xs[xs.length-1]||1;
  const ymax=o.ymax!=null?o.ymax:Math.max(0.0001,...series.flatMap(s=>s.data));
  const X=v=>P+(v-xmin)/((xmax-xmin)||1)*(W-P-8);
  const Y=v=>H-22-(v/ (ymax||1))*(H-22-8);
  let g=`<svg viewBox="0 0 ${W} ${H}">`;
  for(let i=0;i<=4;i++){const yy=8+i*(H-30)/4;g+=`<line x1="${P}" y1="${yy}" x2="${W-8}" y2="${yy}" stroke="#22304a" stroke-width="1"/>`;}
  // x ticks
  for(let i=0;i<=5;i++){const t=xmin+(xmax-xmin)*i/5;g+=`<text x="${X(t)}" y="${H-6}" fill="${C.mut}" font-size="10" text-anchor="middle">${t.toFixed(0)}s</text>`;}
  g+=`<text x="6" y="12" fill="${C.mut}" font-size="10">${(o.ylab||"")}</text>`;
  // episode shading (safety)
  (o.bands||[]).forEach(b=>{g+=`<rect x="${X(b[0])}" y="8" width="${Math.max(2,X(b[1])-X(b[0]))}" height="${H-30}" fill="#ef4444" opacity="0.10"/>`;});
  (o.vlines||[]).forEach(v=>{g+=`<line x1="${X(v.x)}" y1="8" x2="${X(v.x)}" y2="${H-22}" stroke="${v.c}" stroke-width="1" opacity="0.8"/>`;});
  series.forEach(s=>{let d="";xs.forEach((t,i)=>{d+=(i?"L":"M")+X(t).toFixed(1)+" "+Y(s.data[i]).toFixed(1)+" ";});
    g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="1.8"/>`;});
  return g+`</svg>`;
}
function svgHeat(grid,o){o=o||{};const gh=grid.length,gw=grid[0].length,cell=o.cell||26,W=gw*cell,H=gh*cell;
  let m=0;grid.forEach(r=>r.forEach(v=>m=Math.max(m,Math.abs(v))));m=m||1;
  const col=v=>{if(o.seq){const t=Math.min(1,Math.abs(v)/m);const r=Math.round(13+t*9),g=Math.round(26+t*186),b=Math.round(38+t*150);return `rgb(${r},${g},${b})`;}
    const t=Math.max(-1,Math.min(1,v/m));if(t>=0){return `rgba(239,68,68,${t.toFixed(2)})`;}return `rgba(34,211,238,${(-t).toFixed(2)})`;};
  let g=`<svg viewBox="0 0 ${W} ${H}">`;
  for(let r=0;r<gh;r++)for(let c=0;c<gw;c++){g+=`<rect x="${c*cell}" y="${r*cell}" width="${cell-1}" height="${cell-1}" fill="${col(grid[r][c])}"/>`;}
  return g+`</svg>`;
}
function kpi(v,l,s){return `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div><div class="s">${s||""}</div></div>`;}

function render(){
  const d=DATA[cur];
  $("#meta").textContent = `· ${d.video} · ${d.duration_s}s · ${d.sampled_frames} 샘플 @ step ${d.step}`;
  // source toggle
  $("#srcTg").innerHTML = SRC.map(s=>`<button data-s="${s}" class="${s==cur?'on':''}">${s.toUpperCase()}</button>`).join("");
  $("#srcTg").querySelectorAll("button").forEach(b=>b.onclick=()=>{cur=b.dataset.s;render();});
  // kpis
  const su=d.summary;
  $("#kpis").innerHTML =
    kpi(su.peak_people, "최대 인원", "peak / 평균 "+su.mean_count) +
    kpi(pct(su.mean_flow_eff)+"%", "평균 흐름효율", "1=정렬, 0=혼잡") +
    kpi(pct(su.max_counterflow)+"%", "최대 역류비", "주류 반대 이동") +
    kpi(su.n_episodes, "이상 에피소드", "안전 렌즈");
  // tabs
  $("#tabs").querySelectorAll("button").forEach(b=>{b.classList.toggle("on",b.dataset.t==tab);b.onclick=()=>{tab=b.dataset.t;render();};});
  const v=$("#view"); let h="";
  if(tab=="flow"){
    h+=`<div class="card"><h3>흐름 동역학 (시간축)</h3><div class="cap">흐름효율(정렬도)과 역류비 — 충돌·정체 구조를 정량화</div>`+
       svgLine([{data:d.flow_efficiency.map(x=>x*100),c:C.teal},{data:d.counterflow.map(x=>x*100),c:C.amber}],d.ts,{ymax:100,ylab:"%"})+
       `<div class="legend"><span><i class="dot" style="background:${C.teal}"></i>흐름효율 %</span><span><i class="dot" style="background:${C.amber}"></i>역류비 %</span></div></div>`;
    const gr=d.peak_grids||{};
    if(gr.div){h+=`<div class="row g3">
      <div class="card"><h3>Divergence ∇·v</h3><div class="cap">t=${d.peak_t}s · <span style="color:${C.red}">확산/이탈</span> vs <span style="color:${C.cyan}">쏠림/수렴</span></div>${svgHeat(gr.div)}</div>
      <div class="card"><h3>Curl ∇×v (와류)</h3><div class="cap">t=${d.peak_t}s · 회전·소용돌이 강도</div>${svgHeat(gr.curl)}</div>
      <div class="card"><h3>Speed |v|</h3><div class="cap">t=${d.peak_t}s · 이동 속도장</div>${svgHeat(gr.speed,{seq:1})}</div></div>`;}
    h+=`<div class="card"><h3>와류·확산 (시간축)</h3><div class="cap">|curl|, |divergence| 평균 — 비정상 흐름 패턴</div>`+
       svgLine([{data:d.curl,c:C.vio},{data:d.divergence,c:C.cyan}],d.ts,{ylab:""})+
       `<div class="legend"><span><i class="dot" style="background:${C.vio}"></i>|curl|</span><span><i class="dot" style="background:${C.cyan}"></i>|divergence|</span></div></div>`;
  }else if(tab=="density"){
    h+=`<div class="card"><h3>인원·밀도 (시간축)</h3><div class="cap">검출 인원수 — 점유·혼잡 추이 (밀도는 정규화 상대값)</div>`+
       svgLine([{data:d.count,c:C.cyan}],d.ts,{ylab:"명"})+
       `<div class="legend"><span><i class="dot" style="background:${C.cyan}"></i>검출 인원</span></div></div>`;
    const gr=d.peak_grids||{};
    if(gr.speed){h+=`<div class="row g2">
      <div class="card"><h3>이동 밀집 맵</h3><div class="cap">t=${d.peak_t}s · 속도 가중 활동 밀집</div>${svgHeat(gr.speed,{seq:1})}</div>
      <div class="card"><h3>요약</h3><div class="cap">현 클립</div>
      <table><tr><th>지표</th><th>값</th></tr>
      <tr><td>최대 인원</td><td>${d.summary.peak_people}</td></tr>
      <tr><td>평균 인원</td><td>${d.summary.mean_count}</td></tr>
      <tr><td>피크 시각</td><td>${d.peak_t}s</td></tr>
      <tr><td>평균 흐름효율</td><td>${pct(d.summary.mean_flow_eff)}%</td></tr></table></div></div>`;}
  }else{
    const bands=d.episodes.map(e=>[e.onset, e.alarm!=null?Math.max(e.onset,e.alarm)+3:e.onset+3]);
    const vlines=[].concat(...d.episodes.map(e=>[{x:e.onset,c:C.mut},e.alarm!=null?{x:e.alarm,c:C.grn}:null].filter(Boolean)));
    h+=`<div class="card"><h3>안전 — 흐름-압력 위험 타임라인</h3><div class="cap">P=ρ·Var(v) 정규화 · 빨강 음영=이상 에피소드 · 초록선=경보 · 회색선=onset</div>`+
       svgLine([{data:d.risk,c:C.red}],d.ts,{ymax:1,ylab:"risk",bands:bands,vlines:vlines})+
       `<div class="legend"><span><i class="dot" style="background:${C.red}"></i>위험 P</span><span><i class="dot" style="background:${C.grn}"></i>경보</span><span><i class="dot" style="background:${C.mut}"></i>onset</span></div></div>`;
    let rows=d.episodes.map((e,i)=>`<tr><td>#${i+1}</td><td>${e.onset}</td><td>${e.alarm==null?'—':e.alarm}</td><td>${e.lead==null?'—':(e.lead>=0?'+':'')+e.lead}</td></tr>`).join("");
    h+=`<div class="card"><h3>이상 에피소드 (${d.episodes.length})</h3><div class="cap">경보가 onset을 선행(lead&gt;0)하면 조기탐지</div>
      <table><tr><th>#</th><th>onset(s)</th><th>alarm(s)</th><th>lead(s)</th></tr>${rows||'<tr><td colspan=4>—</td></tr>'}</table></div>`;
    h+=`<div class="warn">⚠ 정직한 한계: 이 경보는 <b>고recall·저정밀</b>(스윕상 recall≈1.0, precision 0.12~0.28) — 잘 잡지만 과발화. 에피소드 병합/히스테리시스로 정제 예정.</div>`;
  }
  v.innerHTML=h;
}
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
