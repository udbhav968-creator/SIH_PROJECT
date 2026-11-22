"""Builds web/detect_everything.html -- a self-contained frontend for the
detect-everything pipeline. It runs LIVE against /api/v1/detect/everything when
served by the engine, and falls back to a gallery of REAL captured model outputs
(embedded here, not fabricated) when no engine is reachable."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED = os.environ.get("EMBED_JSON", "/tmp/demo/embed.json")
OUT = os.path.join(ROOT, "web", "detect_everything.html")

demo = json.load(open(EMBED)) if os.path.exists(EMBED) else []

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>ROAD-SHIELD — Detect Everything</title>
<style>
  :root{
    color-scheme: light dark;
    --bg:#f4f6fb; --surface:#ffffff; --surface-2:#eef1f8; --ink:#141925; --muted:#5b6678;
    --line:#dde3ef; --brand:#2f6df0; --brand-ink:#ffffff; --ok:#10b981; --warn:#f59e0b;
    --cand:#ef4444; --shadow:0 1px 3px rgba(20,25,37,.08),0 8px 30px rgba(20,25,37,.06);
    --r:16px;
    padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
    box-sizing:border-box;
  }
  @media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
    --bg:#0c0f16; --surface:#141a24; --surface-2:#1b222e; --ink:#eef2fb; --muted:#9aa6bb;
    --line:#26303f; --brand:#5b8cff; --shadow:0 1px 3px rgba(0,0,0,.4),0 10px 40px rgba(0,0,0,.35);
  }}
  :root[data-theme="dark"]{
    --bg:#0c0f16; --surface:#141a24; --surface-2:#1b222e; --ink:#eef2fb; --muted:#9aa6bb;
    --line:#26303f; --brand:#5b8cff; --shadow:0 1px 3px rgba(0,0,0,.4),0 10px 40px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Inter,Arial,sans-serif;
    -webkit-font-smoothing:antialiased}
  a{color:var(--brand)}
  .wrap{max-width:1080px;margin:0 auto;padding:22px 18px 60px}
  header.hero{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px}
  .logo{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,var(--brand),#8b5cf6);
    display:grid;place-items:center;font-size:22px;box-shadow:var(--shadow)}
  h1{font-size:21px;margin:0;letter-spacing:-.02em}
  .sub{color:var(--muted);font-size:13px;margin-top:1px}
  .pill{margin-left:auto;display:inline-flex;align-items:center;gap:7px;padding:6px 12px;border-radius:999px;
    background:var(--surface);border:1px solid var(--line);font-size:12.5px;font-weight:600;box-shadow:var(--shadow)}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
  .dot.live{background:var(--ok)} .dot.demo{background:var(--warn)}
  .note{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--warn);
    border-radius:12px;padding:11px 14px;color:var(--muted);font-size:13px;margin:16px 0}
  .note b{color:var(--ink)}
  .grid{display:grid;grid-template-columns:1fr;gap:18px;margin-top:8px}
  @media(min-width:860px){.grid{grid-template-columns:1.25fr 1fr}}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
  .card .hd{padding:13px 16px;border-bottom:1px solid var(--line);font-weight:650;font-size:14px;display:flex;align-items:center;gap:8px}
  .card .bd{padding:16px}
  /* dropzone */
  #drop{border:2px dashed var(--line);border-radius:14px;padding:26px 18px;text-align:center;cursor:pointer;
    transition:.15s;background:var(--surface-2)}
  #drop.drag{border-color:var(--brand);background:color-mix(in srgb,var(--brand) 10%,var(--surface-2))}
  #drop .big{font-size:30px}
  #drop .t{font-weight:650;margin-top:6px}
  #drop .s{color:var(--muted);font-size:12.5px;margin-top:3px}
  .controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;font-size:13px;color:var(--muted)}
  .controls label{display:flex;align-items:center;gap:7px}
  input[type=range]{accent-color:var(--brand)}
  .switch{position:relative;width:38px;height:22px}
  .switch input{display:none}
  .slider{position:absolute;inset:0;background:var(--line);border-radius:999px;transition:.15s}
  .slider:before{content:"";position:absolute;width:16px;height:16px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.15s}
  .switch input:checked+.slider{background:var(--brand)}
  .switch input:checked+.slider:before{transform:translateX(16px)}
  button.go{margin-top:14px;width:100%;padding:11px 16px;border:0;border-radius:12px;background:var(--brand);
    color:var(--brand-ink);font-weight:700;font-size:14px;cursor:pointer}
  button.go:disabled{opacity:.5;cursor:default}
  /* result image */
  .imgwrap{position:relative;background:#000;border-radius:12px;overflow:hidden;min-height:120px}
  .imgwrap img{display:block;width:100%;height:auto}
  .vbadge{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.62);color:#fff;font-size:11.5px;
    padding:4px 9px;border-radius:999px;font-weight:600;backdrop-filter:blur(4px)}
  .summary{margin:12px 0 2px;font-size:15px;font-weight:600}
  /* counts dashboard */
  .counts{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:10px;margin-top:4px}
  .chip{background:var(--surface-2);border:1px solid var(--line);border-radius:12px;padding:10px;text-align:center}
  .chip .n{font-size:22px;font-weight:800;letter-spacing:-.02em}
  .chip .l{font-size:11.5px;color:var(--muted);margin-top:1px}
  .chip .e{font-size:16px}
  .chip.cand{border-style:dashed;border-color:var(--cand)}
  /* detections list */
  .dets{margin-top:4px;max-height:280px;overflow:auto}
  .det{display:flex;align-items:center;gap:10px;padding:8px 4px;border-bottom:1px solid var(--line);font-size:13px}
  .det:last-child{border-bottom:0}
  .swatch{width:12px;height:12px;border-radius:3px;flex:0 0 auto}
  .det .nm{font-weight:600}
  .det .cf{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums}
  .tag{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:999px;text-transform:uppercase;letter-spacing:.03em}
  .tag.v{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
  .tag.c{background:color-mix(in srgb,var(--cand) 16%,transparent);color:var(--cand)}
  /* gallery */
  .gal{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:4px}
  .gal button{border:1px solid var(--line);background:var(--surface-2);border-radius:12px;padding:0;cursor:pointer;overflow:hidden;text-align:left}
  .gal img{width:100%;height:74px;object-fit:cover;display:block}
  .gal .c{padding:7px 9px;font-size:12px;font-weight:600}
  .gal button.on{border-color:var(--brand);box-shadow:0 0 0 2px color-mix(in srgb,var(--brand) 35%,transparent)}
  footer{margin-top:26px;color:var(--muted);font-size:12.5px;line-height:1.7}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--muted)}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .ln{width:22px;height:0;border-top:3px solid var(--ok)}
  .ln.c{border-top:3px dashed var(--cand)}
  .err{color:var(--cand);font-size:13px;margin-top:8px;display:none}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="logo">🛣️</div>
    <div>
      <h1>ROAD-SHIELD — Detect Everything</h1>
      <div class="sub">Potholes · cracks · zebra crossings · people · vehicles · traffic lights · signs — boxed and counted.</div>
    </div>
    <span class="pill"><span id="statusDot" class="dot demo"></span><span id="statusText">captured demo</span></span>
  </header>

  <div class="note" id="honesty">
    <b>Honest by design.</b> People, vehicles and crossings come from pretrained detectors used as published.
    <b>Road-damage boxes are an unverified candidate</b> — the detector has not passed this repo’s verification gate,
    so every pothole/crack box is drawn <b>dashed with “?”</b> and labelled <i>candidate</i>. Counts are exact counts of the
    boxes drawn, not a validated accuracy. Nothing on this page is fabricated.
  </div>

  <div class="grid">
    <div class="card">
      <div class="hd">🔎 Analyse an image</div>
      <div class="bd">
        <div id="drop">
          <div class="big">📤</div>
          <div class="t">Drop an image here, or tap to choose</div>
          <div class="s">JPG / PNG · analysed by the live engine when it’s running</div>
          <input id="file" type="file" accept="image/*" hidden>
        </div>
        <div class="controls">
          <label>Confidence <input id="conf" type="range" min="0.05" max="0.6" step="0.05" value="0.25"><span id="confv">0.25</span></label>
          <label>Show candidate damage
            <span class="switch"><input id="cand" type="checkbox" checked><span class="slider"></span></span>
          </label>
        </div>
        <button class="go" id="go" disabled>Analyse</button>
        <div class="err" id="err"></div>
        <div class="legend">
          <span><i class="ln"></i> verified</span>
          <span><i class="ln c"></i> candidate (unverified)</span>
        </div>

        <div style="margin-top:16px;font-size:12.5px;color:var(--muted);font-weight:600">Or try a captured sample</div>
        <div class="gal" id="gal"></div>
      </div>
    </div>

    <div class="card">
      <div class="hd">📊 Result</div>
      <div class="bd">
        <div class="imgwrap"><span class="vbadge" id="vbadge"></span><img id="out" alt="annotated result"></div>
        <div class="summary" id="summary">—</div>
        <div class="counts" id="counts"></div>
        <div style="margin-top:14px;font-size:12.5px;color:var(--muted);font-weight:600">Detections</div>
        <div class="dets" id="dets"></div>
      </div>
    </div>
  </div>

  <footer>
    <div><b>Live mode</b> posts to <code>/api/v1/detect/everything</code> on this page’s origin — start the engine with
      <code>python -m api.server</code> and open this page from it. When no engine answers, the page shows captured real model output.</div>
    <div>To graduate the road-damage detector from <i>candidate</i> to <i>verified</i>, see <code>docs/DETECTION_PIPELINE.md</code>.</div>
  </footer>
</div>

<script>
const DEMO_DATA = /*__DEMO_DATA__*/[];

const META = {
  potholes:{e:"🕳️",l:"Potholes",c:"#fb7185"}, cracks:{e:"〰️",l:"Cracks",c:"#f59e0b"},
  zebra_crossings:{e:"🦓",l:"Zebra",c:"#06b6d4"}, road_arrows:{e:"➡️",l:"Arrows",c:"#22d3ee"},
  persons:{e:"🚶",l:"People",c:"#ec4899"}, cars:{e:"🚗",l:"Cars",c:"#10b981"},
  buses:{e:"🚌",l:"Buses",c:"#3b82f6"}, trucks:{e:"🚚",l:"Trucks",c:"#6366f1"},
  motorcycles:{e:"🏍️",l:"Motorbikes",c:"#8b5cf6"}, bicycles:{e:"🚲",l:"Bicycles",c:"#a855f7"},
  trains:{e:"🚆",l:"Trains",c:"#64748b"}, traffic_lights:{e:"🚦",l:"Signals",c:"#84cc16"},
  stop_signs:{e:"🛑",l:"Stop signs",c:"#f43f5e"}, other:{e:"❓",l:"Other",c:"#94a3b8"}
};
const CAND = new Set(["potholes","cracks"]);
const $=s=>document.querySelector(s);
let fileData=null;

function render(r){
  $("#out").src = r.image || r.annotated_image_base64 || "";
  $("#summary").textContent = r.summary || "No objects detected.";
  const v = r.damage_verification;
  const vb=$("#vbadge");
  vb.textContent = v==="verified" ? "road damage: verified"
                 : v==="candidate_unverified" ? "road damage: candidate (unverified)"
                 : "road damage: hidden";
  vb.style.display = v ? "block":"none";

  const counts=r.counts||{}; const cg=$("#counts"); cg.innerHTML="";
  Object.keys(META).forEach(k=>{
    const n=counts[k]||0; if(!n) return;
    const m=META[k]; const d=document.createElement("div");
    d.className="chip"+(CAND.has(k)?" cand":"");
    d.innerHTML=`<div class="e">${m.e}</div><div class="n">${n}</div><div class="l">${m.l}</div>`;
    cg.appendChild(d);
  });
  if(!cg.children.length){ cg.innerHTML=`<div class="chip"><div class="n">0</div><div class="l">objects</div></div>`; }

  const dl=$("#dets"); dl.innerHTML="";
  (r.detections||[]).sort((a,b)=>(b.confidence||0)-(a.confidence||0)).forEach(d=>{
    const cand = d.verification_status==="candidate_unverified";
    const row=document.createElement("div"); row.className="det";
    const colour = d.color || d.colour_hex || "#38bdf8";
    row.innerHTML=`<span class="swatch" style="background:${colour}"></span>
      <span class="nm">${d.class_name}</span>
      <span class="tag ${cand?'c':'v'}">${cand?'candidate':'verified'}</span>
      <span class="cf">${Math.round((d.confidence||0)*100)}%</span>`;
    dl.appendChild(row);
  });
  if(!dl.children.length) dl.innerHTML=`<div class="det"><span class="nm" style="color:var(--muted)">No detections.</span></div>`;
}

function setStatus(live){
  $("#statusDot").className="dot "+(live?"live":"demo");
  $("#statusText").textContent = live?"live engine":"captured demo";
}

// gallery of real captured outputs
function buildGallery(){
  const g=$("#gal"); g.innerHTML="";
  DEMO_DATA.forEach((s,i)=>{
    const b=document.createElement("button");
    b.innerHTML=`<img src="${s.image}" alt=""><div class="c">${s.title}</div>`;
    b.onclick=()=>{[...g.children].forEach(x=>x.classList.remove("on"));b.classList.add("on");render(s);setStatus(false);};
    g.appendChild(b);
  });
  if(DEMO_DATA.length){ g.firstChild.classList.add("on"); render(DEMO_DATA[0]); }
}

async function analyse(){
  if(!fileData) return;
  $("#err").style.display="none";
  $("#go").disabled=true; $("#go").textContent="Analysing…";
  try{
    const resp=await fetch("/api/v1/detect/everything",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image_base64:fileData,conf_threshold:parseFloat($("#conf").value),
        candidate_damage:$("#cand").checked,annotate:true})
    });
    if(!resp.ok) throw new Error("engine returned "+resp.status);
    const r=await resp.json();
    r.image=r.annotated_image_base64;
    render(r); setStatus(true);
  }catch(e){
    $("#err").textContent="Live engine not reachable ("+e.message+"). Start it with “python -m api.server” and open this page from it. Showing captured demo meanwhile.";
    $("#err").style.display="block"; setStatus(false);
  }finally{ $("#go").disabled=false; $("#go").textContent="Analyse"; }
}

function loadFile(f){
  const rd=new FileReader();
  rd.onload=()=>{ fileData=rd.result; $("#out").src=rd.result; $("#go").disabled=false;
    $("#summary").textContent="Ready — tap Analyse."; };
  rd.readAsDataURL(f);
}

const drop=$("#drop"),file=$("#file");
drop.onclick=()=>file.click();
file.onchange=e=>{ if(e.target.files[0]) loadFile(e.target.files[0]); };
["dragover","dragenter"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("drag");}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("drag");}));
drop.addEventListener("drop",e=>{ if(e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
$("#go").onclick=analyse;
$("#conf").oninput=e=>$("#confv").textContent=parseFloat(e.target.value).toFixed(2);

buildGallery();
// probe engine quietly to flip the status pill if it's live
fetch("/api/v1/ready").then(r=>{if(r.ok)setStatus(true);}).catch(()=>{});
</script>
</body>
</html>
"""

out_html = HTML.replace("/*__DEMO_DATA__*/[]", json.dumps(demo))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(out_html)
print(f"wrote {OUT} ({len(out_html)//1024} KB, {len(demo)} demo scenes)")
