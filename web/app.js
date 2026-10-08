/* ============================================================================
   Shared runtime for every ROAD-SHIELD page.

     1. One API client with consistent error handling, so a dead endpoint shows
        a message instead of a blank panel. Photograph analysis is routed to the
        live engine (web/config.js) when this site is the static deployment.
     2. The navigation, built once here rather than copy-pasted into every page:
        grouped on desktop, a menu panel on phones.
     3. The formatting helpers that decide how a number is PRESENTED - which on
        this system is a correctness question, not a cosmetic one. A measured
        area and an estimated depth must not look alike.
   ========================================================================= */

/* Endpoints that run a model on the request. On the static site (Vercel) these
   go to the live engine; everything else - the measured reports - is served by
   the site itself, so the pages work even while the engine is asleep. */
const ENGINE_PATHS = new Set([
  "/api/v1/pipeline/deep-audit", "/api/v1/vision/analyze-photo", "/api/v1/vision/analyze-custom-photo",
  "/api/v1/privacy/redact", "/api/v1/detect/vision", "/api/v1/vision/predict", "/api/v1/detect/objects",
  "/api/v1/pedestrian/detect", "/api/v1/telemetry/imu",
  "/api/v1/fleet/telemetry", "/api/v1/gis/map-data", "/api/v1/ledger/defects", "/api/v1/fleet/live",
  "/api/v1/priority/ranking", "/api/v1/priority/score", "/api/v1/live/stream", "/api/v1/fleet/report-defect",
]);

/* Endpoints that change stored state. When the engine locks them (public demo), the operator's key - typed
   once on the page, kept only for this browser tab - is sent with them, and only with them. */
const KEYED_PATHS = new Set(["/api/v1/fleet/report-defect", "/api/v1/dispatch/work-order"]);
function operatorKey() { try { return sessionStorage.getItem("roadShieldApiKey") || ""; } catch { return ""; } }
function setOperatorKey(k) { try { k ? sessionStorage.setItem("roadShieldApiKey", k) : sessionStorage.removeItem("roadShieldApiKey"); } catch {} }

const API = {
  // Empty means "same origin". ?api=http://host:8000 (remembered for the
  // session) points every request at one engine, for local testing.
  base: (new URLSearchParams(location.search).get("api")
         || sessionStorage.getItem("roadShieldApiBase") || ""),
  // The live engine for photograph analysis: ?engine=https://... (e.g. a laptop tunnel for a demo, remembered
  // for the session), else web/config.js. Reports keep coming from this site, so the pages still work - with
  // recorded results - when that engine is off.
  // Only engines of the kinds this project deploys are accepted from a link, so a shared link cannot send
  // a visitor's photographs to an arbitrary server.
  engineUrl: (() => {
    const allowed = (u) => /^https:\/\/[a-z0-9-]+\.(trycloudflare\.com|lhr\.life|hf\.space)\/?$/i.test(u)
                        || /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?\/?$/i.test(u);
    let q = new URLSearchParams(location.search).get("engine");
    if (q && !allowed(q)) q = null;
    if (q) { try { sessionStorage.setItem("roadShieldEngine", q); } catch {} }
    let saved = "";
    try { saved = sessionStorage.getItem("roadShieldEngine") || ""; } catch {}
    if (saved && !allowed(saved)) saved = "";
    return ((q || saved || window.ROAD_SHIELD_ENGINE_URL || "") + "").replace(/\/+$/, "");
  })(),
  siteIsStatic: null,          // true once /api/v1/health says this origin cannot run the models
  engineState: "unknown",      // unknown | online | starting | waking | offline | none
  _healthOnce: null,

  async get(path) { return this._go("GET", path); },
  /** Absolute URL for a streaming endpoint (EventSource cannot go through _go). */
  async urlFor(path) { return (await this.baseFor(path)) + path; },
  async post(path, body) { return this._go("POST", path, body); },

  /** Base URL for a path: the live engine for model endpoints on the static site, else this site. */
  async baseFor(path) {
    if (this.base || !this.engineUrl || !ENGINE_PATHS.has(path.split("?")[0])) return this.base;
    if (this.siteIsStatic === null && this._healthOnce) await this._healthOnce;
    return this.siteIsStatic ? this.engineUrl : this.base;
  },

  async _go(method, path, body) {
    const started = performance.now();
    const base = await this.baseFor(path);
    const toEngine = base && base === this.engineUrl;
    try {
      const headers = body ? { "Content-Type": "application/json" } : {};
      if (method === "POST" && KEYED_PATHS.has(path.split("?")[0]) && operatorKey()) headers["X-API-Key"] = operatorKey();
      const res = await fetch(base + path, {
        method,
        headers: Object.keys(headers).length ? headers : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      const text = await res.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; }
      catch { data = { error: "response was not JSON", raw: text.slice(0, 400) }; }
      const out = { ok: res.ok, status: res.status, data, ms: Math.round(performance.now() - started), engine: toEngine };
      if (toEngine && (!res.ok && (res.status >= 502 || data.raw !== undefined))) return this._waking(out);
      if (toEngine && res.ok) setEngineState("online");
      return out;
    } catch (err) {
      const out = { ok: false, status: 0, data: { error: String(err) }, ms: Math.round(performance.now() - started), engine: toEngine };
      return toEngine ? this._waking(out) : out;
    }
  },

  /** The engine did not answer like an engine: asleep (a Space sleeps when idle), still loading, or switched off. */
  _waking(out) {
    if (API.engineState !== "offline") setEngineState("waking");   // never "un-offline" it on a failed request
    out.status = 503;
    out.data = { error: "The live inference engine is waking up.", engine_state: "waking" };
    return out;
  },
};

/** True when a response means "no engine can run this here", in either form. */
function engineUnavailable(r) {
  if (r.ok) return false;
  if (r.data && r.data.engine_state === "waking") return "waking";
  if (r.status === 503 && /inference engine/i.test((r.data && r.data.error) || "")) return "absent";
  return false;
}

/* ---------------------------------------------------------------- icons -- */
/* A small line-icon set (24px grid, 1.8 stroke), inline so the site needs no icon font or CDN. */
const ICONS = {
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  film: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
  map: '<path d="m9 4-6 2.5v13L9 17l6 3 6-2.5v-13L15 7z"/><path d="M9 4v13M15 7v13"/>',
  clipboard: '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4.5V3h6v1.5M9 10h6M9 14h6M9 18h3"/>',
  chart: '<path d="M4 20V4M4 20h16"/><path d="M8 16v-4M12 16V8M16 16v-6"/>',
  database: '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/>',
  shield: '<path d="M12 3 5 6v5c0 4.5 3 8.3 7 10 4-1.7 7-5.5 7-10V6z"/><path d="m9 12 2 2 4-4"/>',
  receipt: '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6M9 16h4"/>',
  camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v4h16v-4"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  bolt: '<path d="M13 3 5 14h6l-1 7 8-11h-6z"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
};
function icon(name, size = 18) {
  return `<svg class="ico" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.info}</svg>`;
}
function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach(n => {
    if (!n.dataset.iconDone) { n.innerHTML = icon(n.dataset.icon, +(n.dataset.size || 20)); n.dataset.iconDone = "1"; }
  });
}

/* ------------------------------------------------------------------ nav -- */
const PAGES = [
  { href: "/",             id: "home",         label: "Overview",     group: "product" },
  { href: "/inspect",      id: "inspect",      label: "Inspection",   group: "product" },
  { href: "/video",        id: "video",        label: "Video",        group: "product" },
  { href: "/corridor",     id: "corridor",     label: "Road map",     group: "product" },
  { href: "/works",        id: "works",        label: "Works",        group: "product" },
  { href: "/models",       id: "models",       label: "Models",       group: "evidence", note: "accuracy, IoU, model card" },
  { href: "/data",         id: "data",         label: "Data",         group: "evidence", note: "datasets and lineage" },
  { href: "/system",       id: "system",       label: "System",       group: "evidence", note: "what is loaded, live" },
  { href: "/architecture", id: "architecture", label: "Architecture", group: "evidence", note: "claims and corrections" },
  { href: "/design",       id: "design",       label: "Design",       group: "evidence", note: "system design" },
  { href: "/impact",       id: "impact",       label: "Impact",       group: "evidence", note: "cost and coverage" },
  { href: "/api-docs",     id: "api-docs",     label: "API",          group: "evidence", note: "endpoints reference" },
];

function buildNav() {
  const current = document.body.dataset.page;
  const cur = (p) => (p.id === current ? ' aria-current="page"' : "");
  const product = PAGES.filter(p => p.group === "product");
  const evidence = PAGES.filter(p => p.group === "evidence");
  const evidenceActive = evidence.some(p => p.id === current);

  const skip = document.createElement("a");
  skip.className = "skip-link"; skip.href = "#main"; skip.textContent = "Skip to content";
  const main = document.querySelector(".wrap");
  if (main && !main.id) main.id = "main";

  const nav = document.createElement("nav");
  nav.className = "nav";
  nav.setAttribute("aria-label", "Main");
  nav.innerHTML =
    `<a class="brand" href="/" aria-label="ROAD-SHIELD home"><span class="brand-mark">${logoSvg()}</span>ROAD-SHIELD</a>` +
    `<div class="nav-links">` +
      product.map(p => `<a class="link" href="${p.href}"${cur(p)}>${p.label}</a>`).join("") +
      `<div class="menu">
         <button class="link menu-btn${evidenceActive ? " active" : ""}" aria-expanded="false" aria-haspopup="true">
           Evidence ${icon("chevron", 14)}</button>
         <div class="menu-panel" role="menu">
           ${evidence.map(p => `<a role="menuitem" href="${p.href}"${cur(p)}><strong>${p.label}</strong><span>${p.note}</span></a>`).join("")}
         </div>
       </div>` +
    `</div>` +
    `<span class="spacer"></span>` +
    `<span class="status-pill" id="healthPill" role="status"><span class="led"></span><span id="healthText">checking…</span></span>` +
    `<button class="nav-toggle" aria-label="Open menu" aria-expanded="false" aria-controls="navSheet">${icon("menu", 20)}</button>`;

  const sheet = document.createElement("div");
  sheet.className = "nav-sheet"; sheet.id = "navSheet"; sheet.hidden = true;
  sheet.innerHTML =
    `<div class="sheet-group"><div class="sheet-label">Product</div>${product.map(p => `<a href="${p.href}"${cur(p)}>${p.label}</a>`).join("")}</div>` +
    `<div class="sheet-group"><div class="sheet-label">Evidence</div>${evidence.map(p => `<a href="${p.href}"${cur(p)}>${p.label}<span>${p.note}</span></a>`).join("")}</div>`;

  document.body.prepend(sheet);
  document.body.prepend(nav);
  document.body.prepend(skip);

  // desktop dropdown
  const menu = nav.querySelector(".menu"), btn = nav.querySelector(".menu-btn");
  const setMenu = (open) => { menu.classList.toggle("open", open); btn.setAttribute("aria-expanded", String(open)); };
  btn.addEventListener("click", (e) => { e.stopPropagation(); setMenu(!menu.classList.contains("open")); });
  document.addEventListener("click", (e) => { if (!menu.contains(e.target)) setMenu(false); });

  // phone sheet
  const toggle = nav.querySelector(".nav-toggle");
  const setSheet = (open) => {
    sheet.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    toggle.innerHTML = icon(open ? "close" : "menu", 20);
    document.body.classList.toggle("sheet-open", open);
  };
  toggle.addEventListener("click", () => setSheet(sheet.hidden));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { setMenu(false); setSheet(false); } });
}

function logoSvg() {
  return `<svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 2.5 4 5.5v6c0 4.8 3.3 8.9 8 10 4.7-1.1 8-5.2 8-10v-6z" fill="var(--accent)" opacity=".18" stroke="var(--accent)" stroke-width="1.6"/>
    <path d="M10.2 17.5 11.2 7h1.6l1 10.5" fill="none" stroke="var(--accent)" stroke-width="1.6" stroke-linecap="round"/>
    <path d="M12 9.2v1.4M12 12.6v1.4" stroke="var(--white)" stroke-width="1.4" stroke-linecap="round"/></svg>`;
}

/* --------------------------------------------------------------- status -- */
function setEngineState(state) {
  API.engineState = state;
  renderPill();
  document.dispatchEvent(new CustomEvent("engine-state", { detail: state }));
}

let lastHealth = null;
function renderPill() {
  const pill = document.getElementById("healthPill"), text = document.getElementById("healthText");
  if (!pill) return;
  const h = lastHealth;
  const show = (main, detail) => {
    text.innerHTML = `${main}${detail ? `<span class="pill-detail"> · ${detail}</span>` : ""}`;
  };
  if (!h || !h.ok || h.data.status !== "ONLINE") {
    pill.className = "status-pill offline"; show("site offline"); return;
  }
  if (!API.siteIsStatic) {
    const backend = (h.data.models?.vision_distress_net || "").replace("LOADED (", "").replace(")", "");
    pill.className = "status-pill online";
    show("live engine", h.data.public_demo ? "public demo" : (backend || "models loaded"));
    text.title = JSON.stringify(h.data.models, null, 2);
    return;
  }
  const states = {
    none:     ["neutral", "recorded results", "no live engine connected"],
    unknown:  ["neutral", "connecting…", "live engine"],
    online:   ["online",  "live engine", API.engineDegraded ? "online · segmenter not loaded" : "online"],
    starting: ["waking",  "engine starting", "loading models"],
    waking:   ["waking",  API.engineSleeps ? "engine waking" : "connecting…", API.engineSleeps ? "about 1–2 min" : "live engine"],
    offline:  ["offline", "engine offline", "recorded results"],
  };
  const [cls, main, detail] = states[API.engineUrl ? API.engineState : "none"] || states.unknown;
  pill.className = "status-pill " + cls;
  show(main, detail);
  text.title = API.engineUrl ? `Engine: ${API.engineUrl}` : "This site serves measured results; set web/config.js to connect a live engine.";
}

async function pollHealth() {
  const h = await API.get("/api/v1/health");
  lastHealth = h;
  API.siteIsStatic = !!(h.ok && h.data.inference_available === false && !API.base);
  renderPill();
  if (API.siteIsStatic && API.engineUrl && !enginePolling) { enginePolling = true; pollEngine(); }
}

/* A Hugging Face Space sleeps and takes 1-2 minutes to wake, so silence there means "waking" for a while.
   A laptop behind a tunnel never sleeps: if it does not answer within ~20 s, it is off. */
API.engineSleeps = (() => { try { return /\.hf\.space$/i.test(new URL(API.engineUrl).hostname); } catch { return false; } })();

let engineFails = 0, enginePolling = false;
async function pollEngine() {
  let next = 10000;
  try {
    const res = await fetch(API.engineUrl + "/api/v1/ready", { cache: "no-store" });
    const data = await res.json().catch(() => null);
    // ready = every model a frame needs is loaded. A JSON 503 with the classifier loaded can still analyse
    // (areas fall back to box estimates, and the result says so); without it the engine is still loading.
    // Anything that is not JSON is the Space's own page while it wakes.
    if (data && (data.ready || (data.models && data.models.vision_classifier))) {
      engineFails = 0; API.engineDegraded = !data.ready; setEngineState("online"); next = 60000;
    } else if (data) {
      setEngineState("starting");                       // the engine answered: models still loading
    } else {
      throw new Error("not the engine");                // a Space's waking page, or a tunnel error page
    }
  } catch {
    engineFails += 1;
    const patience = API.engineSleeps ? 18 : 2;          // ~3 minutes for a Space, ~20 s for a laptop
    setEngineState(engineFails > patience ? "offline" : "waking");
  }
  setTimeout(pollEngine, next);
}

function buildFooter() {
  const f = document.createElement("footer");
  f.className = "site";
  f.innerHTML =
    `<div><strong>ROAD-SHIELD</strong> · SIH 2026 · SIH26124 · Bharat Electronics Limited</div>` +
    `<div class="foot-links">${PAGES.filter(p => p.group === "evidence").map(p => `<a href="${p.href}">${p.label}</a>`).join("")}</div>` +
    `<div class="mono small">every figure on this site is produced by code in this repository</div>`;
  document.body.appendChild(f);
}

/* ----------------------------------------------------------- formatting -- */
const fmt = {
  /** Indian-format rupees. */
  inr(v) {
    if (v == null || isNaN(v)) return "—";
    return "₹" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 });
  },
  num(v, dp = 2) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toFixed(dp);
  },
  pct(v, dp = 1) {
    if (v == null || isNaN(v)) return "—";
    return (Number(v) * 100).toFixed(dp) + "%";
  },
  when(unix) {
    if (!unix) return "—";
    return new Date(unix * 1000).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
  },

  /**
   * The badge that says whether a number is a measurement or an estimate.
   *
   * This exists because the system produces both and they must never be
   * displayed identically. An area from a segmentation mask on a calibrated
   * camera is measured. The same area from a bounding box on an assumed mount
   * is an estimate that can be an order of magnitude out, and a viewer has a
   * right to know which one they are reading before they sign a bill.
   */
  provenance(kind) {
    const map = {
      segmentation_mask: ["measured", "measured · mask"],
      bounding_box_estimate: ["estimate", "estimate · box"],
      calibrated: ["measured", "calibrated camera"],
      default: ["estimate", "assumed mount"],
    };
    const [cls, label] = map[kind] || ["neutral", kind || "unknown"];
    return `<span class="badge ${cls}">${label}</span>`;
  },
};

/** Render an interval as "central (low – high)". Used everywhere a depth or
 *  cost estimate appears, so a range is never silently collapsed to a point. */
function interval(central, low, high, unit = "") {
  if (central == null) return "—";
  if (low == null || high == null) return `${central}${unit}`;
  return `${central}${unit} <span class="muted small">(${low} – ${high}${unit})</span>`;
}

function setApiBase(url) {
  API.base = url || "";
  try { sessionStorage.setItem("roadShieldApiBase", API.base); } catch {}
  location.reload();
}
window.setApiBase = setApiBase;

function el(id) { return document.getElementById(id); }

function setLoading(node, on, text = "working…") {
  if (!node) return;
  node.innerHTML = on ? `<div class="card loading-card"><span class="spinner"></span><span>${text}</span></div>` : "";
}

function errorCard(message, hint) {
  return `<div class="card notice bad">
    <div class="label">could not load</div>
    <p class="small" style="margin:0">${message}</p>
    ${hint ? `<p class="small muted" style="margin:.5rem 0 0">${hint}</p>` : ""}
  </div>`;
}

document.addEventListener("DOMContentLoaded", () => {
  buildNav();
  buildFooter();
  hydrateIcons();
  API._healthOnce = pollHealth();
  setInterval(pollHealth, 30000);
});
