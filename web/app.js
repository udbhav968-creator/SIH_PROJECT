/* ============================================================================
   Shared runtime for every ROAD-SHIELD page.

   Three jobs:
     1. One API client with consistent error handling, so a dead endpoint shows
        a message instead of a blank panel.
     2. The navigation, built once here rather than copy-pasted into 8 files.
     3. The formatting helpers that decide how a number is PRESENTED - which on
        this system is a correctness question, not a cosmetic one. A measured
        area and an estimated depth must not look alike.
   ========================================================================= */

const API = {
  // Empty means "same origin". A deployment that serves only the site can be
  // pointed at a running engine with ?api=http://host:8000 or setApiBase(...),
  // which is remembered for the session.
  base: (new URLSearchParams(location.search).get("api")
         || sessionStorage.getItem("roadShieldApiBase") || ""),

  async get(path) {
    return this._go("GET", path);
  },
  async post(path, body) {
    return this._go("POST", path, body);
  },
  async _go(method, path, body) {
    const started = performance.now();
    try {
      const res = await fetch(this.base + path, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      const text = await res.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; }
      catch { data = { error: "response was not JSON", raw: text.slice(0, 400) }; }
      return { ok: res.ok, status: res.status, data, ms: Math.round(performance.now() - started) };
    } catch (err) {
      return { ok: false, status: 0, data: { error: String(err) }, ms: Math.round(performance.now() - started) };
    }
  },
};

/* ------------------------------------------------------------------ nav -- */
const PAGES = [
  { href: "/",         id: "home",     label: "Overview" },
  { href: "/inspect",  id: "inspect",  label: "Inspection" },
  { href: "/video",    id: "video",    label: "Video" },
  { href: "/corridor", id: "corridor", label: "Corridor" },
  { href: "/works",    id: "works",    label: "Works" },
  { href: "/models",   id: "models",   label: "Models" },
  { href: "/data",     id: "data",     label: "Data" },
  { href: "/system",   id: "system",   label: "System" },
  { href: "/architecture", id: "architecture", label: "Architecture" },
];

function buildNav() {
  const current = document.body.dataset.page;
  const nav = document.createElement("nav");
  nav.className = "nav";
  nav.innerHTML =
    `<a class="brand" href="/"><span class="dot"></span>ROAD-SHIELD</a>` +
    PAGES.map(p =>
      `<a class="link" href="${p.href}"${p.id === current ? ' aria-current="page"' : ""}>${p.label}</a>`
    ).join("") +
    `<span class="spacer"></span>` +
    `<span class="status-pill" id="healthPill"><span class="led"></span><span id="healthText">checking…</span></span>`;
  document.body.prepend(nav);
}

async function pollHealth() {
  const pill = document.getElementById("healthPill");
  const text = document.getElementById("healthText");
  if (!pill) return;
  const r = await API.get("/api/v1/health");
  if (r.ok && r.data.status === "ONLINE") {
    pill.className = "status-pill online";
    const backend = (r.data.models?.vision_distress_net || "").replace("LOADED (", "").replace(")", "");
    text.textContent = `online · ${backend || "models loaded"}`;
    text.title = JSON.stringify(r.data.models, null, 2);
  } else {
    pill.className = "status-pill offline";
    text.textContent = "engine offline";
  }
}

function buildFooter() {
  const f = document.createElement("footer");
  f.className = "site";
  f.innerHTML =
    `<span>ROAD-SHIELD · SIH 2026 · SIH26124 · Bharat Electronics Limited</span>` +
    `<span class="mono small">every figure on this site is produced by code in this repository</span>`;
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
  node.innerHTML = on ? `<p class="muted small"><span class="spinner"></span> ${text}</p>` : "";
}

function errorCard(message, hint) {
  return `<div class="card" style="border-color:rgba(251,113,133,.4)">
    <div class="label" style="color:var(--bad)">could not load</div>
    <p class="small" style="margin:0">${message}</p>
    ${hint ? `<p class="small muted" style="margin:.5rem 0 0">${hint}</p>` : ""}
  </div>`;
}

document.addEventListener("DOMContentLoaded", () => {
  buildNav();
  buildFooter();
  pollHealth();
  setInterval(pollHealth, 20000);
});
