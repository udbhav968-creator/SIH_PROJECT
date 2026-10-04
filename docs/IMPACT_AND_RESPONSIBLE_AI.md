# ROAD-SHIELD — Impact, Responsible AI, and the pilot

What the project is for, how it holds itself to account, and what a first deployment
would measure. Every statistic is sourced; every cost is an editable assumption, labelled
as one.

---

## 1. The problem, in official numbers

| Figure (India, calendar year 2022) | Value | Source |
|---|---|---|
| Road accidents, all causes | 4,61,312 | MoRTH, *Road Accidents in India 2022* (PIB release) |
| Road deaths, all causes | 1,68,491 | same |
| Accidents attributed to potholes | 4,446 | MoRTH data, as reported by FACTLY |
| Deaths attributed to potholes | 1,856 | same |
| Pothole deaths per year, 2013–2022 average | ~2,342 | same |

Pothole deaths are a small share of all road deaths, and they are the share an inspection
system can act on directly: a pothole is a fixed, findable object. The rest of the case is
money and trust — repairs that are billed twice, or billed and never done.

Sources: [PIB — Road Accidents in India 2022](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=1973295) ·
[FACTLY — pothole accidents and deaths, 2013–2022](https://factly.in/data-fatality-rate-in-accidents-on-potholed-roads-increased-from-0-27-in-2013-to-0-42-in-2022/)

## 2. What changes when buses do the inspecting

| Today | With ROAD-SHIELD | Evidence in this repository |
|---|---|---|
| Roads inspected by people, periodically | every bus route inspected every day it runs | design (capacity estimate in `docs/SYSTEM_DESIGN.md`) |
| A defect is a complaint or an inspector's note | a defect is a record: class, outline, ground area, depth interval, cost range, location | `/api/v1/pipeline/deep-audit` |
| The same pothole reported (and billed) more than once | reports of the same class within 8 m merge into one defect; every sighting kept as evidence | `pipeline/fleet_deduplication_engine.py`, tests |
| A bill can be edited after issue | the work order is sealed with SHA-256; one changed rupee breaks the seal | `/api/v1/dispatch/verify-seal`, tests |
| "Repaired" is a claim | the repair photograph is compared with the original; a recycled photo is rejected | `/api/v1/audit/verify-repair` |

None of these outcomes has been measured on a real fleet yet. The pilot (section 5) is how
they get measured.

## 3. Unit economics (assumptions, not quotes)

The `/impact` page has a calculator. Its defaults are **placeholders to be replaced with
real quotes**:

| Input | Placeholder | Why it matters |
|---|---|---|
| Edge box per bus (camera, compute, IMU, 4G modem) | editable | one-time; the classifier runs on a CPU at 7.2 ms per image, so no GPU is needed on the bus |
| Box lifetime | editable | amortises the box per month |
| Mobile data per bus per month | ~3.8 GB (from the capacity estimate) | ~5% of sampled frames are uploaded |
| Price per GB | editable | |
| Cloud compute per bus per month | editable | ~20–40 cores for 5,000 buses in the design estimate |
| Distance per bus per day | ~170 km | turns cost per bus into cost per km surveyed |

The output is **cost per bus per month** and **cost per kilometre surveyed**, which is the
number a road authority can compare with what it spends on inspection today.

## 4. Responsible AI

Mapped onto Microsoft's six Responsible AI principles. Each row says what exists in the
code, where the evidence is, and what is still missing.

| Principle | What the system does | Evidence | Gap |
|---|---|---|---|
| **Fairness** | Makes no decision about a person; no demographic inference. The real fairness risk is geographic: roads without bus routes are not inspected, and models trained on one country underperform in another (this project measured 33% on Indian roads before Indian data was added, 92.3% after). | `checkpoints/indian_roads_eval_report.json` | No per-city or per-region evaluation yet; coverage of non-bus roads needs another source |
| **Reliability & safety** | Held-out evaluation split by photograph; models chosen on validation before the test set is scored; every deep model has a classical fallback; missing inputs are a 400, never a default; 200+ automated tests | `scripts/select_vision_model.py`, `tests/`, `/api/v1/models/served` | No field validation of depth, PCI or deterioration; no bus footage yet |
| **Privacy & security** | People and number plates blurred before an image is shared; no identity stored; work orders sealed; optional API key on write endpoints; static files served only from an allow-list | `models/privacy_redactor.py`, `api/server.py` | Blur recall not measured; on-bus blurring, TLS and device certificates are designed, not built |
| **Inclusiveness** | Runs on an ordinary CPU (no GPU, no PyTorch at inference), so it fits low-cost hardware; any camera works through a calibration profile; the site works on a phone and without map tiles | `models/camera_calibration.py`, `web/` | Interface is English-only |
| **Transparency** | Every number labelled measured / estimate / range; model cards with per-class scores including the weak classes; the whole-frame classifier's opinion shown beside the measured result; OpenAPI contract; a claims registry that lists withdrawn claims | `/models`, `/architecture`, `/api-docs`, `checkpoints/claims.json` | — |
| **Accountability** | Each served model is identified by a SHA-256 checksum and the rule that chose it; each work order is sealed and traceable to its sightings; a human authority issues and approves repairs — the system recommends, it does not dispatch on its own | `models/served_report.py`, `pipeline/defect_store.py` | No formal governance process with a road authority yet |

Source for the principles: [Microsoft Learn — What is Responsible AI](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai?view=azureml-api-2)

## 5. The 90-day pilot

One route, about 20 buses, run in **shadow mode**: the system reports, people decide, and
its reports are compared with a manual survey of the same road.

| Weeks | Work | Output |
|---|---|---|
| 0–2 | Install edge units on ~20 buses on one route; calibrate each camera with a checkerboard | calibration profiles; first footage |
| 3–6 | Collect footage and IMU; label ~1,000 bus-camera frames; retrain and re-select models on bus data | the first models measured on bus footage |
| 7–10 | Shadow mode: the system reports, inspectors survey the same road independently | paired system-vs-inspector records |
| 11–13 | Measure and report | the KPI table below, with confidence intervals |

| KPI | How it is measured |
|---|---|
| Detection precision and recall | system defects vs the inspectors' survey of the same stretch |
| Area and depth error | system area vs tape-measured area; depth interval vs measured depth |
| Time from first sighting to work order | ledger timestamps |
| Duplicate reports merged | ledger: merged sightings / all sightings |
| Repair photos rejected as recycled | repair-verification log |
| Cost per km surveyed | the section 3 calculator, with real quotes |

## 6. Azure deployment mapping (designed)

How the designed architecture maps onto Azure services. **Not deployed**; the current
engine is cloud-agnostic and runs anywhere Docker runs.

| Component | Azure service |
|---|---|
| On-bus inference and store-and-forward | Azure IoT Edge on the bus unit, running the ONNX models |
| Device identity and telemetry ingest | Azure IoT Hub |
| Public API gateway (auth, rate limits) | Azure API Management |
| Queue between ingest and inference | Azure Event Hubs or Service Bus |
| Inference workers, scaled on queue depth | Azure Container Apps or AKS with KEDA |
| Frames (7-day retention, evidence kept) | Azure Blob Storage with lifecycle management |
| Ledger (defects, sightings, work orders) | Azure Database for PostgreSQL – Flexible Server with PostGIS |
| Maps and geocoding | Azure Maps |
| Training, model registry, Responsible AI dashboard | Azure Machine Learning |
| Logs, metrics, traces | Azure Monitor and Application Insights |
| Secrets and keys | Azure Key Vault |
| Staff sign-in for the authority and contractor portals | Microsoft Entra ID |
