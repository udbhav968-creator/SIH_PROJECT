# ROAD-SHIELD — System Design

Problem statement SIH26124 (Bharat Electronics Limited): use the public bus fleet as a mobile sensing network to
find, measure, cost and track road defects, and to stop the same defect being repaired — or billed — twice.

This document separates two things everywhere, because mixing them is how a design document misleads:

- **Implemented** — exists in this repository, runs, and is covered by tests or a measured report.
- **Designed** — the production architecture the implemented pieces slot into. Not built yet.

---

## 1. Requirements

### Functional

| # | Requirement | Status |
|---|---|---|
| F1 | Classify a road frame into 7 conditions (normal, crack, pothole, waterlogging, missing zebra crossing, missing divider, damaged sign) | Implemented |
| F2 | Outline the defect (pixel mask) and compute its ground area from camera geometry | Implemented |
| F3 | Give depth as an interval, cost as a MoRTH Section 500 range, PCI by ASTM D6433 | Implemented |
| F4 | Merge reports of the same defect class within 8 m into one defect; keep every raw sighting as evidence | Implemented (SQLite) |
| F5 | Issue a tamper-evident (SHA-256-sealed) work order; hold it when the location is unknown | Implemented |
| F6 | Verify a repair photograph is new (not a resubmitted old one) | Implemented |
| F7 | Blur people and number plates before an image leaves the system | Implemented (recall not measured) |
| F8 | Ingest dashcam video, sampling frames by distance travelled, not by time | Implemented |
| F9 | Fuse vision with accelerometer (IMU) evidence when a window is available | Implemented |
| F10 | On-bus edge inference with store-and-forward upload | Designed |
| F11 | Authority and contractor portals with an approval workflow | Designed |

### Non-functional

| Property | Target | Where it stands |
|---|---|---|
| Honesty of numbers | every figure labelled measured / estimate / range; nothing defaulted | Implemented (and tested) |
| Edge latency | < 50 ms per frame on a CPU | EfficientNet-B0 ONNX measured at 16.5 ms per image on CPU |
| Server latency | < 3 s per new defect, end to end | 1.7–2.6 s measured (single CPU core; 2-vCPU Colab VM) |
| Availability | no lost reports when the network drops | Designed: on-bus queue, idempotent upload |
| Privacy | no identifiable person or plate stored | blur implemented; on-device blur designed |
| Auditability | every number traceable to a report, every model to a checksum | claims registry + model registry implemented |
| Portability | no GPU and no PyTorch at inference | Implemented (ONNX Runtime on CPU) |

---

## 2. Capacity estimate

Assumptions are stated so the numbers can be redone for any fleet. They are planning estimates, not measurements.

| Quantity | Assumption / formula | Value |
|---|---|---|
| Fleet | a large city fleet | **5,000 buses** |
| Distance per bus per day | 14 h × ~12 km/h in traffic | ~170 km |
| Frames sampled per bus | one every 8 m of travel (the implemented default) | ~21,000 / day |
| Edge compute per bus | 21,000 × 16.5 ms | ~6 min of CPU per day |
| Frames uploaded | edge keeps non-normal frames + a 1% audit sample, ~5% | ~1,060 / bus / day |
| Upload per bus | 1,060 × ~120 KB | ~127 MB / day (~3.8 GB / month on 4G) |
| Fleet uploads | 5,000 × 1,060 over a 14-hour day | **~105 frames/s average, ~210/s peak** |
| Repeat sightings | a route passes the same defect many times a day; assume 90% match a known, recently confirmed defect | ~10–21 new or changed defects/s |
| Server compute | new/changed defects × ~2 s on one core (repeats only update the ledger) | ~20–40 CPU cores, or one T4 GPU for the U-Net path |
| Object storage | 5.3 M frames/day × 120 KB, 7-day retention; evidence frames kept | ~0.64 TB/day, ~4.5 TB rolling |
| Ledger rows | 5.3 M sightings × ~200 B | ~1 GB/day, ~390 GB/year (partition by month) |

The design hinges on **two filters**: the edge classifier discards ~95% of frames before upload, and the
repeat-sighting check skips the expensive segmentation for defects already confirmed. Without them the same fleet
would need ~400 cores.

---

## 3. Architecture

```
 BUS (edge)                                   CLOUD
 ┌──────────────────────────┐   HTTPS/MQTT   ┌───────────────┐   ┌──────────┐   ┌───────────────────────┐
 │ camera + IMU + GPS        │ ─────────────► │ API gateway    │──►│  queue    │──►│ inference workers     │
 │ distance sampler (8 m)    │  store-and-    │ auth, rate     │   │ (Redis    │   │ classify ▸ segment ▸  │
 │ ONNX classifier (edge)    │  forward,      │ limit, request │   │  Streams) │   │ area ▸ depth ▸ cost   │
 │ on-device privacy blur    │  idempotent    │ IDs            │   └──────────┘   │ (ONNX, autoscaled)    │
 │ upload queue              │  report ids    └───────┬───────┘                  └──────────┬────────────┘
 └──────────────────────────┘                        │                                     │
                                                      ▼                                     ▼
                                              object storage                    fleet dedup service
                                              (frames, 7 days;                  (PostGIS: same class
                                               evidence kept)                    within 8 m → merge)
                                                                                          │
                ┌──────────────────────┬───────────────────────────┬──────────────────────┘
                ▼                      ▼                           ▼
        ledger DB (Postgres)   work-order service          repair verification
        defects, sightings,    SHA-256 seal, approval,     SSIM / sharpness / pHash
        work orders            HELD_NO_GPS                 against the original
                │
                ▼
     dashboard + authority portal + contractor portal + public API (OpenAPI)

 ML PLATFORM:  data lake ▸ labelling ▸ GPU training ▸ model registry (sha256, metrics, selection rule)
               ▸ canary to workers and over-the-air to buses ▸ monitoring (drift, vision-vs-IMU disagreement)
```

### What exists today (implemented)

One process — `python -m api.server` — contains the API, the inference pipeline, the fleet ledger (SQLite), work
orders, repair verification and the website. It is the *monolith that the boxes above are carved out of*: each box
is already a separate module with its own interface, so splitting it is deployment work, not a rewrite.

| Designed component | Implemented module today |
|---|---|
| inference worker | `pipeline/deep_inference_pipeline.py` (`DeepInferencePipeline.audit_image`) |
| classifier | `models/deep_vision_net.py`, `models/cnn_embedder.py` (served one chosen by `checkpoints/vision_model_selection.json`) |
| segmenter | `models/unet_segmenter.py` / `models/defect_segmenter.py` (chosen by `segmenter_selection.json`) |
| road-damage detector | `models/road_damage_detector.py` (YOLOv8, RDD2022 India) |
| geometry, depth, cost, PCI | `models/ipm_homography_engine.py`, `models/depth_estimator.py`, `models/morth_dispatch_agent.py` |
| fleet dedup service | `pipeline/fleet_deduplication_engine.py` + `pipeline/defect_store.py` (SQLite, spatial index on lat/lon) |
| work-order service | `models/morth_dispatch_agent.py` (`generate_work_order`, `verify_work_order_seal`) |
| repair verification | repair forensics in the API (`/api/v1/audit/verify-repair`) |
| privacy | `models/privacy_redactor.py` |
| video ingest | `pipeline/video_ingest.py` (frames by ground distance, pHash suppression) |
| API gateway features | request IDs, structured access log, optional API key on write endpoints (`api/server.py`) |
| public API contract | `api/openapi.py` → `/api/v1/openapi.json`, rendered at `/api-docs` |
| model registry | `models/served_report.py:model_registry` → `/api/v1/models/served` |
| results-only deployment | `api/vercel_app.py` on Vercel (reports, no inference) |
| GPU training | `scripts/colab_train_all.sh`, `scripts/colab_train_extra.sh` |

---

## 4. Request flow — one frame from a bus

1. The bus samples a frame after every 8 m of travel and runs the edge classifier. A *normal* frame is dropped
   (except the 1% audit sample). *(designed; the classifier and the distance sampler are implemented)*
2. Faces and plates are blurred on the device; the frame, GPS fix, IMU window and a client-generated report id are
   queued and uploaded when there is signal. Re-sending the same id is a no-op. *(designed)*
3. `POST /api/v1/pipeline/deep-audit` classifies, segments, projects to ground area with that vehicle's calibration,
   gives a depth interval, a cost range and PCI, and labels each number's provenance. *(implemented)*
4. `POST /api/v1/fleet/report-defect` merges it with a known defect of the same class within 8 m, or registers a new
   one; the raw sighting is stored either way, linked to the defect. *(implemented)*
5. When a defect is confirmed, `POST /api/v1/dispatch/work-order` issues a sealed order (held if there is no GPS).
   Anyone can re-verify the seal. *(implemented)*
6. After repair, the contractor's photograph is compared with the original; a recycled photo is rejected.
   *(implemented)*

---

## 5. Data model

Implemented in `pipeline/defect_store.py` (SQLite); the production schema is the same in Postgres + PostGIS.

| Table | Key columns | Notes |
|---|---|---|
| `defects` | `defect_id` PK, `lat`, `lon`, `defect_class`, `severity_pci`, `area_m2`, `confirmation_count`, `reporting_buses`, `first_seen`, `last_seen` | one row per physical defect; indexed on (lat, lon) and class |
| `reports` | `report_id` PK, `defect_id` FK, `bus_id`, `lat`, `lon`, `defect_class`, `reported_at`, `merged`, `distance_m` | every raw sighting — the evidence a contractor disputes |
| `work_orders` | `order_id` PK, `defect_id` FK, `payload`, `seal_sha256`, `issued_at`, `status` | the sealed payload is stored verbatim |

Production additions *(designed)*: a PostGIS `geography(Point)` column with a GiST index (radius queries in metres),
monthly partitioning of `reports`, a `frames` table pointing to object storage, and an append-only audit log.

---

## 6. API

The contract for clients is OpenAPI 3 — `GET /api/v1/openapi.json`, rendered at `/api-docs`. Conventions,
all implemented:

- Missing or invalid input → `400` with a reason. No default GPS, no default PCI, no invented confidence.
- Every response carries `X-Request-ID`; with `ROAD_SHIELD_ACCESS_LOG=1` each request is one JSON log line with the
  same id, status and latency.
- Write endpoints (`fleet/report-defect`, `dispatch/work-order`, training endpoints, incident reports) require an
  API key when `ROAD_SHIELD_API_KEY` is set (`X-API-Key` or `Authorization: Bearer`), compared in constant time.

---

## 7. ML lifecycle

| Stage | How | Status |
|---|---|---|
| Data | DNIT polygons, RDD2022 India, Kaggle, Wikimedia, real IMU logs; perceptual-hash dedup; label-conflict audit; non-road data excluded by policy | Implemented |
| Split | by source photograph (and time block for IMU); test sets never used for any choice | Implemented |
| Training | Colab T4: fine-tuned CNNs, U-Net, YOLOv8, IMU 1-D CNN; resumable via Drive | Implemented |
| Selection | a rule fixed before the test set is scored decides each served model; losers are reported | Implemented |
| Export | ONNX with a parity check against PyTorch; CPU latency recorded | Implemented |
| Registry | artefact SHA-256, metrics, serving status, deciding rule (`/api/v1/models/served`) | Implemented |
| Claims | `checkpoints/claims.json` rebuilt from the reports; README generated from them | Implemented |
| Deployment | canary on a share of workers, then over-the-air to buses, keyed by registry checksum | Designed |
| Monitoring | confidence distribution, class mix per route, vision-vs-IMU disagreement rate as a drift signal | Designed |
| Feedback | contractor- and inspector-confirmed defects become labelled data | Designed (`/api/v1/training/active-feedback` stub exists) |

---

## 8. Deployment

| Environment | What runs | Status |
|---|---|---|
| Laptop / server | `python -m api.server` or `docker compose up` (engine + persistent volume for the ledger) | Implemented |
| Vercel | the website and every measured result; inference refused with a reason (model stack ~440 MB > 250 MB function limit) | Implemented |
| Production | gateway + workers on Kubernetes (HPA on queue depth), managed Postgres/PostGIS, object storage, Redis Streams | Designed |
| Bus | edge box (ARM CPU, e.g. a BEL-made unit) with ONNX Runtime, camera, IMU, GPS, 4G | Designed |

---

## 9. Scaling and reliability

- **Stateless workers** scale horizontally on queue depth; the queue absorbs morning/evening peaks.
- **Idempotent uploads** (client report ids) make retries safe; the bus keeps a local queue when offline.
- **Dedup is the single writer** for a geohash cell, so two buses reporting the same pothole at once cannot create
  two defects (partition the dedup stream by geohash).
- **Graceful degradation**, already implemented: no segmenter → bounding-box area, labelled as such; no IMU → fusion
  reports *unavailable*; a deep model that fails to load → its classical fallback; no map key → OpenStreetMap or
  an explicit `UNAVAILABLE`.
- **Failure modes and responses**: GPS missing → order `HELD_NO_GPS`; model file corrupt → registry checksum mismatch,
  fallback model; camera moved → calibration profile flagged, areas marked *estimate*.

---

## 10. Security and privacy

| Concern | Control | Status |
|---|---|---|
| Identifiable people / plates | blur before sharing; designed to move on-device so raw faces never leave the bus | Implemented (server) / Designed (edge) |
| Tampered bills | SHA-256 seal over the order; re-verifiable by anyone | Implemented |
| Recycled repair photos | SSIM + sharpness + perceptual hash | Implemented |
| Unauthorised writes | API key on write endpoints, constant-time compare | Implemented (optional) |
| Path traversal | static files served only from an explicit route table; image paths restricted to `datasets/` | Implemented |
| Transport | TLS at the gateway; device certificates for buses | Designed |
| Data protection | DPDP Act 2023: purpose limitation, 7-day raw-frame retention, evidence frames only | Designed |

---

## 11. Observability

Implemented: request ids, JSON access logs, `/api/v1/health` reporting which models loaded, a live self-test on
`/system`, and the model registry. Designed: metrics (p50/p95 latency per stage, queue depth, frames/s per route),
traces across gateway → worker → dedup, and model-quality dashboards (confidence drift, IMU disagreement).

---

## 12. Cost sketch (designed, order of magnitude)

For the 5,000-bus estimate above: ~40 CPU cores or 1–2 GPU nodes for inference, a managed Postgres, ~5 TB of rolling
object storage and ~4 GB/month of mobile data per bus. The edge classifier is what keeps this small — it removes ~95%
of the upload and compute before anything leaves the bus.

---

## 13. Azure mapping, impact and Responsible AI

The Azure service mapping, the sourced problem statistics, the cost-per-km model, the
Responsible AI mapping onto Microsoft's six principles and the 90-day pilot plan are in
[`docs/IMPACT_AND_RESPONSIBLE_AI.md`](IMPACT_AND_RESPONSIBLE_AI.md) and on the site's
`/impact` page.

## 14. Roadmap

1. Bus pilot: one route, edge box, real footage and IMU — the data the project is missing.
2. Postgres/PostGIS ledger and a queue between ingest and inference.
3. On-device privacy blur and store-and-forward upload.
4. Authority and contractor portals with the approval workflow.
5. Ground-truth depth measurements to turn the depth interval into a calibrated model.
