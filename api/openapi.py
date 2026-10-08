"""
OpenAPI 3.0 description of the main ROAD-SHIELD endpoints.

Served at GET /api/v1/openapi.json by both the engine and the static Vercel
deployment, and rendered at /api-docs. Hand-written for the endpoints a client
integrates with; the engine has more (maps, automotive, export) that are
internal or demonstration-only and are not part of this contract.
"""

NUM = {"type": "number"}
STR = {"type": "string"}
INT = {"type": "integer"}
OBJ = {"type": "object"}
ERR = {"description": "Invalid or missing input - the engine never substitutes a default",
       "content": {"application/json": {"schema": {"type": "object", "properties": {"error": STR}}}}}
AUTH_ERR = {"description": "ROAD_SHIELD_API_KEY is set on the server and the request did not carry it"}


def _json(schema, desc="OK"):
    return {"description": desc, "content": {"application/json": {"schema": schema}}}


def _body(props, required=()):
    return {"required": True, "content": {"application/json": {"schema": {
        "type": "object", "properties": props, "required": list(required)}}}}


def spec():
    secured = [{"apiKey": []}]
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "ROAD-SHIELD AI Engine",
            "version": "2026.10",
            "description": ("Road-defect assessment from bus-fleet imagery: classification, segmentation, metric "
                            "area, depth interval, MoRTH cost, sealed work orders, fleet deduplication. Every "
                            "response labels which numbers are measurements and which are estimates. Missing "
                            "inputs are a 400, never a silent default. Bodies over ROAD_SHIELD_MAX_BODY_MB (default 25) "
                            "are a 413; model endpoints answer 429 with Retry-After when the server sets "
                            "ROAD_SHIELD_RATE_LIMIT (requests per minute per client)."),
        },
        "servers": [{"url": "/"}],
        "components": {
            "securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key",
                                           "description": "Required on write endpoints only when the server sets "
                                                          "ROAD_SHIELD_API_KEY. Authorization: Bearer <key> also works."}},
        },
        "paths": {
            "/api/v1/health": {"get": {"summary": "Liveness and which models loaded", "tags": ["system"],
                                       "responses": {"200": _json(OBJ)}}},
            "/api/v1/ready": {"get": {"summary": "Readiness: 200 only when the classifier and segmenter are loaded",
                                      "tags": ["system"],
                                      "responses": {"200": _json(OBJ, "ready"),
                                                    "503": _json(OBJ, "not ready; `missing` names the model")}}},
            "/metrics": {"get": {"summary": "Prometheus metrics: requests, latency histogram, model readiness",
                                 "tags": ["system"],
                                 "responses": {"200": {"description": "Prometheus text format",
                                                       "content": {"text/plain": {"schema": STR}}}}}},
            "/api/v1/pipeline/deep-audit": {"post": {
                "summary": "Full analysis of one road photograph", "tags": ["inference"],
                "requestBody": _body({
                    "image_base64": {**STR, "description": "JPEG/PNG, base64"},
                    "corridor_id": STR, "device_id": {**STR, "description": "calibrated camera profile"},
                    "latitude": NUM, "longitude": NUM, "chainage_km": NUM,
                    "imu_series": {"type": "array", "items": {"type": "array", "items": NUM},
                                   "description": "optional accelerometer window for sensor fusion"},
                    "traffic_esal": NUM, "rain_mm": NUM, "pavement_age_yr": NUM}, ["image_base64"]),
                "responses": {"200": _json(OBJ, "class, mask, area, depth interval, cost range, PCI, scene objects, "
                                                "road-damage boxes, provenance of every figure"), "400": ERR}}},
            "/api/v1/privacy/redact": {"post": {
                "summary": "Blur people and number plates before an image is shared", "tags": ["inference"],
                "requestBody": _body({"image_base64": STR}, ["image_base64"]),
                "responses": {"200": _json({"type": "object", "properties": {
                    "redacted_image_base64": STR, "report": OBJ}}), "400": ERR}}},
            "/api/v1/video/ingest": {"post": {
                "summary": "Decode a clip, sample frames by ground distance, audit each", "tags": ["inference"],
                "requestBody": _body({"video_path": STR, "gps_track": {"type": "array", "items": OBJ},
                                      "sample_every_m": NUM, "sample_every_s": NUM, "max_frames": INT,
                                      "bus_id": STR, "device_id": STR}),
                "responses": {"200": _json(OBJ), "400": ERR}}},
            "/api/v1/fleet/report-defect": {"post": {
                "summary": "A bus reports a defect; same-class reports within 8 m merge", "tags": ["fleet"],
                "security": secured,
                "requestBody": _body({"bus_id": STR, "lat": NUM, "lon": NUM, "defect_class": STR,
                                      "area_m2": NUM, "severity_pci": NUM},
                                     ["bus_id", "lat", "lon", "defect_class", "area_m2", "severity_pci"]),
                "responses": {"200": _json(OBJ, "action REGISTERED_NEW_DEFECT or DEDUPLICATED_AND_UPDATED, with the defect"), "400": ERR,
                              "401": AUTH_ERR}}},
            "/api/v1/fleet/ingest-sealed": {"post": {
                "summary": "Encrypted packets from bus agents (AES-256-GCM, edge/crypto.py); applied in order, "
                           "replays acknowledged but not applied", "tags": ["fleet"],
                "requestBody": _body({"packets": {"type": "array", "items": OBJ}}, ["packets"]),
                "responses": {"200": _json(OBJ, "accepted_in_order tells the bus how many to drop from its queue"),
                              "400": ERR, "503": _json(OBJ, "ROAD_SHIELD_FLEET_KEY not set on the server")}}},
            "/api/v1/fleet/live": {"get": {"summary": "Bus positions (last 10 min), IMU-only shocks, recent events",
                                           "tags": ["fleet"], "responses": {"200": _json(OBJ)}}},
            "/api/v1/live/stream": {"get": {
                "summary": "Server-sent events: bus_position, defect, shock, work_order; resumes from Last-Event-ID",
                "tags": ["fleet"],
                "responses": {"200": {"description": "text/event-stream",
                                      "content": {"text/event-stream": {"schema": STR}}},
                              "503": _json(OBJ, "too many open streams")}}},
            "/api/v1/works/orders": {
                "get": {"summary": "Work orders with SLA, overdue flag, fleet passes since repair; ?status=",
                        "tags": ["works"], "responses": {"200": _json(OBJ), "400": ERR}},
                "post": {"summary": "Issue a sealed work order for a ledger defect (its measured class, area, depth, "
                                    "PCI and location)", "tags": ["works"], "security": secured,
                         "requestBody": _body({"defect_id": STR, "depth_cm": NUM, "note": STR}, ["defect_id"]),
                         "responses": {"200": _json(OBJ), "400": ERR, "401": AUTH_ERR,
                                       "409": _json(OBJ, "the defect already has an open order")}}},
            "/api/v1/works/order": {"get": {"summary": "One order with its hash-chained history (?id=)", "tags": ["works"],
                                            "responses": {"200": _json(OBJ), "404": _json(OBJ)}}},
            "/api/v1/works/status": {"post": {
                "summary": "Move an order: ASSIGNED (needs contractor), IN_PROGRESS, REPAIRED, VERIFIED, REOPENED, "
                           "CANCELLED; the fleet verifies or reopens repaired orders by itself", "tags": ["works"],
                "security": secured,
                "requestBody": _body({"work_order_id": STR, "status": STR, "contractor": STR, "note": STR, "actor": STR},
                                     ["work_order_id", "status"]),
                "responses": {"200": _json(OBJ), "401": AUTH_ERR, "404": _json(OBJ),
                              "409": _json(OBJ, "not allowed from the current status")}}},
            "/api/v1/ledger/export": {"get": {"summary": "The ledger for GIS: ?format=geojson or csv", "tags": ["fleet"],
                                              "responses": {"200": {"description": "file download"}, "400": ERR}}},
            "/api/v1/citizen/report": {"post": {
                "summary": "A citizen's photograph and location: analysed, pinned as pending, confirmed later by a bus "
                           "or an operator; the photograph is not stored", "tags": ["fleet"],
                "requestBody": _body({"image_base64": STR, "lat": NUM, "lon": NUM, "location_source": STR},
                                     ["image_base64", "lat", "lon"]),
                "responses": {"200": _json(OBJ, "report (null when no defect was found) and a message"), "400": ERR,
                              "429": _json(OBJ, "too many reports from this connection")}}},
            "/api/v1/citizen/reports": {"get": {"summary": "Citizen reports; ?status=pending|confirmed|dismissed",
                                                "tags": ["fleet"], "responses": {"200": _json(OBJ)}}},
            "/api/v1/citizen/review": {"post": {
                "summary": "Operator decision on a pending citizen report: promote (into the ledger) or dismiss",
                "tags": ["fleet"], "security": secured,
                "requestBody": _body({"report_id": STR, "action": STR}, ["report_id", "action"]),
                "responses": {"200": _json(OBJ), "401": AUTH_ERR, "404": _json(OBJ), "409": _json(OBJ)}}},
            "/api/v1/mlops/overview": {"get": {
                "summary": "Model registry, what is serving, drift and OOD rate, shadow results, labelling queue, "
                           "traffic estimate and recent training runs", "tags": ["mlops"],
                "responses": {"200": _json(OBJ)}}},
            "/api/v1/mlops/drift": {"get": {"summary": "Production drift (PSI against the training reference), "
                                                       "OOD rate and latency; ?hours=168&source=api|citizen|video|fleet",
                                            "tags": ["mlops"], "responses": {"200": _json(OBJ), "400": ERR}}},
            "/api/v1/mlops/versions": {"get": {"summary": "All versions of one model and its registry events; ?model=",
                                               "tags": ["mlops"], "responses": {"200": _json(OBJ), "404": ERR}}},
            "/api/v1/mlops/gate": {"get": {"summary": "Would this version pass promotion? ?model=&version=",
                                           "tags": ["mlops"], "responses": {"200": _json(OBJ), "404": ERR}}},
            "/api/v1/mlops/runs": {"get": {"summary": "Training runs with parameters and metrics; ?experiment=",
                                           "tags": ["mlops"], "responses": {"200": _json(OBJ)}}},
            "/api/v1/mlops/run": {"get": {"summary": "One training run with its metric history; ?id=",
                                          "tags": ["mlops"], "responses": {"200": _json(OBJ), "404": ERR}}},
            "/api/v1/mlops/promote": {"post": {
                "summary": "Deploy a registered version into checkpoints/ after its gate, then reload the models",
                "tags": ["mlops"], "security": secured,
                "requestBody": _body({"model": STR, "version": NUM, "force": {"type": "boolean"}, "reason": STR},
                                     ["model", "version"]),
                "responses": {"200": _json(OBJ), "401": AUTH_ERR, "409": _json(OBJ, "gate failed or files changed")}}},
            "/api/v1/mlops/rollback": {"post": {
                "summary": "Put the previous production version of a model back and reload", "tags": ["mlops"],
                "security": secured, "requestBody": _body({"model": STR, "reason": STR}, ["model"]),
                "responses": {"200": _json(OBJ), "401": AUTH_ERR, "409": _json(OBJ)}}},
            "/api/v1/mlops/stage": {"post": {
                "summary": "Move a version to candidate, staging (runs in shadow) or archived", "tags": ["mlops"],
                "security": secured, "requestBody": _body({"model": STR, "version": NUM, "stage": STR},
                                                          ["model", "version", "stage"]),
                "responses": {"200": _json(OBJ), "401": AUTH_ERR, "409": _json(OBJ)}}},
            "/api/v1/mlops/register": {"post": {
                "summary": "Register the model's current files in checkpoints/ as a candidate and report its gate",
                "tags": ["mlops"], "security": secured, "requestBody": _body({"model": STR, "note": STR}, ["model"]),
                "responses": {"200": _json(OBJ), "401": AUTH_ERR, "409": _json(OBJ)}}},
            "/api/v1/mlops/rebaseline": {"post": {
                "summary": "Make the last N hours of production the drift reference (or {reset: true} for the training one)",
                "tags": ["mlops"], "security": secured, "requestBody": _body({"hours": NUM, "reset": {"type": "boolean"}}, []),
                "responses": {"200": _json(OBJ), "400": ERR, "401": AUTH_ERR}}},
            "/api/v1/mlops/reload": {"post": {"summary": "Reload every model from checkpoints/ in the background",
                                              "tags": ["mlops"], "security": secured,
                                              "responses": {"202": _json(OBJ), "401": AUTH_ERR}}},
            "/api/v1/mlops/al/queue": {"get": {
                "summary": "Photographs the models were least sure about, for labelling; ?status=pending|labelled|exported",
                "tags": ["mlops"], "security": secured, "responses": {"200": _json(OBJ), "401": AUTH_ERR}}},
            "/api/v1/mlops/al/image": {"get": {"summary": "One queued photograph (people and plates blurred); ?id=",
                                               "tags": ["mlops"], "security": secured,
                                               "responses": {"200": {"description": "image/jpeg"}, "401": AUTH_ERR,
                                                             "404": ERR}}},
            "/api/v1/mlops/al/label": {"post": {
                "summary": "Label a queued photograph", "tags": ["mlops"], "security": secured,
                "requestBody": _body({"item_id": STR, "label": STR}, ["item_id", "label"]),
                "responses": {"200": _json(OBJ), "400": ERR, "401": AUTH_ERR, "404": ERR}}},
            "/api/v1/mlops/al/export": {"post": {
                "summary": "Zip of the labelled photographs, one folder per class, with a manifest",
                "tags": ["mlops"], "security": secured,
                "responses": {"200": {"description": "application/zip"}, "401": AUTH_ERR, "409": ERR}}},
            "/api/v1/traffic/cells": {"get": {"summary": "Traffic per 100 m road cell estimated from bus cameras",
                                              "tags": ["mlops"], "responses": {"200": _json(OBJ)}}},
            "/api/v1/traffic/estimate": {"get": {"summary": "Traffic estimate at a point; ?lat=&lon=",
                                                 "tags": ["mlops"], "responses": {"200": _json(OBJ), "400": ERR}}},
            "/api/v1/alerts": {"get": {"summary": "Recent alerts (P1 defects, overdue orders) and webhook status",
                                       "tags": ["fleet"], "responses": {"200": _json(OBJ)}}},
            "/api/v1/fleet/telemetry": {"get": {"summary": "Ledger counts and deduplication rate", "tags": ["fleet"],
                                                "responses": {"200": _json(OBJ)}}},
            "/api/v1/gis/map-data": {"get": {"summary": "Deduplicated defects for the map", "tags": ["fleet"],
                                             "responses": {"200": _json(OBJ)}}},
            "/api/v1/dispatch/work-order": {"post": {
                "summary": "Issue a sealed repair work order (HMAC-SHA256 when the server holds "
                           "ROAD_SHIELD_SEAL_KEY, plain SHA-256 otherwise)", "tags": ["works"], "security": secured,
                "requestBody": _body({"corridor_id": STR, "distress_class": STR, "area_sqm": NUM, "depth_cm": NUM,
                                      "pci_score": NUM, "latitude": NUM, "longitude": NUM},
                                     ["distress_class", "area_sqm", "depth_cm", "pci_score"]),
                "responses": {"200": _json(OBJ, "order with quantities, cost, seal; HELD_NO_GPS without a location"),
                              "400": ERR, "401": AUTH_ERR}}},
            "/api/v1/dispatch/verify-seal": {"post": {
                "summary": "Re-compute an order's seal and say whether it was altered", "tags": ["works"],
                "requestBody": _body({"work_order": OBJ}, ["work_order"]),
                "responses": {"200": _json({"type": "object", "properties": {
                    "is_valid": {"type": "boolean"}, "work_order_id": STR, "seal_algorithm": STR,
                    "status": {**STR, "enum": ["SEAL_VERIFIED_AUTHENTIC", "CORRUPTED_OR_TAMPERED",
                                               "KEY_REQUIRED_TO_VERIFY", "UNKEYED_SEAL_REJECTED",
                                               "MALFORMED_WORK_ORDER"]}}})}}},
            "/api/v1/priority/score": {"post": {
                "summary": "Repair Priority Index for one defect: PI = w1(100-PCI) + w2 Vol + w3 Traffic",
                "tags": ["works"],
                "requestBody": _body({"pci": NUM, "area_m2": NUM, "depth_cm": NUM, "volume_m3": NUM,
                                      "traffic_pcu_per_day": NUM, "reporting_buses": {"type": "array", "items": STR},
                                      "weights": {"type": "array", "items": NUM}}, ["pci"]),
                "responses": {"200": _json(OBJ, "index, band, each term, weights applied, missing terms"),
                              "400": ERR}}},
            "/api/v1/priority/ranking": {"get": {
                "summary": "The ledger in repair order by Priority Index; ?weights=w1,w2,w3 and ?stability=1 "
                           "(Kendall tau under +/-0.1 weight shifts)", "tags": ["works"],
                "responses": {"200": _json(OBJ), "400": ERR}}},
            "/api/v1/training/metrics": {"get": {"summary": "Held-out results of every served model",
                                                 "tags": ["models"], "responses": {"200": _json(OBJ)}}},
            "/api/v1/models/served": {"get": {"summary": "Model registry: artefacts with SHA-256, metrics, serving "
                                                         "status and the rule that chose each", "tags": ["models"],
                                              "responses": {"200": _json(OBJ)}}},
            "/api/v1/segmentation/status": {"get": {"summary": "Which segmenter serves, and its IoU",
                                                    "tags": ["models"], "responses": {"200": _json(OBJ)}}},
            "/api/v1/claims": {"get": {"summary": "Claims registry: every claim, its evidence, and withdrawn claims",
                                       "tags": ["models"], "responses": {"200": _json(OBJ)}}},
            "/api/v1/openapi.json": {"get": {"summary": "This document", "tags": ["system"],
                                             "responses": {"200": _json(OBJ)}}},
        },
    }
