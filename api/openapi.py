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
                            "inputs are a 400, never a silent default."),
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
            "/api/v1/fleet/telemetry": {"get": {"summary": "Ledger counts and deduplication rate", "tags": ["fleet"],
                                                "responses": {"200": _json(OBJ)}}},
            "/api/v1/gis/map-data": {"get": {"summary": "Deduplicated defects for the map", "tags": ["fleet"],
                                             "responses": {"200": _json(OBJ)}}},
            "/api/v1/dispatch/work-order": {"post": {
                "summary": "Issue a SHA-256-sealed repair work order", "tags": ["works"], "security": secured,
                "requestBody": _body({"corridor_id": STR, "distress_class": STR, "area_sqm": NUM, "depth_cm": NUM,
                                      "pci_score": NUM, "latitude": NUM, "longitude": NUM},
                                     ["distress_class", "area_sqm", "depth_cm", "pci_score"]),
                "responses": {"200": _json(OBJ, "order with quantities, cost, seal; HELD_NO_GPS without a location"),
                              "400": ERR, "401": AUTH_ERR}}},
            "/api/v1/dispatch/verify-seal": {"post": {
                "summary": "Re-hash an order and say whether it was altered", "tags": ["works"],
                "requestBody": _body({"work_order": OBJ}, ["work_order"]),
                "responses": {"200": _json({"type": "object", "properties": {
                    "is_valid": {"type": "boolean"}, "work_order_id": STR,
                    "status": {**STR, "enum": ["SEAL_VERIFIED_AUTHENTIC", "CORRUPTED_OR_TAMPERED"]}}})}}},
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
