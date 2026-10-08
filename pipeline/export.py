"""
The defect ledger as files a municipality's GIS team can open: GeoJSON (QGIS, ArcGIS, Google Earth via
conversion, any web map) and CSV (Excel). One row / feature per deduplicated defect, with its priority and
the status of its latest work order.

CSV cells that start with = + - or @ are prefixed with an apostrophe so a spreadsheet does not run them as
formulas (a defect class or address is text from the field, and must stay text).
"""
import csv
import io
import json
import time

FIELDS = ("defect_id", "defect_class", "lat", "lon", "severity_pci", "area_m2", "depth_cm", "confirmations",
          "distinct_sources", "first_seen_utc", "last_seen_utc", "priority_index", "priority_band", "repair_status",
          "work_order_id", "address")


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(t))) if t else None


def _row(d, priority_fn, repair):
    pr = priority_fn(d) or {}
    rp = repair.get(d["defect_id"]) or {}
    return {
        "defect_id": d["defect_id"], "defect_class": d.get("defect_class"), "lat": d.get("lat"), "lon": d.get("lon"),
        "severity_pci": d.get("severity_pci"), "area_m2": d.get("area_m2"), "depth_cm": d.get("depth_cm"),
        "confirmations": d.get("confirmation_count"), "distinct_sources": len(set(d.get("reporting_buses") or [])),
        "first_seen_utc": _iso(d.get("first_seen_timestamp")), "last_seen_utc": _iso(d.get("last_seen_timestamp")),
        "priority_index": pr.get("priority_index"), "priority_band": pr.get("band"),
        "repair_status": rp.get("status", "NO_ORDER"), "work_order_id": rp.get("work_order_id"),
        "address": d.get("address"),
    }


def ledger_geojson(defects, priority_fn, repair):
    feats = []
    for d in defects:
        r = _row(d, priority_fn, repair)
        feats.append({"type": "Feature", "id": r["defect_id"],
                      "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                      "properties": {k: v for k, v in r.items() if k not in ("lat", "lon")}})
    return json.dumps({"type": "FeatureCollection", "name": "road_shield_defects",
                       "generated_utc": _iso(time.time()), "features": feats}, default=str)


def _safe_cell(v):
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def ledger_csv(defects, priority_fn, repair):
    buf = io.StringIO()
    buf.write("\ufeff")                      # byte-order mark: Excel then reads non-English addresses correctly
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    for d in defects:
        w.writerow({k: _safe_cell(v) for k, v in _row(d, priority_fn, repair).items()})
    return buf.getvalue()
