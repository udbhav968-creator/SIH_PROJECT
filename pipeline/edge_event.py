"""
The packet a bus sends when it sees something: compact, signed, under 1 KB.

A bus never uploads video. For each analysed frame with findings it sends one
JSON event built from the perception result:

  * road findings (potholes, cracks, zebra crossings) with class code,
    confidence and normalised box - what the ledger needs to place a defect
  * counts of people and vehicles, never their boxes or crops (DPDP Act 2023
    data minimisation: the server has no use for where a person stood)
  * the alerts raised on the bus
  * an HMAC-SHA256 over the canonical payload, keyed per device, so the
    server can tell which bus sent it and that nothing was altered in transit

The 1,024-byte limit is enforced, not hoped for: when a frame has more
findings than fit, the least important ones are dropped (potholes before
cracks before markings, then by confidence) and the packet records how many
were dropped. A bare SHA-256 would not do here: anyone who edits the packet
can recompute it. An HMAC cannot be recomputed without the device key.
"""

import base64
import hashlib
import hmac
import json
import os
import time

MAX_EVENT_BYTES = 1024
SCHEMA_VERSION = 1

# Short wire codes. RDD2022's D-codes for damage; two letters for the rest.
CLASS_CODES = {
    "pothole": "D40",
    "alligator_crack": "D20",
    "transverse_crack": "D10",
    "longitudinal_crack": "D00",
    "crosswalk": "ZC",
    "guide_arrows": "GA",
}
CODE_CLASSES = {code: name for name, code in CLASS_CODES.items()}
# Lower sorts first = kept first when the packet has to shed findings.
IMPORTANCE = {"D40": 0, "D20": 1, "D10": 2, "D00": 3, "ZC": 4, "GA": 5}
COUNTED_CLASSES = {"person": "ped", "bicycle": "cyc", "motorcycle": "2w", "car": "car",
                   "bus": "bus", "truck": "trk", "cow": "ani", "dog": "ani"}


class EventError(ValueError):
    """A packet that is malformed, oversize or fails signature verification."""


def device_key(bus_id, env=os.environ):
    """
    The HMAC key for a bus. Provisioned per device in production
    (ROAD_SHIELD_DEVICE_KEY_<BUS_ID>); a fleet-wide ROAD_SHIELD_DEVICE_KEY is
    accepted for demos. None means packets go unsigned and say so.
    """
    specific = env.get("ROAD_SHIELD_DEVICE_KEY_" + bus_id.upper().replace("-", "_"))
    value = specific or env.get("ROAD_SHIELD_DEVICE_KEY")
    return value.encode("utf-8") if value else None


def _canonical(payload):
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("ascii")


def _sign(payload, key):
    return base64.urlsafe_b64encode(hmac.new(key, _canonical(payload), hashlib.sha256).digest()[:16]).decode()


def encode_event(perception_result, bus_id, lat, lon, timestamp=None, key=None, max_bytes=MAX_EVENT_BYTES):
    """
    Build the wire packet for one analysed frame. Returns bytes <= max_bytes.

    `perception_result` is the dict returned by RoadScenePerception.analyze.
    """
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise EventError("latitude/longitude out of range")

    findings = []
    counts = {}
    for det in perception_result.get("detections", []):
        code = CLASS_CODES.get(det["class_name"])
        if code:
            x, y, w, h = det["bbox_normalized"]
            findings.append([code, round(det["confidence"], 2), round(x, 3), round(y, 3), round(w, 3), round(h, 3)])
        elif det["class_name"] in COUNTED_CLASSES:
            label = COUNTED_CLASSES[det["class_name"]]
            counts[label] = counts.get(label, 0) + 1
    findings.sort(key=lambda f: (IMPORTANCE[f[0]], -f[1]))

    payload = {
        "v": SCHEMA_VERSION,
        "bus": str(bus_id)[:24],
        "ts": int(timestamp if timestamp is not None else time.time()),
        "ll": [round(lat, 6), round(lon, 6)],  # ~0.1 m: finer than GPS can resolve
        "f": findings,
        "n": counts,
        "a": [a["code"] for a in perception_result.get("alerts", [])][:4],
        "drop": 0,
    }

    total = len(findings)
    while True:
        payload["f"] = findings[: total - payload["drop"]]
        packet = dict(payload)
        if key is not None:
            packet["sig"] = _sign(payload, key)
        wire = _canonical(packet)
        if len(wire) <= max_bytes:
            return wire
        if payload["drop"] >= total:
            raise EventError(f"event exceeds {max_bytes} bytes even with no findings")
        payload["drop"] += 1


def decode_event(wire, key=None, require_signature=False, max_bytes=MAX_EVENT_BYTES):
    """
    Parse and verify a packet. Returns the payload with findings expanded to
    dicts. Raises EventError on anything suspicious.
    """
    if len(wire) > max_bytes:
        raise EventError(f"packet is {len(wire)} bytes; the limit is {max_bytes}")
    try:
        packet = json.loads(wire)
    except (ValueError, UnicodeDecodeError):
        raise EventError("packet is not JSON") from None
    if not isinstance(packet, dict) or packet.get("v") != SCHEMA_VERSION:
        raise EventError("unknown packet schema")

    signature = packet.pop("sig", None)
    if signature is None:
        if require_signature or key is not None:
            raise EventError("packet is unsigned")
        verified = False
    else:
        if key is None:
            raise EventError("packet is signed but no device key is configured to verify it")
        if not hmac.compare_digest(signature, _sign(packet, key)):
            raise EventError("signature does not match: packet altered or wrong device key")
        verified = True

    try:
        findings = [
            {"class_name": CODE_CLASSES[code], "code": code, "confidence": conf,
             "bbox_normalized": [x, y, w, h]}
            for code, conf, x, y, w, h in packet["f"]
        ]
    except (KeyError, TypeError, ValueError):
        raise EventError("malformed findings") from None
    return {
        "bus_id": packet["bus"], "timestamp": packet["ts"], "lat": packet["ll"][0], "lon": packet["ll"][1],
        "findings": findings, "counts": packet.get("n", {}), "alerts": packet.get("a", []),
        "findings_dropped_for_size": packet.get("drop", 0), "signature_verified": verified,
        "bytes": len(wire),
    }
