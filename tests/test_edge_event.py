"""Edge event packets: size guarantee, privacy minimisation, and tamper detection."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import edge_event as ev

KEY = b"test-device-key"


def det(class_name, conf, box=(0.1, 0.5, 0.2, 0.1)):
    return {"class_name": class_name, "confidence": conf, "bbox_normalized": list(box)}


def result(detections, alerts=()):
    return {"detections": detections, "alerts": [{"code": c} for c in alerts]}


class EdgeEventTests(unittest.TestCase):
    def test_round_trip_with_signature(self):
        scene = result([det("pothole", 0.91), det("crosswalk", 0.8), det("person", 0.9), det("person", 0.7),
                        det("car", 0.95)], alerts=["PEDESTRIAN_ON_CROSSING"])
        wire = ev.encode_event(scene, "BUS-KA01-204", 12.9716, 77.5946, timestamp=1_790_000_000, key=KEY)
        decoded = ev.decode_event(wire, key=KEY)
        self.assertTrue(decoded["signature_verified"])
        self.assertEqual([f["class_name"] for f in decoded["findings"]], ["pothole", "crosswalk"])
        self.assertEqual(decoded["counts"], {"ped": 2, "car": 1})
        self.assertEqual(decoded["alerts"], ["PEDESTRIAN_ON_CROSSING"])
        self.assertEqual((decoded["lat"], decoded["lon"]), (12.9716, 77.5946))

    def test_people_are_counted_never_located(self):
        wire = ev.encode_event(result([det("person", 0.9, (0.4, 0.4, 0.05, 0.2))]), "B1", 0, 0)
        self.assertNotIn("0.4", wire.decode())  # no person box leaves the bus
        self.assertEqual(json.loads(wire)["n"], {"ped": 1})

    def test_never_exceeds_one_kilobyte_and_keeps_potholes(self):
        crowded = [det("longitudinal_crack", 0.5 + i / 1000) for i in range(200)]
        crowded += [det("pothole", 0.6), det("pothole", 0.9)]
        wire = ev.encode_event(result(crowded), "BUS-KA01-204", 12.9716, 77.5946, key=KEY)
        self.assertLessEqual(len(wire), ev.MAX_EVENT_BYTES)
        decoded = ev.decode_event(wire, key=KEY)
        self.assertGreater(decoded["findings_dropped_for_size"], 0)
        self.assertEqual([f["code"] for f in decoded["findings"][:2]], ["D40", "D40"])
        self.assertEqual(len(decoded["findings"]) + decoded["findings_dropped_for_size"], 202)

    def test_tampering_is_detected(self):
        wire = ev.encode_event(result([det("pothole", 0.9)]), "B1", 12.0, 77.0, key=KEY)
        forged = wire.replace(b'"D40",0.9', b'"D40",0.1')
        self.assertNotEqual(forged, wire)
        with self.assertRaises(ev.EventError):
            ev.decode_event(forged, key=KEY)
        with self.assertRaises(ev.EventError):
            ev.decode_event(wire, key=b"another-bus-key")

    def test_unsigned_packets_are_labelled_and_can_be_refused(self):
        wire = ev.encode_event(result([det("pothole", 0.9)]), "B1", 12.0, 77.0)
        self.assertFalse(ev.decode_event(wire)["signature_verified"])
        with self.assertRaises(ev.EventError):
            ev.decode_event(wire, require_signature=True)
        with self.assertRaises(ev.EventError):
            ev.decode_event(wire, key=KEY)  # a configured key means signatures are expected

    def test_rejects_bad_input(self):
        with self.assertRaises(ev.EventError):
            ev.encode_event(result([]), "B1", 95.0, 0.0)
        for bad in (b"not json", b'{"v": 99}', b"x" * 2000):
            with self.assertRaises(ev.EventError):
                ev.decode_event(bad)

    def test_device_key_lookup(self):
        env = {"ROAD_SHIELD_DEVICE_KEY_BUS_KA01_204": "specific", "ROAD_SHIELD_DEVICE_KEY": "fleet"}
        self.assertEqual(ev.device_key("BUS-KA01-204", env), b"specific")
        self.assertEqual(ev.device_key("BUS-OTHER", env), b"fleet")
        self.assertIsNone(ev.device_key("BUS-OTHER", {}))


if __name__ == "__main__":
    unittest.main()
