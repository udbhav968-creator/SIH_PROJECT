"""
Dashcam video -> deduplicated defect reports.

The premise of this project is a bus fleet as a mobile sensing network. Until
this module existed, nothing in the codebase decoded video: the pipeline took
one photograph at a time, and a "video" endpoint existed that required the
client to POST individual base64 frames. That is not a fleet ingest, it is a
photograph uploader with extra steps.

What this does
--------------
Decodes a video file, samples frames at a fixed interval, runs each through the
existing inference pipeline, and feeds the results into the deduplication
engine with a GPS position - so the same pothole seen in forty consecutive
frames becomes one defect, not forty.

Two things make that work:

1. **Sampling by distance, not by frame.** At 40 km/h a bus covers 11 m per
   second. Sampling every frame wastes 95% of the compute on near-identical
   views; sampling every N frames regardless of speed either misses defects in
   slow traffic or floods the ledger at speed. When a GPS track is supplied,
   frames are chosen to be roughly `sample_every_m` apart on the ground.

2. **Perceptual-hash suppression between frames.** Consecutive frames of the
   same defect look alike. Before a frame is analysed at all, it is hashed
   against the last few accepted frames and skipped if it is a near-duplicate.
   This is the same DCT pHash the forensic auditor uses, and it removes most
   of the redundancy that survives distance sampling.

GPS
---
A track is a list of ``{"t": seconds_from_start, "lat": .., "lon": ..}``.
Positions are linearly interpolated between fixes. If no track is supplied,
frames get no position and the reports are not deduplicated spatially - the
run still produces per-frame analysis, and says plainly that it could not
locate anything. Inventing a location would be worse than having none.

    from pipeline.video_ingest import VideoIngestor
    ing = VideoIngestor(pipeline, dedup_engine)
    summary = ing.process("dashcam.mp4", gps_track=track, bus_id="KA01F1234")
"""

import json
import os
import time

import numpy as np


class GPSTrack:
    """Interpolates a position for any timestamp inside the track."""

    def __init__(self, fixes):
        self.fixes = sorted(
            [f for f in (fixes or []) if {"t", "lat", "lon"} <= set(f)],
            key=lambda f: float(f["t"]))

    def __len__(self):
        return len(self.fixes)

    def position_at(self, t):
        """(lat, lon) at time t seconds, or None outside the track."""
        if not self.fixes:
            return None
        t = float(t)
        if t <= float(self.fixes[0]["t"]):
            return float(self.fixes[0]["lat"]), float(self.fixes[0]["lon"])
        if t >= float(self.fixes[-1]["t"]):
            return float(self.fixes[-1]["lat"]), float(self.fixes[-1]["lon"])
        for a, b in zip(self.fixes, self.fixes[1:]):
            ta, tb = float(a["t"]), float(b["t"])
            if ta <= t <= tb:
                span = max(tb - ta, 1e-6)
                w = (t - ta) / span
                return (float(a["lat"]) + w * (float(b["lat"]) - float(a["lat"])),
                        float(a["lon"]) + w * (float(b["lon"]) - float(a["lon"])))
        return None

    @staticmethod
    def haversine_m(lat1, lon1, lat2, lon2):
        import math
        r = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @classmethod
    def from_file(cls, path):
        """A JSON list of fixes, or a GPX file."""
        if path.lower().endswith(".gpx"):
            import xml.etree.ElementTree as ET
            root = ET.parse(path).getroot()
            ns = {"g": "http://www.topografix.com/GPX/1/1"}
            pts, t0 = [], None
            for pt in root.iterfind(".//g:trkpt", ns):
                tel = pt.find("g:time", ns)
                if tel is None:
                    continue
                from datetime import datetime
                ts = datetime.fromisoformat(tel.text.replace("Z", "+00:00")).timestamp()
                t0 = ts if t0 is None else t0
                pts.append({"t": ts - t0, "lat": float(pt.get("lat")), "lon": float(pt.get("lon"))})
            return cls(pts)
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))


class VideoIngestor:
    """Decodes a video and turns it into deduplicated defect reports."""

    def __init__(self, pipeline, dedup_engine=None, hasher=None):
        self.pipeline = pipeline
        self.dedup = dedup_engine
        if hasher is None:
            from models.forensic_audit_engine import ForensicDuplicateHasher
            hasher = ForensicDuplicateHasher(hash_size=8)
        self.hasher = hasher

    # ------------------------------------------------------------------
    @staticmethod
    def probe(path):
        """Duration, fps and frame count without decoding the whole file."""
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"fps": round(fps, 2), "frame_count": frames,
                "duration_s": round(frames / fps, 2) if fps else None,
                "width": w, "height": h}

    def process(self, path, gps_track=None, bus_id="UNKNOWN", sample_every_s=1.0,
                sample_every_m=8.0, max_frames=400, hash_distance=6,
                min_confidence=0.35, device_id=None, progress=None):
        """
        Run a video through the pipeline.

        `sample_every_m` takes precedence when a GPS track is supplied;
        `sample_every_s` is the fallback when it is not.
        """
        import cv2

        info = self.probe(path)
        fps = info["fps"] or 25.0
        track = gps_track if isinstance(gps_track, GPSTrack) else GPSTrack(gps_track)
        located = len(track) > 0

        cap = cv2.VideoCapture(path)
        step_frames = max(1, int(round(fps * sample_every_s)))

        recent_hashes = []
        detections, skipped_dup, analysed, frame_idx = [], 0, 0, 0
        last_pos = None
        t_start = time.time()

        while cap.isOpened() and analysed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps

            take = (frame_idx % step_frames == 0)
            pos = track.position_at(t) if located else None
            if located and pos is not None:
                if last_pos is None:
                    take = True
                else:
                    moved = GPSTrack.haversine_m(last_pos[0], last_pos[1], pos[0], pos[1])
                    take = moved >= sample_every_m
            if not take:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Cheap redundancy filter before the expensive part.
            digest = self.hasher.compute_hash(cv2.resize(rgb, (64, 64)))
            if any(self.hasher.hamming_distance(digest, h) <= hash_distance
                   for h in recent_hashes):
                skipped_dup += 1
                frame_idx += 1
                continue
            recent_hashes.append(digest)
            recent_hashes = recent_hashes[-12:]

            result = self.pipeline.audit_image(image_input=rgb)
            analysed += 1
            last_pos = pos or last_pos
            if progress and analysed % 10 == 0:
                progress(analysed, frame_idx, info["frame_count"])

            top = (result.get("all_detections") or [{}])[0]
            conf = float(top.get("confidence", 0.0))
            cls_name = top.get("class_name", "")
            is_defect = conf >= min_confidence and "Normal" not in cls_name

            rec = {
                "frame_index": frame_idx, "timestamp_s": round(t, 2),
                "class_name": cls_name, "confidence": round(conf, 4),
                "area_m2": top.get("surface_area_m2"),
                "pci": (result.get("astm_d6433_pci") or {}).get("pci_score"),
                "lat": pos[0] if pos else None, "lon": pos[1] if pos else None,
                "is_defect": bool(is_defect),
            }
            if is_defect and self.dedup is not None and pos is not None:
                merge = self.dedup.ingest_fleet_detection(
                    bus_id=bus_id, lat=pos[0], lon=pos[1], defect_class=cls_name,
                    severity_pci=rec["pci"] or 100.0, area_m2=rec["area_m2"] or 0.0,
                    image_timestamp=time.time(), enrich_location=False)
                rec["dedup"] = merge
            detections.append(rec)
            frame_idx += 1

        cap.release()
        defects = [d for d in detections if d["is_defect"]]
        unique = len({d["dedup"]["defect_id"] for d in defects if d.get("dedup")})

        return {
            "video": {"path": os.path.basename(path), **info},
            "bus_id": bus_id,
            "device_id": device_id,
            "sampling": {
                "mode": "by_distance" if located else "by_time",
                "sample_every_m": sample_every_m if located else None,
                "sample_every_s": None if located else sample_every_s,
                "max_frames": max_frames,
            },
            "gps": {
                "available": located,
                "fixes": len(track),
                "note": None if located else
                        "No GPS track supplied. Frames were analysed but nothing could be "
                        "located or deduplicated spatially; no position has been invented.",
            },
            "frames_analysed": analysed,
            "frames_skipped_as_near_duplicates": skipped_dup,
            "defect_frames": len(defects),
            "unique_defects_after_dedup": unique if located else None,
            "seconds_elapsed": round(time.time() - t_start, 1),
            "ms_per_analysed_frame": round(1000 * (time.time() - t_start) / max(1, analysed), 1),
            "detections": detections,
        }


def synthesise_test_video(path, frames=60, size=(640, 480), fps=15):
    """
    A short synthetic clip: a grey road with a dark blob that grows.

    Used by the tests. Real dashcam footage is not in this repository and
    fabricating a "sample video" that looked like real road footage would be
    exactly the kind of thing this project exists to avoid - this one is
    obviously synthetic and is only ever used to prove the decoder works.
    """
    import cv2
    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(path.replace(".mp4", ".avi"),
                                 cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        path = path.replace(".mp4", ".avi")
    rng = np.random.default_rng(0)
    for i in range(frames):
        img = np.full((h, w, 3), 140, np.uint8)
        img = cv2.add(img, rng.integers(0, 12, (h, w, 3), dtype=np.uint8))
        r = 12 + i // 3
        cv2.circle(img, (w // 2, int(h * 0.72)), r, (55, 55, 55), -1)
        writer.write(img)
    writer.release()
    return path
