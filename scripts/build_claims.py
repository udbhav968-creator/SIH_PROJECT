"""
Regenerate checkpoints/claims.json from the reports on disk.

    python -m scripts.build_claims

The claims registry is what /api/v1/claims serves and what the Architecture
page renders. It is generated from the training reports rather than written by
hand, so a claim about accuracy cannot drift away from the number the training
script actually produced - which is exactly how the five corrections listed in
it came to be needed in the first place.
"""
import json, os, sys, time, glob

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
TARGET = os.path.join(CKPT, "claims.json")


def main():
    if not os.path.exists(TARGET):
        sys.exit(f"{TARGET} not found. It is committed to the repository; "
                 f"restore it from git rather than regenerating from nothing.")
    with open(TARGET, encoding="utf-8") as fh:
        claims = json.load(fh)

    def rep(name):
        p = os.path.join(CKPT, name)
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    heads = {}
    for p in sorted(glob.glob(os.path.join(CKPT, "cnn_head_*_report.json"))):
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        heads[r.get("backbone")] = r
    seg = rep("defect_segmenter_report.json")
    imu = rep("imu_shock_report.json")

    # Refresh only the measured numbers; the prose is deliberate and stays.
    for sub in claims["subsystems"]:
        m = sub.setdefault("measured", {})
        if sub["id"] == "M1" and "resnet50" in heads:
            r = heads["resnet50"]
            m.update(held_out_accuracy=r.get("held_out_test_accuracy"),
                     held_out_macro_f1=r.get("held_out_test_macro_f1"),
                     test_images=r.get("held_out_test_images"),
                     test_photographs=r.get("held_out_test_photographs"))
        elif sub["id"] == "M1-fallback" and "resnet50" in heads:
            b = heads["resnet50"].get("handcrafted_baseline", {})
            m.update(held_out_accuracy=b.get("accuracy"), held_out_macro_f1=b.get("macro_f1"))
        elif sub["id"] == "M1-edge" and "mobilenetv2" in heads:
            r = heads["mobilenetv2"]
            m.update(held_out_accuracy=r.get("held_out_test_accuracy"),
                     held_out_macro_f1=r.get("held_out_test_macro_f1"))
        elif sub["id"] == "M_SEG" and seg:
            io = seg.get("iou", {})
            fp = seg.get("false_positives_on_clean_roads") or {}
            m.update(clean_road_false_blob_rate=fp.get("photo_rate_any_blob"),
                     clean_photographs_scored=fp.get("clean_photographs_scored"),
                     crack_iou=io.get("crack", {}).get("iou"),
                     crack_dice=io.get("crack", {}).get("dice"),
                     pothole_iou=io.get("pothole", {}).get("iou"),
                     pothole_dice=io.get("pothole", {}).get("dice"),
                     test_photographs=seg.get("trained_on", {}).get("test_photographs"),
                     training_pixels=seg.get("trained_on", {}).get("training_pixels"))
        elif sub["id"] == "M_GATE":
            q = rep("proposal_tuning_report.json") or {}
            # The sweep's own best row was measured with the brightness grid
            # still able to raise defects. The committed build closes it, so the
            # authoritative numbers are the end-to-end ones.
            e2e = rep("detection_quality_report.json") or {}
            rec = dict(q.get("recommended") or {})
            if e2e:
                rec["false_positive_rate"] = e2e.get("false_positive_rate")
                rec["detection_rate"] = e2e.get("detection_rate")
                rec["min_component_fraction"] = 0.004
                rec["min_component_confidence"] = 0.45
                q = dict(q, clean_photographs=e2e.get("clean_photographs"),
                         defect_photographs=e2e.get("defect_photographs"),
                         measured_through=e2e.get("measured_through"))
            if rec:
                # Replace, never merge: the earlier keys here held the 27.8%
                # figure that was withdrawn, and a stale key beside a fresh one
                # is how a withdrawn number survives a refresh.
                m.clear()
                m.update(
                    false_positive_rate_before=0.444,
                    detection_rate_before=1.0,
                    before_note="the pre-fix build, measured through audit_image()",
                    max_heuristic_box_fraction=0.25,
                    min_component_fraction=rec.get("min_component_fraction"),
                    min_component_confidence=rec.get("min_component_confidence"),
                    false_positive_rate=rec.get("false_positive_rate"),
                    detection_rate=rec.get("detection_rate"),
                    clean_photographs=q.get("clean_photographs"),
                    defect_photographs=q.get("defect_photographs"),
                    measured_through=q.get("measured_through"))
        elif sub["id"] == "M4" and imu:
            m["held_out_accuracy"] = imu.get("held_out_validation_accuracy")

    claims["generated_unix"] = int(time.time())
    with open(TARGET, "w", encoding="utf-8") as fh:
        json.dump(claims, fh, indent=2)
    print(f"refreshed {TARGET}")
    print(f"  {len(claims['subsystems'])} subsystems, {len(claims['corrections'])} corrections")


if __name__ == "__main__":
    main()
