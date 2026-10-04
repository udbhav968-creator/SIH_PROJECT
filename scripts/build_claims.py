"""
Regenerate checkpoints/claims.json from the reports on disk.

    python -m scripts.build_claims

The claims registry is what /api/v1/claims serves and what the Architecture
page renders. It is generated from the training reports rather than written by
hand, so a claim about accuracy cannot drift away from the number the training
script actually produced - which is exactly how the five corrections listed in
it came to be needed in the first place.
"""
import json, os, re, sys, time, glob

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
    # M1 must describe the head the server actually serves. The loader prefers
    # ResNet-50 only when its backbone is on disk; otherwise MobileNetV2 serves.
    # Filling M1 from the ResNet-50 report regardless meant the architecture
    # page quoted a model that was not running.
    def _served():
        for bb in ("resnet50", "mobilenetv2"):
            if bb in heads and os.path.exists(os.path.join(CKPT, f"cnn_backbone_{bb}.onnx")):
                return bb
        return None
    served = _served()
    seg = rep("defect_segmenter_report.json")
    imu = rep("imu_shock_report.json")
    imu_deep = rep("imu_deep_report.json")
    imu_sel = rep("imu_model_selection.json")
    vsel = rep("vision_model_selection.json")
    ft = rep("finetune_summary.json")
    deep_served = (vsel.get("served") == "deep_cnn" and ft
                   and glob.glob(os.path.join(CKPT, "deep_vision_*.onnx")))

    # Refresh only the measured numbers; the prose is deliberate and stays.
    for sub in claims["subsystems"]:
        m = sub.setdefault("measured", {})
        if sub["id"] == "M1" and deep_served:
            arch = ft["chosen_arch"]
            fr = rep(f"finetune_{ft['served']}_report.json")
            onnx = ft.get("served_onnx") or {}
            sub["architecture"] = (f"{arch} (ImageNet-pretrained, fine-tuned end to end on the road corpus, "
                                   f"refit on train+val), ONNX Runtime, flip TTA")
            sub["why"] = ("Chosen over the frozen MobileNetV2 + scikit-learn head by a rule fixed before "
                          "the test set was read: it had to beat the head on both accuracy and macro-F1 on "
                          "non-test data. " + vsel.get("evidence", ""))
            sub["rejected_alternative"] = ("Frozen-embedding head (kept as the fallback); other fine-tuned "
                                           "architectures lost on validation - see finetune_summary.json.")
            sub["evidence"] = f"checkpoints/finetune_{ft['served']}_report.json, checkpoints/finetune_summary.json, checkpoints/vision_model_selection.json"
            sub["reproduce"] = "python -m training.train_finetune_cnn && python -m scripts.select_vision_model"
            m.clear()
            m.update(served_model=ft["served"], held_out_accuracy=ft["served_test"]["accuracy"],
                     held_out_macro_f1=ft["served_test"]["macro_f1"], test_images=ft["served_test"]["images"],
                     test_photographs=(fr.get("split") or {}).get("test_photographs"),
                     indian_roads=ft.get("served_indian_roads"),
                     onnx_cpu_ms_per_image=onnx.get("onnx_cpu_ms_per_image"),
                     onnx_size_mb=onnx.get("onnx_size_mb"))
            sub["not_claimed"] = (f"Latency is {onnx.get('onnx_cpu_ms_per_image')} ms per image (single "
                                  f"image, ONNX Runtime on the {onnx.get('cpu_threads')}-thread training "
                                  "machine's CPU, before flip TTA doubles it); it was not measured on a "
                                  "Raspberry Pi or Jetson. The rare municipal classes have 7-8 test images each.")
        elif sub["id"] == "M1" and served:
            r = heads[served]
            sub["architecture"] = (f"{'ResNet-50' if served == 'resnet50' else 'MobileNetV2'} "
                                   f"(ImageNet, frozen, ONNX Runtime) -> class-balanced "
                                   f"{r.get('head', 'logistic')} head")
            sub["evidence"] = f"checkpoints/cnn_head_{served}_report.json"
            sub["reproduce"] = f"python -m training.train_cnn_head --backbone {served} --compare"
            sub["why"] = ("Transfer learning from ImageNet with a frozen backbone and a head selected by grouped "
                          "cross-validation." + (" End-to-end fine-tuning was trained and compared on the identical "
                          "split but did not beat this head on non-test data: " + vsel.get("evidence", "")
                          if ft else ""))
            sub["not_claimed"] = ("Latency of the MobileNetV2 embedding is about 15 ms per pass on a laptop "
                                  "CPU, doubled by flip TTA; not measured on edge hardware.")
            m.update(served_backbone=served,
                     held_out_accuracy=r.get("held_out_test_accuracy"),
                     held_out_macro_f1=r.get("held_out_test_macro_f1"),
                     test_images=r.get("held_out_test_images"),
                     test_photographs=r.get("held_out_test_photographs"))
        elif sub["id"] == "M1-fallback" and served:
            b = heads[served].get("handcrafted_baseline", {})
            sub["evidence"] = f"checkpoints/cnn_head_{served}_report.json (scored on the identical split)"
            m.update(held_out_accuracy=b.get("accuracy"), held_out_macro_f1=b.get("macro_f1"))
            # Measured from the fitted model, never carried over by hand: this
            # value was 0.451 in the file while the retrained model kept 0.415.
            try:
                import joblib
                blob = joblib.load(os.path.join(CKPT, "vision_distress_model.joblib"))
                pipe = blob.get("pipeline") or blob.get("model") if isinstance(blob, dict) else blob
                pca = next(st for name, st in pipe.steps if "pca" in name.lower())
                var = round(float(pca.explained_variance_ratio_.sum()), 4)
                m.update(pca_components=int(pca.n_components_), pca_variance_retained=var)
                sub["not_claimed"] = (f"PCA retains {var * 100:.1f}% of variance at "
                                      f"{int(pca.n_components_)} components, measured from the fitted model.")
            except Exception as e:
                print(f"  could not measure PCA variance: {e}")
        elif sub["id"] == "M1-edge" and "mobilenetv2" in heads:
            r = heads["mobilenetv2"]
            sub["architecture"] = f"MobileNetV2 (14 MB ONNX) -> {r.get('head', 'logistic')} head"
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
        elif sub["id"] == "M4" and imu_sel.get("served") == "cnn" and imu_deep:
            h = imu_deep["held_out"]["cnn"]
            sub["architecture"] = "1-D residual CNN (3-seed ensemble), ONNX Runtime"
            sub["evidence"] = "checkpoints/imu_deep_report.json, checkpoints/imu_model_selection.json"
            sub["reproduce"] = "python -m training.train_imu_deep"
            m.clear()
            m.update(held_out_accuracy=h["accuracy"], held_out_macro_f1=h["macro_f1"],
                     held_out_windows=imu_deep["data"]["held_out_windows"],
                     random_forest_held_out_accuracy=imu_deep["held_out"]["random_forest"]["accuracy"],
                     random_forest_held_out_macro_f1=imu_deep["held_out"]["random_forest"]["macro_f1"])
        elif sub["id"] == "M4" and imu:
            m["held_out_accuracy"] = imu.get("held_out_validation_accuracy")
            if imu_deep:
                m["cnn_compared"] = {"served": imu_sel.get("served"),
                                     "cv": imu_sel.get("cv"),
                                     "held_out": imu_sel.get("held_out_for_reporting")}

    _refresh_prose(claims, rep, imu_sel, imu_deep)
    _deep_extras(claims, rep)
    claims["generated_unix"] = int(time.time())
    with open(TARGET, "w", encoding="utf-8") as fh:
        json.dump(claims, fh, indent=2)
    print(f"refreshed {TARGET}")
    print(f"  {len(claims['subsystems'])} subsystems, {len(claims['corrections'])} corrections")


AUDIT_CORRECTIONS = [
    ("The IMU 1-D CNN beats the RandomForest (5-fold CV)", "CORRECTED (protocol fixed)",
     "The first comparison shuffled individual 1-second windows into CV folds, so neighbouring windows from the "
     "same drive sat on both sides of a fold. CV favoured the CNN (accuracy 0.869 vs 0.849); the time-separated "
     "held-out logs then scored the CNN below the RandomForest (0.787 vs 0.872). CV now uses contiguous time "
     "blocks (training/train_imu_deep.py), and the comparison is re-run with that protocol."),
    ("Faces and number plates are blurred on the bus before transmission", "FIXED (implemented)",
     "No such code existed. models/privacy_redactor.py now blurs the head region of detected people and "
     "plates found inside detected vehicles (/api/v1/privacy/redact). Its recall is not measured - there is "
     "no annotated face/plate set - and no bus deployment exists."),
    ("A request without GPS is located in Delhi / Bengaluru", "FIXED",
     "The API and pipeline defaulted missing coordinates to 28.7041, 77.1025 or 12.9716, 77.5946. Location is "
     "now null and work orders without GPS are issued HELD_NO_GPS."),
    ("A work order is sealed even when its measurements are missing", "FIXED",
     "Missing fields became a 2.2 m2, 6.5 cm pothole at PCI 42, then SHA-256 sealed. The endpoint now returns "
     "400 with the missing field list; the Works page also sent field names the server ignored."),
    ("Detections carry a model confidence", "CORRECTED",
     "Rule decisions reported a fixed 0.99, agreement with DAN-DAG raised confidence by max(p, 0.55p+0.45q+0.03), "
     "and ALPR kinematics used a formula with a 0.75 floor. All removed; rule decisions report no confidence."),
    ("The ledger shows the city's defects", "QUALIFIED",
     "Five invented bus reports were seeded on every fresh start. Demo rows are now opt-in (ROAD_SHIELD_SEED_DEMO=1)."),
    ("IMU classes include expansion joints and rumble strips", "CORRECTED",
     "The drive logs contain unmarked and marked speed breakers; the classes are now named for what they contain."),
    ("Laplacian-variance gate (42.5), 4th-order Butterworth filter, CLAHE, 2.45 m camera", "WITHDRAWN",
     "None of these exist in the code. The gate is luminance std < 6.5, IMU windows are mean-centred with FFT "
     "band energies, there is no CLAHE step, and the default camera height is an assumed 1.45 m."),
]


def _refresh_prose(claims, rep, imu_sel, imu_deep):
    """Prose that quotes a number is regenerated from the report that measured it,
    so text and figure cannot disagree (they did: '5.5% false blobs' beside a
    measured 23.3%, '8.3% false positives' beside a measured 11.1%)."""
    subs = {s["id"]: s for s in claims["subsystems"]}
    seg = rep("defect_segmenter_report.json")
    gate = rep("semantic_gate_report.json")
    if "M_SEG" in subs and seg:
        fp = (seg.get("false_positives_on_clean_roads") or {})
        io = seg.get("iou", {})
        r = (gate.get("results") or {})
        a, c = r.get("A_segmenter", {}), r.get("C_seg_gated_0.5", {})
        subs["M_SEG"]["not_claimed"] = (
            f"IoU of {io.get('crack', {}).get('iou')} (crack) and {io.get('pothole', {}).get('iou')} (pothole) is a "
            f"working model, not a solved problem. On held-out clean roads the segmenter alone draws a false blob on "
            f"{fp.get('photo_rate_any_blob', 0) * 100:.1f}% of {fp.get('clean_photographs_scored')} photographs. "
            + (f"The CNN semantic gate in front of it reduced clean roads with a false blob from "
               f"{a.get('clean_photos_with_false_blob')} to {c.get('clean_photos_with_false_blob')} and raised "
               f"pothole IoU from {a.get('pothole', {}).get('iou')} to {c.get('pothole', {}).get('iou')} "
               f"(checkpoints/semantic_gate_report.json, measured with the frozen-embedding classifier). "
               if a and c else "")
            + "Water-filled potholes are still under-detected.")
    if "M_GATE" in subs:
        m = subs["M_GATE"].get("measured", {})
        fpr, det = m.get("false_positive_rate"), m.get("detection_rate")
        nc, nd = m.get("clean_photographs"), m.get("defect_photographs")
        if fpr is not None and det is not None:
            subs["M_GATE"]["not_claimed"] = (
                f"{fpr * 100:.1f}% false positives is better, not solved - {round(fpr * (nc or 0))} of {nc} clean "
                f"photographs are still reported as defective. Detection is {det * 100:.1f}% of {nd} defect "
                f"photographs. Measured through audit_image() by scripts/measure_detection_quality.py.")
    if "M4" in subs:
        s = subs["M4"]
        s["role"] = "SMOOTH_ASPHALT / UNMARKED_SPEED_BREAKER / MARKED_SPEED_BREAKER / POTHOLE_IMPACT"
        s["rejected_alternative"] = (
            "A 1-D residual CNN is trained and compared on the same windows (training/train_imu_deep.py); it is "
            "served only if it beats the forest on cross-validation over the training windows.")
        s["not_claimed"] = (
            "The windows are REAL accelerometer recordings, but from 10 car drive logs published by one GitHub "
            "project (VishalSingh25/Pothole-Project), not from a bus fleet. 164 held-out windows; the speed-breaker "
            "classes have 11-32 of them, so their scores move with single windows.")
    if "M_DET" in subs:
        subs["M_DET"]["why"] = (
            "Single-stage, 12.8 MB, runs on CPU. Detects that a person is PRESENT and how far away, without storing "
            "any identity; the same boxes drive the privacy redactor that blurs heads and plates before sharing.")
    if "M_PRIVACY" not in subs:
        claims["subsystems"].append({
            "id": "M_PRIVACY", "name": "Privacy redaction",
            "architecture": "COCO person / vehicle boxes -> head-region and contour-localised plate Gaussian blur "
                            "(+ Haar frontal-face cascade when installed)",
            "role": "Blur people and number plates before an image is shared (DPDP Act 2023)",
            "why": "The report promised it; it did not exist. Uses detections the pipeline already computes.",
            "measured": {"recall_measured": False},
            "evidence": "models/privacy_redactor.py, tests/test_integrity_fixes.py PrivacyRedaction",
            "not_claimed": "No recall figure: there is no annotated face/plate set in this project. A missed face "
                           "or plate is possible; the API reports which detectors ran.",
            "status_endpoint": "/api/v1/privacy/redact"})
    for c in claims["corrections"]:
        if c.get("claim") == "PCA retains 56% of variance":
            m = (subs.get("M1-fallback") or {}).get("measured", {})
            v = m.get("pca_variance_retained")
            if v:
                c["status"] = f"CORRECTED to {v * 100:.1f}%"
                c["reason"] = (f"Measured from the fitted model: {m.get('pca_components')} components, "
                               f"explained_variance_ratio_.sum() = {v}.")
        if c.get("claim") == "Reported defects are trustworthy without a gate":
            gm = (subs.get("M_GATE") or {}).get("measured", {})
            if gm.get("false_positive_rate") is not None:
                c["reason"] = re.sub(r"Now [\d.]+% false positives with [\d.]+% detection",
                                     f"Now {gm['false_positive_rate'] * 100:.1f}% false positives with "
                                     f"{gm['detection_rate'] * 100:.1f}% detection", c["reason"])
    have = {c.get("claim") for c in claims["corrections"]}
    for claim, status, reason in AUDIT_CORRECTIONS:
        if claim not in have:
            claims["corrections"].append({"claim": claim, "status": status, "reason": reason})


def _deep_extras(claims, rep):
    """U-Net segmenter and YOLOv8 road-damage detector, from their own reports.
    Nothing is written for a model that has not been trained."""
    subs = {s["id"]: s for s in claims["subsystems"]}
    sel = rep("segmenter_selection.json")
    unet = rep("defect_segmenter_unet.json")
    if sel and unet and not sel.get("smoke") and "M_SEG" in subs:
        seg = subs["M_SEG"]
        t = sel.get("test") or {}
        tu, tp = t.get("unet") or {}, t.get("pixel_classifier") or {}

        def io(d, c):
            return (d.get(c) or {}).get("iou")
        cmp_txt = (f"Held-out test (same 500 photographs): U-Net crack IoU {io(tu, 'crack')}, pothole IoU "
                   f"{io(tu, 'pothole')}; pixel classifier crack {io(tp, 'crack')}, pothole {io(tp, 'pothole')}.")
        if sel.get("served") == "unet" and os.path.exists(os.path.join(CKPT, "defect_segmenter_unet.onnx")):
            fp = unet.get("false_positives_on_clean_roads") or {}
            seg["architecture"] = ("U-Net with a ResNet-18 encoder (ImageNet-pretrained, all layers trained on the DNIT "
                                   "polygons), flip TTA, ONNX Runtime; per-class thresholds tuned on the calibration split")
            seg["why"] = "Chosen by a rule fixed before the test set was scored: " + sel.get("rule", "") + " " + sel.get("why", "")
            seg["rejected_alternative"] = ("HistGradientBoosting pixel classifier on 11 hand-designed features (kept as "
                                           "the fallback). " + cmp_txt)
            seg["evidence"] = ("checkpoints/defect_segmenter_unet.json, checkpoints/segmenter_selection.json, "
                               "training/train_unet_segmenter.py")
            seg["reproduce"] = "python -m training.train_unet_segmenter"
            seg["measured"] = {"crack_iou": io(tu, "crack"), "pothole_iou": io(tu, "pothole"),
                               "crack_dice": (tu.get("crack") or {}).get("dice"),
                               "pothole_dice": (tu.get("pothole") or {}).get("dice"),
                               "clean_road_false_blob_rate": fp.get("photo_rate_any_blob"),
                               "clean_photographs_scored": fp.get("clean_photographs_scored"),
                               "test_photographs": (unet.get("trained_on") or {}).get("test_photographs"),
                               "pixel_classifier_crack_iou": io(tp, "crack"),
                               "pixel_classifier_pothole_iou": io(tp, "pothole")}
            seg["not_claimed"] = (f"IoU of {io(tu, 'crack')} (crack) and {io(tu, 'pothole')} (pothole) on DNIT "
                                  f"(Brazilian) photographs; not measured on Indian or bus-camera frames. "
                                  f"On held-out clean roads it draws a false blob on "
                                  f"{(fp.get('photo_rate_any_blob') or 0) * 100:.1f}% of "
                                  f"{fp.get('clean_photographs_scored')} photographs.")
        else:
            seg["deep_alternative_tried"] = ("A U-Net (ResNet-18 encoder) was trained on the same polygons and did NOT "
                                             "replace this model: " + sel.get("why", "") + " " + cmp_txt)
        dc = sel.get("deployment_check") or {}
        if dc and dc.get("passed") is False and sel.get("iou_selection_served") == "unet":
            claim = (f"The U-Net segmenter serves (pothole IoU {io(tu, 'pothole') or 0:.3f} vs "
                     f"{io(tp, 'pothole') or 0:.3f})" + (" - multi-dataset run" if sel.get("trained_with") else ""))
            if sel.get("trained_with"):
                reason = ("Retrained on several pixel-labelled datasets, it won the IoU selection on DNIT calibration "
                          "photographs, but the end-to-end check through audit_image() on other datasets kept the pixel "
                          "classifier: " + dc.get("why", "") + ".")
            else:
                reason = ("It won mask IoU on the DNIT test photographs and was briefly served on that basis. A pipeline "
                          "test then failed on Kaggle pothole photographs, and the end-to-end check through audit_image() "
                          "on other datasets reversed it: " + dc.get("why", "") + ". The pixel classifier serves; the "
                          "U-Net stays on disk, measured. A better mask on its training dataset was not a better product.")
            if claim not in {c.get("claim") for c in claims["corrections"]}:
                claims["corrections"].append({"claim": claim, "status": "REVERSED", "reason": reason})
    det = rep("road_damage_detector_report.json")
    if det and os.path.exists(os.path.join(CKPT, "damage_rdd2022_india.onnx")):
        from models.road_damage_detector import detector_blocked_by
        blocked = detector_blocked_by(det)
        t = det.get("test") or {}
        entry = {
            "id": "M_RDD", "name": "Road-damage detector (RDD2022 India)",
            "architecture": det.get("model"),
            "role": "Draw boxes around cracks (D00/D10/D20) and potholes (D40) in a road frame",
            "why": ("A whole-image classifier cannot say where in a frame the damage is. Trained on the RDD2022 India "
                    "boxes; weights and serving threshold chosen on the validation split, test scored once."),
            "measured": {"test_map50": t.get("map50"), "test_map50_95": t.get("map50_95"),
                         "test_precision": t.get("precision"), "test_recall": t.get("recall"),
                         "per_class": t.get("per_class"),
                         "test_photographs": ((det.get("data") or {}).get("photographs") or {}).get("test"),
                         "serving_confidence": det.get("serving_confidence")},
            "evidence": "checkpoints/road_damage_detector_report.json, training/train_rdd_detector.py",
            "reproduce": "python -m training.train_rdd_detector",
            "not_claimed": det.get("not_claimed"),
            "served": blocked is None,
            "checks": {"artefact": det.get("artefact_check"), "clean_roads": det.get("deployment_check")},
        }
        if blocked:
            entry["role"] = ("Trained and measured, NOT shown: the " + blocked.replace("_", " ") + " in "
                             "scripts/verify_rdd_detector.py is missing or failed")
        if "M_RDD" in subs:
            subs["M_RDD"].update(entry)
        else:
            claims["subsystems"].append(entry)


if __name__ == "__main__":
    main()
