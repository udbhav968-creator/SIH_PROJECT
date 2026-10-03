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

    claims["generated_unix"] = int(time.time())
    with open(TARGET, "w", encoding="utf-8") as fh:
        json.dump(claims, fh, indent=2)
    print(f"refreshed {TARGET}")
    print(f"  {len(claims['subsystems'])} subsystems, {len(claims['corrections'])} corrections")


if __name__ == "__main__":
    main()
