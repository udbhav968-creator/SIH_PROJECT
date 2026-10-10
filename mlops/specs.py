"""
What each served model is, on disk and on paper.

files      glob patterns under checkpoints/ that together are one version of the model
metrics    report file -> {metric name: dotted path in that JSON}; recorded with every version
gate       the promotion rule: the primary metric must be at least `floor` and may not fall more
           than `max_drop` below the version in production; each guard metric has its own limit
train      how to retrain it: a command run on this machine (CPU minutes), or "colab" when it
           needs a GPU (notebooks/ and scripts/colab_*); the registry then takes the result in
           with `python -m mlops register <name>` after the files are copied back
"""

SPECS = {
    "vision_classifier": {
        "title": "Road-condition classifier (fine-tuned CNN)",
        "task": "7-class whole-frame classification",
        "files": ["deep_vision_*.onnx", "deep_vision_*.json", "finetune_summary.json", "finetune_*_report.json",
                  "vision_model_selection.json", "cnn_head_mobilenetv2.joblib", "cnn_head_mobilenetv2_report.json",
                  "vision_ensemble.json", "ensemble_report.json"],
        "metrics": {"finetune_summary.json": {"test_macro_f1": "served_test.macro_f1",
                                              "test_accuracy": "served_test.accuracy",
                                              "indian_roads_macro_f1": "served_indian_roads.macro_f1",
                                              "cpu_ms_per_image": "served_onnx.onnx_cpu_ms_per_image"}},
        "gate": {"primary": "test_macro_f1", "direction": "max", "floor": 0.75, "max_drop": 0.01,
                 "guards": {"indian_roads_macro_f1": {"direction": "max", "max_drop": 0.02},
                            "cpu_ms_per_image": {"direction": "min", "max_rise": 15.0}}},
        "train": "colab",
        "shadow": "vision",
    },
    "segmenter": {
        "title": "Defect segmenter (U-Net / pixel classifier)",
        "task": "crack and pothole pixel masks",
        "files": ["defect_segmenter_unet.onnx", "defect_segmenter_unet.json", "defect_segmenter.joblib",
                  "defect_segmenter_report.json", "segmenter_selection.json", "semantic_gate_report.json"],
        "metrics": {"segmenter_selection.json": {"pothole_iou_unet": "test.unet.pothole.iou",
                                                 "crack_iou_unet": "test.unet.crack.iou",
                                                 "clean_false_blob_rate_unet": "test.unet.clean.photo_rate_any_blob"}},
        "gate": {"primary": "pothole_iou_unet", "direction": "max", "floor": 0.4, "max_drop": 0.02,
                 "guards": {"crack_iou_unet": {"direction": "max", "max_drop": 0.03},
                            "clean_false_blob_rate_unet": {"direction": "min", "max_rise": 0.02}}},
        "train": "colab",
    },
    "road_damage_detector": {
        "title": "Road-damage detector (YOLOv8, RDD2022 India)",
        "task": "boxes for D00/D10/D20/D40",
        "files": ["damage_rdd2022_india.onnx", "damage_rdd2022_india.json"],
        "metrics": {"damage_rdd2022_india.json": {"test_map50": "test.map50", "test_map50_95": "test.map50_95"}},
        "gate": {"primary": "test_map50", "direction": "max", "floor": 0.3, "max_drop": 0.02},
        "train": "colab",
    },
    "coco_detector": {
        "title": "COCO object detector (people, vehicles, signs)",
        "task": "80-class detection; vehicle counts feed the traffic estimate",
        "files": ["road_shield_detector.onnx"],
        "metrics": {},
        "gate": {"primary": None, "manual": "published weights, not trained here: promote by hand after review"},
        "train": "python -m scripts.fetch_detector",
    },
    "imu_classifier": {
        "title": "IMU shock classifier",
        "task": "4-class accelerometer windows",
        "files": ["imu_shock_model.joblib", "imu_shock_report.json", "imu_shock_cnn.onnx", "imu_shock_cnn.json",
                  "imu_model_selection.json", "imu_deep_report.json"],
        "metrics": {"imu_shock_report.json": {"held_out_accuracy": "held_out_validation_accuracy",
                                              "held_out_macro_f1": "per_class_report.macro avg.f1-score",
                                              "pothole_recall": "per_class_report.Pothole Impact.recall"}},
        "gate": {"primary": "held_out_macro_f1", "direction": "max", "floor": 0.6, "max_drop": 0.02,
                 "guards": {"pothole_recall": {"direction": "max", "max_drop": 0.05}}},
        "train": ["python", "-m", "training.train_imu"],
    },
    "pci_regressor": {
        "title": "PCI regressor (ASTM D6433 deduct curves)",
        "task": "pavement condition index from distress densities",
        "files": ["pci_model.joblib", "pci_model_report.json"],
        "metrics": {"pci_model_report.json": {"held_out_r2": "held_out_r2", "held_out_mae": "held_out_mae"}},
        "gate": {"primary": "held_out_r2", "direction": "max", "floor": 0.85, "max_drop": 0.01,
                 "guards": {"held_out_mae": {"direction": "min", "max_rise": 0.5}}},
        "train": ["python", "-m", "training.train_civil_models"],
    },
    "depth_estimator": {
        "title": "Pothole depth estimator",
        "task": "depth in cm from photometric features",
        "files": ["depth_estimator_model.joblib", "depth_estimator_report.json"],
        "metrics": {"depth_estimator_report.json": {"held_out_pothole_r2": "held_out_pothole_r2",
                                                    "held_out_pothole_mae_cm": "held_out_pothole_mae_cm"}},
        "gate": {"primary": "held_out_pothole_r2", "direction": "max", "floor": 0.5, "max_drop": 0.02},
        "train": ["python", "-m", "training.train_civil_models"],
    },
    "deterioration_forecaster": {
        "title": "Monsoon deterioration forecaster",
        "task": "area and depth growth of a defect",
        "files": ["deterioration_model.joblib", "deterioration_model_report.json"],
        "metrics": {"deterioration_model_report.json": {"held_out_area_r2": "held_out_area_r2",
                                                        "held_out_depth_r2": "held_out_depth_r2"}},
        "gate": {"primary": "held_out_area_r2", "direction": "max", "floor": 0.8, "max_drop": 0.02},
        "train": ["python", "-m", "training.train_civil_models"],
    },
    "crack_verifier": {
        "title": "Crack verifier (gate on segmenter crack components)",
        "task": "is the crop around a proposed crack a pavement crack",
        "files": ["crack_verifier.npz", "crack_verifier_report.json"],
        "metrics": {"crack_verifier_report.json": {"auroc_road_scene": "test.summary.auroc_crack_vs_road_scene",
                                                   "crack_kept": "test.summary.crack_kept",
                                                   "road_scene_rejected": "test.summary.road_scene_rejected"}},
        "gate": {"primary": "auroc_road_scene", "direction": "max", "floor": 0.9, "max_drop": 0.02,
                 "guards": {"crack_kept": {"direction": "max", "max_drop": 0.02}}},
        "train": ["python", "-m", "training.train_crack_verifier"],
        "note": "a retrained verifier is not served until python -m scripts.select_crack_gate measures it end to end",
    },
    "ood_guard": {
        "title": "Input guard (out-of-distribution)",
        "task": "is this a usable road photograph",
        "files": ["ood_guard.npz", "ood_guard_report.json", "monitoring_reference.json"],
        "metrics": {"ood_guard_report.json": {"auroc_combined": "test.combined.auroc",
                                              "ood_caught_rate": "test.combined.ood_caught_rate",
                                              "road_flagged_rate": "test.combined.in_distribution_flagged_rate",
                                              "road_refused_rate": "test.refusals.road_photos_refused_rate"}},
        "gate": {"primary": "auroc_combined", "direction": "max", "floor": 0.9, "max_drop": 0.01,
                 "guards": {"road_refused_rate": {"direction": "min", "max_rise": 0.01, "ceiling": 0.03},
                            "road_flagged_rate": {"direction": "min", "max_rise": 0.02, "ceiling": 0.08}}},
        "train": ["python", "-m", "training.train_ood_guard", "--ood-dir", "{ood_dir}"],
    },
}


def spec(name):
    if name not in SPECS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(SPECS)}")
    return SPECS[name]
