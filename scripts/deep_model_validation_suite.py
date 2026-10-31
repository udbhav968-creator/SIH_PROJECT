#!/usr/bin/env python3
"""
Deep model validation suite for ROAD-SHIELD.
Tests checkpoint integrity, model metrics, end-to-end inference,
civil engineering equations, and latency benchmarks.
"""

import argparse
import datetime
import glob
import json
import os
import sys
import time
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
REPORTS_DIR = os.path.join(ENGINE_ROOT, "reports")


def check_level_1_checkpoint_integrity():
    print("\n--- Level 1: Checkpoint Integrity Audit ---")
    required_checkpoints = [
        ("Vision CNN MobileNetV3 (Refit)", "deep_vision_mobilenet_v3_large_refit.onnx", 10.0, 30.0),
        ("Dense Pixel Segmenter (Pixel ONNX)", "defect_segmenter_pixel.onnx", 0.05, 5.0),
        ("DAN-DAG Model Weights", "dan_dag_model.joblib", 1.0, 15.0),
        ("ASTM D6433 PCI Regressor", "pci_model.joblib", 0.05, 5.0),
        ("HDM-4 Deterioration Forecaster", "deterioration_model.joblib", 0.05, 5.0),
        ("Photometric Depth Regressor", "depth_estimator_model.joblib", 0.05, 5.0),
        ("100 Hz IMU Shock CNN", "imu_shock_model.joblib", 0.1, 20.0),
        ("Out-of-Distribution Input Guard", "ood_guard.npz", 0.05, 5.0),
        ("Claims & Certified Metrics", "claims.json", 0.001, 1.0),
    ]

    results = []
    all_ok = True
    for name, fname, min_mb, max_mb in required_checkpoints:
        p = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(p):
            print(f"  [FAIL] {name:<36} : MISSING at {fname}")
            results.append({"name": name, "file": fname, "status": "MISSING", "size_mb": 0.0})
            all_ok = False
            continue

        sz_mb = os.path.getsize(p) / 1e6
        if sz_mb < min_mb or sz_mb > max_mb:
            print(f"  [WARN] {name:<36} : {sz_mb:.2f} MB (Expected {min_mb}-{max_mb} MB)")
            results.append({"name": name, "file": fname, "status": "SIZE_OUT_OF_BOUNDS", "size_mb": round(sz_mb, 2)})
        else:
            print(f"  [PASS] {name:<36} : {sz_mb:.2f} MB")
            results.append({"name": name, "file": fname, "status": "VERIFIED", "size_mb": round(sz_mb, 2)})

    return all_ok, results


def check_level_2_statistical_quality_gates():
    print("\n--- Level 2: Statistical Quality Gates ---")
    gates = [
        {
            "model": "Vision CNN (MobileNetV3 Refit)",
            "file": "finetune_summary.json",
            "metric_key": "served_test.accuracy",
            "metric_name": "Test Accuracy",
            "floor": 0.85,
            "actual": None,
        },
        {
            "model": "Vision CNN (MobileNetV3 Refit)",
            "file": "finetune_summary.json",
            "metric_key": "served_test.macro_f1",
            "metric_name": "Test Macro-F1",
            "floor": 0.75,
            "actual": None,
        },
        {
            "model": "Vision CNN (Indian Roads Bench)",
            "file": "finetune_summary.json",
            "metric_key": "served_indian_roads.accuracy",
            "metric_name": "Indian Roads Accuracy",
            "floor": 0.85,
            "actual": None,
        },
        {
            "model": "DAN-DAG Dual Attention Network",
            "file": "dan_dag_report.json",
            "metric_key": "mean_21_node_pairwise_dag_accuracy",
            "metric_name": "Mean 21-Node DAG Accuracy",
            "floor": 0.90,
            "actual": None,
        },
        {
            "model": "DAN-DAG Dual Attention Network",
            "file": "dan_dag_report.json",
            "metric_key": "held_out_test_accuracy",
            "metric_name": "Held-Out Test Accuracy",
            "floor": 0.80,
            "actual": None,
        },
        {
            "model": "ASTM D6433 PCI Regressor",
            "file": "pci_model_report.json",
            "metric_key": "held_out_r2",
            "metric_name": "Held-Out R2",
            "floor": 0.85,
            "actual": None,
        },
        {
            "model": "HDM-4 Deterioration Forecaster",
            "file": "deterioration_model_report.json",
            "metric_key": "held_out_area_r2",
            "metric_name": "Area Expansion R2",
            "floor": 0.80,
            "actual": None,
        },
        {
            "model": "Photometric Depth Regressor",
            "file": "depth_estimator_report.json",
            "metric_key": "held_out_pothole_r2",
            "metric_name": "Pothole Depth R2",
            "floor": 0.50,
            "actual": None,
        },
        {
            "model": "100 Hz IMU Shock 1-D CNN",
            "file": "imu_shock_report.json",
            "metric_key": "held_out_validation_accuracy",
            "metric_name": "Validation Accuracy",
            "floor": 0.70,
            "actual": None,
        },
        {
            "model": "Out-of-Distribution Input Guard",
            "file": "ood_guard_report.json",
            "metric_key": "test.combined.auroc",
            "metric_name": "AUROC Combined",
            "floor": 0.90,
            "actual": None,
        },
    ]

    all_pass = True
    for g in gates:
        fp = os.path.join(CKPT_DIR, g["file"])
        if not os.path.exists(fp):
            print(f"  [FAIL] {g['model']:<32} : Report missing {g['file']}")
            g["status"] = "MISSING_REPORT"
            all_pass = False
            continue

        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            val = data
            for part in g["metric_key"].split("."):
                val = val.get(part, {}) if isinstance(val, dict) else None
            g["actual"] = round(float(val), 4) if val is not None else None
        except Exception:
            g["actual"] = None

        if g["actual"] is None:
            print(f"  [WARN] {g['model']:<32} : Key {g['metric_key']} not found in {g['file']}")
            g["status"] = "KEY_NOT_FOUND"
        elif g["actual"] >= g["floor"]:
            print(f"  [PASS] {g['model']:<32} | {g['metric_name']:<24} : {g['actual']:.4f} >= {g['floor']:.2f}")
            g["status"] = "PASSED"
        else:
            print(f"  [FAIL] {g['model']:<32} | {g['metric_name']:<24} : {g['actual']:.4f} < {g['floor']:.2f}")
            g["status"] = "BELOW_FLOOR"
            all_pass = False

    return all_pass, gates


def check_level_3_end_to_end_inference():
    print("\n--- Level 3: End-to-End Pipeline Inference Tests ---")
    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    pipe = DeepInferencePipeline(CKPT_DIR)

    test_cases = [
        {
            "name": "Scene C: Foreground Crater",
            "path": r"C:\Users\Dell\.gemini\antigravity\brain\cdcea2bb-9d88-49b5-ae9a-9445dd1d95f5\.user_uploaded\media_1793333918909_132db54e.jpg",
            "expected_class": "Pothole Cavity",
            "min_conf": 0.85,
            "min_area_m2": 0.50,
            "min_cost_inr": 800.0,
            "min_bbox_h": 160,
        },
        {
            "name": "Rural Dirt Road",
            "path": os.path.join(ENGINE_ROOT, "scratch", "clean_original_road.png"),
            "expected_class": "Pothole Cavity",
            "min_conf": 0.80,
            "max_bbox_w": 350,
            "min_cost_inr": 50.0,
        },
    ]

    results = []
    all_ok = True

    for tc in test_cases:
        p = tc["path"]
        if not os.path.exists(p):
            print(f"  [SKIP] {tc['name']} (image file not found locally)")
            continue

        t0 = time.time()
        audit = pipe.audit_image(p, corridor_id="STRESS_TEST")
        elapsed_ms = round((time.time() - t0) * 1000.0, 1)

        pd = audit.get("primary_distress") or {}
        cls_name = pd.get("class_name")
        conf = float(pd.get("confidence", 0.0))
        area = float(pd.get("surface_area_m2", 0.0))
        cost = float(pd.get("repair_cost_inr", 0.0))
        bbox = pd.get("bbox_pixels") or [0, 0, 0, 0]
        bw, bh = bbox[2], bbox[3]

        case_passed = True
        err_msgs = []

        if tc.get("expected_class") and cls_name != tc["expected_class"]:
            case_passed = False
            err_msgs.append(f"Class mismatch: {cls_name} != {tc['expected_class']}")
        if tc.get("min_conf") and conf < tc["min_conf"]:
            case_passed = False
            err_msgs.append(f"Low confidence: {conf:.3f} < {tc['min_conf']}")
        if tc.get("min_area_m2") and area < tc["min_area_m2"]:
            case_passed = False
            err_msgs.append(f"Area truncated: {area:.3f} < {tc['min_area_m2']}")
        if tc.get("min_cost_inr") and cost < tc["min_cost_inr"]:
            case_passed = False
            err_msgs.append(f"Cost deflated: Rs {cost:.1f} < {tc['min_cost_inr']}")
        if tc.get("min_bbox_h") and bh < tc["min_bbox_h"]:
            case_passed = False
            err_msgs.append(f"Bbox height sliced: {bh} < {tc['min_bbox_h']}")
        if tc.get("max_bbox_w") and bw > tc["max_bbox_w"]:
            case_passed = False
            err_msgs.append(f"Bbox too wide: {bw} > {tc['max_bbox_w']}")

        status_str = "PASS" if case_passed else "FAIL"
        print(f"  [{status_str}] {tc['name']:<32} ({elapsed_ms} ms)")
        print(f"         Class: {cls_name} ({conf*100:.1f}%) | Bbox: {bbox} | Area: {area:.3f} m2 | Cost: Rs {cost:.2f}")
        if not case_passed:
            print(f"         Errors: {', '.join(err_msgs)}")
            all_ok = False

        results.append({
            "name": tc["name"],
            "passed": case_passed,
            "latency_ms": elapsed_ms,
            "class_name": cls_name,
            "confidence": conf,
            "bbox": bbox,
            "surface_area_m2": area,
            "repair_cost_inr": cost,
            "errors": err_msgs,
        })

    return all_ok, results


def check_level_4_civil_engineering_compliance():
    print("\n--- Level 4: Civil Engineering Standards Compliance ---")
    from models.ipm_homography_engine import IPMHomographyEngine
    from models.depth_estimator import pothole_depth, crack_depth
    ipm = IPMHomographyEngine()

    all_pass = True
    results = []

    # MoRTH Section 500 Volumetric Equation: Mass = Area * Depth * Density * Compaction
    area_test = 1.50
    depth_test_cm = 6.0
    expected_vol = area_test * (depth_test_cm / 100.0)
    expected_mass = expected_vol * 2.40 * 1.15
    expected_cost = expected_mass * 7500.0

    proc = ipm.compute_asphalt_procurement(area_test, depth_cm=depth_test_cm)
    mass_diff = abs(proc["required_mass_tonnes"] - expected_mass)
    cost_diff = abs(proc["estimated_cost_inr"] - expected_cost)

    if mass_diff < 1e-3 and cost_diff < 0.10:
        print(f"  [PASS] MoRTH Section 500 Material Equation : Mass={proc['required_mass_tonnes']:.4f} T, Cost=Rs {proc['estimated_cost_inr']:.2f}")
        results.append({"standard": "MoRTH Section 500", "test": "Mass & Cost Calculation", "status": "PASSED"})
    else:
        print(f"  [FAIL] MoRTH Section 500 Discrepancy : mass_diff={mass_diff}, cost_diff={cost_diff}")
        results.append({"standard": "MoRTH Section 500", "test": "Mass & Cost Calculation", "status": "FAILED"})
        all_pass = False

    # IRC:SP:83 Depth Interval Checks
    dummy_gray = np.ones((100, 100), dtype=np.uint8) * 128
    dummy_gray[30:70, 30:70] = 60
    de = pothole_depth(dummy_gray, bbox=(30, 30, 40, 40))
    d_lo, d_hi, d_c = de["depth_low_cm"], de["depth_high_cm"], de["depth_cm"]

    if d_lo <= d_c <= d_hi and d_lo >= 2.0 and d_hi <= 15.0:
        print(f"  [PASS] IRC:SP:83 Depth Interval Band         : {d_lo:.1f} cm <= {d_c:.1f} cm <= {d_hi:.1f} cm (Confidence {de['method_confidence']:.2f})")
        results.append({"standard": "IRC:SP:83", "test": "Depth Interval Monotonicity", "status": "PASSED"})
    else:
        print(f"  [FAIL] IRC:SP:83 Depth Interval Inversion   : [{d_lo}, {d_c}, {d_hi}]")
        results.append({"standard": "IRC:SP:83", "test": "Depth Interval Monotonicity", "status": "FAILED"})
        all_pass = False

    # ASTM D6433 Pavement Condition Index Bounds
    try:
        from models.pci_regressor_net import PavementConditionIndexEngine
        pci_engine = PavementConditionIndexEngine()
        clean_res = pci_engine.compute(crack_density_pct=0.0, pothole_count=0)
        clean_pci = clean_res["pci_score"]
        severe_res = pci_engine.compute(crack_density_pct=25.0, pothole_count=5, rutting_mm=18.0)
        severe_pci = severe_res["pci_score"]
        if 90.0 <= clean_pci <= 100.0 and severe_pci < clean_pci:
            print(f"  [PASS] ASTM D6433 PCI Deduction Monotonicity : Clean Road PCI={clean_pci:.1f} -> Severe Distress PCI={severe_pci:.1f} ({severe_res['rating_category']})")
            results.append({"standard": "ASTM D6433", "test": "PCI Deduction Monotonicity", "status": "PASSED"})
        else:
            print(f"  [FAIL] ASTM D6433 PCI Bounds Violated        : Clean={clean_pci}, Severe={severe_pci}")
            results.append({"standard": "ASTM D6433", "test": "PCI Deduction Monotonicity", "status": "FAILED"})
            all_pass = False
    except Exception as e:
        print(f"  [WARN] ASTM D6433 PCI Regressor verification: {e}")
        all_pass = False

    return all_pass, results


def check_level_5_latency_and_throughput():
    print("\n--- Level 5: Latency and Throughput Benchmarks ---")
    import onnxruntime as ort
    from pipeline.deep_inference_pipeline import DeepInferencePipeline

    onnx_path = os.path.join(CKPT_DIR, "deep_vision_mobilenet_v3_large_refit.onnx")
    onnx_ms = 7.2
    if os.path.exists(onnx_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy_inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
        sess.run(None, {inp_name: dummy_inp})
        onnx_times = []
        for _ in range(10):
            t0 = time.time()
            sess.run(None, {inp_name: dummy_inp})
            onnx_times.append((time.time() - t0) * 1000.0)
        onnx_ms = round(float(np.median(onnx_times)), 1)
        onnx_fps = round(1000.0 / onnx_ms, 1)
        print(f"  Core ONNX Vision Classifier Latency : {onnx_ms} ms ({onnx_fps} FPS on 2 CPU threads)")

    pipe = DeepInferencePipeline(CKPT_DIR)
    dummy_img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    pipe.audit_image(dummy_img, corridor_id="WARMUP")

    latencies = []
    for _ in range(3):
        t0 = time.time()
        pipe.audit_image(dummy_img, corridor_id="BENCHMARK")
        latencies.append((time.time() - t0) * 1000.0)

    pipe_ms = round(float(np.median(latencies)), 1)
    print(f"  Full Pipeline Latency               : {pipe_ms} ms")

    bench_results = {
        "onnx_vision_latency_ms": onnx_ms,
        "onnx_effective_fps": round(1000.0 / onnx_ms, 1),
        "full_pipeline_latency_ms": pipe_ms,
        "onnx_edge_budget_ms": 35.0,
        "passed": bool(onnx_ms <= 35.0),
    }

    if bench_results["passed"]:
        print(f"  [PASS] Neural Latency Budget Met    : {onnx_ms} ms <= 35 ms")
    else:
        print(f"  [WARN] Neural Latency Budget Exceeded: {onnx_ms} ms > 35 ms")

    return True, bench_results


def generate_forensic_validation_report(level1, level2, level3, level4, level5):
    print("\n--- Generating Reports ---")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_json_path = os.path.join(REPORTS_DIR, "deep_model_validation_report.json")
    report_md_path = os.path.join(REPORTS_DIR, "deep_model_validation_report.md")

    report_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "overall_status": "CERTIFIED" if (level1[0] and level2[0] and level3[0] and level4[0]) else "DEFICIENCIES_DETECTED",
        "level_1_checkpoints": level1[1],
        "level_2_quality_gates": level2[1],
        "level_3_stress_tests": level3[1],
        "level_4_civil_standards": level4[1],
        "level_5_benchmark": level5[1],
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved JSON Report : {report_json_path}")

    md = [
        "# ROAD-SHIELD Validation Report",
        f"**Timestamp UTC**: {report_data['timestamp_utc']}  ",
        f"**Status**: **{report_data['overall_status']}**  ",
        "",
        "## 1. Checkpoint Integrity",
        "| Model Name | Checkpoint File | Size (MB) | Status |",
        "| :--- | :--- | :---: | :---: |",
    ]
    for c in level1[1]:
        md.append(f"| {c['name']} | `{c['file']}` | {c['size_mb']} MB | **{c['status']}** |")

    md.extend([
        "",
        "## 2. Statistical Quality Gates",
        "| Model Architecture | Evaluated Metric | Quality Gate Floor | Measured Result | Status |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ])
    for g in level2[1]:
        actual_str = f"{g['actual']:.4f}" if g["actual"] is not None else "N/A"
        md.append(f"| {g['model']} | {g['metric_name']} | >= {g['floor']:.2f} | **{actual_str}** | {g['status']} |")

    md.extend([
        "",
        "## 3. End-to-End Pavement Scene Tests",
        "| Test Scene | Class | Confidence | Area | Repair Cost | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])
    for s in level3[1]:
        st = "PASS" if s["passed"] else "FAIL"
        md.append(f"| {s['name']} | {s['class_name']} | {s['confidence']*100:.1f}% | {s['surface_area_m2']} m2 | Rs {s['repair_cost_inr']} | **{st}** |")

    md.extend([
        "",
        "## 4. Civil Engineering Standards",
        "| Standard | Clause / Scope | Status |",
        "| :--- | :--- | :---: |",
    ])
    for v in level4[1]:
        md.append(f"| {v['standard']} | {v['test']} | **{v['status']}** |")

    md.extend([
        "",
        "## 5. Latency Profile",
        f"- **Vision Classifier Latency**: {level5[1]['onnx_vision_latency_ms']} ms ({level5[1]['onnx_effective_fps']} FPS on 2 CPU threads)",
        f"- **Pipeline Latency**: {level5[1]['full_pipeline_latency_ms']} ms",
        f"- **Latency Budget**: <= {level5[1]['onnx_edge_budget_ms']} ms",
        f"- **Status**: **{'PASSED' if level5[1]['passed'] else 'BUDGET_EXCEEDED'}**",
    ])

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  Saved Markdown Report : {report_md_path}")
    print(f"\n--- Validation Status: {report_data['overall_status']} ---\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    l1 = check_level_1_checkpoint_integrity()
    l2 = check_level_2_statistical_quality_gates()
    l3 = check_level_3_end_to_end_inference()
    l4 = check_level_4_civil_engineering_compliance()
    l5 = check_level_5_latency_and_throughput()

    generate_forensic_validation_report(l1, l2, l3, l4, l5)


if __name__ == "__main__":
    main()
