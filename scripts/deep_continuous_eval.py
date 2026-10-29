"""
ROAD-SHIELD: Automated Continuous Deep Evaluation & Visual Verification Suite
=============================================================================
Runs rigorous, multi-condition end-to-end audits across real road photographs
and outputs both quantitative MLOps metrics AND rendered visual overlay images
in reports/visual_verification/ for human visual audit.
"""

import os
import sys
import json
import time
import glob
import base64
import numpy as np
from PIL import Image, ImageDraw

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE_ROOT)

from api.vercel_inference import audit_image_base64

VISUAL_DIR = os.path.join(ENGINE_ROOT, "reports", "visual_verification")
os.makedirs(VISUAL_DIR, exist_ok=True)

def encode_image(img_path):
    with open(img_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

def find_test_images():
    candidates = []
    
    # 1. Real user test scenes
    rural_road = os.path.join(ENGINE_ROOT, "scratch", "clean_original_road.png")
    if os.path.exists(rural_road):
        candidates.append({"type": "rural_flooded_puddles", "path": rural_road})

    asphalt_road = os.path.join(ENGINE_ROOT, "scratch", "user_test_scene_2.png")
    if os.path.exists(asphalt_road):
        candidates.append({"type": "asphalt_center_pothole", "path": asphalt_road})

    # 2. RDD2022 Indian road samples
    rdd_files = glob.glob(os.path.join(ENGINE_ROOT, "datasets", "01_rdd2022_india", "**", "*.jpg"), recursive=True)
    for p in rdd_files[:4]:
        candidates.append({"type": "rdd2022_asphalt_distress", "path": p})

    # 3. Kaggle Potholes
    pothole_files = glob.glob(os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "**", "*.jpg"), recursive=True)
    for p in pothole_files[:4]:
        candidates.append({"type": "kaggle_asphalt_pothole", "path": p})

    # 4. CRACK500 Fatigue cracks
    crack_files = glob.glob(os.path.join(ENGINE_ROOT, "datasets", "03_crack500_fatigue", "**", "*.jpg"), recursive=True)
    for p in crack_files[:3]:
        candidates.append({"type": "crack500_fatigue_crack", "path": p})

    # 5. Monsoon waterlogging hazard
    water_files = glob.glob(os.path.join(ENGINE_ROOT, "datasets", "09_waterlogging_hazard", "**", "*.jpg"), recursive=True)
    for p in water_files[:3]:
        candidates.append({"type": "monsoon_waterlogging", "path": p})

    return candidates

def render_visual_verification_image(orig_path, res, out_name):
    """Draws segmentation mask and defect bounding boxes directly onto original image."""
    try:
        orig_im = Image.open(orig_path).convert("RGBA")
        seg = res.get("segmentation", {})
        mask_url = seg.get("mask_data_url")
        
        composite = orig_im
        if mask_url and "base64," in mask_url:
            raw_png = base64.b64decode(mask_url.split("base64,")[1])
            mask_im = Image.open(io.BytesIO(raw_png)).convert("RGBA")
            if mask_im.size != orig_im.size:
                mask_im = mask_im.resize(orig_im.size, Image.NEAREST)
            composite = Image.alpha_composite(orig_im, mask_im)

        # Draw bounding boxes and labels
        draw = ImageDraw.Draw(composite)
        dets = res.get("all_detections", [])
        for d in dets:
            if not d.get("is_distress"):
                continue
            box = d.get("bbox_pixels")
            if not box:
                continue
            bx, by, bw, bh = box
            cls_name = d.get("class_name", "Defect")
            conf = int(d.get("confidence", 0.90) * 100)
            cost = d.get("repair_cost_inr", 0.0)
            # Cyan box for pothole, yellow for crack
            color = "#06b6d4" if d.get("class_id") == 2 else "#eab308"
            draw.rectangle([bx, by, bx + bw, by + bh], outline=color, width=2)
            label = f"{cls_name} {conf}% | Rs.{cost:.0f}"
            draw.rectangle([bx, max(0, by - 16), bx + len(label) * 7, by], fill=color)
            draw.text((bx + 2, max(0, by - 15)), label, fill="#000000")

        out_path = os.path.join(VISUAL_DIR, out_name)
        composite.convert("RGB").save(out_path)
        return out_path
    except Exception as e:
        return f"Error rendering: {e}"

import io

def run_suite():
    print("=" * 80)
    print("  ROAD-SHIELD CONTINUOUS DEEP EVALUATION & VISUAL VERIFICATION SUITE")
    print("=" * 80)
    images = find_test_images()
    print(f"Discovered {len(images)} representative evaluation test photographs.\n")

    results = []
    crashes = 0
    giant_box_violations = 0
    unpriced_violations = 0
    area_overflow_violations = 0

    print(f"{'IMAGE':<32} {'TYPE':<25} {'DETS':<6} {'GIANT':<6} {'UNPRICED':<9} {'COST (INR)':<12} {'VISUAL OVERLAY'}")
    print("-" * 115)

    for i, info in enumerate(images):
        path = info["path"]
        cat = info["type"]
        b64 = encode_image(path)
        
        t0 = time.time()
        try:
            res = audit_image_base64(b64)
            elapsed_ms = round((time.time() - t0) * 1000.0, 1)
        except Exception as e:
            crashes += 1
            print(f"{os.path.basename(path)[:30]:<32} {cat:<25} CRASH: {str(e)[:40]}")
            continue

        dets = res.get("all_detections", [])
        seg = res.get("segmentation", {})
        ledger = res.get("morth_civil_ledger", {})
        
        giant_boxes = [d for d in dets if d.get("area_diagnostics", {}).get("box_share_of_frame", 0.0) > 0.35 and d.get("is_distress")]
        unpriced_defects = [d for d in dets if d.get("is_distress") and d.get("needs_manual_survey")]
        area_overflows = [d for d in dets if d.get("is_distress") and d.get("surface_area_m2", 0.0) > 3.0]
        
        giant_box_violations += len(giant_boxes)
        unpriced_violations += len(unpriced_defects)
        area_overflow_violations += len(area_overflows)

        out_name = f"eval_{i+1:02d}_{os.path.splitext(os.path.basename(path))[0]}.png"
        v_path = render_visual_verification_image(path, res, out_name)

        cost_val = ledger.get("total_estimated_repair_inr", 0.0) if ledger else 0.0
        cost_str = f"Rs.{cost_val:,.2f}" if cost_val > 0 else "Rs.0.00"
        distress_count = len([d for d in dets if d.get("is_distress")])
        
        print(f"{os.path.basename(path)[:30]:<32} {cat:<25} {distress_count:<6} {len(giant_boxes):<6} {len(unpriced_defects):<9} {cost_str:<12} {os.path.basename(out_name)}")

        results.append({
            "path": os.path.basename(path),
            "category": cat,
            "distress_detections_count": distress_count,
            "giant_boxes_count": len(giant_boxes),
            "unpriced_defects_count": len(unpriced_defects),
            "area_overflow_count": len(area_overflows),
            "total_cost_inr": cost_val,
            "visual_overlay": out_name,
            "elapsed_ms": elapsed_ms,
        })

    print("=" * 115)
    print("\nEVALUATION SUMMARY & QUALITY GATES:")
    print(f"  Total Images Audited         : {len(images)}")
    print(f"  Pipeline Crashes             : {crashes}  (Target: 0)")
    print(f"  Giant Box Violations (>35%)  : {giant_box_violations}  (Target: 0)")
    print(f"  Unpriced Defect Violations   : {unpriced_violations}  (Target: 0)")
    print(f"  Area Overflow (>3.0 m2)      : {area_overflow_violations}  (Target: 0)")
    print(f"  Visual Verification Images   : {len(images)} saved to reports/visual_verification/")

    report_path = os.path.join(ENGINE_ROOT, "reports", "continuous_deep_eval_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "total_images": len(images),
            "crashes": crashes,
            "giant_box_violations": giant_box_violations,
            "unpriced_violations": unpriced_violations,
            "area_overflow_violations": area_overflow_violations,
            "results": results
        }, f, indent=2)

    all_passed = (crashes == 0 and giant_box_violations == 0 and unpriced_violations == 0 and area_overflow_violations == 0)
    if all_passed:
        print("\n>>> ALL MLOPS QUALITY GATES PASSED PERFECTLY! <<<")
    else:
        print("\n>>> QUALITY GATES FAILED - REVIEW LOGS ABOVE <<<")

if __name__ == "__main__":
    run_suite()
