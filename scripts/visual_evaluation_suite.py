"""Visual test evaluation suite: renders actual annotated test photographs with masks,
bounding boxes, depth profiles, and civil repair estimates so developers and judges
can visually inspect the model's predictions on real images."""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from pipeline.deep_inference_pipeline import DeepInferencePipeline

OUTPUT_DIR = os.path.join(ENGINE_ROOT, "reports", "visual_verification")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def annotate_visual_frame(img_bgr, audit_result, title="Visual Test"):
    """Overlays clean bounding boxes, labels, and civil metrics onto the photograph."""
    H, W = img_bgr.shape[:2]
    canvas = img_bgr.copy()
    
    # Overlay segmentation mask if present
    seg = audit_result.get("defect_segmentation") or audit_result.get("segmentation") or {}
    mask_work = seg.get("mask_work")
    if mask_work is not None:
        mask_full = cv2.resize(mask_work.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros_like(canvas)
        color_mask[mask_full == 1] = [180, 105, 255] # Cracks: purple
        color_mask[mask_full == 2] = [45, 212, 212]  # Potholes: cyan
        alpha = 0.35
        has_mask = (mask_full == 1) | (mask_full == 2)
        canvas[has_mask] = cv2.addWeighted(canvas[has_mask], 1 - alpha, color_mask[has_mask], alpha, 0)
    
    all_dets = audit_result.get("all_detections", [])
    road_dets = [d for d in all_dets if d.get("is_distress") and (d.get("bbox_pixels") or d.get("bbox_normalized"))]
    
    # Draw individual defect boxes with non-colliding #1, #2 tags
    top_y = 1e9
    top_x = 0
    drawn_badges = []
    for idx, d in enumerate(road_dets):
        bn = d.get("bbox_normalized")
        bp = d.get("bbox_pixels")
        if bn:
            bx = int(round(bn[0] * W))
            by = int(round(bn[1] * H))
            bw = int(round(bn[2] * W))
            bh = int(round(bn[3] * H))
        elif bp:
            bx, by, bw, bh = bp
        else:
            continue
            
        if by < top_y:
            top_y = by
            top_x = bx
            
        color = (212, 212, 45) if d.get("area_method") == "segmentation_mask" else (43, 169, 240)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, 2)
        
        # Clean non-colliding index badge #1, #2...
        badge = f"#{idx+1}"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bx_b, by_b = bx, by
        for (ox, oy, ow, oh) in drawn_badges:
            if abs(bx_b - ox) < (tw + 12) and abs(by_b - oy) < (th + 10):
                by_b = min(H - th - 6, by + bh + 2)
                break
        drawn_badges.append((bx_b, by_b, tw + 6, th + 6))
        cv2.rectangle(canvas, (bx_b, by_b), (bx_b + tw + 6, by_b + th + 6), color, -1)
        cv2.putText(canvas, badge, (bx_b + 3, by_b + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 15, 23), 1, cv2.LINE_AA)

    # Consolidated header banner on top
    if len(road_dets) > 1 and top_y < 1e8:
        avg_conf = int(round(np.mean([d.get("confidence", 0.0) for d in road_dets]) * 100))
        dominant_cls = (road_dets[0].get("class_name") or "Defect").split()[0]
        unit = "cavities" if "Pothole" in dominant_cls else "defects"
        banner = f"{dominant_cls}s ({len(road_dets)} {unit}) | Avg {avg_conf}%"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        by = max(10, top_y - bh - 8)
        bx = max(10, min(W - bw - 16, top_x))
        cv2.rectangle(canvas, (bx, by), (bx + bw + 12, by + bh + 8), (10, 15, 23), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw + 12, by + bh + 8), (212, 212, 45), 2)
        cv2.putText(canvas, banner, (bx + 6, by + bh + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (212, 212, 45), 2, cv2.LINE_AA)
    elif len(road_dets) == 1:
        d = road_dets[0]
        bp = d.get("bbox_pixels") or [0, 0, 50, 50]
        pct = int(round(d.get("confidence", 0.0) * 100))
        lbl = f"{d.get('class_name', 'Defect').split()[0]} {pct}%"
        cv2.putText(canvas, lbl, (bp[0], max(20, bp[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (212, 212, 45), 2, cv2.LINE_AA)

    # Top status bar
    pci = audit_result.get("astm_d6433_pci", {}).get("pci_score", "N/A")
    surf = audit_result.get("surface_material", {}).get("name", "Unknown Surface")
    status_bar = f"ROAD-SHIELD VISUAL AUDIT | {title} | ASTM D6433 PCI: {pci} | {surf}"
    cv2.rectangle(canvas, (0, 0), (W, 28), (15, 23, 42), -1)
    cv2.putText(canvas, status_bar, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1, cv2.LINE_AA)
    
    return canvas

def run_visual_verification():
    print("[VISUAL SUITE] Initializing deep inference pipeline...")
    pipe = DeepInferencePipeline()
    
    test_cases = [
        {
            "id": "01_user_upload_multipothole_earthen",
            "name": "User Upload: Waterlogged Earthen Road Potholes",
            "path": os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images", "user_upload_media_1794667697851_69045724.jpg"),
        },
        {
            "id": "02_kaggle_pothole_cavity",
            "name": "Kaggle Real Asphalt Pothole Cavity",
            "path": os.path.join(ENGINE_ROOT, "datasets", "02_kaggle_pothole_600", "real_images", "kaggle_real_pothole_01.jpg"),
        },
        {
            "id": "03_crack500_fatigue_crack",
            "name": "CRACK500 Longitudinal & Alligator Fatigue Cracking",
            "path": os.path.join(ENGINE_ROOT, "datasets", "03_crack500_fatigue", "real_images", "1014620_RS_386_386RS124739_30026_RAW.jpg"),
        },
        {
            "id": "04_missing_zebra_crossing",
            "name": "MoRTH Class 4: Missing Pedestrian Zebra Crossing",
            "path": os.path.join(ENGINE_ROOT, "datasets", "10_missing_zebra_crossing", "real_images", "WM_10_zebra_001_RAW.png"),
        },
        {
            "id": "05_damaged_traffic_sign",
            "name": "MoRTH Class 6: Damaged Regulatory Traffic Sign",
            "path": os.path.join(ENGINE_ROOT, "datasets", "12_damaged_traffic_signs", "real_images", "real_01.jpg"),
        },
    ]

    saved_images = []
    print(f"[VISUAL SUITE] Processing {len(test_cases)} actual test frames with visual overlay...")
    for tc in test_cases:
        p = tc["path"]
        if not os.path.exists(p):
            print(f"  [SKIP] {tc['name']} (image not found: {p})")
            continue
            
        img = cv2.imread(p)
        if img is None:
            continue
            
        t0 = time.time()
        res = pipe.audit_image(p)
        elapsed_ms = round((time.time() - t0) * 1000.0, 1)
        
        annotated = annotate_visual_frame(img, res, title=tc["name"])
        out_path = os.path.join(OUTPUT_DIR, f"{tc['id']}.png")
        cv2.imwrite(out_path, annotated)
        
        dets = [d for d in res.get("all_detections", []) if d.get("is_distress")]
        print(f"  [PASS] {tc['name']:<50} -> {out_path} ({elapsed_ms} ms, {len(dets)} defects)")
        saved_images.append({
            "test_id": tc["id"],
            "name": tc["name"],
            "image_path": out_path,
            "distress_count": len(dets),
            "primary_distress": (res.get("primary_distress") or {}).get("class_name"),
            "primary_conf": (res.get("primary_distress") or {}).get("confidence"),
            "surface": (res.get("surface_material") or {}).get("name"),
            "pci": (res.get("astm_d6433_pci") or {}).get("pci_score"),
        })

    print(f"\n[VISUAL SUITE] Successfully generated {len(saved_images)} visual verification inspection images in {OUTPUT_DIR}")
    return saved_images

if __name__ == "__main__":
    run_visual_verification()
