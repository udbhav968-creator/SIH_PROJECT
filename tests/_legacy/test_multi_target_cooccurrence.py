"""
Automated Test Suite: Multi-Target Co-Occurrence Architecture (Person + Pothole)
Validates simultaneous independent detection, non-zero civil metrics, and dual ADAS alerts.
Standards: MoRTH / NHAI SIH2026-MORTH-TRANS-018 and Automotive OEM Tier-1 Safety
"""
import os
import sys
import base64
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOWNLOADS_DIR = os.path.abspath(os.path.join(ENGINE_ROOT, '..'))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from models.cv_cavity_detector import CVCavityDetector
from models.vision_distress_net import VisionDistressNet
from pipeline.deep_inference_pipeline import DeepInferencePipeline

def find_image_path(filename):
    for d in [ENGINE_ROOT, DOWNLOADS_DIR]:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in {ENGINE_ROOT} or {DOWNLOADS_DIR}")

def test_multi_target_cooccurrence():
    print('=' * 75)
    print('VERIFYING MULTI-TARGET CO-OCCURRENCE DETECTION (PERSON + ROAD CAVITY)')
    print('=' * 75)

    ckpt_dir = os.path.join(ENGINE_ROOT, 'checkpoints')
    vision_model = VisionDistressNet(in_features=64, hidden_dims=[512, 256, 128], num_classes=10)
    vis_ckpt = os.path.join(ckpt_dir, 'vision_distress_weights.npz')
    if os.path.exists(vis_ckpt):
        vision_model.load_weights(vis_ckpt)
        print('  ✓ Loaded Model M1 weights from checkpoint.')

    detector = CVCavityDetector()
    pipeline = DeepInferencePipeline(ckpt_dir)

    boy_path = find_image_path('boy.webp')
    uddu_path = find_image_path('uddu.webp')

    # -------------------------------------------------------------------------
    # TEST 1: Dual Target Image (boy.webp: Child/VRU + Road Distress)
    # -------------------------------------------------------------------------
    print('\n[Test 1] Analyzing Dual-Hazard Image (boy.webp)...')
    assert os.path.exists(boy_path), f'Missing {boy_path}'
    
    with open(boy_path, 'rb') as f:
        boy_b64 = base64.b64encode(f.read()).decode('utf-8')

    cv_res_boy = detector.analyze_image(boy_b64, vision_model=vision_model)
    print(f'  Total Detections: {cv_res_boy["detections_count"]}')
    print(f'  Pedestrians Found: {cv_res_boy["pedestrians_count"]}')
    print(f'  Distress Found: {cv_res_boy["distress_count"]}')
    print(f'  Has Dual Targets: {cv_res_boy["has_dual_targets"]}')

    assert cv_res_boy['pedestrians_count'] >= 1, 'Must detect child/pedestrian in boy.webp!'
    assert cv_res_boy['primary_pedestrian'] is not None, 'primary_pedestrian must be populated!'
    assert cv_res_boy['has_dual_targets'] is True, 'has_dual_targets must be True when person and distress co-occur!'
    assert cv_res_boy['primary_distress'] is not None, 'primary_distress must be populated!'
    
    dist_cost = cv_res_boy['primary_distress']['physical_dimensions']['estimated_repair_cost_inr']
    print(f'  Primary Pedestrian: {cv_res_boy["primary_pedestrian"]["class_name"]} ({cv_res_boy["primary_pedestrian"]["confidence"]:.3f})')
    print(f'  Primary Distress: {cv_res_boy["primary_distress"]["class_name"]} - Repair Cost: INR {dist_cost}')
    assert dist_cost > 0, 'Road distress repair cost MUST NOT be zeroed out in dual-target frame!'

    # -------------------------------------------------------------------------
    # TEST 2: Pure Pothole Image (uddu.webp: Cavity Only, 0 False Pedestrians)
    # -------------------------------------------------------------------------
    print('\n[Test 2] Analyzing Pavement Pothole Image (uddu.webp)...')
    assert os.path.exists(uddu_path), f'Missing {uddu_path}'

    with open(uddu_path, 'rb') as f:
        uddu_b64 = base64.b64encode(f.read()).decode('utf-8')

    cv_res_uddu = detector.analyze_image(uddu_b64, vision_model=vision_model)
    print(f'  Total Detections: {cv_res_uddu["detections_count"]}')
    print(f'  Pedestrians Found: {cv_res_uddu["pedestrians_count"]}')
    print(f'  Distress Found: {cv_res_uddu["distress_count"]}')
    print(f'  Has Dual Targets: {cv_res_uddu["has_dual_targets"]}')

    assert cv_res_uddu['pedestrians_count'] == 0, 'Water reflection must NOT trigger false pedestrian in uddu.webp!'
    assert cv_res_uddu['primary_pedestrian'] is None, 'primary_pedestrian must be None for pure pothole!'
    assert cv_res_uddu['has_dual_targets'] is False, 'has_dual_targets must be False for single-target cavity!'
    assert cv_res_uddu['primary_distress']['class_id'] == 4, 'Must classify primary distress as D40 Pothole!'
    assert cv_res_uddu['primary_distress']['physical_dimensions']['estimated_repair_cost_inr'] > 0, 'Must calculate repair cost for D40 cavity!'

    # -------------------------------------------------------------------------
    # TEST 3: Deep 12-Stage Inference Pipeline on Dual-Target Scene
    # -------------------------------------------------------------------------
    print('\n[Test 3] Executing 12-Stage Deep Pipeline Audit on boy.webp...')
    audit_boy = pipeline.audit_image(boy_b64, corridor_id='NH-44 Urban Zone')
    
    assert audit_boy['status'] == 'ANALYSIS_COMPLETE'
    assert audit_boy['vulnerable_safety_alert'] is True, 'Pipeline must activate vulnerable road user alert!'
    assert audit_boy['has_dual_targets'] is True, 'Pipeline must flag dual-hazard co-occurrence!'
    assert audit_boy['morth_civil_ledger']['total_estimated_repair_inr'] > 0, 'Pipeline civil repair budget must be preserved (>0)!'
    assert audit_boy['cryptographic_work_order'] is not None, 'Pipeline must generate cryptographic work order for the road distress!'
    print(f'  Dual Summary: {audit_boy["dual_target_summary"][:80]}...')
    print(f'  Civil Ledger Total: INR {audit_boy["morth_civil_ledger"]["total_estimated_repair_inr"]}')
    print(f'  Cryptographic Seal: {audit_boy["cryptographic_work_order"]["sha256_cryptographic_seal"][:20]}...')

    # -------------------------------------------------------------------------
    # TEST 4: Deep 12-Stage Inference Pipeline on Pure Pothole Scene
    # -------------------------------------------------------------------------
    print('\n[Test 4] Executing 12-Stage Deep Pipeline Audit on uddu.webp...')
    audit_uddu = pipeline.audit_image(uddu_b64, corridor_id='NH-44 Section KM 108.4')

    assert audit_uddu['status'] == 'ANALYSIS_COMPLETE'
    assert audit_uddu['vulnerable_safety_alert'] is False, 'Pure pothole must not trigger pedestrian alert!'
    assert audit_uddu['has_dual_targets'] is False, 'Pure pothole has dual targets == False!'
    assert audit_uddu['primary_distress']['class_id'] == 4, 'Primary distress must be D40 cavity!'
    assert audit_uddu['bayesian_sensor_fusion']['verdict'] == 'CONFIRMED_POTHOLE'
    assert audit_uddu['morth_civil_ledger']['total_estimated_repair_inr'] > 0
    print(f'  Bayesian Gate Verdict: {audit_uddu["bayesian_sensor_fusion"]["verdict"]}')
    print(f'  ASTM D6433 PCI Score: {audit_uddu["astm_d6433_pci"]["pci_score"]} ({audit_uddu["astm_d6433_pci"]["rating_category"]})')
    print(f'  Civil Ledger Total: INR {audit_uddu["morth_civil_ledger"]["total_estimated_repair_inr"]}')

    print('\n' + '=' * 75)
    print('✅ MULTI-TARGET CO-OCCURRENCE & DEEP PIPELINE VERIFIED SUCCESSFULLY!')
    print('=' * 75)
    return True

if __name__ == '__main__':
    test_multi_target_cooccurrence()
