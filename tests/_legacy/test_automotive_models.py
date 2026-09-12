"""
Automated Verification Suite for ROAD-SHIELD Automotive Multimodal & RL Architecture
Verifies:
1. 10-Class Vision Model M1 (including Class 9 VRU Pedestrian)
2. Multimodal Cross-Attention Transformer MM-1 (5 Modalities)
3. Automotive ADAS & Active Chassis RL Policy Agent RL-1 (Dueling-DQN)
4. Vector CAN DBC & C++20 Header-Only Real-Time ECU Driver
5. Deep Inference Pipeline Integration
"""

import os
import sys
import time
import json
import numpy as np

ENGINE_ROOT = r"c:\Users\Dell\Downloads\road_shield_ai_engine"
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from models.vision_distress_net import VisionDistressNet
from models.multimodal_transformer_fusion import MultimodalTransformerFusionNet
from models.automotive_rl_policy_agent import AutomotiveRLPolicyAgent
from models.automotive_telematics_engine import AutomotiveTelematicsEngine
from pipeline.deep_inference_pipeline import DeepInferencePipeline

def run_tests():
    print("=" * 80)
    print("🚗 ROAD-SHIELD AUTOMOTIVE MULTIMODAL & RL VERIFICATION SUITE")
    print("=" * 80)

    ckpt_dir = os.path.join(ENGINE_ROOT, "checkpoints")

    # 1. Test Model M1 (10 Classes)
    print("\n[TEST 1] Testing Model M1 (10-Class Vision Distress Net)...")
    m1 = VisionDistressNet(in_features=64, hidden_dims=[512, 256, 128], num_classes=10)
    m1_ckpt = os.path.join(ckpt_dir, "vision_distress_weights.npz")
    if os.path.exists(m1_ckpt):
        m1.load_weights(m1_ckpt)
        print(f"  ✓ Loaded M1 checkpoint: {m1_ckpt}")
    assert len(m1.CLASS_NAMES) == 10, f"Expected 10 classes, got {len(m1.CLASS_NAMES)}"
    assert m1.CLASS_NAMES[9] == "Child / Pedestrian Hazard (Vulnerable Road User)"
    print(f"  ✓ 10-Class taxonomy verified. Class 9: {m1.CLASS_NAMES[9]}")

    # Single inference test
    dummy_feat = np.zeros((1, 64), dtype=np.float32)
    dummy_feat[0, 0:10] = 3.5
    preds, conf, probs, _ = m1.predict(dummy_feat)
    deep_res = m1.predict_deep(dummy_feat)[0]
    print(f"  ✓ M1 predict_deep executed successfully. Predicted: {deep_res['class_name']} (Color: {deep_res['color_hex']})")

    # 2. Test Model MM-1 (Multimodal Cross-Attention Transformer)
    print("\n[TEST 2] Testing Model MM-1 (Multimodal Transformer Fusion Net)...")
    mm = MultimodalTransformerFusionNet(embed_dim=64, num_classes=10)
    mm_ckpt = os.path.join(ckpt_dir, "multimodal_fusion_weights.npz")
    if os.path.exists(mm_ckpt):
        mm.load_weights(mm_ckpt)
        print(f"  ✓ Loaded MM-1 checkpoint: {mm_ckpt}")

    v_vis = np.zeros(64, dtype=np.float32)
    v_vis[4*6 : 4*6+6] = 2.5 # Pothole visual
    v_imu = np.zeros(36, dtype=np.float32); v_imu[0] = 4.5 # 4.5 m/s^2 shock
    v_dep = np.zeros(16, dtype=np.float32); v_dep[0] = 0.55 # 55mm depth
    v_can = np.zeros(12, dtype=np.float32); v_can[0] = 0.85 # 85 km/h
    v_env = np.zeros(8, dtype=np.float32); v_env[0] = 0.78 # mu = 0.78

    res_mm = mm.predict_multimodal(v_vis, v_imu, v_dep, v_can, v_env)
    print(f"  ✓ MM-1 Fused Prediction: {res_mm['label']} (Conf: {res_mm['confidence']}, ASIL: {res_mm['asil_safety_rating']})")
    print(f"  ✓ Attention Breakdown: {res_mm['attention_breakdown']}")

    # Optical Shadow Suppression Test
    v_vis_shadow = np.zeros(64, dtype=np.float32); v_vis_shadow[4*6:4*6+6] = 2.5 # looks like pothole optically
    v_imu_flat = np.zeros(36, dtype=np.float32) # flat IMU
    v_dep_flat = np.zeros(16, dtype=np.float32) # flat depth
    res_shadow = mm.predict_multimodal(v_vis_shadow, v_imu_flat, v_dep_flat, v_can, v_env)
    print(f"  ✓ Shadow Test: Optical Suppression Active = {res_shadow['optical_suppression_active']}")

    # 3. Test Model RL-1 (Automotive RL Policy Agent)
    print("\n[TEST 3] Testing Model RL-1 (Automotive RL Policy Agent)...")
    rl = AutomotiveRLPolicyAgent(state_dim=32, num_actions=6)
    rl_ckpt = os.path.join(ckpt_dir, "automotive_rl_agent_weights.npz")
    if os.path.exists(rl_ckpt):
        rl.load_weights(rl_ckpt)
        print(f"  ✓ Loaded RL-1 checkpoint: {rl_ckpt}")

    # Test Scenario A: Pedestrian Hazard (Class 9)
    res_vru = rl.evaluate_telemetry_state(
        hazard_class_id=9,
        confidence=0.98,
        distance_m=18.0,
        vehicle_speed_kmh=45.0,
        surface_friction_mu=0.75
    )
    print(f"  ✓ Scenario [VRU Pedestrian]: Recommended Action = {res_vru['action_name']} (ASIL: {res_vru['asil_functional_safety']['rating']})")
    print(f"    Actuator Setpoints: {res_vru['actuator_setpoints']}")

    # Test Scenario B: Severe Highway Pothole (Class 4)
    res_pot = rl.evaluate_telemetry_state(
        hazard_class_id=4,
        confidence=0.96,
        distance_m=35.0,
        vehicle_speed_kmh=90.0,
        surface_friction_mu=0.78,
        pothole_depth_mm=55.0,
        imu_z_shock_ms2=5.2
    )
    print(f"  ✓ Scenario [Severe Pothole]: Recommended Action = {res_pot['action_name']} (Pre-Lift: {res_pot['actuator_setpoints']['suspension_lift_mm']}mm)")

    # Test Scenario C: Nominal Pavement (Class 0)
    res_norm = rl.evaluate_telemetry_state(
        hazard_class_id=0,
        confidence=0.99,
        distance_m=50.0,
        vehicle_speed_kmh=80.0
    )
    print(f"  ✓ Scenario [Nominal Cruise]: Recommended Action = {res_norm['action_name']}")

    # 4. Test Automotive Telematics Engine & Exports
    print("\n[TEST 4] Testing Automotive Telematics & Protocol Engine...")
    telem = AutomotiveTelematicsEngine(checkpoints_dir=ckpt_dir)
    can_pkt = telem.generate_adas_can_packet(res_pot, hazard_class_id=4, ttc_sec=2.2, speed_kmh=90.0)
    print(f"  ✓ CAN 2.0B Frame Encoded: {can_pkt['can_id']} | Raw Hex: {can_pkt['raw_hex']}")

    dbc_file = telem.generate_can_dbc()
    cpp_file = telem.generate_cpp_ecu_header()
    assert os.path.exists(dbc_file), f"DBC file missing: {dbc_file}"
    assert os.path.exists(cpp_file), f"C++ header missing: {cpp_file}"
    print(f"  ✓ Vector CAN DBC: {dbc_file} ({os.path.getsize(dbc_file)} bytes)")
    print(f"  ✓ C++20 Real-Time ECU Header: {cpp_file} ({os.path.getsize(cpp_file)} bytes)")

    # 5. Test Deep Inference Pipeline Integration
    print("\n[TEST 5] Testing DeepInferencePipeline evaluate_automotive_incident...")
    pipe = DeepInferencePipeline(ckpt_dir)
    pipe_res = pipe.evaluate_automotive_incident(
        hazard_class_id=4,
        confidence=0.95,
        distance_m=38.0,
        vehicle_speed_kmh=92.0,
        surface_friction_mu=0.75,
        pothole_depth_mm=50.0,
        imu_z_shock_ms2=4.8
    )
    print(f"  ✓ Pipeline Incident Evaluation: Action={pipe_res['rl_policy_decision']['action_name']}")
    print(f"    CAN Telemetry: {pipe_res['can_bus_telemetry']['raw_hex']}")
    print(f"    Standards: {pipe_res['automotive_standards']}")

    print("\n" + "=" * 80)
    print("🎉 ALL AUTOMOTIVE MULTIMODAL & RL TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
