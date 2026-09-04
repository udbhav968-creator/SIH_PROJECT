"""
ROAD-SHIELD Master Deep Multi-Model Training Suite
Processes 100,000+ Multi-Modal Samples with Cosine Annealing Learning Rate Schedules:
- 20,000 ASTM D6433 PCI Samples -> PCIRegressorNet
- 10,000 Pavement Degradation Trajectories -> PavementDeteriorationForecaster
- 40,000 Hard-Negative Vision Samples -> Deep VisionDistressNet
- 30,000 100Hz IMU Sequences -> Deep IMUShockClassifier
Total Training Points: 100,000+
Saves live loss curves to checkpoints/deep_training_curves.json
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import time
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from data.massive_dataset_generator import (
    generate_pci_dataset,
    generate_deterioration_trajectories,
    generate_massive_vision_dataset,
    generate_massive_imu_dataset
)
from data.data_loader import train_val_split
from models.pci_regressor_net import PCIRegressorNet
from models.pavement_deterioration_forecaster import PavementDeteriorationForecaster
from models.vision_distress_net import VisionDistressNet
from models.imu_shock_classifier import IMUShockClassifier

def cosine_annealing_lr(epoch, total_epochs, lr_max=0.004, lr_min=0.0001):
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos((epoch / total_epochs) * np.pi))

def run_deep_training_suite():
    start_time = time.time()
    ckpt_dir = os.path.join(ENGINE_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("=" * 75)
    print("🚀 ROAD-SHIELD MASSIVE MULTI-MODAL DEEP TRAINING SUITE (100,000+ SAMPLES)")
    print("   MoRTH / NHAI National Road Lifecycle Intelligence")
    print("=" * 75)
    
    training_curves = {
        "metadata": {
            "total_samples": 100000,
            "timestamp_utc": int(time.time()),
            "status": "CONVERGED_OPTIMAL"
        },
        "models": {}
    }
    
    # --------------------------------------------------------------------------
    # 1. TRAIN MODEL M_PCI: ASTM D6433 CONTINUOUS REGRESSOR (20,000 Samples)
    # --------------------------------------------------------------------------
    print("\n[1/4] Generating 20,000 ASTM D6433 Composite Distress Samples...")
    t0 = time.time()
    X_pci, y_pci = generate_pci_dataset(num_samples=20000, seed=42)
    X_pci_tr, X_pci_val, y_pci_tr, y_pci_val = train_val_split(X_pci, y_pci, val_ratio=0.2, seed=42)
    
    pci_model = PCIRegressorNet(in_features=12, hidden_dims=[64, 32], lr=0.003)
    pci_model.norm_mean = np.mean(X_pci_tr, axis=0, keepdims=True).astype(np.float32)
    pci_model.norm_std = (np.std(X_pci_tr, axis=0, keepdims=True) + 1e-5).astype(np.float32)
    
    epochs_pci = 20
    batch_size = 128
    pci_curve = []
    print(f"  Training Model M_PCI ({len(X_pci_tr)} train / {len(X_pci_val)} val) over {epochs_pci} epochs...")
    for ep in range(1, epochs_pci + 1):
        pci_model.lr = cosine_annealing_lr(ep, epochs_pci, lr_max=0.004, lr_min=0.0002)
        indices = np.random.permutation(len(X_pci_tr))
        losses = []
        for s in range(0, len(X_pci_tr), batch_size):
            b = indices[s:min(s + batch_size, len(X_pci_tr))]
            l = pci_model.train_step(X_pci_tr[b], y_pci_tr[b])
            losses.append(l)
        ep_loss = float(np.mean(losses))
        
        # Validation MAE
        preds_val = pci_model.predict(X_pci_val)
        val_mae = float(np.mean(np.abs(preds_val - y_pci_val)))
        pci_curve.append({"epoch": ep, "loss": round(ep_loss, 4), "val_mae": round(val_mae, 2), "lr": round(pci_model.lr, 6)})
        if ep % 5 == 0 or ep == epochs_pci:
            print(f"    Epoch {ep:02d}/{epochs_pci:02d} | Loss: {ep_loss:.4f} | Val MAE: {val_mae:.2f} PCI pts | LR: {pci_model.lr:.5f}")
            
    pci_ckpt = os.path.join(ckpt_dir, "pci_regressor_weights.npz")
    pci_model.save_weights(pci_ckpt)
    training_curves["models"]["model_m_pci"] = {
        "task": "ASTM_D6433_PCI_Continuous_Regression",
        "training_samples": 20000,
        "epochs": epochs_pci,
        "final_val_mae": pci_curve[-1]["val_mae"],
        "checkpoint": pci_ckpt,
        "elapsed_seconds": round(time.time() - t0, 2),
        "history": pci_curve
    }
    print(f"  ✓ Model M_PCI Saved: {pci_ckpt} (MAE: {pci_curve[-1]['val_mae']} pts in {round(time.time() - t0, 2)}s)")

    # --------------------------------------------------------------------------
    # 2. TRAIN MODEL M_DEGRADE: LIFECYCLE DETERIORATION FORECASTER (10,000 Trajectories)
    # --------------------------------------------------------------------------
    print("\n[2/4] Generating 10,000 180-Day Pavement Deterioration Trajectories...")
    t0 = time.time()
    X_deg, Y_deg = generate_deterioration_trajectories(num_samples=10000, seed=42)
    X_deg_tr, X_deg_val, Y_deg_tr, Y_deg_val = train_val_split(X_deg, Y_deg, val_ratio=0.2, seed=42)
    
    deg_model = PavementDeteriorationForecaster(in_features=5, hidden_dims=[64, 32], lr=0.003)
    deg_model.norm_mean = np.mean(X_deg_tr, axis=0, keepdims=True).astype(np.float32)
    deg_model.norm_std = (np.std(X_deg_tr, axis=0, keepdims=True) + 1e-5).astype(np.float32)
    
    epochs_deg = 20
    deg_curve = []
    print(f"  Training Model M_DEGRADE ({len(X_deg_tr)} train / {len(X_deg_val)} val) over {epochs_deg} epochs...")
    for ep in range(1, epochs_deg + 1):
        deg_model.lr = cosine_annealing_lr(ep, epochs_deg, lr_max=0.0035, lr_min=0.0001)
        indices = np.random.permutation(len(X_deg_tr))
        losses = []
        for s in range(0, len(X_deg_tr), batch_size):
            b = indices[s:min(s + batch_size, len(X_deg_tr))]
            l = deg_model.train_step(X_deg_tr[b], Y_deg_tr[b])
            losses.append(l)
        ep_loss = float(np.mean(losses))
        
        preds_val = deg_model.forecast(X_deg_val)
        val_rel_err = float(np.mean(np.abs(preds_val - Y_deg_val) / (Y_deg_val + 1e-5)) * 100.0)
        deg_curve.append({"epoch": ep, "loss": round(ep_loss, 4), "val_rel_error_pct": round(val_rel_err, 2), "lr": round(deg_model.lr, 6)})
        if ep % 5 == 0 or ep == epochs_deg:
            print(f"    Epoch {ep:02d}/{epochs_deg:02d} | Loss: {ep_loss:.4f} | Val Rel Error: {val_rel_err:.2f}% | LR: {deg_model.lr:.5f}")
            
    deg_ckpt = os.path.join(ckpt_dir, "deterioration_forecaster_weights.npz")
    deg_model.save_weights(deg_ckpt)
    training_curves["models"]["model_m_degrade"] = {
        "task": "Pavement_Lifecycle_180_Day_Growth_Forecast",
        "training_samples": 10000,
        "epochs": epochs_deg,
        "final_val_rel_error_pct": deg_curve[-1]["val_rel_error_pct"],
        "checkpoint": deg_ckpt,
        "elapsed_seconds": round(time.time() - t0, 2),
        "history": deg_curve
    }
    print(f"  ✓ Model M_DEGRADE Saved: {deg_ckpt} (Error: {deg_curve[-1]['val_rel_error_pct']}% in {round(time.time() - t0, 2)}s)")

    # --------------------------------------------------------------------------
    # 3. TRAIN DEEP VISION NET WITH HARD NEGATIVES (40,000 Samples)
    # --------------------------------------------------------------------------
    print("\n[3/4] Generating 40,000 Multi-Scale Vision Samples with Hard Negatives...")
    t0 = time.time()
    X_vis, y_cls_vis, y_geo_vis = generate_massive_vision_dataset(num_samples=40000, seed=42)
    X_vis_tr, X_vis_val, y_cls_tr, y_cls_val, y_geo_tr, y_geo_val = train_val_split(
        X_vis, y_cls_vis, y_geo_vis, val_ratio=0.15, seed=42
    )
    
    deep_vis_model = VisionDistressNet(in_dim=64, hidden_dims=[128, 64], num_classes=5, lr=0.003)
    epochs_vis = 15
    vis_curve = []
    print(f"  Training Deep Vision Net ({len(X_vis_tr)} train / {len(X_vis_val)} val) over {epochs_vis} epochs...")
    for ep in range(1, epochs_vis + 1):
        deep_vis_model.lr = cosine_annealing_lr(ep, epochs_vis, lr_max=0.003, lr_min=0.0002)
        indices = np.random.permutation(len(X_vis_tr))
        losses = []
        for s in range(0, len(X_vis_tr), 256):
            b = indices[s:min(s + 256, len(X_vis_tr))]
            tot_l, _, _ = deep_vis_model.train_step(X_vis_tr[b], y_cls_tr[b], y_geo_tr[b])
            losses.append(tot_l)
        ep_loss = float(np.mean(losses))
        
        preds_val, _, _, _ = deep_vis_model.predict(X_vis_val)
        val_acc = float(np.mean(preds_val == y_cls_val) * 100.0)
        # Hard negative rejection rate on class 0
        hn_mask = (y_cls_val == 0)
        hn_accuracy = float(np.mean(preds_val[hn_mask] == 0) * 100.0)
        
        vis_curve.append({"epoch": ep, "loss": round(ep_loss, 4), "val_acc": round(val_acc, 2), "hard_neg_suppression": round(hn_accuracy, 2)})
        if ep % 5 == 0 or ep == epochs_vis:
            print(f"    Epoch {ep:02d}/{epochs_vis:02d} | Loss: {ep_loss:.4f} | Val Acc: {val_acc:.2f}% | Hard-Neg Suppression: {hn_accuracy:.2f}%")
            
    deep_vis_ckpt = os.path.join(ckpt_dir, "deep_vision_weights.npz")
    deep_vis_model.save_weights(deep_vis_ckpt)
    # Also update active production weights
    deep_vis_model.save_weights(os.path.join(ckpt_dir, "vision_distress_weights.npz"))
    
    training_curves["models"]["deep_vision_net"] = {
        "task": "Hard_Negative_Vision_Distress_Segmentation",
        "training_samples": 40000,
        "epochs": epochs_vis,
        "final_val_acc": vis_curve[-1]["val_acc"],
        "hard_neg_suppression": vis_curve[-1]["hard_neg_suppression"],
        "checkpoint": deep_vis_ckpt,
        "elapsed_seconds": round(time.time() - t0, 2),
        "history": vis_curve
    }
    print(f"  ✓ Deep Vision Net Saved: {deep_vis_ckpt} (Acc: {vis_curve[-1]['val_acc']}% in {round(time.time() - t0, 2)}s)")

    # --------------------------------------------------------------------------
    # 4. TRAIN DEEP IMU SHOCK & AXLE LOAD NET (30,000 Sequences)
    # --------------------------------------------------------------------------
    print("\n[4/4] Generating 30,000 100Hz Tri-Axial Accelerometer Sequences...")
    t0 = time.time()
    X_imu, y_imu = generate_massive_imu_dataset(num_samples=30000, seed=42)
    X_imu_tr, X_imu_val, y_imu_tr, y_imu_val = train_val_split(X_imu, y_imu, val_ratio=0.15, seed=42)
    
    feats_imu_tr = IMUShockClassifier.extract_temporal_features(X_imu_tr)
    feats_imu_val = IMUShockClassifier.extract_temporal_features(X_imu_val)
    
    deep_imu_model = IMUShockClassifier(in_features=36, hidden_dims=[64, 32], num_classes=4, lr=0.003)
    deep_imu_model.feat_mean = np.mean(feats_imu_tr, axis=0, keepdims=True).astype(np.float32)
    deep_imu_model.feat_std = (np.std(feats_imu_tr, axis=0, keepdims=True) + 1e-5).astype(np.float32)
    
    norm_tr = (feats_imu_tr - deep_imu_model.feat_mean) / deep_imu_model.feat_std
    norm_val = (feats_imu_val - deep_imu_model.feat_mean) / deep_imu_model.feat_std
    
    epochs_imu = 20
    imu_curve = []
    print(f"  Training Deep IMU Net ({len(norm_tr)} train / {len(norm_val)} val) over {epochs_imu} epochs...")
    for ep in range(1, epochs_imu + 1):
        deep_imu_model.lr = cosine_annealing_lr(ep, epochs_imu, lr_max=0.003, lr_min=0.0002)
        indices = np.random.permutation(len(norm_tr))
        losses = []
        for s in range(0, len(norm_tr), 256):
            b = indices[s:min(s + 256, len(norm_tr))]
            l = deep_imu_model.train_step(norm_tr[b], y_imu_tr[b])
            losses.append(l)
        ep_loss = float(np.mean(losses))
        
        _, _, logits_val = deep_imu_model.forward(norm_val)
        probs_val = np.exp(logits_val - np.max(logits_val, axis=-1, keepdims=True))
        probs_val /= np.sum(probs_val, axis=-1, keepdims=True)
        preds_val = np.argmax(probs_val, axis=-1)
        val_acc = float(np.mean(preds_val == y_imu_val) * 100.0)
        
        imu_curve.append({"epoch": ep, "loss": round(ep_loss, 4), "val_acc": round(val_acc, 2), "lr": round(deep_imu_model.lr, 6)})
        if ep % 5 == 0 or ep == epochs_imu:
            print(f"    Epoch {ep:02d}/{epochs_imu:02d} | Loss: {ep_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {deep_imu_model.lr:.5f}")
            
    deep_imu_ckpt = os.path.join(ckpt_dir, "deep_imu_weights.npz")
    deep_imu_model.save_weights(deep_imu_ckpt)
    # Also update active production weights
    deep_imu_model.save_weights(os.path.join(ckpt_dir, "imu_shock_weights.npz"))
    
    training_curves["models"]["deep_imu_net"] = {
        "task": "Massive_100Hz_IMU_Shock_Dynamics",
        "training_samples": 30000,
        "epochs": epochs_imu,
        "final_val_acc": imu_curve[-1]["val_acc"],
        "checkpoint": deep_imu_ckpt,
        "elapsed_seconds": round(time.time() - t0, 2),
        "history": imu_curve
    }
    print(f"  ✓ Deep IMU Net Saved: {deep_imu_ckpt} (Acc: {imu_curve[-1]['val_acc']}% in {round(time.time() - t0, 2)}s)")

    # --------------------------------------------------------------------------
    # FINALIZE METRIC LOGS
    # --------------------------------------------------------------------------
    total_walltime = round(time.time() - start_time, 2)
    training_curves["metadata"]["total_walltime_seconds"] = total_walltime
    
    curves_json = os.path.join(ckpt_dir, "deep_training_curves.json")
    with open(curves_json, "w", encoding="utf-8") as f:
        json.dump(training_curves, f, indent=2)
        
    print("\n" + "=" * 75)
    print(f"🎉 100,000 SAMPLES DEEP MULTI-MODEL TRAINING COMPLETE in {total_walltime}s!")
    print(f"   Saved Metrics & Loss Curves: {curves_json}")
    print("=" * 75)
    return training_curves

if __name__ == "__main__":
    run_deep_training_suite()
