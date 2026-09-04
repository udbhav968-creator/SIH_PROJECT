import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Training pipeline for Model M4 (100 Hz IMU Shock Classifier).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_generator import generate_imu_dataset
from data.data_loader import train_val_split
from models.imu_shock_classifier import IMUShockClassifier

def run_training(epochs=20, batch_size=64, save_dir=None):
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    
    print("[M4 IMU Shock Classifier] Generating 100 Hz tri-axial accelerometer data...")
    X_raw, y = generate_imu_dataset(num_samples=4000, timesteps=100, seed=42)
    
    X_tr_raw, X_val_raw, y_tr, y_val = train_val_split(X_raw, y, val_ratio=0.2, seed=42)
    
    print("  Extracting multi-scale temporal dynamic features...")
    feats_tr = IMUShockClassifier.extract_temporal_features(X_tr_raw)
    feats_val = IMUShockClassifier.extract_temporal_features(X_val_raw)
    
    # Feature normalization
    mean = np.mean(feats_tr, axis=0, keepdims=True)
    std = np.std(feats_tr, axis=0, keepdims=True) + 1e-5
    feats_tr = (feats_tr - mean) / std
    feats_val = (feats_val - mean) / std
    
    model = IMUShockClassifier(in_features=feats_tr.shape[1], hidden_dims=[64, 32], num_classes=4, lr=0.003)
    model.feat_mean = mean.astype(np.float32)
    model.feat_std = std.astype(np.float32)
    
    print("  Starting Adam optimization...")
    history = []
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(len(feats_tr))
        epoch_losses = []
        
        for start_idx in range(0, len(feats_tr), batch_size):
            b_idx = indices[start_idx:min(start_idx + batch_size, len(feats_tr))]
            loss = model.train_step(feats_tr[b_idx], y_tr[b_idx])
            epoch_losses.append(loss)
            
        mean_loss = float(np.mean(epoch_losses))
        
        # Validation
        _, _, logits_val = model.forward(feats_val)
        probs_val = np.exp(logits_val - np.max(logits_val, axis=-1, keepdims=True))
        probs_val /= np.sum(probs_val, axis=-1, keepdims=True)
        preds_val = np.argmax(probs_val, axis=-1)
        val_acc = float(np.mean(preds_val == y_val) * 100.0)
        
        pothole_mask = (y_val == 3)
        pothole_recall = float(np.mean(preds_val[pothole_mask] == 3) * 100.0)
        
        history.append({
            "epoch": epoch,
            "loss": round(mean_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "shock_detection_rate": round(pothole_recall, 2)
        })
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} - Loss: {mean_loss:.4f} - Val Acc: {val_acc:.2f}% - Pothole Shock Recall: {pothole_recall:.2f}%")
            
    ckpt_path = os.path.join(save_dir, "imu_shock_weights.npz")
    model.save_weights(ckpt_path)
    print(f"  [SUCCESS] Saved checkpoint: {ckpt_path}")
    
    return {
        "model": "IMUShockClassifier",
        "final_loss": history[-1]["loss"],
        "val_accuracy": history[-1]["val_accuracy"],
        "shock_detection_rate": history[-1]["shock_detection_rate"],
        "checkpoint": ckpt_path
    }

if __name__ == "__main__":
    run_training()
