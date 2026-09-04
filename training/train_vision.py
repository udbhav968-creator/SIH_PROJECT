import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Training pipeline for Model M1 (Vision Distress Net).
"""
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_generator import generate_vision_dataset
from data.data_loader import train_val_split, get_batches
from models.vision_distress_net import VisionDistressNet

def run_training(epochs=15, batch_size=64, save_dir=None):
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    
    print("[M1 Vision Distress Net] Generating dataset (RDD2022 + IDD distribution)...")
    X, y_cls, y_geo, y_area = generate_vision_dataset(num_samples=5000, seed=42)
    
    X_tr, X_val, y_cls_tr, y_cls_val, y_geo_tr, y_geo_val = train_val_split(
        X, y_cls, y_geo, val_ratio=0.2, seed=42
    )
    print(f"  Training samples: {X_tr.shape[0]} | Validation samples: {X_val.shape[0]}")
    
    model = VisionDistressNet(in_dim=64, hidden_dims=[128, 64], num_classes=5, lr=0.0025)
    
    print("  Starting Adam optimization...")
    history = []
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(len(X_tr))
        epoch_losses = []
        
        for start_idx in range(0, len(X_tr), batch_size):
            b_idx = indices[start_idx:min(start_idx + batch_size, len(X_tr))]
            tot_loss, cls_l, geo_l = model.train_step(X_tr[b_idx], y_cls_tr[b_idx], y_geo_tr[b_idx])
            epoch_losses.append(tot_loss)
            
        mean_loss = float(np.mean(epoch_losses))
        
        # Validation
        preds_val, _, probs_val, geo_preds_val = model.predict(X_val)
        val_acc = float(np.mean(preds_val == y_cls_val) * 100.0)
        pothole_mask = (y_cls_val == 4)
        pothole_recall = float(np.mean(preds_val[pothole_mask] == 4) * 100.0)
        
        history.append({
            "epoch": epoch,
            "loss": round(mean_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "pothole_recall": round(pothole_recall, 2)
        })
        
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} - Loss: {mean_loss:.4f} - Val Acc: {val_acc:.2f}% - Pothole Recall: {pothole_recall:.2f}%")
            
    ckpt_path = os.path.join(save_dir, "vision_distress_weights.npz")
    model.save_weights(ckpt_path)
    print(f"  [SUCCESS] Saved checkpoint: {ckpt_path}")
    
    return {
        "model": "VisionDistressNet",
        "final_loss": history[-1]["loss"],
        "val_accuracy": history[-1]["val_accuracy"],
        "pothole_recall": history[-1]["pothole_recall"],
        "checkpoint": ckpt_path
    }

if __name__ == "__main__":
    run_training()
