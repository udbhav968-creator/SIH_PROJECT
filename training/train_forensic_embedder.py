import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Training pipeline for Models M7 & M8 (Anti-Fraud Metric Embedder).
Trained using Triplet Margin Loss: L = max(0, ||a - p||^2 - ||a - n||^2 + margin).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_generator import generate_forensic_triplets
from models.forensic_audit_engine import ForensicMetricEmbedder

def run_training(epochs=15, batch_size=64, margin=0.3, save_dir=None):
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    
    print("[M7/M8 Forensic Embedder] Generating 2,000 anchor-positive-negative triplets...")
    anchors, positives, negatives = generate_forensic_triplets(num_triplets=2000, embed_dim=48, seed=42)
    
    # Split 80/20
    split = 1600
    a_tr, a_val = anchors[:split], anchors[split:]
    p_tr, p_val = positives[:split], positives[split:]
    n_tr, n_val = negatives[:split], negatives[split:]
    
    model = ForensicMetricEmbedder(in_dim=48, hidden_dim=64, embed_dim=32, lr=0.003)
    
    print("  Optimizing metric embedding space with Triplet Loss...")
    history = []
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(len(a_tr))
        losses = []
        
        for start in range(0, len(a_tr), batch_size):
            b_idx = indices[start:min(start + batch_size, len(a_tr))]
            a_b, p_b, n_b = a_tr[b_idx], p_tr[b_idx], n_tr[b_idx]
            
            # Forward
            emb_a = model.forward(a_b)
            emb_p = model.forward(p_b)
            emb_n = model.forward(n_b)
            
            # Distance squared
            d_pos = np.sum((emb_a - emb_p)**2, axis=-1)
            d_neg = np.sum((emb_a - emb_n)**2, axis=-1)
            loss_per_sample = np.maximum(0.0, d_pos - d_neg + margin)
            loss = float(np.mean(loss_per_sample))
            losses.append(loss)
            
            # Analytical gradient update on weights
            # Gradient where margin violated:
            active = (loss_per_sample > 0).astype(np.float32)[:, None]
            grad_a = 2.0 * (emb_n - emb_p) * active / len(b_idx)
            
            # Update output layer
            h1 = np.maximum(0, a_b @ model.w1 + model.b1)
            model.w2 -= model.lr * (h1.T @ grad_a)
            model.b2 -= model.lr * np.sum(grad_a, axis=0, keepdims=True)
            
        mean_loss = float(np.mean(losses))
        
        # Validation Cosine Separation
        emb_val_a = model.forward(a_val)
        emb_val_p = model.forward(p_val)
        emb_val_n = model.forward(n_val)
        
        sim_pos = np.mean(np.sum(emb_val_a * emb_val_p, axis=-1))
        sim_neg = np.mean(np.sum(emb_val_a * emb_val_n, axis=-1))
        margin_gap = float(sim_pos - sim_neg)
        
        history.append({
            "epoch": epoch,
            "triplet_loss": round(mean_loss, 4),
            "sim_positive": round(float(sim_pos), 4),
            "sim_negative": round(float(sim_neg), 4),
            "margin_gap": round(margin_gap, 4)
        })
        
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} - Loss: {mean_loss:.4f} - Positive Sim: {sim_pos:.3f} - Negative Sim: {sim_neg:.3f} - Margin Gap: {margin_gap:.3f}")
            
    ckpt_path = os.path.join(save_dir, "forensic_embedder_weights.npz")
    model.save_weights(ckpt_path)
    print(f"  [SUCCESS] Saved checkpoint: {ckpt_path}")
    
    return {
        "model": "ForensicMetricEmbedder",
        "final_loss": history[-1]["triplet_loss"],
        "sim_positive": history[-1]["sim_positive"],
        "sim_negative": history[-1]["sim_negative"],
        "margin_gap": history[-1]["margin_gap"],
        "checkpoint": ckpt_path
    }

if __name__ == "__main__":
    run_training()
