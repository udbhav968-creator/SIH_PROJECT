"""Google Colab GPU Deep Training Script for ROAD-SHIELD AI Engine.

Execute this script in a Google Colab notebook with free T4 / V100 GPU:
    !git clone https://github.com/udbhav968-creator/SIH_PROJECT.git
    %cd SIH_PROJECT
    !pip install -r requirements.txt
    !python colab/train_road_shield_gpu.py --epochs 65 --batch-size 64
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

def verify_gpu_environment():
    print("=" * 72)
    print("ROAD-SHIELD GOOGLE COLAB GPU TRAINING ENGINE")
    print("=" * 72)
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (Fallback)"
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {has_cuda}")
    print(f"Compute Device  : {device_name}")
    if has_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory Total: {props.total_memory / (1024**3):.2f} GB")
    print("=" * 72)
    return torch.device("cuda" if has_cuda else "cpu")

def train_dan_dag_on_gpu(device, epochs=65, batch_size=64):
    print(f"\n[GPU Stage 1/2] Training DAN-DAG Dual Attention Network on {device}...")
    from training.train_cnn_head import collect_files, grouped_split, embed_items
    from models.cnn_embedder import CNNEmbedder

    embedder = CNNEmbedder(prefer=("mobilenetv2", "resnet50"))
    items = collect_files()
    print(f"  Loaded {len(items)} real road distress images across 7 civil classes.")

    train_items, val_items, test_items = grouped_split(items, seed=42)
    print(f"  Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    # Extract embeddings
    X_train, y_train, P_train, dom_train, _, _ = embed_items(embedder, train_items, "training")
    X_test, y_test, P_test, dom_test, _, _ = embed_items(embedder, test_items, "held-out")

    # Move tensors to GPU
    z_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    p_t = torch.tensor(P_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    D = X_train.shape[1]
    P = P_train.shape[1]
    n_classes = len(np.unique(y_train))

    class ColabDANDAGHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.cam_fc1 = nn.Linear(D, 160)
            self.cam_fc2 = nn.Linear(160, D)
            self.fc1 = nn.Linear(D + P, 512)
            self.ln1 = nn.LayerNorm(512)
            self.fc2 = nn.Linear(512, 256)
            self.ln2 = nn.LayerNorm(256)
            self.skip = nn.Linear(D + P, 256, bias=False)
            self.fc3 = nn.Linear(256, n_classes)
            self.drop = nn.Dropout(0.20)

        def forward(self, z, p):
            gate = torch.sigmoid(self.cam_fc2(F.relu(self.cam_fc1(z))))
            z_att = z * (1.0 + 0.45 * gate)
            x = torch.cat([z_att, 0.65 * p], dim=-1)
            h1 = F.silu(self.ln1(self.fc1(x)))
            h1 = self.drop(h1)
            h2 = F.silu(self.ln2(self.fc2(h1)) + self.skip(x))
            h2 = self.drop(h2)
            return self.fc3(h2)

    net = ColabDANDAGHead().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.03)

    t0 = time.time()
    n_samples = len(y_train)
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n_samples, device=device)
        ep_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = net(z_t[idx], p_t[idx])
            loss = crit(out, y_t[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        sched.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs} | Cross-Entropy Loss: {ep_loss / n_samples:.4f} | Elapsed: {time.time() - t0:.1f}s")

    # Evaluation on held-out test split
    net.eval()
    with torch.no_grad():
        z_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        p_test = torch.tensor(P_test, dtype=torch.float32, device=device)
        test_logits = net(z_test, p_test)
        preds = test_logits.argmax(dim=-1).cpu().numpy()
        test_acc = float((preds == y_test).mean())
        print(f"\n  [COLAB EVAL] Held-Out 7-Class Test Accuracy on GPU: {test_acc * 100:.2f}%")

    # Save GPU trained weights
    os.makedirs(os.path.join(ENGINE_ROOT, "checkpoints"), exist_ok=True)
    out_ckpt = os.path.join(ENGINE_ROOT, "checkpoints", "colab_gpu_dan_dag.pt")
    torch.save(net.state_dict(), out_ckpt)
    print(f"  Saved GPU model checkpoint -> {out_ckpt}")
    return net

def run_visual_verification_stage():
    print("\n[GPU Stage 2/2] Running Visual Evaluation Suite on Real Test Frames...")
    from scripts.visual_evaluation_suite import run_visual_verification
    saved = run_visual_verification()
    print(f"  Verified {len(saved)} visual test frames with masks, bounding boxes, and PCI scores.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=65)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-visual", action="store_true")
    args = ap.parse_args()

    device = verify_gpu_environment()
    train_dan_dag_on_gpu(device, epochs=args.epochs, batch_size=args.batch_size)
    if not args.skip_visual:
        run_visual_verification_stage()
    print("\n[COMPLETE] Google Colab GPU training & visual verification finished successfully.")

if __name__ == "__main__":
    main()
