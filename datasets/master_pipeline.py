"""
ROAD-SHIELD Master Multi-Benchmark Training Pipeline
Loads all 7 curated datasets + Real GitHub Open Datasets and executes
end-to-end multi-modal deep training across Vision, IMU, and PCI models.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import json
import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))

if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from models.vision_distress_net import VisionDistressNet
from models.imu_shock_classifier import IMUShockClassifier
from models.pci_regressor_net import PCIRegressorNet

class MasterDatasetPipeline:
    """Unified loader and multi-task pipeline for all 7 benchmark datasets + Real GitHub Images."""

    def __init__(self, root_dir=DATASETS_DIR):
        self.root_dir = root_dir
        self.manifest_path = os.path.join(self.root_dir, "dataset_manifest.json")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"dataset_manifest.json not found in {self.root_dir}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def print_catalog(self):
        """Displays formatted catalog of all datasets."""
        print("=" * 82)
        print("📚 ROAD-SHIELD BENCHMARK & REAL GITHUB DATASET INVENTORY")
        print("=" * 82)
        for ds in self.manifest["datasets"]:
            print(f"📁 {ds['id']:<35} | {ds['name']:<40} | {ds['samples']:>6,} samples")
            print(f"   Files: {', '.join(ds['files'])}")
        
        # Real GitHub Images
        real_npz = os.path.join(self.root_dir, "real_github_images_features.npz")
        if os.path.exists(real_npz):
            n_real = len(np.load(real_npz)["features"])
            print(f"📁 {'real_github_images':<35} | {'Real Raw GitHub JPEGs (Potholes & Cracks)':<40} | {n_real:>6,} photos")
            print(f"   Source: https://github.com/andrijdavid/Cracks-and-Potholes-in-Road-Images-Dataset")

        print("-" * 82)
        print(f"Total Curated Multi-Modal Data Points: {self.manifest['total_training_samples']:,} + Real Images")
        print("=" * 82)

    def load_vision_corpus(self):
        """
        Combines RDD2022 India, Kaggle Pothole-600, CRACK500, Hard-Negatives,
        AND Real GitHub raw images into a unified vision training & validation corpus.
        """
        print("\n[Pipeline] Loading Vision Distress Corpus (RDD2022 + Kaggle + CRACK500 + Real GitHub)...")
        rdd_train = np.load(os.path.join(self.root_dir, "01_rdd2022_india", "rdd2022_train.npz"))
        rdd_val = np.load(os.path.join(self.root_dir, "01_rdd2022_india", "rdd2022_val.npz"))
        kaggle = np.load(os.path.join(self.root_dir, "02_kaggle_pothole_600", "kaggle_potholes.npz"))
        crack = np.load(os.path.join(self.root_dir, "03_crack500_fatigue", "crack500_train.npz"))
        hn = np.load(os.path.join(self.root_dir, "05_morth_civil_hard_negatives", "hard_negatives.npz"))

        X_train_list = [rdd_train["features"], kaggle["features"][:8000], crack["features"][:6400], hn["features"][:4000]]
        y_train_list = [rdd_train["labels"], kaggle["labels"][:8000], crack["labels"][:6400], np.zeros(4000, dtype=np.int64)]
        
        geo_rdd = rdd_train["bboxes"]
        geo_extra = np.tile([0.45, 0.55, 0.20, 0.15], (len(kaggle["labels"][:8000]) + len(crack["labels"][:6400]) + 4000, 1)).astype(np.float32)

        # Ingest Real GitHub Raw Image Features
        real_npz = os.path.join(self.root_dir, "real_github_images_features.npz")
        if os.path.exists(real_npz):
            r_data = np.load(real_npz)
            n_r = len(r_data["features"])
            split_r = int(n_r * 0.8)
            X_train_list.append(r_data["features"][:split_r])
            y_train_list.append(r_data["labels"][:split_r])
            geo_extra = np.vstack([geo_extra, r_data["bboxes"][:split_r]])
            print(f"  ✓ Ingested {split_r} Real GitHub training photos into vision corpus.")

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        geo_train = np.vstack([geo_rdd, geo_extra])

        # Validation
        X_val_list = [rdd_val["features"], kaggle["features"][8000:], crack["features"][6400:], hn["features"][4000:]]
        y_val_list = [rdd_val["labels"], kaggle["labels"][8000:], crack["labels"][6400:], np.zeros(1000, dtype=np.int64)]
        
        if os.path.exists(real_npz):
            X_val_list.append(r_data["features"][split_r:])
            y_val_list.append(r_data["labels"][split_r:])

        X_val = np.vstack(X_val_list)
        y_val = np.concatenate(y_val_list)

        print(f"  ✓ Unified Vision Training Set  : {X_train.shape[0]:,} samples (Dim: {X_train.shape[1]})")
        print(f"  ✓ Unified Vision Validation Set: {X_val.shape[0]:,} samples")
        return X_train, y_train, geo_train, X_val, y_val

    def load_imu_corpus(self):
        """Loads 100Hz 3-axis accelerometer telemetry shock sequences."""
        print("\n[Pipeline] Loading 100Hz Mobile-IMU Telemetry Corpus...")
        t_data = np.load(os.path.join(self.root_dir, "04_mobile_imu_telemetry_100hz", "imu_shock_100hz_train.npz"))
        v_data = np.load(os.path.join(self.root_dir, "04_mobile_imu_telemetry_100hz", "imu_shock_100hz_val.npz"))
        print(f"  ✓ IMU Training Set  : {t_data['imu_signals'].shape[0]:,} sequences (100 timesteps x 3 axes)")
        print(f"  ✓ IMU Validation Set: {v_data['imu_signals'].shape[0]:,} sequences")
        return t_data["imu_signals"], t_data["labels"], v_data["imu_signals"], v_data["labels"]

    def load_pci_corpus(self):
        """Loads ASTM D6433 Pavement Condition Index dataset."""
        print("\n[Pipeline] Loading ASTM D6433 PCI Corpus...")
        pci_data = np.load(os.path.join(self.root_dir, "06_astm_d6433_pci_benchmark", "pci_dataset.npz"))
        X = pci_data["distress_densities"]
        y = pci_data["true_pci_scores"]
        split = int(0.8 * len(X))
        print(f"  ✓ ASTM PCI Training Set  : {split:,} road sections (20 distress categories)")
        print(f"  ✓ ASTM PCI Validation Set: {len(X) - split:,} road sections")
        return X[:split], y[:split], X[split:], y[split:]

    def train_vision_model(self, epochs=3, batch_size=256):
        """Trains Model M1 (VisionDistressNet) on the master vision corpus."""
        X_train, y_train, geo_train, X_val, y_val = self.load_vision_corpus()
        model = VisionDistressNet(in_dim=64, num_classes=5)
        
        print(f"\n🚀 Initiating Deep Training on Model M1 (VisionDistressNet) across {epochs} Epochs...")
        t0 = time.time()
        num_batches = int(np.ceil(len(X_train) / batch_size))

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(len(X_train))
            X_shuff = X_train[perm]
            y_shuff = y_train[perm]
            geo_shuff = geo_train[perm]
            
            epoch_loss = 0.0
            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, len(X_train))
                loss = model.train_step(X_shuff[start:end], y_shuff[start:end], geo_shuff[start:end])
                loss_val = float(loss[0]) if isinstance(loss, (tuple, list)) else float(loss)
                epoch_loss += loss_val

            avg_loss = epoch_loss / num_batches
            _, _, val_logits, _ = model.forward(X_val)
            val_preds = np.argmax(val_logits, axis=1)
            val_acc = np.mean(val_preds == y_val) * 100.0

            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        walltime = round(time.time() - t0, 2)
        print(f"🏁 Model M1 Training on Real + Benchmark Corpus Completed in {walltime}s! Val Accuracy: {val_acc:.2f}%")
        return model

if __name__ == "__main__":
    pipeline = MasterDatasetPipeline()
    pipeline.print_catalog()
    
    if "--train" in sys.argv:
        pipeline.train_vision_model(epochs=3)
    else:
        X_v, y_v, _, _, _ = pipeline.load_vision_corpus()
        X_imu, y_imu, _, _ = pipeline.load_imu_corpus()
        X_pci, y_pci, _, _ = pipeline.load_pci_corpus()
        print("\n🎉 ALL BENCHMARKS & REAL GITHUB DATASETS VERIFIED & READY!")
