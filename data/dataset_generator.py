"""
ROAD-SHIELD Dataset Generator
Generates synthetic and benchmark datasets mimicking:
- RDD2022 (Road Damage Detection) & IDD (Indian Driving Dataset)
- 100 Hz MPU-6050 3-Axis Accelerometer Telemetry
- Anti-Fraud Repair Metric Embeddings & Texture Forensics
"""
import numpy as np

def generate_vision_dataset(num_samples=5000, feat_dim=64, seed=42):
    np.random.seed(seed)
    y_cls = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.25, 0.20, 0.15, 0.20, 0.20])
    
    X = np.zeros((num_samples, feat_dim), dtype=np.float32)
    y_geo = np.zeros((num_samples, 4), dtype=np.float32)
    y_area = np.zeros(num_samples, dtype=np.float32)
    
    for i in range(num_samples):
        cls = y_cls[i]
        base = np.random.normal(loc=0.0, scale=0.3, size=feat_dim).astype(np.float32)
        
        if cls == 0:  # Normal Road
            X[i] = base
            y_geo[i] = [0.0, 0.0, 0.0, 0.0]
            y_area[i] = 0.0
        elif cls == 1:  # D00 Longitudinal Crack
            base[0:16] += np.random.normal(loc=2.0, scale=0.35, size=16)
            X[i] = base
            y_geo[i] = [np.random.uniform(0.3, 0.6), np.random.uniform(0.1, 0.3), np.random.uniform(0.05, 0.15), np.random.uniform(0.4, 0.8)]
            y_area[i] = np.random.uniform(0.1, 0.4)
        elif cls == 2:  # D10 Transverse Crack
            base[16:32] += np.random.normal(loc=2.0, scale=0.35, size=16)
            X[i] = base
            y_geo[i] = [np.random.uniform(0.1, 0.3), np.random.uniform(0.4, 0.7), np.random.uniform(0.4, 0.8), np.random.uniform(0.05, 0.15)]
            y_area[i] = np.random.uniform(0.15, 0.5)
        elif cls == 3:  # D20 Alligator Crack
            base[32:48] += np.random.normal(loc=2.3, scale=0.45, size=16)
            X[i] = base
            y_geo[i] = [np.random.uniform(0.2, 0.5), np.random.uniform(0.3, 0.6), np.random.uniform(0.2, 0.45), np.random.uniform(0.2, 0.45)]
            y_area[i] = np.random.uniform(0.5, 1.8)
        elif cls == 4:  # D40 Pothole Cavity
            base[48:64] += np.random.normal(loc=2.9, scale=0.5, size=16)
            X[i] = base
            y_geo[i] = [np.random.uniform(0.25, 0.65), np.random.uniform(0.4, 0.75), np.random.uniform(0.15, 0.4), np.random.uniform(0.15, 0.35)]
            y_area[i] = np.random.uniform(0.3, 3.5)
            
    return X, y_cls, y_geo, y_area

def generate_imu_dataset(num_samples=4000, timesteps=100, seed=42):
    np.random.seed(seed)
    y = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.35, 0.20, 0.20, 0.25])
    X = np.zeros((num_samples, timesteps, 3), dtype=np.float32)
    t = np.linspace(0, 1.0, timesteps)
    
    for i in range(num_samples):
        cls = y[i]
        ax = np.random.normal(0.0, 0.2, timesteps)
        ay = np.random.normal(0.0, 0.2, timesteps)
        az = np.random.normal(9.81, 0.35, timesteps)
        
        if cls == 0:  # Smooth Asphalt
            pass
        elif cls == 1:  # Expansion Joint (narrow sharp impulse)
            idx = np.random.randint(40, 60)
            az[idx:idx+3] += np.random.uniform(2.2, 3.5)
            az[idx+3:idx+5] -= np.random.uniform(1.0, 1.8)
        elif cls == 2:  # Rumble Strip (20Hz oscillation)
            freq = np.random.uniform(18.0, 24.0)
            phase = np.random.uniform(0, np.pi)
            osc = np.sin(2 * np.pi * freq * t + phase) * np.random.uniform(1.8, 2.9)
            az += osc
            ax += osc * 0.3
        elif cls == 3:  # Pothole Impact (cavity drop + massive bottom-out rebound)
            idx = np.random.randint(40, 55)
            az[idx:idx+4] -= np.random.uniform(3.0, 5.0)
            az[idx+4:idx+8] += np.random.uniform(5.5, 9.0)
            ax[idx+4:idx+8] += np.random.uniform(1.5, 3.2)
            ay[idx+4:idx+8] += np.random.uniform(1.2, 2.8)
            
        X[i, :, 0] = ax
        X[i, :, 1] = ay
        X[i, :, 2] = az
        
    return X, y

def generate_forensic_triplets(num_triplets=2000, embed_dim=48, seed=42):
    np.random.seed(seed)
    anchors = np.zeros((num_triplets, embed_dim), dtype=np.float32)
    positives = np.zeros((num_triplets, embed_dim), dtype=np.float32)
    negatives = np.zeros((num_triplets, embed_dim), dtype=np.float32)
    
    for i in range(num_triplets):
        proto = np.random.randn(embed_dim).astype(np.float32)
        proto /= (np.linalg.norm(proto) + 1e-8)
        
        pos = proto + np.random.normal(0, 0.10, embed_dim).astype(np.float32)
        pos /= (np.linalg.norm(pos) + 1e-8)
        
        neg = np.random.randn(embed_dim).astype(np.float32)
        neg /= (np.linalg.norm(neg) + 1e-8)
        
        anchors[i] = proto
        positives[i] = pos
        negatives[i] = neg
        
    return anchors, positives, negatives
