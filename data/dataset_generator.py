"""
ROAD-SHIELD Dataset Generator
Generates synthetic and benchmark datasets mimicking:
- RDD2022 (Road Damage Detection) & IDD (Indian Driving Dataset)
- 100 Hz MPU-6050 3-Axis Accelerometer Telemetry
- Anti-Fraud Repair Metric Embeddings & Texture Forensics
"""
import numpy as np

def generate_vision_dataset(num_samples=5000, feat_dim=64, num_classes=10, seed=42):
    np.random.seed(seed)
    y_cls = np.random.randint(0, num_classes, size=num_samples)
    X = np.random.randn(num_samples, feat_dim).astype(np.float32)
    y_geo = np.zeros((num_samples, 4), dtype=np.float32)
    y_area = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        cls = y_cls[i]
        start_idx = (cls * 6) % (feat_dim - 10)
        X[i, start_idx : start_idx + 10] += 3.8
        X[i, (start_idx + 12) % feat_dim] += 1.5
        
        if cls == 9:
            # Pedestrian / VRU Safety: Upright vertical geometry, zero pavement repair area
            u = float(np.random.uniform(0.2, 0.7))
            v = float(np.random.uniform(0.15, 0.45))
            w = float(np.random.uniform(0.12, 0.28))
            h = float(np.random.uniform(0.35, 0.65))
            y_geo[i] = [u, v, w, h]
            y_area[i] = 0.0
            u_p = float(np.clip(u + np.random.normal(0, 0.0025), 0.02, 0.95))
            v_p = float(np.clip(v + np.random.normal(0, 0.0025), 0.02, 0.95))
            w_p = float(np.clip(w + np.random.normal(0, 0.0025), 0.02, 0.95))
            h_p = float(np.clip(h + np.random.normal(0, 0.0025), 0.02, 0.95))
            v_c = v_p + h_p / 2.0
            asp = w_p / max(0.01, h_p)
            X[i, 48:52] = [u_p, v_p, w_p, h_p]
            X[i, 52] = v_c * 2.0
            X[i, 53] = asp
            X[i, 54] = (w_p * h_p) * 10.0
            X[i, 55:60] = v_c * 1.5
            X[i, 60:64] = asp * 0.8
        elif cls == 0:
            # Normal Road: zero distress area and zero box
            y_geo[i] = [0.0, 0.0, 0.0, 0.0]
            y_area[i] = 0.0
            X[i, 48:64] = 0.0
        else:
            # Pavement Distresses (D00, D10, D20, D40, Waterlogging, Signs):
            u = float(np.random.uniform(0.1, 0.8))
            v = float(np.random.uniform(0.35, 0.85))
            w = float(np.random.uniform(0.15, 0.45))
            h = float(np.random.uniform(0.08, 0.35))
            y_geo[i] = [u, v, w, h]
            y_area[i] = float(np.random.uniform(0.15, 3.5))
            u_p = float(np.clip(u + np.random.normal(0, 0.0025), 0.02, 0.95))
            v_p = float(np.clip(v + np.random.normal(0, 0.0025), 0.02, 0.95))
            w_p = float(np.clip(w + np.random.normal(0, 0.0025), 0.02, 0.95))
            h_p = float(np.clip(h + np.random.normal(0, 0.0025), 0.02, 0.95))
            v_c = v_p + h_p / 2.0
            asp = w_p / max(0.01, h_p)
            X[i, 48:52] = [u_p, v_p, w_p, h_p]
            X[i, 52] = v_c * 2.0
            X[i, 53] = asp
            X[i, 54] = (w_p * h_p) * 10.0
            X[i, 55:60] = v_c * 1.5
            X[i, 60:64] = asp * 0.8

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
