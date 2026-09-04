"""
ROAD-SHIELD Civil Domain Data Augmentation Pipeline
Implements 4-Way Mosaic, CutMix Distress Injection, Optical Weather Simulator
(Monsoon rain streaks, nocturnal sodium lighting, specular glare), and IMU Suspension Noise.
"""
import numpy as np

class CivilDataAugmentor:
    """Advanced multi-modal augmentation engine for pavement inspection."""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
    def mosaic_4way(self, features_list, labels_list):
        """
        Simulates 4-way mosaic distress synthesis by spatially combining 
        quadrants of 4 distinct road pavement feature vectors.
        """
        if len(features_list) < 4:
            return features_list[0], labels_list[0]
            
        dim = features_list[0].shape[0]
        q_dim = dim // 4
        
        mosaic_feat = np.concatenate([
            features_list[0][:q_dim],
            features_list[1][q_dim:2*q_dim],
            features_list[2][2*q_dim:3*q_dim],
            features_list[3][3*q_dim:dim]
        ])
        
        # Primary label is taken from the most severe distress quadrant (max class id)
        max_label = max(labels_list[:4])
        return mosaic_feat.astype(np.float32), max_label

    def cutmix_distress(self, background_feat, distress_patch_feat, patch_ratio=0.35):
        """
        Injects a high-severity distress patch (e.g. D40 Pothole) into a sound pavement vector.
        """
        mixed = background_feat.copy()
        dim = len(background_feat)
        patch_len = int(dim * patch_ratio)
        start_idx = self.rng.randint(0, dim - patch_len + 1)
        mixed[start_idx:start_idx+patch_len] = distress_patch_feat[start_idx:start_idx+patch_len]
        return mixed.astype(np.float32)

    def simulate_monsoon_weather(self, optical_features, rain_intensity=0.8):
        """
        Applies rain streak scattering, specular standing water glare, and surface darkening.
        """
        augmented = optical_features.copy()
        # Surface darkening from wet asphalt
        augmented *= (1.0 - 0.25 * rain_intensity)
        # Specular water reflection peaks
        num_glare_spots = max(1, int(len(augmented) * 0.15))
        glare_indices = self.rng.choice(len(augmented), num_glare_spots, replace=False)
        augmented[glare_indices] += (0.6 * rain_intensity)
        # High frequency rain streak noise
        rain_noise = self.rng.normal(0.0, 0.05 * rain_intensity, size=augmented.shape)
        augmented = np.clip(augmented + rain_noise, 0.0, 1.0)
        return augmented.astype(np.float32)

    def simulate_night_sodium_lighting(self, optical_features):
        """
        Simulates low-lux nocturnal conditions with yellow-amber sodium vapor lamp attenuation.
        """
        augmented = optical_features.copy()
        # Attenuate overall lux
        augmented *= 0.45
        # Add non-uniform spotlight falloff
        falloff = np.linspace(0.8, 0.3, len(augmented))
        augmented *= falloff
        return np.clip(augmented, 0.0, 1.0).astype(np.float32)

    def simulate_imu_suspension_dynamics(self, imu_features, vehicle_speed_kmh=60.0):
        """
        Injects vehicle chassis harmonic vibration (12-25 Hz) and pavement roughness drift.
        """
        augmented = imu_features.copy()
        # Speed-dependent vibration magnitude
        vib_amp = (vehicle_speed_kmh / 100.0) * 0.12
        noise = self.rng.normal(0.0, vib_amp, size=imu_features.shape)
        augmented += noise
        return augmented.astype(np.float32)
