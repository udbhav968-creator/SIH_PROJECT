"""
Model M5: Urban Traffic Density & Vulnerable Pedestrian Detection Network
Classifies Vehicles (Car, Bus, Truck, Two-Wheeler) and Pedestrians (Pedestrian, Child Crossing).
Calculates real-time Urban Congestion Index (UCI) and Vulnerable Crossing Warnings.
"""
import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class UrbanTrafficNet:
    CLASS_NAMES = [
        "Car", "City Bus", "Heavy Truck", "Two-Wheeler", 
        "Pedestrian", "Vulnerable Child Crossing", "Clear Roadway"
    ]
    
    def __init__(self, in_features=48, hidden_dims=[256, 128], num_classes=7, lr=0.002, seed=42):
        np.random.seed(seed)
        self.in_features = in_features
        self.num_classes = num_classes
        self.lr = lr
        
        # Deep CNN/FFN Hybrid backbone
        dims = [in_features] + hidden_dims
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]).astype(np.float32) * np.sqrt(2.0 / dims[i])
            b = np.zeros((1, dims[i+1]), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
            
        # Classification Head
        self.w_cls = np.random.randn(hidden_dims[-1], num_classes).astype(np.float32) * np.sqrt(2.0 / hidden_dims[-1])
        self.b_cls = np.zeros((1, num_classes), dtype=np.float32)
        
    def forward(self, X):
        a = X
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            a = gelu(z)
        logits = a @ self.w_cls + self.b_cls
        return logits
        
    def predict(self, X):
        logits = self.forward(X)
        probs = softmax(logits)
        preds = np.argmax(probs, axis=-1)
        conf = np.max(probs, axis=-1)
        return preds, conf, probs

    def calculate_congestion_index(self, vehicle_counts, road_capacity=40):
        """
        Computes Urban Congestion Index (0.0 to 1.0) and Bottleneck status.
        Weights: Truck (2.5 PCU), Bus (2.0 PCU), Car (1.0 PCU), Two-Wheeler (0.5 PCU)
        """
        pcu = (
            vehicle_counts.get("Car", 0) * 1.0 +
            vehicle_counts.get("City Bus", 0) * 2.0 +
            vehicle_counts.get("Heavy Truck", 0) * 2.5 +
            vehicle_counts.get("Two-Wheeler", 0) * 0.5
        )
        density_ratio = min(1.0, round(pcu / road_capacity, 3))
        
        if density_ratio >= 0.80:
            status = "SEVERELY_CONGESTED_BOTTLENECK"
            color = "#ef4444"
        elif density_ratio >= 0.50:
            status = "MODERATE_FLOW"
            color = "#f59e0b"
        else:
            status = "OPTIMAL_FREE_FLOW"
            color = "#10b981"
            
        return {
            "pcu_equivalent": round(pcu, 1),
            "congestion_index": density_ratio,
            "status": status,
            "indicator_color": color,
            "estimated_delay_mins": round(density_ratio * 18.5, 1)
        }

    def compute_urban_congestion_index(self, feature_vec, road_capacity=40):
        """
        Compute Urban Congestion Index (UCI) scalar from a raw 48-dim feature vector.
        Returns float (total PCU equivalent, 0-300 range).
        """
        feat = np.array(feature_vec, dtype=np.float32).flatten()
        car_count   = max(0, int(abs(feat[0]) * 20 + 10))
        bus_count   = max(0, int(abs(feat[1]) * 5 + 2))
        truck_count = max(0, int(abs(feat[2]) * 4 + 1))
        bike_count  = max(0, int(abs(feat[3]) * 15 + 5))
        counts = {"Car": car_count, "City Bus": bus_count,
                  "Heavy Truck": truck_count, "Two-Wheeler": bike_count}
        result = self.calculate_congestion_index(counts, road_capacity=road_capacity)
        return round(float(result["pcu_equivalent"]), 1)

    def get_congestion_details(self, feature_vec, road_capacity=40):
        """Full congestion analysis dict from raw feature vector."""
        feat = np.array(feature_vec, dtype=np.float32).flatten()
        car_count   = max(0, int(abs(feat[0]) * 20 + 10))
        bus_count   = max(0, int(abs(feat[1]) * 5 + 2))
        truck_count = max(0, int(abs(feat[2]) * 4 + 1))
        bike_count  = max(0, int(abs(feat[3]) * 15 + 5))
        counts = {"Car": car_count, "City Bus": bus_count,
                  "Heavy Truck": truck_count, "Two-Wheeler": bike_count}
        result = self.calculate_congestion_index(counts, road_capacity=road_capacity)
        result["vehicle_counts"] = counts
        return result

    def save_weights(self, path):
        np.savez_compressed(
            path,
            w0=self.weights[0], b0=self.biases[0],
            w1=self.weights[1], b1=self.biases[1],
            w_cls=self.w_cls, b_cls=self.b_cls
        )

    def load_weights(self, path):
        data = np.load(path)
        self.weights[0] = data["w0"]
        self.biases[0] = data["b0"]
        self.weights[1] = data["w1"]
        self.biases[1] = data["b1"]
        self.w_cls = data["w_cls"]
        self.b_cls = data["b_cls"]
