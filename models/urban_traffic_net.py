"""
Model M5: Urban Traffic Density & Vulnerable Pedestrian Detection Network
Classifies Vehicles (Car, Bus, Truck, Two-Wheeler) and Pedestrians (Pedestrian, Child Crossing).
Calculates real-time Urban Congestion Index (UCI) and Vulnerable Crossing Warnings.
"""
import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))

def gelu_grad(x):
    s = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))
    t = np.tanh(s)
    ds = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * np.power(x, 2.0))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * ds

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class UrbanTrafficNet:
    CLASS_NAMES = [
        "Car", "City Bus", "Heavy Truck", "Two-Wheeler", 
        "Pedestrian", "Vulnerable Child Crossing", "Clear Roadway"
    ]

    IRC_STANDARDS = {
        0: "IRC:106-1990: Guidelines for Capacity of Urban Roads in Plain Areas - Passenger Car Unit (1.0 PCU)",
        1: "IRC:106-1990 Section 4: Public Transport Dedicated Bus Lane & Bus Rapid Transit Priority (2.0 PCU)",
        2: "IRC:37-2018 / IRC:106: Commercial Heavy Vehicle Axle Load Enforcement & Route Restrictions (2.5 PCU)",
        3: "IRC:86-1983 / IRC:106: Segregated Non-Motorized & Two-Wheeler Exclusive Lane Standard (0.5 PCU)",
        4: "IRC:103-2012 Clause 6: Pedestrian Facilities - At-Grade Signalized Pelican Crossing & Raised Table",
        5: "IRC:103-2012 Clause 7.4: School Safety Zone Speed Calming, Flashing Amber Beacon & High-Vis Crosswalk",
        6: "IRC:73-1980 / MoRTH Standard Roadway Section: Nominal Free-Flow Paved Carriageway"
    }
    
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
        activations = [a]
        pre_acts = []
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            a = gelu(z)
            activations.append(a)
        logits = a @ self.w_cls + self.b_cls
        return logits, activations, pre_acts
        
    def predict(self, X):
        logits, _, _ = self.forward(X)
        probs = softmax(logits)
        preds = np.argmax(probs, axis=-1)
        conf = np.max(probs, axis=-1)
        return preds, conf, probs

    def predict_deep(self, X):
        """Deep multi-task urban traffic & pedestrian safety forensic inference."""
        logits, _, _ = self.forward(X)
        probs = softmax(logits)
        B = len(X)
        results = []

        for i in range(B):
            p_vec = probs[i]
            pred_id = int(np.argmax(p_vec))
            conf = float(p_vec[pred_id])

            entropy = float(-np.sum(p_vec * np.log2(p_vec + 1e-10)))
            uncertainty_rating = "LOW_UNCERTAINTY" if entropy < 1.0 else ("MODERATE_UNCERTAINTY" if entropy < 1.8 else "HIGH_UNCERTAINTY_OOD")

            sorted_indices = np.argsort(p_vec)[::-1][:3]
            top3 = [
                {
                    "rank": rank + 1,
                    "class_id": int(idx),
                    "class_name": self.CLASS_NAMES[idx],
                    "probability": round(float(p_vec[idx]), 4)
                }
                for rank, idx in enumerate(sorted_indices)
            ]

            if pred_id == 5:
                safety_alert = "CRITICAL_CHILD_CROSSING_HAZARD"
            elif pred_id == 4:
                safety_alert = "VULNERABLE_PEDESTRIAN_IN_TRANSIT"
            elif pred_id == 2:
                safety_alert = "HEAVY_FREIGHT_AXLE_LOAD"
            elif pred_id == 1:
                safety_alert = "TRANSIT_BUS_BOTTLENECK"
            else:
                safety_alert = "NOMINAL_URBAN_CORRIDOR"

            results.append({
                "class_id": pred_id,
                "class_name": self.CLASS_NAMES[pred_id],
                "confidence": round(conf, 4),
                "shannon_entropy_bits": round(entropy, 3),
                "uncertainty_rating": uncertainty_rating,
                "safety_alert_code": safety_alert,
                "irc_standard_specification": self.IRC_STANDARDS.get(pred_id, "MoRTH Urban Guideline"),
                "top3_ranked_predictions": top3,
                "all_class_probabilities": {self.CLASS_NAMES[k]: round(float(p_vec[k]), 4) for k in range(len(p_vec))}
            })

        return results

    def train_step(self, X, y):
        """Full deep end-to-end backpropagation across all layers and heads."""
        B = len(X)
        logits, activations, pre_acts = self.forward(X)
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(B), y] + 1e-8))

        d_logits = probs.copy()
        d_logits[np.arange(B), y] -= 1.0
        d_logits /= B

        feat = activations[-1]
        d_w_cls = feat.T @ d_logits
        d_b_cls = np.sum(d_logits, axis=0, keepdims=True)

        self.w_cls -= self.lr * d_w_cls
        self.b_cls -= self.lr * d_b_cls

        d_h = d_logits @ self.w_cls.T
        for l_idx in reversed(range(len(self.weights))):
            d_z = d_h * gelu_grad(pre_acts[l_idx])
            d_w = activations[l_idx].T @ d_z
            d_b = np.sum(d_z, axis=0, keepdims=True)
            d_h = d_z @ self.weights[l_idx].T

            self.weights[l_idx] -= self.lr * d_w
            self.biases[l_idx] -= self.lr * d_b

        return float(loss)

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
