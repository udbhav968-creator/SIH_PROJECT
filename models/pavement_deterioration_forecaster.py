"""
Model M_DEGRADE: Temporal Pavement Lifecycle Forecaster
Predicts future cavity expansion at 30, 60, 90, and 180 days
based on initial geometry, daily ESAL truck traffic, and monsoon rainfall (mm).
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))

def gelu_grad(x):
    s = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))
    t = np.tanh(s)
    ds = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * np.power(x, 2.0))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * ds

class PavementDeteriorationForecaster:
    HORIZONS = [30, 60, 90, 180]
    
    def __init__(self, in_features=5, hidden_dims=[64, 32], lr=0.003, seed=42):
        np.random.seed(seed)
        self.lr = lr
        dims = [in_features] + hidden_dims + [4] # 4 future horizons
        self.weights = []
        self.biases = []
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]).astype(np.float32) * np.sqrt(2.0 / dims[i])
            b = np.zeros((1, dims[i+1]), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))
            
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self.norm_mean = np.zeros((1, in_features), dtype=np.float32)
        self.norm_std = np.ones((1, in_features), dtype=np.float32)

    def forward(self, X):
        X_norm = (X - self.norm_mean) / (self.norm_std + 1e-5)
        activations = [X_norm]
        pre_acts = []
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            a = gelu(z)
            activations.append(a)
        out = activations[-1] @ self.weights[-1] + self.biases[-1]
        return activations, pre_acts, out

    def forecast(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _, _, out = self.forward(X)
        # Enforce positive area
        pred_areas = np.maximum(X[:, 0:1], out)
        return pred_areas

    def predict_lifecycle_roi(self, init_area_m2, depth_cm, esal_trucks, rain_mm, age_yr=3.5):
        """
        Calculates the financial ROI of immediate preventive patching vs 180-day delayed monsoon repair.
        """
        X = np.array([[init_area_m2, depth_cm, esal_trucks, rain_mm, age_yr]], dtype=np.float32)
        preds = self.forecast(X)[0]
        # Physical Pavement Mechanics: Cumulative deterioration is strictly non-decreasing over time
        a30 = max(float(init_area_m2), float(preds[0]))
        a60 = max(a30, float(preds[1]))
        a90 = max(a60, float(preds[2]))
        a180 = max(a90, float(preds[3]))
        
        # Immediate repair cost (MoRTH Sec 500 DBM @ 7,500/Tonne)
        mass_today = init_area_m2 * (depth_cm / 100.0) * 2.40 * 1.15
        cost_today = round(mass_today * 7500.0, 2)
        
        # 180-day collapsed repair cost (depth increases by 2.2x due to monsoon sub-base scouring)
        depth_180 = depth_cm * 2.2
        mass_180 = a180 * (depth_180 / 100.0) * 2.40 * 1.15
        cost_180 = round(mass_180 * 7500.0, 2)
        savings_inr = round(cost_180 - cost_today, 2)
        
        return {
            "initial_area_m2": round(init_area_m2, 2),
            "forecast_30_days_area_m2": round(a30, 2),
            "forecast_60_days_area_m2": round(a60, 2),
            "forecast_90_days_area_m2": round(a90, 2),
            "forecast_180_days_area_m2": round(a180, 2),
            "immediate_repair_cost_inr": cost_today,
            "delayed_repair_cost_180d_inr": cost_180,
            "municipal_savings_preventive_inr": savings_inr,
            "growth_factor_180d": round(a180 / (init_area_m2 + 1e-5), 2)
        }

    def train_step(self, X, y_targets):
        self.t += 1
        B = X.shape[0]
        activations, pre_acts, out = self.forward(X)
        
        diff = out - y_targets
        loss = np.mean(0.5 * (diff**2))
        d_out = diff / B
        
        d_a = d_out
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            if i == len(self.weights) - 1:
                d_z = d_a
            else:
                d_z = d_a * gelu_grad(pre_acts[i])
                
            d_w = a_prev.T @ d_z
            d_b = np.sum(d_z, axis=0, keepdims=True)
            
            if i > 0:
                d_a = d_z @ self.weights[i].T
                
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * d_w
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (d_w**2)
            m_hat_w = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2**self.t)
            self.weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * d_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (d_b**2)
            m_hat_b = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2**self.t)
            self.biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
            
        return float(loss)

    def save_weights(self, path):
        np.savez_compressed(
            path,
            w0=self.weights[0], b0=self.biases[0],
            w1=self.weights[1], b1=self.biases[1],
            w2=self.weights[2], b2=self.biases[2],
            norm_mean=self.norm_mean, norm_std=self.norm_std
        )

    def load_weights(self, path):
        data = np.load(path)
        self.weights[0] = data["w0"]
        self.biases[0] = data["b0"]
        self.weights[1] = data["w1"]
        self.biases[1] = data["b1"]
        self.weights[2] = data["w2"]
        self.biases[2] = data["b2"]
        if "norm_mean" in data:
            self.norm_mean = data["norm_mean"]
        if "norm_std" in data:
            self.norm_std = data["norm_std"]
