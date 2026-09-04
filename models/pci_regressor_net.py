"""
Model M_PCI: ASTM D6433 Continuous Pavement Condition Index (PCI) Regressor
Maps composite road distress density vectors into continuous 0-100 rating.
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

class PCIRegressorNet:
    def __init__(self, in_features=12, hidden_dims=[64, 32], lr=0.003, seed=42):
        np.random.seed(seed)
        self.lr = lr
        dims = [in_features] + hidden_dims + [1]
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

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _, _, out = self.forward(X)
        pci = np.clip(out.flatten(), 0.0, 100.0)
        return pci

    def get_rating_category(self, pci_score):
        if pci_score >= 85:
            return "EXCELLENT", "Optimal surface texture; routine monitoring only."
        elif pci_score >= 70:
            return "SATISFACTORY", "Minor hairline cracks; schedule preventive seal coating."
        elif pci_score >= 55:
            return "FAIR", "Moderate distress; bituminous patch repair required within 30 days."
        elif pci_score >= 40:
            return "POOR", "Significant alligator fatigue; structural overlay needed."
        elif pci_score >= 25:
            return "VERY_POOR", "Severe sub-base pumping; axle load failure hazard."
        else:
            return "FAILED", "Complete structural collapse; emergency full-depth reconstruction mandatory."

    def train_step(self, X, y):
        self.t += 1
        B = X.shape[0]
        y_col = y.reshape(-1, 1)
        activations, pre_acts, out = self.forward(X)
        
        # Smooth L1 (Huber) Loss
        diff = out - y_col
        abs_diff = np.abs(diff)
        loss = np.mean(np.where(abs_diff < 1.0, 0.5 * (diff**2), abs_diff - 0.5))
        d_out = np.where(abs_diff < 1.0, diff, np.sign(diff)) / B
        
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
