"""
Model M4: 100 Hz 3-Axis IMU Shock Classifier
Extracts multi-scale temporal dynamic features from sliding 100-sample windows
and classifies them into [Smooth, Expansion Joint, Rumble Strip, Pothole Impact].
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

class IMUShockClassifier:
    CLASS_NAMES = ["Smooth Asphalt", "Expansion Joint", "Rumble Strip", "Pothole Impact"]
    
    def __init__(self, in_features=36, hidden_dims=[64, 32], num_classes=4, lr=0.003, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.num_classes = num_classes
        
        dims = [in_features] + hidden_dims + [num_classes]
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
        self.feat_mean = np.zeros((1, in_features), dtype=np.float32)
        self.feat_std = np.ones((1, in_features), dtype=np.float32)

    @staticmethod
    def extract_temporal_features(X_raw):
        """
        Converts (B, 100, 3) raw time series into 36 high-order temporal dynamics features.
        """
        B, T, C = X_raw.shape
        feats = []
        for c in range(C):
            sig = X_raw[:, :, c]
            mean = np.mean(sig, axis=1, keepdims=True)
            std = np.std(sig, axis=1, keepdims=True)
            var = np.var(sig, axis=1, keepdims=True)
            mx = np.max(sig, axis=1, keepdims=True)
            mn = np.min(sig, axis=1, keepdims=True)
            ptp = mx - mn
            # Zero crossing rate around mean
            centered = sig - mean
            zcr = np.mean(np.abs(np.diff(np.sign(centered), axis=1)) > 0, axis=1, keepdims=True)
            # Energy
            energy = np.mean(sig**2, axis=1, keepdims=True)
            # Derivative (jerk)
            diff1 = np.diff(sig, axis=1)
            jerk_max = np.max(np.abs(diff1), axis=1, keepdims=True)
            jerk_mean = np.mean(np.abs(diff1), axis=1, keepdims=True)
            # Half-window power ratio (asymmetry)
            p1 = np.mean(sig[:, :T//2]**2, axis=1, keepdims=True)
            p2 = np.mean(sig[:, T//2:]**2, axis=1, keepdims=True)
            ratio = (p2 + 1e-5) / (p1 + 1e-5)
            
            feats.extend([mean, std, var, mx, mn, ptp, zcr, energy, jerk_max, jerk_mean, p1, ratio])
            
        return np.concatenate(feats, axis=1).astype(np.float32)

    def forward(self, feats):
        activations = [feats]
        pre_acts = []
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            a = gelu(z)
            activations.append(a)
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        return activations, pre_acts, logits

    def predict(self, X_raw):
        feats = self.extract_temporal_features(X_raw)
        feats = (feats - self.feat_mean) / (self.feat_std + 1e-5)
        _, _, logits = self.forward(feats)
        probs = softmax(logits)
        preds = np.argmax(probs, axis=-1)
        pothole_conf = probs[:, 3]
        return preds, pothole_conf, probs

    def train_step(self, feats, y):
        self.t += 1
        B = feats.shape[0]
        activations, pre_acts, logits = self.forward(feats)
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(B), y] + 1e-8))
        
        d_out = probs.copy()
        d_out[np.arange(B), y] -= 1.0
        d_out /= B
        
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
                
            self._adam_update(self.weights[i], d_w, self.m_w[i], self.v_w[i])
            self._adam_update(self.biases[i], d_b, self.m_b[i], self.v_b[i])
            
        return loss

    def _adam_update(self, param, grad, m, v):
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad**2)
        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def save_weights(self, path):
        np.savez_compressed(
            path,
            w0=self.weights[0], b0=self.biases[0],
            w1=self.weights[1], b1=self.biases[1],
            w2=self.weights[2], b2=self.biases[2],
            feat_mean=self.feat_mean,
            feat_std=self.feat_std
        )

    def load_weights(self, path):
        data = np.load(path)
        self.weights[0] = data["w0"]
        self.biases[0] = data["b0"]
        self.weights[1] = data["w1"]
        self.biases[1] = data["b1"]
        self.weights[2] = data["w2"]
        self.biases[2] = data["b2"]
        if "feat_mean" in data:
            self.feat_mean = data["feat_mean"]
        if "feat_std" in data:
            self.feat_std = data["feat_std"]
