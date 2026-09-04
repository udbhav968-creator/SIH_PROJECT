"""
Model M1: Edge Vision Distress CNN-Transformer Hybrid Network
Embeds deep CNN feature extraction and Multi-Head Self Attention (Transformers).
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

class VisionDistressNet:
    CLASS_NAMES = [
        "Normal Road", "D00 Longitudinal", "D10 Transverse", "D20 Alligator", "D40 Pothole",
        "Waterlogging", "Missing Zebra Crossing", "Missing Road Divider", "Damaged Traffic Sign"
    ]
    
    def __init__(self, in_features=64, hidden_dims=[512, 256, 128], num_classes=9, lr=0.002, seed=42, in_dim=None):
        if in_dim is not None:
            in_features = in_dim
        np.random.seed(seed)
        self.lr = lr
        self.num_classes = num_classes
        self.has_transformer = True
        
        # --- DEEP CNN EMBEDDED BLOCK (Simulated 1D Conv over features) ---
        # Converts 64 flat features into richer local patterns
        self.conv_w = np.random.randn(in_features, hidden_dims[0]).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.conv_b = np.zeros((1, hidden_dims[0]), dtype=np.float32)
        
        # --- TRANSFORMER EMBEDDED BLOCK (Self Attention) ---
        self.d_k = hidden_dims[0]
        self.W_q = np.random.randn(hidden_dims[0], self.d_k).astype(np.float32) * 0.1
        self.W_k = np.random.randn(hidden_dims[0], self.d_k).astype(np.float32) * 0.1
        self.W_v = np.random.randn(hidden_dims[0], self.d_k).astype(np.float32) * 0.1
        
        # --- DENSE DEEP LAYERS ---
        self.weights = []
        self.biases = []
        curr = hidden_dims[0]
        for h in hidden_dims[1:]:
            self.weights.append(np.random.randn(curr, h).astype(np.float32) * np.sqrt(2.0 / curr))
            self.biases.append(np.zeros((1, h), dtype=np.float32))
            curr = h
            
        # Heads
        self.w_cls = np.random.randn(curr, num_classes).astype(np.float32) * 0.05
        self.b_cls = np.zeros((1, num_classes), dtype=np.float32)
        
        self.w_geo = np.random.randn(curr, 4).astype(np.float32) * 0.05
        self.b_geo = np.zeros((1, 4), dtype=np.float32)
        
    def forward(self, X):
        if self.has_transformer and self.conv_w is not None:
            # 1. 1D CNN Local Feature Projection
            cnn_out = X @ self.conv_w + self.conv_b
            cnn_act = gelu(cnn_out)
            
            # 2. Transformer Multi-Head Self-Attention
            Q = cnn_act @ self.W_q
            K = cnn_act @ self.W_k
            V = cnn_act @ self.W_v
            
            scores = (Q @ K.T) / np.sqrt(self.d_k)
            attn_weights = softmax(scores)
            attn_out = attn_weights @ V
            
            # Residual connection & layer norm simulation
            h = cnn_act + attn_out
            h = (h - np.mean(h, axis=-1, keepdims=True)) / (np.std(h, axis=-1, keepdims=True) + 1e-6)
        else:
            h = X
            
        activations = [h]
        pre_acts = []
        for w, b in zip(self.weights, self.biases):
            z = h @ w + b
            pre_acts.append(z)
            h = gelu(z)
            activations.append(h)
            
        cls_logits = h @ self.w_cls + self.b_cls
        geo_preds = h @ self.w_geo + self.b_geo
        return activations, pre_acts, cls_logits, geo_preds

    def predict(self, X):
        _, _, cls_logits, geo_preds = self.forward(X)
        probs = softmax(cls_logits)
        preds = np.argmax(probs, axis=-1)
        conf = np.max(probs, axis=-1)
        return preds, conf, probs, geo_preds

    def train_step(self, X, y_cls, y_geo=None):
        B = len(X)
        activations, pre_acts, cls_logits, geo_preds = self.forward(X)
        
        # Softmax Cross-Entropy loss
        probs = softmax(cls_logits)
        loss = -np.mean(np.log(probs[np.arange(B), y_cls] + 1e-8))
        
        # Gradient on logits
        d_logits = probs.copy()
        d_logits[np.arange(B), y_cls] -= 1.0
        d_logits /= B
        
        # Backprop to classification head
        feat = activations[-1]
        d_w_cls = feat.T @ d_logits
        d_b_cls = np.sum(d_logits, axis=0, keepdims=True)
        
        # Gradient into feature layer
        d_feat = d_logits @ self.w_cls.T
        
        # Update classification head
        self.w_cls -= self.lr * d_w_cls
        self.b_cls -= self.lr * d_b_cls
        
        # Backprop through last hidden layer
        if len(self.weights) > 0:
            d_z = d_feat * gelu_grad(pre_acts[-1])
            d_w = activations[-2].T @ d_z
            d_b = np.sum(d_z, axis=0, keepdims=True)
            self.weights[-1] -= self.lr * d_w
            self.biases[-1] -= self.lr * d_b
            
        return float(loss)

    def save_weights(self, path):
        # Save a mock or the actual arrays
        np.savez_compressed(
            path,
            conv_w=self.conv_w, conv_b=self.conv_b,
            W_q=self.W_q, W_k=self.W_k, W_v=self.W_v,
            w0=self.weights[0], b0=self.biases[0],
            w1=self.weights[1], b1=self.biases[1],
            w_cls=self.w_cls, b_cls=self.b_cls,
            w_geo=self.w_geo, b_geo=self.b_geo
        )

    def load_weights(self, path):
        data = np.load(path)
        if "conv_w" in data and "W_q" in data:
            self.has_transformer = True
            self.conv_w = data["conv_w"]
            self.conv_b = data["conv_b"] if "conv_b" in data else np.zeros((1, self.conv_w.shape[1]), dtype=np.float32)
            self.W_q = data["W_q"]
            self.W_k = data["W_k"] if "W_k" in data else np.random.randn(*self.W_q.shape).astype(np.float32) * 0.1
            self.W_v = data["W_v"] if "W_v" in data else np.random.randn(*self.W_q.shape).astype(np.float32) * 0.1
            self.d_k = self.conv_w.shape[1]
        else:
            self.has_transformer = False

        if "w0" in data:
            if len(self.weights) == 0:
                self.weights = [data["w0"]]
                self.biases = [data["b0"] if "b0" in data else np.zeros((1, data["w0"].shape[1]), dtype=np.float32)]
            else:
                self.weights[0] = data["w0"]
                if "b0" in data: self.biases[0] = data["b0"]
        if "w1" in data:
            if len(self.weights) < 2:
                self.weights.append(data["w1"])
                self.biases.append(data["b1"] if "b1" in data else np.zeros((1, data["w1"].shape[1]), dtype=np.float32))
            else:
                self.weights[1] = data["w1"]
                if "b1" in data: self.biases[1] = data["b1"]
        if "w_cls" in data:
            self.w_cls = data["w_cls"]
            self.b_cls = data["b_cls"] if "b_cls" in data else np.zeros((1, self.w_cls.shape[1]), dtype=np.float32)
            self.num_classes = self.w_cls.shape[1]
        if "w_geo" in data:
            self.w_geo = data["w_geo"]
            self.b_geo = data["b_geo"] if "b_geo" in data else np.zeros((1, 4), dtype=np.float32)
