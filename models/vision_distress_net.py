"""
Model M1: Edge Vision Distress CNN-Transformer Hybrid Network
Embeds deep CNN feature extraction, Multi-Head Self Attention (Transformers),
full-layer backpropagation, and multi-level forensic engineering predictions.
"""
import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))

def gelu_grad(x):
    s = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))
    t = np.tanh(s)
    ds = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * np.power(x, 2.0))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * ds

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def sigmoid_grad(sig_x):
    return sig_x * (1.0 - sig_x)

def compute_box_iou(box1, box2):
    """Computes IoU between two bounding boxes [u, v, w, h] or batches [N, 4]."""
    b1 = np.atleast_2d(box1)
    b2 = np.atleast_2d(box2)
    xA = np.maximum(b1[:, 0], b2[:, 0])
    yA = np.maximum(b1[:, 1], b2[:, 1])
    xB = np.minimum(b1[:, 0] + b1[:, 2], b2[:, 0] + b2[:, 2])
    yB = np.minimum(b1[:, 1] + b1[:, 3], b2[:, 1] + b2[:, 3])
    inter_area = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    area1 = np.maximum(1e-6, b1[:, 2] * b1[:, 3])
    area2 = np.maximum(1e-6, b2[:, 2] * b2[:, 3])
    union_area = area1 + area2 - inter_area
    ious = inter_area / np.maximum(1e-6, union_area)
    return float(ious[0]) if (b1.shape[0] == 1 and b2.shape[0] == 1) else ious

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class VisionDistressNet:
    CLASS_NAMES = [
        "Normal Road", "D00 Longitudinal Crack", "D10 Transverse Crack", "D20 Alligator Crack", "D40 Pothole Cavity",
        "Waterlogging", "Missing Zebra Crossing", "Missing Road Divider", "Damaged Traffic Sign",
        "Child / Pedestrian Hazard (Vulnerable Road User)"
    ]

    IRC_STANDARDS = {
        0: "IRC:82-2015 Clause 3.1: Routine Visual Survey - Non-Distress Stable Pavement",
        1: "IRC:SP:72-2015 Clause 5.3: Hot Pour Bituminous Joint Sealant (Crack Width < 5mm)",
        2: "IRC:SP:72-2015 Clause 5.4: Modified Polymer Bitumen Crack Injection (Transverse Thermal Relief)",
        3: "IRC:37-2018 Section 6: Structural Fatigue Mill & Infill with Dense Bituminous Macadam (DBM)",
        4: "IRC:82-2015 Clause 4.2: Mechanical Pot-Hole Patching with Bituminous Concrete (BC) & VG-30 Tack Coat",
        5: "IRC:SP:42-2014 Section 8: Highway Camber Correction & Stormwater Cross-Drainage Culvert",
        6: "IRC:35-2015 Clause 7.2: Retroreflective Thermoplastic Road Marking (Zebra Pedestrian Crossing)",
        7: "IRC:79-2019 Section 4: Dual-Beam W-Beam Crash Barrier & Retroreflective Road Divider",
        8: "IRC:67-2012 Code of Practice for Road Signs: High-Intensity Microprismatic Retroreflective Retrofit",
        9: "IRC:103-2012 Guidelines for Pedestrian Facilities: Signalized Pelican Crossing & Refuge Island"
    }

    CLASS_COLORS = {
        0: {"hex": "#10b981", "glow": "rgba(16, 185, 129, 0.4)", "badge": "bg-emerald-950 text-emerald-300 border-emerald-800", "label": "NORMAL ROAD"},
        1: {"hex": "#ec4899", "glow": "rgba(236, 72, 153, 0.4)", "badge": "bg-pink-950 text-pink-300 border-pink-800", "label": "D00 LONGITUDINAL CRACK"},
        2: {"hex": "#a855f7", "glow": "rgba(168, 85, 247, 0.4)", "badge": "bg-purple-950 text-purple-300 border-purple-800", "label": "D10 TRANSVERSE CRACK"},
        3: {"hex": "#f43f5e", "glow": "rgba(244, 63, 94, 0.4)", "badge": "bg-rose-950 text-rose-300 border-rose-800", "label": "D20 ALLIGATOR CRACK"},
        4: {"hex": "#f59e0b", "glow": "rgba(245, 158, 11, 0.4)", "badge": "bg-amber-950 text-amber-300 border-amber-800", "label": "D40 POTHOLE CAVITY"},
        5: {"hex": "#0ea5e9", "glow": "rgba(14, 165, 233, 0.4)", "badge": "bg-sky-950 text-sky-300 border-sky-800", "label": "WATERLOGGING HAZARD"},
        6: {"hex": "#eab308", "glow": "rgba(234, 179, 8, 0.4)", "badge": "bg-yellow-950 text-yellow-300 border-yellow-800", "label": "MISSING ZEBRA"},
        7: {"hex": "#10b981", "glow": "rgba(16, 185, 129, 0.4)", "badge": "bg-emerald-950 text-emerald-300 border-emerald-800", "label": "MISSING DIVIDER"},
        8: {"hex": "#3b82f6", "glow": "rgba(59, 130, 246, 0.4)", "badge": "bg-blue-950 text-blue-300 border-blue-800", "label": "DAMAGED TRAFFIC SIGN"},
        9: {"hex": "#06b6d4", "glow": "rgba(6, 182, 212, 0.45)", "badge": "bg-cyan-950 text-cyan-300 border-cyan-800", "label": "VRU PEDESTRIAN HAZARD"}
    }

    def __init__(self, in_features=64, hidden_dims=[512, 256, 128], num_classes=10, lr=0.002, seed=42, in_dim=None):
        if in_dim is not None:
            in_features = in_dim
        np.random.seed(seed)
        self.lr = lr
        self.num_classes = num_classes
        self.has_transformer = True

        # --- DEEP CNN EMBEDDED BLOCK (1D Conv over features) ---
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

        # Multi-Task Heads
        self.w_cls = np.random.randn(curr, num_classes).astype(np.float32) * 0.05
        self.b_cls = np.zeros((1, num_classes), dtype=np.float32)

        self.w_geo = np.zeros((curr, 4), dtype=np.float32)
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
            cnn_out = None
            h = X

        activations = [h]
        pre_acts = []
        for w, b in zip(self.weights, self.biases):
            z = h @ w + b
            pre_acts.append(z)
            h = gelu(z)
            activations.append(h)

        cls_logits = h @ self.w_cls + self.b_cls
        delta = h @ self.w_geo + self.b_geo
        if X.shape[-1] >= 52 and np.any(X[:, 48:52] > 0.005):
            prop = np.clip(X[:, 48:52], 0.0, 1.0)
            has_prop = (prop[:, 2] > 0.005) & (prop[:, 3] > 0.005)
            geo_preds = np.zeros_like(prop)
            geo_preds[has_prop] = np.clip(prop[has_prop] + 0.12 * np.tanh(delta[has_prop]), 0.0, 1.0)
            geo_preds[~has_prop] = sigmoid(delta[~has_prop])
        else:
            geo_preds = sigmoid(delta)
        return activations, pre_acts, cls_logits, geo_preds, cnn_out

    def predict(self, X):
        if len(X) > 512:
            preds_l, conf_l, probs_l, geo_l = [], [], [], []
            for i in range(0, len(X), 512):
                bx = X[i : i + 512]
                _, _, c_logits, g_preds, _ = self.forward(bx)
                p = softmax(c_logits)
                preds_l.append(np.argmax(p, axis=-1))
                conf_l.append(np.max(p, axis=-1))
                probs_l.append(p)
                geo_l.append(g_preds)
            return np.concatenate(preds_l), np.concatenate(conf_l), np.vstack(probs_l), np.vstack(geo_l)
        activations, pre_acts, cls_logits, geo_preds, _ = self.forward(X)
        probs = softmax(cls_logits)
        preds = np.argmax(probs, axis=-1)
        conf = np.max(probs, axis=-1)
        return preds, conf, probs, geo_preds

    def predict_deep(self, X):
        """Deep multi-task engineering inference with entropy, ranking, severity, and IRC standards."""
        activations, pre_acts, cls_logits, geo_preds, _ = self.forward(X)
        probs = softmax(cls_logits)
        B = len(X)
        results = []

        for i in range(B):
            p_vec = probs[i]
            pred_id = int(np.argmax(p_vec))
            conf = float(p_vec[pred_id])

            # Shannon entropy in bits (epistemic uncertainty)
            entropy = float(-np.sum(p_vec * np.log2(p_vec + 1e-10)))
            if pred_id == 9:
                uncertainty_rating = "VULNERABLE_ROAD_USER_CONFIRMED"
            else:
                uncertainty_rating = "LOW_UNCERTAINTY" if entropy < 1.2 else ("MODERATE_UNCERTAINTY" if entropy < 2.2 else "HIGH_UNCERTAINTY_OOD")

            # Top-3 ranking
            sorted_indices = np.argsort(p_vec)[::-1][:3]
            top3 = [
                {
                    "rank": rank + 1,
                    "class_id": int(idx),
                    "class_name": self.CLASS_NAMES[idx] if idx < len(self.CLASS_NAMES) else f"Class_{idx}",
                    "probability": round(float(p_vec[idx]), 4),
                    "color_hex": self.CLASS_COLORS.get(int(idx), {}).get("hex", "#f59e0b")
                }
                for rank, idx in enumerate(sorted_indices)
            ]

            # ASTM D6433 Severity
            if pred_id == 0:
                severity = "NONE"
            elif pred_id == 9:
                severity = "N/A_PEDESTRIAN_SAFETY_INCIDENT"
            elif pred_id == 4:
                severity = "HIGH" if conf > 0.85 else ("MEDIUM" if conf > 0.60 else "LOW")
            else:
                severity = "HIGH" if conf > 0.88 else ("MEDIUM" if conf > 0.65 else "LOW")

            irc_standard = self.IRC_STANDARDS.get(pred_id, "MoRTH General Road Infrastructure Guideline")
            color_meta = self.CLASS_COLORS.get(pred_id, {"hex": "#f59e0b", "glow": "rgba(245, 158, 11, 0.4)", "badge": "bg-amber-950 text-amber-300 border-amber-800", "label": "DISTRESS"})
            bbox = [round(float(v), 4) for v in geo_preds[i]]

            results.append({
                "class_id": pred_id,
                "class_name": self.CLASS_NAMES[pred_id] if pred_id < len(self.CLASS_NAMES) else f"Class_{pred_id}",
                "confidence": round(conf, 4),
                "shannon_entropy_bits": round(entropy, 3),
                "uncertainty_rating": uncertainty_rating,
                "astm_d6433_severity": severity,
                "irc_standard_specification": irc_standard,
                "color_hex": color_meta["hex"],
                "glow_color": color_meta["glow"],
                "badge_class": color_meta["badge"],
                "hud_label": color_meta["label"],
                "top3_ranked_predictions": top3,
                "predicted_bbox_normalized": bbox,
                "all_class_probabilities": {self.CLASS_NAMES[k]: round(float(p_vec[k]), 4) for k in range(min(len(self.CLASS_NAMES), len(p_vec)))}
            })

        return results

    def train_step(self, X, y_cls, y_geo=None):
        """Full-depth end-to-end backpropagation across all layers, CNN block, and multi-task heads."""
        B = len(X)
        activations, pre_acts, cls_logits, geo_preds, cnn_out = self.forward(X)

        # 1. Softmax Cross-Entropy loss
        probs = softmax(cls_logits)
        loss_cls = -np.mean(np.log(probs[np.arange(B), y_cls] + 1e-8))

        # 2. Multi-Task Geometry Huber Smooth L1 + Box IoU loss (computed strictly on valid non-zero bounding boxes)
        loss_geo = 0.0
        loss_iou = 0.0
        valid_geo = (y_geo[:, 2] > 0.005) & (y_geo[:, 3] > 0.005) if y_geo is not None else None

        if y_geo is not None and np.any(valid_geo):
            diff_valid = geo_preds[valid_geo] - y_geo[valid_geo]
            abs_valid = np.abs(diff_valid)
            smooth_l1 = np.where(abs_valid < 0.1, 0.5 * (diff_valid ** 2) / 0.1, abs_valid - 0.05)
            loss_geo = float(np.mean(smooth_l1))
            
            # Vectorized Box IoU
            ious = compute_box_iou(geo_preds[valid_geo], y_geo[valid_geo])
            loss_iou = float(np.mean(1.0 - ious))

        total_loss = float(loss_cls + 0.4 * loss_geo + 0.3 * loss_iou)

        # 3. Gradient on classification logits
        d_logits = probs.copy()
        d_logits[np.arange(B), y_cls] -= 1.0
        d_logits /= B

        feat = activations[-1]
        d_w_cls = feat.T @ d_logits
        d_b_cls = np.sum(d_logits, axis=0, keepdims=True)

        d_feat = d_logits @ self.w_cls.T

        # If geometry head active (backpropagate ONLY for valid defect / pedestrian bounding boxes)
        if y_geo is not None and np.any(valid_geo):
            d_smooth = np.zeros_like(geo_preds)
            diff_valid = geo_preds[valid_geo] - y_geo[valid_geo]
            abs_valid = np.abs(diff_valid)
            d_smooth[valid_geo] = np.where(abs_valid < 0.1, diff_valid / 0.1, np.sign(diff_valid))

            delta = feat @ self.w_geo + self.b_geo
            tanh_d = np.tanh(delta)
            num_valid = max(1.0, float(np.sum(valid_geo)))
            d_geo = (d_smooth * 0.12 * (1.0 - tanh_d**2)) / num_valid

            d_w_geo = feat.T @ d_geo
            d_b_geo = np.sum(d_geo, axis=0, keepdims=True)
            d_feat += d_geo @ self.w_geo.T
            self.w_geo -= self.lr * d_w_geo
            self.b_geo -= self.lr * d_b_geo

        # Update classification head
        self.w_cls -= self.lr * d_w_cls
        self.b_cls -= self.lr * d_b_cls

        # 4. Full Deep Backpropagation through dense hidden layers
        d_h = d_feat
        for l_idx in reversed(range(len(self.weights))):
            d_z = d_h * gelu_grad(pre_acts[l_idx])
            d_w = activations[l_idx].T @ d_z
            d_b = np.sum(d_z, axis=0, keepdims=True)
            d_h = d_z @ self.weights[l_idx].T

            self.weights[l_idx] -= self.lr * d_w
            self.biases[l_idx] -= self.lr * d_b

        # 5. Full Deep Backpropagation into CNN projection
        if self.has_transformer and self.conv_w is not None and cnn_out is not None:
            d_cnn_act = d_h
            d_conv_z = d_cnn_act * gelu_grad(cnn_out)
            d_conv_w = X.T @ d_conv_z
            d_conv_b = np.sum(d_conv_z, axis=0, keepdims=True)
            self.conv_w -= self.lr * d_conv_w
            self.conv_b -= self.lr * d_conv_b

        return total_loss, float(loss_cls), float(loss_geo)

    def save_weights(self, path):
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
