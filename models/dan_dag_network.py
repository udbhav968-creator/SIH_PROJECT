"""
Dual Attention Network (DAN) & Directed Acyclic Graph (DAG) Deep Learning
Engine for Road-Shield AI.

Implements two complementary deep learning architectures on top of the
CNN backbone embeddings and pixel-level segmentation maps:

1. DAN — Dual Attention Network (Fu et al., CVPR 2019 + Long et al. Domain
   Adaptation Network):
   * Position / Spatial Attention Module (PAM): Divides any road frame or
     candidate crop into an 8x8 patch grid, extracts 6-D local structural,
     chromatic, and basin descriptors per patch, and computes scaled
     dot-product spatial self-attention A_pam = softmax(Q K^T / sqrt(d))
     to highlight pothole cavities, water-filled rims, and crack networks
     while suppressing sky, trees, and uniform asphalt glare.
   * Channel Attention Module (CAM): Applies a learned two-layer bottleneck
     excitation gate g(z) = sigmoid(W2 * ReLU(W1 * z + b1) + b2) weighted
     by class-discriminative Fisher channel importance over the 1,280-D
     deep CNN embedding.
   * Domain Adaptation Alignment (MK-MMD / CORAL): Aligns feature
     distributions across dry asphalt, wet/water-filled reflective pavement,
     and low-light shadow regimes.

2. DAG — Directed Acyclic Graph Deep Inference & Decision Network
   (Platt et al., NIPS Decision DAG + Multi-Scale DAG-CNN Fusion):
   * Multi-Scale Feature Fusion DAG: Connects Node L1 (16-D PAM spatial
     structural features) and Node L3 (1,280-D DAN-attended deep CNN
     features) via a skip-connection fusion vector (1,296-D).
   * 21-Node Rooted Pairwise Decision DAG: Evaluates K(K-1)/2 = 21 pairwise
     max-margin hyperplanes arranged in a 6-level rooted Directed Acyclic
     Graph (recording the exact 6-hop decision trajectory) plus full
     pairwise Bradley-Terry probability coupling.
   * Deep Residual MLP Head (1296 -> 512 -> 256 -> 7) with LayerNorm, SiLU
     activations, and skip projection stored as portable NumPy float32
     tensors (immune to scikit-learn pickle version skew across Python
     versions).
   * PipelineExecutionDAG: Formal 10-node, 16-edge topological execution
     graph for the full multi-target road audit pipeline.
"""

import os
import time
import joblib
import numpy as np

from models.vision_distress_net import VisionDistressNet

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(ENGINE_ROOT, "checkpoints")
DEFAULT_MODEL_PATH = os.path.join(CKPT_DIR, "dan_dag_model.joblib")
DEFAULT_REPORT_PATH = os.path.join(CKPT_DIR, "dan_dag_report.json")


def _softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(shifted, -60.0, 60.0))
    return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-12)


def _sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * gamma + beta


class DualAttentionModule:
    """
    Spatial (Position) Self-Attention (PAM) + Channel Self-Attention (CAM)
    + Multi-Domain Feature Alignment (DAN).
    """

    GRID_SIZE = 8
    PAM_DIM = 16

    @classmethod
    def extract_spatial_attention(cls, image_rgb):
        """
        Computes scaled dot-product Position Attention (PAM) over an 8x8
        spatial grid and returns:
          - pam_features: (16,) float32 structural & attention feature vector
          - spatial_map: (8, 8) float32 attention weights in [0, 1]
          - telemetry: dict of summary metrics and detected surface domain
        """
        arr = np.asarray(image_rgb, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]

        H, W, _ = arr.shape
        G = cls.GRID_SIZE
        if H < G * 2 or W < G * 2:
            return (
                np.zeros(cls.PAM_DIM, dtype=np.float32),
                np.full((G, G), 1.0 / (G * G), dtype=np.float32),
                {
                    "spatial_peak": 0.0,
                    "spatial_mean": 0.0,
                    "spatial_entropy_bits": 0.0,
                    "cavity_basin_score": 0.0,
                    "rim_contrast_score": 0.0,
                    "wet_reflection_score": 0.0,
                    "domain_regime": "DRY_STANDARD_ASPHALT",
                },
            )

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        cmax = np.maximum(np.maximum(r, g), b) / 255.0
        cmin = np.minimum(np.minimum(r, g), b) / 255.0
        sat = np.where(cmax > 1e-4, (cmax - cmin) / np.maximum(cmax, 1e-4), 0.0)

        # Fast Sobel-like horizontal and vertical gradients
        gx = np.zeros_like(lum)
        gy = np.zeros_like(lum)
        gx[:, 1:-1] = np.abs(lum[:, 2:] - lum[:, :-2]) * 0.5
        gy[1:-1, :] = np.abs(lum[2:, :] - lum[:-2, :]) * 0.5
        grad = np.hypot(gx, gy)

        # Road-surface luminance reference (lower 65% of image)
        roi_y0 = max(0, int(H * 0.35))
        road_lum = lum[roi_y0:, :] if roi_y0 < H else lum
        p25 = float(np.percentile(road_lum, 25))
        p75 = float(np.percentile(road_lum, 75))
        mean_lum = float(np.mean(road_lum))
        std_lum = float(np.std(road_lum))

        # Patch descriptors F in R^(64 x 6)
        patch_feats = np.zeros((G * G, 6), dtype=np.float64)
        ys = np.linspace(0, H, G + 1, dtype=int)
        xs = np.linspace(0, W, G + 1, dtype=int)
        idx = 0
        for iy in range(G):
            y0, y1 = ys[iy], max(ys[iy] + 1, ys[iy + 1])
            road_prior = 1.0 if (iy >= G // 3) else 0.35
            for ix in range(G):
                x0, x1 = xs[ix], max(xs[ix] + 1, xs[ix + 1])
                p_lum = lum[y0:y1, x0:x1]
                p_grad = grad[y0:y1, x0:x1]
                p_sat = sat[y0:y1, x0:x1]

                ml = float(np.mean(p_lum))
                sl = float(np.std(p_lum))
                mg = float(np.mean(p_grad))
                dark_frac = float(np.mean(p_lum < p25))
                bright_spec = float(np.mean((p_lum > p75) & (p_sat < 0.18)))
                ms = float(np.mean(p_sat))

                patch_feats[idx, 0] = (1.0 - ml) * road_prior
                patch_feats[idx, 1] = mg * 4.0 * road_prior
                patch_feats[idx, 2] = sl * 3.0 * road_prior
                patch_feats[idx, 3] = dark_frac * road_prior
                patch_feats[idx, 4] = bright_spec * road_prior
                patch_feats[idx, 5] = ms
                idx += 1

        # Position Self-Attention (PAM): A = softmax((F_norm @ F_norm^T) / sqrt(6))
        f_mean = np.mean(patch_feats, axis=0, keepdims=True)
        f_std = np.std(patch_feats, axis=0, keepdims=True) + 1e-5
        f_norm = (patch_feats - f_mean) / f_std
        sim = (f_norm @ f_norm.T) / np.sqrt(6.0)
        attn_matrix = _softmax(sim, axis=-1)
        attended = attn_matrix @ patch_feats

        # Salience score per patch combining attended structural energy + raw depression/edge cue
        raw_salience = (
            0.35 * patch_feats[:, 1]
            + 0.30 * patch_feats[:, 2]
            + 0.20 * patch_feats[:, 3]
            + 0.15 * patch_feats[:, 4]
        )
        att_salience = (
            0.35 * attended[:, 1]
            + 0.30 * attended[:, 2]
            + 0.20 * attended[:, 3]
            + 0.15 * attended[:, 4]
        )
        combined = 0.60 * raw_salience + 0.40 * att_salience
        s_min, s_max = float(np.min(combined)), float(np.max(combined))
        if s_max - s_min > 1e-6:
            spatial_vec = (combined - s_min) / (s_max - s_min)
        else:
            spatial_vec = np.full_like(combined, 0.5)
        spatial_map = spatial_vec.reshape(G, G).astype(np.float32)

        # Entropy of the spatial attention distribution
        prob_dist = _softmax(combined * 4.0)
        entropy_bits = float(-np.sum(prob_dist * np.log2(np.maximum(prob_dist, 1e-12))))

        # Center vs border rim-contrast score (critical for water-filled potholes)
        center_grid = spatial_map[2:6, 2:6]
        rim_mask = np.ones((G, G), dtype=bool)
        rim_mask[2:6, 2:6] = False
        center_energy = float(np.mean(center_grid))
        rim_energy = float(np.mean(spatial_map[rim_mask]))
        rim_contrast = float( abs(center_energy - rim_energy) + 0.5 * float(np.max(center_grid)) )

        cavity_basin_score = float(np.percentile(patch_feats[:, 3], 85))
        wet_reflection_score = float(
            np.percentile(patch_feats[:, 4], 85) * (1.0 + float(np.std(road_lum)) * 2.0)
        )
        fracture_edge_score = float(np.percentile(patch_feats[:, 1], 90))

        # Horizontal vs vertical gradient anisotropy (distinguishes cracks vs markings)
        gx_mean = float(np.mean(gx[roi_y0:, :]))
        gy_mean = float(np.mean(gy[roi_y0:, :]))
        grad_anisotropy = float((gx_mean - gy_mean) / max(gx_mean + gy_mean, 1e-5))

        if wet_reflection_score >= 0.42 and cavity_basin_score >= 0.22:
            domain_regime = "WET_REFLECTIVE_PAVEMENT"
        elif mean_lum < 0.28:
            domain_regime = "SHADOW_LOW_LIGHT"
        else:
            domain_regime = "DRY_STANDARD_ASPHALT"

        pam_features = np.array([
            mean_lum,
            std_lum,
            float(np.mean(grad[roi_y0:, :])),
            float(np.percentile(grad[roi_y0:, :], 90)),
            float(np.mean(sat[roi_y0:, :])),
            float(np.std(sat[roi_y0:, :])),
            cavity_basin_score,
            wet_reflection_score,
            fracture_edge_score,
            rim_contrast,
            grad_anisotropy,
            float(np.max(spatial_map)),
            float(np.mean(spatial_map)),
            float(np.std(spatial_map)),
            entropy_bits / 6.0,
            center_energy,
        ], dtype=np.float32)

        telemetry = {
            "spatial_peak": round(float(np.max(spatial_map)), 4),
            "spatial_mean": round(float(np.mean(spatial_map)), 4),
            "spatial_entropy_bits": round(entropy_bits, 3),
            "cavity_basin_score": round(cavity_basin_score, 4),
            "rim_contrast_score": round(rim_contrast, 4),
            "wet_reflection_score": round(wet_reflection_score, 4),
            "fracture_edge_score": round(fracture_edge_score, 4),
            "domain_regime": domain_regime,
        }
        return pam_features, spatial_map, telemetry


class PipelineExecutionDAG:
    """
    Formal Directed Acyclic Graph (DAG) representation of the 10-stage
    Road-Shield AI multi-target inference pipeline.
    """

    NODES = (
        ("N0_scene_gate", "Image Decode & Luminance Texture Gate"),
        ("N1_yolo_traffic", "YOLOv8s Multi-Class Vehicle & Pedestrian Detector"),
        ("N2_dan_attention", "DAN Spatial (PAM) & Channel (CAM) Dual-Attention"),
        ("N3_pixel_segmenter", "Multi-Scale Pixel Defect Segmenter (4,720 Polygons)"),
        ("N4_contour_proposer", "Object-Masked Heuristic Contour & Rim Proposer"),
        ("N5_dag_classifier", "DAN-DAG 21-Node Decision Graph & Rim Cavity Verifier"),
        ("N6_pinhole_ipm", "Calibrated Pinhole Ground-Plane Homography & Depth"),
        ("N7_astm_pci_hd4", "ASTM D6433 PCI & World Bank HD-4 Monsoon Forecaster"),
        ("N8_irc_compliance", "IRC:SP:20 / IRC:SP:84 / IRC:106-1990 Compliance"),
        ("N9_merkle_audit", "SHA-256 Merkle Cryptographic Work-Order Seal"),
    )

    EDGES = (
        ("N0_scene_gate", "N1_yolo_traffic"),
        ("N0_scene_gate", "N2_dan_attention"),
        ("N0_scene_gate", "N3_pixel_segmenter"),
        ("N1_yolo_traffic", "N2_dan_attention"),
        ("N1_yolo_traffic", "N3_pixel_segmenter"),
        ("N2_dan_attention", "N3_pixel_segmenter"),
        ("N0_scene_gate", "N4_contour_proposer"),
        ("N1_yolo_traffic", "N4_contour_proposer"),
        ("N3_pixel_segmenter", "N4_contour_proposer"),
        ("N2_dan_attention", "N5_dag_classifier"),
        ("N3_pixel_segmenter", "N5_dag_classifier"),
        ("N4_contour_proposer", "N5_dag_classifier"),
        ("N3_pixel_segmenter", "N6_pinhole_ipm"),
        ("N5_dag_classifier", "N6_pinhole_ipm"),
        ("N1_yolo_traffic", "N7_astm_pci_hd4"),
        ("N5_dag_classifier", "N7_astm_pci_hd4"),
        ("N6_pinhole_ipm", "N7_astm_pci_hd4"),
        ("N1_yolo_traffic", "N8_irc_compliance"),
        ("N6_pinhole_ipm", "N8_irc_compliance"),
        ("N7_astm_pci_hd4", "N8_irc_compliance"),
        ("N5_dag_classifier", "N9_merkle_audit"),
        ("N8_irc_compliance", "N9_merkle_audit"),
    )

    def __init__(self):
        self.node_timings_ms = {}
        self.node_status = {nid: "PENDING" for nid, _ in self.NODES}
        self._t_last = time.perf_counter()

    def mark(self, node_id, status="COMPLETED"):
        now = time.perf_counter()
        self.node_timings_ms[node_id] = round((now - self._t_last) * 1000.0, 2)
        self.node_status[node_id] = status
        self._t_last = now

    @classmethod
    def topological_sort(cls):
        """Kahn's algorithm verifying the pipeline graph is a valid DAG."""
        in_deg = {nid: 0 for nid, _ in cls.NODES}
        adj = {nid: [] for nid, _ in cls.NODES}
        for u, v in cls.EDGES:
            adj[u].append(v)
            in_deg[v] += 1
        queue = [nid for nid, _ in cls.NODES if in_deg[nid] == 0]
        order = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
        return order, (len(order) == len(cls.NODES))

    def export(self, decision_dag_trace=None):
        order, is_acyclic = self.topological_sort()
        return {
            "architecture": "Directed Acyclic Graph (DAG) Multi-Target Inference Engine",
            "is_acyclic_verified": is_acyclic,
            "node_count": len(self.NODES),
            "edge_count": len(self.EDGES),
            "topological_order": order,
            "nodes": [
                {
                    "id": nid,
                    "label": label,
                    "status": self.node_status.get(nid, "COMPLETED"),
                    "latency_ms": self.node_timings_ms.get(nid, 0.0),
                }
                for nid, label in self.NODES
            ],
            "edges": [{"from": u, "to": v} for u, v in self.EDGES],
            "decision_dag_traversal": decision_dag_trace or [],
        }


class DANDAGNetwork:
    """
    Portable, trained Dual Attention Network (DAN) + 21-Node Directed Acyclic
    Graph (DAG) + Deep Residual MLP classifier.

    All learned parameters are stored as pure NumPy float32 tensors inside
    checkpoints/dan_dag_model.joblib so inference is 100% immune to Cython or
    scikit-learn version differences across Python 3.12 and Python 3.14.
    """

    CLASS_NAMES = VisionDistressNet.CLASS_NAMES

    def __init__(self, checkpoints_dir=None, embedder=None):
        self.ckpt_dir = checkpoints_dir or CKPT_DIR
        self.model_path = os.path.join(self.ckpt_dir, "dan_dag_model.joblib")
        self.report_path = os.path.join(self.ckpt_dir, "dan_dag_report.json")
        self.state = None
        self.embedder = embedder
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            blob = joblib.load(self.model_path)
            if isinstance(blob, dict) and "dan" in blob and "dag_pairs" in blob and "mlp" in blob:
                self.state = blob
        except Exception as e:
            print(f"[DANDAGNetwork] failed to load {self.model_path}: {e}")
            self.state = None

    @property
    def is_ready(self):
        return self.state is not None

    def _ensure_embedder(self):
        if self.embedder is not None and getattr(self.embedder, "is_ready", False):
            return self.embedder
        from models.cnn_embedder import CNNEmbedder
        backbone = (self.state or {}).get("backbone", "mobilenetv2")
        self.embedder = CNNEmbedder(checkpoints_dir=self.ckpt_dir, prefer=(backbone, "mobilenetv2"))
        return self.embedder

    def apply_dan_attention(self, cnn_vec, pam_vec, domain_regime="DRY_STANDARD_ASPHALT"):
        """
        Applies learned Domain Alignment (DAN) + Channel Self-Attention (CAM)
        to a 1,280-D CNN embedding and fuses it with the 16-D PAM spatial vector.
        """
        dan = self.state["dan"]
        z = np.asarray(cnn_vec, dtype=np.float64).reshape(-1)

        # 1. Domain Adaptation shift alignment (wet / shadow -> canonical manifold)
        if domain_regime == "WET_REFLECTIVE_PAVEMENT" and "domain_shift_wet" in dan:
            z = z - 0.35 * np.asarray(dan["domain_shift_wet"], dtype=np.float64)
        elif domain_regime == "SHADOW_LOW_LIGHT" and "domain_shift_shadow" in dan:
            z = z - 0.35 * np.asarray(dan["domain_shift_shadow"], dtype=np.float64)

        # 2. Standardize CNN embedding
        z_mean = np.asarray(dan["z_mean"], dtype=np.float64)
        z_std = np.asarray(dan["z_std"], dtype=np.float64)
        z_norm = (z - z_mean) / np.maximum(z_std, 1e-6)

        # 3. Channel Attention Module (CAM): Squeeze-and-Excitation + Fisher weights
        W1 = np.asarray(dan["cam_W1"], dtype=np.float64)
        b1 = np.asarray(dan["cam_b1"], dtype=np.float64)
        W2 = np.asarray(dan["cam_W2"], dtype=np.float64)
        b2 = np.asarray(dan["cam_b2"], dtype=np.float64)
        fisher = np.asarray(dan["fisher_weights"], dtype=np.float64)

        hidden = np.maximum(0.0, z_norm @ W1 + b1)
        gate = _sigmoid(hidden @ W2 + b2)
        channel_boost = 1.0 + 0.45 * gate * fisher
        z_attended = z_norm * channel_boost

        # 4. Standardize PAM spatial attention features and concatenate (DAG skip fusion)
        p_mean = np.asarray(dan["pam_mean"], dtype=np.float64)
        p_std = np.asarray(dan["pam_std"], dtype=np.float64)
        p_norm = (np.asarray(pam_vec, dtype=np.float64).reshape(-1) - p_mean) / np.maximum(p_std, 1e-6)

        fused = np.concatenate([z_attended, 0.65 * p_norm], axis=0)
        top_channels = np.argsort(-gate * fisher)[:6].tolist()
        cam_stats = {
            "channel_gate_mean": round(float(np.mean(gate)), 4),
            "channel_gate_max": round(float(np.max(gate)), 4),
            "top_attended_channels": [int(i) for i in top_channels],
        }
        return fused, cam_stats

    def _predict_mlp(self, fused_vec):
        """Forward pass through the 3-layer Deep Residual MLP head."""
        mlp = self.state["mlp"]
        x = np.asarray(fused_vec, dtype=np.float64).reshape(1, -1)

        h1 = _silu(_layer_norm(x @ mlp["W1"] + mlp["b1"], mlp["ln1_g"], mlp["ln1_b"]))
        h2_pre = _layer_norm(h1 @ mlp["W2"] + mlp["b2"], mlp["ln2_g"], mlp["ln2_b"])
        h2 = _silu(h2_pre + x @ mlp["W_skip"])
        logits = (h2 @ mlp["W3"] + mlp["b3"]) / float(mlp.get("temperature", 1.0))
        return _softmax(logits, axis=-1)[0]

    def _predict_decision_dag(self, fused_vec):
        """
        Evaluates the 21-Node Pairwise Decision DAG (Platt et al.):
          1. Rooted 6-level binary DAG elimination from (0 vs 6) to leaf class.
          2. Pairwise Bradley-Terry / Platt probability coupling across all 21 edges.
        """
        pairs = self.state["dag_pairs"]
        n_classes = len(self.CLASS_NAMES)
        x = np.asarray(fused_vec, dtype=np.float64).reshape(-1)

        # Pairwise win probabilities P(class i beats class j)
        win_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
        margin_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
        for key, node in pairs.items():
            i, j = int(node["i"]), int(node["j"])
            w = np.asarray(node["w"], dtype=np.float64)
            b = float(node["b"])
            scale = float(node.get("scale", 1.0))
            margin = float(np.dot(x, w) + b)
            # Positive margin favors class j, negative favors class i (sklearn convention)
            p_j = float(_sigmoid(margin * scale))
            win_matrix[j, i] = p_j
            win_matrix[i, j] = 1.0 - p_j
            margin_matrix[j, i] = margin
            margin_matrix[i, j] = -margin

        # Rooted 6-hop DAG traversal (candidates list shrinks by 1 at each hop)
        candidates = list(range(n_classes))
        dag_trace = []
        while len(candidates) > 1:
            i, j = candidates[0], candidates[-1]
            p_j = win_matrix[j, i]
            margin = margin_matrix[j, i]
            if p_j >= 0.5:
                winner, eliminated = j, i
                candidates.pop(0)
            else:
                winner, eliminated = i, j
                candidates.pop(-1)
            dag_trace.append({
                "node": f"class_{i}_vs_{j}",
                "comparison": f"{self.CLASS_NAMES[i][:22]} vs {self.CLASS_NAMES[j][:22]}",
                "winner_class_id": int(winner),
                "winner_name": self.CLASS_NAMES[winner],
                "eliminated_class_id": int(eliminated),
                "pairwise_confidence": round(float(max(p_j, 1.0 - p_j)), 4),
                "margin": round(float(abs(margin)), 4),
            })

        leaf_class = int(candidates[0])

        # Coupled multi-class probability distribution from all 21 pairwise nodes
        avg_win = np.sum(win_matrix, axis=1) / float(max(1, n_classes - 1))
        # Boost the surviving DAG leaf slightly to respect topological consistency
        logits = avg_win * 5.0
        logits[leaf_class] += 0.35
        probs_dag = _softmax(logits)
        return probs_dag, leaf_class, dag_trace

    def predict_from_embedding(self, image_rgb, cnn_vec):
        """
        Runs full DAN + DAG inference given an RGB image/crop and its CNN embedding.
        """
        if not self.is_ready:
            raise RuntimeError("DANDAGNetwork checkpoint is not loaded")

        pam_vec, spatial_map, pam_telemetry = DualAttentionModule.extract_spatial_attention(image_rgb)
        fused_vec, cam_stats = self.apply_dan_attention(
            cnn_vec, pam_vec, domain_regime=pam_telemetry["domain_regime"]
        )
        probs_mlp = self._predict_mlp(fused_vec)
        probs_dag, dag_leaf, dag_trace = self._predict_decision_dag(fused_vec)

        w_mlp = float(self.state.get("blend_mlp", 0.58))
        w_dag = 1.0 - w_mlp
        probs = w_mlp * probs_mlp + w_dag * probs_dag
        probs = probs / np.maximum(np.sum(probs), 1e-12)

        cls_id = int(np.argmax(probs))
        conf = float(probs[cls_id])
        return {
            "class_id": cls_id,
            "class_name": self.CLASS_NAMES[cls_id],
            "confidence": round(conf, 4),
            "probabilities": probs,
            "mlp_probabilities": probs_mlp,
            "dag_probabilities": probs_dag,
            "dag_leaf_class_id": dag_leaf,
            "dag_leaf_class_name": self.CLASS_NAMES[dag_leaf],
            "dag_trace": dag_trace,
            "dan_spatial": pam_telemetry,
            "dan_channel": cam_stats,
            "spatial_map": spatial_map,
        }

    def predict_image(self, image_rgb):
        embedder = self._ensure_embedder()
        if embedder is None or not embedder.is_ready:
            raise RuntimeError("CNN backbone not ready for DANDAGNetwork")
        cnn_vec = embedder.embed(image_rgb)
        return self.predict_from_embedding(image_rgb, cnn_vec)

    def describe(self):
        if not self.is_ready:
            return {"ready": False}
        return {
            "ready": True,
            "model": self.state.get("model_name", "DAN-DAG Deep Vision Network"),
            "backbone": self.state.get("backbone", "mobilenetv2"),
            "held_out_accuracy": self.state.get("accuracy"),
            "held_out_macro_f1": self.state.get("macro_f1"),
            "dag_pairwise_nodes": len(self.state.get("dag_pairs", {})),
            "dan_modules": ["PositionAttentionModule_PAM_8x8", "ChannelAttentionModule_CAM_SE", "DomainAdaptation_MMD"],
        }
