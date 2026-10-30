"""
Export the served pixel segmenter (checkpoints/defect_segmenter.joblib) to ONNX so the
Vercel build can run it with onnxruntime alone, and measure the port.

    python -m scripts.export_pixel_segmenter_onnx            # write the .onnx + .json
    python -m scripts.export_pixel_segmenter_onnx --verify   # compare with the original

The ONNX graph is built directly from the fitted trees (skl2onnx's converter does not
handle every scikit-learn version), so it needs only the `onnx` package.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, "checkpoints")
SRC = os.path.join(CKPT, "defect_segmenter.joblib")


def export():
    import joblib
    import onnx
    from onnx import helper as H, TensorProto
    from models.pixel_segmenter_lite import ONNX_PATH, META_PATH

    m = joblib.load(SRC)
    clf = m["classifier"]
    K = len(clf.classes_)
    base = np.asarray(clf._baseline_prediction).reshape(-1)
    A = {k: [] for k in ("tid", "nid", "fid", "val", "mode", "t", "f", "miss")}
    C = {k: [] for k in ("tid", "nid", "cid", "w")}
    tid = 0
    for it in clf._predictors:
        for k, pred in enumerate(it):
            nodes = pred.nodes
            if nodes["is_categorical"].any():
                raise SystemExit("categorical splits are not supported by this exporter")
            for i, n in enumerate(nodes):
                A["tid"].append(tid); A["nid"].append(i)
                if n["is_leaf"]:
                    A["fid"].append(0); A["val"].append(0.0); A["mode"].append("LEAF")
                    A["t"].append(0); A["f"].append(0); A["miss"].append(0)
                    C["tid"].append(tid); C["nid"].append(i); C["cid"].append(k); C["w"].append(float(n["value"]))
                else:
                    A["fid"].append(int(n["feature_idx"])); A["val"].append(float(n["num_threshold"]))
                    A["mode"].append("BRANCH_LEQ"); A["t"].append(int(n["left"])); A["f"].append(int(n["right"]))
                    A["miss"].append(int(bool(n["missing_go_to_left"])))
            tid += 1
    node = H.make_node(
        "TreeEnsembleClassifier", ["X"], ["label", "probabilities"], domain="ai.onnx.ml",
        nodes_treeids=A["tid"], nodes_nodeids=A["nid"], nodes_featureids=A["fid"], nodes_values=A["val"],
        nodes_modes=A["mode"], nodes_truenodeids=A["t"], nodes_falsenodeids=A["f"],
        nodes_missing_value_tracks_true=A["miss"], class_treeids=C["tid"], class_nodeids=C["nid"],
        class_ids=C["cid"], class_weights=C["w"], classlabels_int64s=[int(c) for c in clf.classes_],
        base_values=[float(b) for b in base], post_transform="SOFTMAX")
    g = H.make_graph([node], "road_shield_pixel_segmenter",
                     [H.make_tensor_value_info("X", TensorProto.FLOAT, [None, 11])],
                     [H.make_tensor_value_info("label", TensorProto.INT64, [None]),
                      H.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, K])])
    model = H.make_model(g, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("ai.onnx.ml", 3)],
                         producer_name="road-shield")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, ONNX_PATH)

    import onnxruntime as ort
    rng = np.random.RandomState(1)
    X = (rng.rand(20000, 11) * [255, 255, 255, 200, 200, 60, 60, 80, 1, 1, 100]
         - [0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 50]).astype(np.float32)
    p_onnx = ort.InferenceSession(ONNX_PATH).run(None, {"X": X})[1]
    diff = float(np.abs(p_onnx - clf.predict_proba(X)).max())
    meta = {
        "source": "checkpoints/defect_segmenter.joblib",
        "source_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
        "thresholds": m["thresholds"],
        "trees": tid,
        "max_probability_difference_vs_sklearn": diff,
        "note": "same trees, same thresholds; features ported to numpy in models/pixel_segmenter_lite.py",
    }
    json.dump(meta, open(META_PATH, "w", encoding="utf-8"), indent=2)
    print(f"wrote {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1e6:.2f} MB, {tid} trees); "
          f"max probability difference vs scikit-learn {diff:.1e}")


def verify(n=40):
    import cv2
    from models.defect_segmenter import DefectSegmenter, extract_pixel_features as cv_feats, mask_from_proba
    from models import pixel_segmenter_lite as lite

    seg = DefectSegmenter(SRC)
    files = []
    for folder in ("02_kaggle_pothole_600", "03_crack500_fatigue", "05_morth_civil_hard_negatives"):
        files += sorted(glob.glob(os.path.join(ROOT, "datasets", folder, "**", "*.jpg"), recursive=True))[: n // 3 + 1]
    agree, feat_err, blob_same = [], [], 0
    for p in files:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        f_cv, shape = cv_feats(img)
        f_np, _ = lite.extract_pixel_features(img)
        feat_err.append(float(np.abs(f_cv - f_np).max()))
        m_cv = mask_from_proba(seg.clf.predict_proba(f_cv), shape, img.shape[:2], seg.thresholds)
        m_np = mask_from_proba(lite.predict_proba(img), shape, img.shape[:2], seg.thresholds)
        a, b = m_cv["mask_work"] if "mask_work" in m_cv else m_cv["mask"], m_np["mask_work"] if "mask_work" in m_np else m_np["mask"]
        agree.append(float((a == b).mean()))
        blob_same += int(bool((a > 0).any()) == bool((b > 0).any()))
    print(f"{len(files)} photographs: pixel labels agree {100 * np.mean(agree):.2f}% (worst {100 * min(agree):.2f}%); "
          f"same defect / no-defect verdict on {blob_same}/{len(files)}; max feature difference {max(feat_err):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else export()
