"""
INT8 quantization of the served ONNX models, with the accuracy cost measured.

    python -m scripts.quantize_models                  # classifier backbone + trained detectors
    python -m scripts.quantize_models --only classifier

Static, per-channel QDQ quantization (onnxruntime.quantization) calibrated on
TRAINING images only, then scored on the same held-out test images as the
FP32 model:

* classifier backbone: through the real serving path
  (load_best_vision_model(...).predict_image), once with the FP32 session and
  once with the INT8 session swapped in, on the grouped held-out split that
  produced the published accuracy. Reports accuracy, macro-F1, how often the
  two agree, and backbone latency.
* YOLO detectors: box mAP on each detector's test split, FP32 ONNX vs INT8
  ONNX, via Ultralytics' ONNX backend.

INT8 files go to checkpoints/int8/ and are NOT served automatically: whether
the speed is worth the accuracy change is a decision for the numbers in
checkpoints/quantization_report.json.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Windows: load scikit-learn before PyTorch. Its wheel ships a current VC++ runtime
# (msvcp140.dll); with an older system runtime, torch's c10.dll fails to load
# (WinError 1114) unless a newer copy is already in the process.
import sklearn  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
INT8_DIR = CKPT / "int8"
CALIBRATION = {"method": "percentile"}  # set from --calibration; best of the three on the classifier


class _ArrayReader:
    """CalibrationDataReader over a list of preprocessed input arrays."""

    def __init__(self, input_name, arrays):
        self._items = iter([{input_name: a} for a in arrays])

    def get_next(self):
        return next(self._items, None)


def yolo_head_nodes(model_path):
    """
    Names of the nodes in a YOLO export's final block (the detection head).

    The head concatenates box coordinates (0..input size, in pixels) with
    class probabilities (0..1) into one output. Quantized with one shared
    INT8 scale, the probabilities collapse: measured on the crossing
    detector, mAP50 fell from 0.864 to 0.100. Keeping the head in FP32 and
    quantizing the backbone and neck avoids that.
    """
    import re

    import onnx
    graph = onnx.load(str(model_path)).graph
    blocks = [int(m.group(1)) for n in graph.node if (m := re.match(r"^/model\.(\d+)/", n.name))]
    if not blocks:
        return []
    head = f"/model.{max(blocks)}/"
    return [n.name for n in graph.node if n.name.startswith(head)]


def quantize(fp32_path, int8_path, input_name, calibration_arrays, nodes_to_exclude=()):
    import onnx
    from onnx import version_converter
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    source = fp32_path
    model = onnx.load(str(fp32_path))
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    if opset < 13:  # per-channel DequantizeLinear (the `axis` attribute) needs opset 13
        source = int8_path.with_suffix(".opset13.onnx")
        onnx.save(version_converter.convert_version(model, 13), str(source))
    prepared = int8_path.with_suffix(".prep.onnx")
    quant_pre_process(str(source), str(prepared))
    quantize_static(str(prepared), str(int8_path), _ArrayReader(input_name, calibration_arrays),
                    quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, nodes_to_exclude=list(nodes_to_exclude),
                    calibrate_method={"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
                                      "entropy": CalibrationMethod.Entropy}[CALIBRATION["method"]])
    prepared.unlink(missing_ok=True)
    if source != fp32_path:
        source.unlink(missing_ok=True)
    return int8_path


def _session(path):
    import onnxruntime as ort

    from models.cnn_embedder import _session_options
    return ort.InferenceSession(str(path), _session_options(ort), providers=["CPUExecutionProvider"])


def _latency_ms(session, feed, runs=30):
    session.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 2)


def classifier(seed=42, calibration_images=128):
    from sklearn.metrics import accuracy_score, f1_score

    from data.image_dataset import load_image
    from models.deep_vision_net import load_best_vision_model
    from training.train_deep_vision import collect_files, grouped_split

    model, backend = load_best_vision_model(str(CKPT), verbose=False)
    embedder = getattr(model, "embedder", None)
    if embedder is None or not embedder.is_ready:
        return {"skipped": f"serving model {backend!r} has no ONNX CNN backbone"}

    # Same capped, grouped split as training/train_cnn_head.py.
    rng = np.random.default_rng(seed)
    by_class = {}
    for item in collect_files():
        by_class.setdefault(item[1], []).append(item)
    capped = []
    for _cls, items in sorted(by_class.items()):
        if len(items) > 1500:
            items = [items[i] for i in sorted(rng.choice(len(items), 1500, replace=False))]
        capped.extend(items)
    train_items, _val, test_items = grouped_split(capped, seed=seed)

    fp32_path = CKPT / f"cnn_backbone_{embedder.name}.onnx"
    calib_pick = np.random.default_rng(0).choice(len(train_items), min(calibration_images, len(train_items)),
                                                 replace=False)
    calibration = [embedder.preprocess(load_image(train_items[i][0]))[None] for i in calib_pick]
    int8_path = quantize(fp32_path, INT8_DIR / f"cnn_backbone_{embedder.name}.int8.onnx",
                         embedder._input_name, calibration)

    images = [load_image(path) for path, _cls, _grp in test_items]
    truth = [cls for _p, cls, _g in test_items]
    # predict_proba columns follow the head's own class order.
    classes = np.asarray(getattr(model.head, "classes_", np.arange(len(model.predict_probabilities(images[0])))))

    def predict_all():
        return [int(classes[int(np.argmax(model.predict_probabilities(img)))]) for img in images]

    fp32_session = embedder._session
    fp32_pred = predict_all()
    embedder._session = _session(int8_path)
    int8_pred = predict_all()
    int8_session, embedder._session = embedder._session, fp32_session

    feed = {embedder._input_name: calibration[0]}
    return {
        "backbone": embedder.name,
        "held_out_test_images": len(images),
        "fp32": {"accuracy": round(accuracy_score(truth, fp32_pred), 4),
                 "macro_f1": round(f1_score(truth, fp32_pred, average="macro"), 4),
                 "backbone_ms": _latency_ms(fp32_session, feed), "mb": round(fp32_path.stat().st_size / 2**20, 1)},
        "int8": {"accuracy": round(accuracy_score(truth, int8_pred), 4),
                 "macro_f1": round(f1_score(truth, int8_pred, average="macro"), 4),
                 "backbone_ms": _latency_ms(int8_session, feed), "mb": round(int8_path.stat().st_size / 2**20, 1)},
        "prediction_agreement": round(float(np.mean(np.array(fp32_pred) == np.array(int8_pred))), 4),
        "int8_file": str(int8_path.relative_to(ROOT)),
        "calibration_images": len(calibration),
    }


def detector(name, calibration_images=128):
    from PIL import Image
    from ultralytics import YOLO

    from models.onnx_object_detector import letterbox
    from training.train_detector import DetectorConfig, split_images

    config = DetectorConfig.load(ROOT / "configs" / "detectors" / f"{name}.yaml")
    fp32_path = CKPT / "detectors" / f"{name}.onnx"
    if not fp32_path.exists() or not Path(config.data).exists():
        return {"skipped": "model or dataset not present"}
    card = json.loads((CKPT / "detectors" / f"{name}.json").read_text(encoding="utf-8"))
    size = card["input_size"]

    pairs = split_images(Path(config.data), "train")
    pick = np.random.default_rng(0).choice(len(pairs), min(calibration_images, len(pairs)), replace=False)
    calibration = []
    for i in pick:
        canvas, *_ = letterbox(np.asarray(Image.open(pairs[i][0]).convert("RGB")), size)
        calibration.append((np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None])
    input_name = _session(fp32_path).get_inputs()[0].name
    head = yolo_head_nodes(fp32_path)
    int8_path = quantize(fp32_path, INT8_DIR / f"{name}.int8.onnx", input_name, calibration,
                         nodes_to_exclude=head)

    result = {}
    for label, path in (("fp32", fp32_path), ("int8", int8_path)):
        metrics = YOLO(str(path), task="detect").val(data=config.data, split="test", imgsz=size, batch=1,
                                                     device="cpu", plots=False, verbose=False)
        result[label] = {"mAP50": round(float(metrics.box.map50), 4),
                         "mAP50_95": round(float(metrics.box.map), 4),
                         "ms_per_image": _latency_ms(_session(path), {input_name: calibration[0]}),
                         "mb": round(path.stat().st_size / 2**20, 1)}
    result["int8_file"] = str(int8_path.relative_to(ROOT))
    result["calibration_images"] = len(calibration)
    result["fp32_nodes_kept"] = f"{len(head)} detection-head nodes left in FP32"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=["classifier", "road_damage", "crosswalk", "license_plate"])
    parser.add_argument("--calibration", choices=["minmax", "percentile", "entropy"], default="percentile")
    parser.add_argument("--calibration-images", type=int, default=128,
                        help="images used to calibrate; lower it if calibration runs out of memory")
    args = parser.parse_args(argv)
    CALIBRATION["method"] = args.calibration

    targets = ["classifier", "road_damage", "crosswalk", "license_plate"]
    if args.only:
        targets = [args.only]
    report_path = CKPT / "quantization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["method"] = (f"onnxruntime static QDQ, per-channel INT8 weights, UINT8 activations, "
                        f"{args.calibration} calibration on training images; scored on held-out test splits")
    for target in targets:
        print(f"== {target}", flush=True)
        report[target] = (classifier(calibration_images=args.calibration_images) if target == "classifier"
                          else detector(target, calibration_images=args.calibration_images))
        print(json.dumps(report[target], indent=2), flush=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
