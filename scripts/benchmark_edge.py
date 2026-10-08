"""
How fast does each model run on THIS machine? Run it on the Raspberry Pi 5 / Jetson the bus would carry.

    python -m scripts.benchmark_edge                    # every .onnx in checkpoints/ and checkpoints/int8/
    python -m scripts.benchmark_edge --threads 4 --runs 50
    python -m scripts.benchmark_edge --quantize         # first build an INT8 copy of the served classifier
    python -m scripts.benchmark_edge --pipeline 20      # also time the full photograph pipeline on 20 photos

Writes checkpoints/edge_benchmark_<machine>.json: the device (Pi model string from /proc/device-tree when
there is one, else CPU name), ONNX Runtime version, thread count, and per model the file size and latency
(median, 95th percentile, mean over --runs after --warmup untimed runs) on a batch of one.

Inputs are random tensors of the model's own input shape (free dimensions set to the size the project
uses), so the numbers are speed only. --quantize is different: it builds
checkpoints/int8/deep_vision_<arch>.int8.onnx with static INT8 quantization (QDQ, per-channel weights),
calibrated on --calib road photographs from the training folders, and then checks it on --check different
photographs: how often its class agrees with the FP32 model, and the largest probability difference. An
INT8 model that is faster but disagrees is not a drop-in replacement, and the report says so.
Quantization needs the `onnx` package (pip install onnx); timing does not.
"""
import argparse
import glob
import json
import os
import platform
import random
import socket
import sys
import time

import numpy as np

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
FREE_DIM = {2: 224, 3: 224}
PHOTO_DIRS = ("datasets/01_rdd2022_india", "datasets/02_kaggle_pothole_600", "datasets/03_crack500_fatigue")


def device_info():
    model = None
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(p, "rb") as fh:
                model = fh.read().decode("utf-8", "replace").strip("\x00").strip()
                break
        except OSError:
            pass
    cpu = platform.processor() or None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "hardware")):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as fh:
            mem_gb = round(int(fh.readline().split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    import onnxruntime as ort
    return {"board": model, "cpu": cpu, "machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "memory_gb": mem_gb, "os": platform.platform(), "python": platform.python_version(),
            "onnxruntime": ort.__version__, "host": socket.gethostname()}


def concrete_shape(shape, model_name):
    out = []
    for i, d in enumerate(shape):
        if isinstance(d, int) and d > 0:
            out.append(d)
        elif i == 0:
            out.append(1)
        else:
            out.append(100 if "imu" in model_name and i == 2 else FREE_DIM.get(i, 3))
    return out


def random_input(inp, model_name, rng):
    shape = concrete_shape(inp.shape, model_name)
    if "uint8" in inp.type:
        return rng.integers(0, 255, size=shape, dtype=np.uint8)
    if "int64" in inp.type:
        return rng.integers(0, 10, size=shape).astype(np.int64)
    return rng.standard_normal(shape).astype(np.float32)


def time_model(path, threads, runs, warmup):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    t0 = time.perf_counter()
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000
    rng = np.random.default_rng(0)
    name = os.path.basename(path)
    feeds = {i.name: random_input(i, name, rng) for i in sess.get_inputs()}
    for _ in range(warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    return {"file": os.path.relpath(path, ENGINE_ROOT).replace("\\", "/"),
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "input_shapes": {k: list(v.shape) for k, v in feeds.items()},
            "load_ms": round(load_ms, 1), "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "mean_ms": round(float(a.mean()), 2),
            "runs": runs, "precision": "int8" if ".int8." in name or os.sep + "int8" + os.sep in path else "fp32"}


def photographs(n, seed, exclude=()):
    files = [f for d in PHOTO_DIRS for f in glob.glob(os.path.join(ENGINE_ROOT, d, "**", "*.jpg"), recursive=True)]
    files = sorted(set(files) - set(exclude))
    random.Random(seed).shuffle(files)
    return files[:n]


def quantize_classifier(n_calib, n_check):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from PIL import Image
    from models.deep_vision_net import DeepVisionNet
    net = DeepVisionNet()
    if not net.is_ready or not (net.weights_path or "").endswith(".onnx"):
        raise SystemExit("no served ONNX classifier to quantize")
    src = net.weights_path
    os.makedirs(os.path.join(CKPT, "int8"), exist_ok=True)
    dst = os.path.join(CKPT, "int8", os.path.basename(src).replace(".onnx", ".int8.onnx"))
    input_name = net._session.get_inputs()[0].name

    def tensor(path):
        with Image.open(path) as im:
            x = net._preprocess(np.asarray(im.convert("RGB")))
        return x[:1] if x.ndim == 4 else x[None]

    calib = photographs(n_calib, seed=1)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            p = next(self.it, None)
            return None if p is None else {input_name: tensor(p).astype(np.float32)}

    quantize_static(src, dst, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    import onnxruntime as ort
    s32 = net._session
    s8 = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    agree, max_diff, n = 0, 0.0, 0
    for p in photographs(n_check, seed=2, exclude=calib):
        x = tensor(p).astype(np.float32)
        a = s32.run(None, {input_name: x})[0][0]
        b = s8.run(None, {input_name: x})[0][0]
        pa, pb = np.exp(a - a.max()), np.exp(b - b.max())
        pa, pb = pa / pa.sum(), pb / pb.sum()
        agree += int(pa.argmax() == pb.argmax())
        max_diff = max(max_diff, float(np.abs(pa - pb).max()))
        n += 1
    return {"source": os.path.relpath(src, ENGINE_ROOT).replace("\\", "/"),
            "int8": os.path.relpath(dst, ENGINE_ROOT).replace("\\", "/"),
            "method": "static INT8, QDQ, per-channel weights, MinMax calibration",
            "calibration_photographs": len(calib), "check_photographs": n,
            "top1_agreement_with_fp32": round(agree / n, 4) if n else None,
            "max_probability_difference": round(max_diff, 4),
            "note": "agreement with the FP32 model on photographs not used for calibration; not an accuracy"}


def time_pipeline(n):
    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    p = DeepInferencePipeline()
    files = photographs(n, seed=3)
    times = []
    for f in files:
        t = time.perf_counter()
        p.audit_image(f)
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times[1:] or times)
    return {"photographs": len(files), "median_ms": round(float(np.median(a)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1), "note": "first photograph excluded (warm-up)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = its default)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--check", type=int, default=200)
    ap.add_argument("--pipeline", type=int, default=0, help="also time the full pipeline on N photographs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    report = {"device": device_info(), "threads": a.threads or "onnxruntime default",
              "measured_unix": int(time.time()), "models": []}
    if a.quantize:
        report["quantization"] = quantize_classifier(a.calib, a.check)
        print("[bench] quantized:", report["quantization"])
    paths = sorted(glob.glob(os.path.join(CKPT, "*.onnx")) + glob.glob(os.path.join(CKPT, "int8", "*.onnx")))
    for p in paths:
        try:
            r = time_model(p, a.threads, a.runs, a.warmup)
        except Exception as e:
            r = {"file": os.path.relpath(p, ENGINE_ROOT).replace("\\", "/"), "error": str(e)[:200]}
        report["models"].append(r)
        print(f"  {r['file']:<60} " + (f"{r['median_ms']:>8.1f} ms median  {r['p95_ms']:>8.1f} p95  "
                                         f"{r['size_mb']:>6.1f} MB" if "median_ms" in r else r["error"]))
    if a.pipeline:
        report["full_pipeline"] = time_pipeline(a.pipeline)
        print("[bench] full pipeline:", report["full_pipeline"])
    tag = (report["device"]["board"] or report["device"]["machine"] or "machine").lower()
    tag = "".join(c if c.isalnum() else "_" for c in tag)[:40].strip("_")
    out = a.out or os.path.join(CKPT, f"edge_benchmark_{tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"[bench] written {os.path.relpath(out, ENGINE_ROOT)}")
    return report


if __name__ == "__main__":
    main()
