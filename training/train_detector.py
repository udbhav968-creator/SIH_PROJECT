"""Train, evaluate and export a YOLO detector from a declarative config.

Every model that ships in ``checkpoints/detectors/`` is produced by this
script from a YAML file in ``configs/detectors/``, so a weight file can always
be traced back to the exact data, hyper-parameters and budget that made it.

Usage::

    python -m training.train_detector configs/detectors/road_damage.yaml
    python -m training.train_detector configs/detectors/crosswalk.yaml --hours 0.1  # smoke run
    python -m training.train_detector configs/detectors/crosswalk.yaml --eval-only
    python -m training.train_detector configs/detectors/crosswalk.yaml --resume     # after a crash/reboot

Outputs, per config ``name``:

* ``checkpoints/detectors/<name>.pt``   - best PyTorch weights
* ``checkpoints/detectors/<name>.onnx`` - ONNX export for CPU serving
* ``checkpoints/detectors/<name>.json`` - model card: config, metrics, provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Windows: load scikit-learn before PyTorch. Its wheel ships a current VC++ runtime
# (msvcp140.dll); with an older system runtime, torch's c10.dll fails to load
# (WinError 1114) unless a newer copy is already in the process.
import sklearn  # noqa: F401
import yaml

LOG = logging.getLogger("train_detector")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "detectors"


@dataclass
class DetectorConfig:
    name: str
    data: str
    base_model: str = "yolo11n.pt"
    imgsz: int = 640
    epochs: int = 100
    hours: float | None = None  # wall-clock budget; overrides epochs when set
    batch: int = 16
    patience: int = 20
    workers: int = 4
    cache: bool | str = False
    seed: int = 0
    device: str = "cpu"
    description: str = ""
    dataset_license: str = ""
    # Passed verbatim to Ultralytics (augmentation, optimiser, ...).
    train_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> DetectorConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        config = cls(**raw)
        data_path = Path(config.data)
        if not data_path.is_absolute():
            config.data = str((PROJECT_ROOT / data_path).resolve())
        return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics_to_dict(metrics: Any, names: dict[int, str]) -> dict[str, Any]:
    box = metrics.box
    per_class = {}
    for i, class_index in enumerate(box.ap_class_index):
        per_class[names[int(class_index)]] = {
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "mAP50": round(float(box.ap50[i]), 4),
            "mAP50_95": round(float(box.ap[i]), 4),
        }
    return {
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(v, 2) for k, v in metrics.speed.items()},
    }


def serving_thresholds(f1_curve: np.ndarray, px: np.ndarray, class_indices: list[int],
                       names: dict[int, str], floor: float = 0.05, ceiling: float = 0.9,
                       default: float = 0.25) -> dict[str, float]:
    """Per-class confidence threshold that maximises F1 on the validation split.

    ``f1_curve`` has one row per class that had labels in the split, aligned
    with ``class_indices``; classes absent from the split get ``default``.
    The clamp stops a noisy curve on a rare class from picking a threshold so
    low it floods the output, or so high it never fires.
    """
    thresholds = {name: default for name in names.values()}
    kernel = np.ones(21) / 21  # light smoothing: the raw curve is jagged at the tails
    for row, class_index in enumerate(class_indices):
        smoothed = np.convolve(f1_curve[row], kernel, mode="same")
        best = float(px[int(smoothed.argmax())])
        thresholds[names[int(class_index)]] = round(min(max(best, floor), ceiling), 3)
    return thresholds


def evaluate(weights: Path, config: DetectorConfig, split: str) -> tuple[dict[str, Any], Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=config.data, split=split, imgsz=config.imgsz, batch=config.batch,
                        device=config.device, workers=config.workers, plots=False, verbose=False)
    return _metrics_to_dict(metrics, model.names), metrics


def split_images(data_yaml: Path, split: str) -> list[tuple[Path, Path]]:
    """(image, label) pairs for a split, using the YOLO images/ -> labels/ convention."""
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(spec.get("path") or data_yaml.parent)
    image_dir = root / spec[split]
    label_dir = root / Path(spec[split].replace("images", "labels", 1))
    return [
        (image, (label_dir / image.relative_to(image_dir)).with_suffix(".txt"))
        for image in sorted(image_dir.rglob("*"))
        if image.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def image_level_metrics(weights: Path, config: DetectorConfig, split: str,
                        thresholds: dict[str, float]) -> dict[str, Any]:
    """Presence metrics per image at the serving thresholds.

    Box mAP answers "how well are boxes placed". Operators ask simpler
    questions: does this frame contain a pothole, and how often does a clean
    road raise an alarm. Both are answered here, on images the thresholds
    were not tuned on.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    index_of = {name: index for index, name in names.items()}
    images = split_images(Path(config.data), split)
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in names.values()}
    background_images = background_alarms = 0

    min_conf = min(thresholds.values())
    for batch_start in range(0, len(images), 32):
        batch = images[batch_start:batch_start + 32]
        results = model.predict([str(image) for image, _ in batch], imgsz=config.imgsz,
                                conf=min_conf, device=config.device, verbose=False)
        for (_, label_path), result in zip(batch, results, strict=True):
            truth: set[int] = set()
            if label_path.exists():
                truth = {int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()}
            predicted = {
                int(cls) for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True)
                if conf >= thresholds[names[int(cls)]]
            }
            if not truth:
                background_images += 1
                background_alarms += bool(predicted)
            for name, stats in counts.items():
                index = index_of[name]
                stats["tp"] += index in truth and index in predicted
                stats["fp"] += index not in truth and index in predicted
                stats["fn"] += index in truth and index not in predicted

    per_class = {}
    for name, stats in counts.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        per_class[name] = {
            **stats,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
        }
    return {
        "images": len(images),
        "background_images": background_images,
        "background_false_alarm_rate": (round(background_alarms / background_images, 4)
                                        if background_images else None),
        "per_class": per_class,
    }


def train(config: DetectorConfig, run_dir: Path, resume: bool = False) -> Path:
    from ultralytics import YOLO

    last = run_dir / "weights" / "last.pt"
    if resume:
        # Continues the interrupted run with its saved optimiser state, epoch
        # counter and arguments, so a reboot costs one partial epoch, not the run.
        if not last.exists():
            raise FileNotFoundError(f"--resume given but {last} does not exist")
        LOG.info("resuming %s from %s", config.name, last)
        YOLO(str(last)).train(resume=True)
        return run_dir / "weights" / "best.pt"

    model = YOLO(config.base_model)
    args = dict(
        data=config.data, imgsz=config.imgsz, epochs=config.epochs, batch=config.batch,
        patience=config.patience, workers=config.workers, cache=config.cache, seed=config.seed,
        deterministic=True, device=config.device, project=str(run_dir.parent), name=run_dir.name,
        exist_ok=True, plots=True,
        amp=config.device != "cpu",  # mixed precision only pays off (and is only supported) on GPU
    )
    if config.hours:
        args["time"] = config.hours
    args.update(config.train_args)
    LOG.info("training %s from %s (%s)", config.name, config.base_model,
             f"{config.hours} h budget" if config.hours else f"{config.epochs} epochs")
    model.train(**args)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training finished without {best}")
    return best


def export_onnx(weights: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    exported = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, dynamic=False,
                                         simplify=True, opset=17)
    return Path(exported)


def _hardware(device: str) -> str:
    if device != "cpu":
        try:
            import torch

            return f"{torch.cuda.get_device_name(0)} (cuda:{device})"
        except Exception:  # the card should still be written without a GPU name
            pass
    return f"{platform.processor() or platform.machine()} ({device})"


def write_model_card(config: DetectorConfig, weights: Path, onnx_path: Path | None,
                     metrics: dict[str, Any], thresholds: dict[str, float],
                     train_seconds: float | None, epochs_trained: int | None = None) -> Path:
    from ultralytics import YOLO

    names = YOLO(str(weights)).names
    card = {
        "name": config.name,
        "description": config.description,
        "classes": [names[i] for i in sorted(names)],
        "serving_thresholds": thresholds,
        "architecture": Path(config.base_model).stem,
        "input_size": config.imgsz,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_seconds": round(train_seconds, 1) if train_seconds else None,
        # With a time budget, config.epochs is only the ceiling; this is what ran.
        "epochs_trained": epochs_trained,
        "hardware": _hardware(config.device),
        "dataset": {"yaml": Path(config.data).name, "license": config.dataset_license},
        "config": asdict(config) | {"data": Path(config.data).name},
        "metrics": metrics,
        "artifacts": {
            "pt": {"file": weights.name, "sha256": _sha256(weights)},
            **({"onnx": {"file": onnx_path.name, "sha256": _sha256(onnx_path)}} if onnx_path else {}),
        },
    }
    path = weights.with_suffix(".json")
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path


def apply_overrides(config: DetectorConfig, args: argparse.Namespace) -> DetectorConfig:
    """Apply command-line overrides; a fixed --epochs replaces any time budget."""
    if getattr(args, "hours", None) is not None:
        config.hours = args.hours
    if getattr(args, "epochs", None) is not None:
        config.epochs, config.hours = args.epochs, None
    if getattr(args, "imgsz", None) is not None:
        if args.imgsz % 32:
            raise ValueError("--imgsz must be a multiple of 32")
        config.imgsz = args.imgsz
    for key in ("device", "batch"):
        if getattr(args, key, None) is not None:
            setattr(config, key, getattr(args, key))
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--hours", type=float, help="override the config's time budget")
    # Overrides for running the same config on other hardware, e.g. a Colab GPU:
    #   --device 0 --epochs 100 --imgsz 640 --batch 32
    parser.add_argument("--device", help="'cpu' or a CUDA index such as 0")
    parser.add_argument("--epochs", type=int, help="train for a fixed epoch count (drops the time budget)")
    parser.add_argument("--imgsz", type=int, help="training and serving input size (multiple of 32)")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--eval-only", action="store_true",
                        help="re-evaluate the shipped checkpoint without training")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from runs/detect/<name>/weights/last.pt")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = DetectorConfig.load(args.config)
    try:
        apply_overrides(config, args)
    except ValueError as exc:
        parser.error(str(exc))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    shipped = CHECKPOINT_DIR / f"{config.name}.pt"

    train_seconds = None
    if not args.eval_only:
        started = time.monotonic()
        best = train(config, args.runs_dir / config.name, resume=args.resume)
        train_seconds = time.monotonic() - started
        shutil.copy2(best, shipped)
        LOG.info("saved %s", shipped)
    elif not shipped.exists():
        LOG.error("no checkpoint at %s; train first", shipped)
        return 1

    metrics: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for split in ("val", "test"):
        metrics[split], raw = evaluate(shipped, config, split)
        LOG.info("%s: mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f", split, metrics[split]["mAP50"],
                 metrics[split]["mAP50_95"], metrics[split]["precision"], metrics[split]["recall"])
        if split == "val":  # thresholds are tuned on val and only ever scored on test
            thresholds = serving_thresholds(raw.box.f1_curve, raw.box.px,
                                            [int(i) for i in raw.box.ap_class_index], raw.names)
            LOG.info("serving thresholds (max-F1 on val): %s", thresholds)
    metrics["test_image_level"] = image_level_metrics(shipped, config, "test", thresholds)
    LOG.info("test image-level: background false-alarm rate %s, per class %s",
             metrics["test_image_level"]["background_false_alarm_rate"],
             {k: (v["precision"], v["recall"]) for k, v in metrics["test_image_level"]["per_class"].items()})

    onnx_path = None
    if not args.no_export:
        onnx_path = export_onnx(shipped, config.imgsz)
        LOG.info("exported %s", onnx_path)
    results_csv = args.runs_dir / config.name / "results.csv"
    epochs_trained = (sum(1 for line in results_csv.read_text().splitlines()[1:] if line.strip())
                      if results_csv.exists() else None)
    previous_card = shipped.with_suffix(".json")
    if train_seconds is None and previous_card.exists():
        # --eval-only re-scores an existing model; keep the training time it recorded.
        train_seconds = json.loads(previous_card.read_text(encoding="utf-8")).get("train_seconds")
    card = write_model_card(config, shipped, onnx_path, metrics, thresholds, train_seconds, epochs_trained)
    LOG.info("model card %s", card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
