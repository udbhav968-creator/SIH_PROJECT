# Where the real data comes from, and how much of it exists

An honest inventory first, because the plan depends on it.

## What actually exists for road damage

| Source | Real size | Annotations | Access |
|---|---|---|---|
| **RDD2022** (Road Damage Detection challenge) | ~47,000 images, 6 countries incl. India | Pascal VOC boxes: D00/D10/D20 cracks, D40 potholes | Open, no login. github.com/sekilab/RoadDamageDetector |
| RDD2020 (previous edition) | ~26,000 images | Same scheme | Open, and mirrored on Kaggle |
| Kaggle pothole detection sets | 600–5,000 each | VOC or YOLO boxes | Kaggle API |
| CRACK500 / DeepCrack / Crack Forest | 500–3,300 each | Pixel masks | Open, GitHub |
| GAPs (German Asphalt Pavement Distress) | ~1,900 high-res | Boxes | Registration required |
| Roboflow Universe road-damage projects | 1,000–20,000 each | YOLO, ready-made splits | Free account |

Adding these honestly gets you to roughly **60,000–80,000 real annotated road
images**. There is no million-image road dataset in existence, publicly or
otherwise. Anyone claiming one at a hackathon is describing COCO or ImageNet,
which are general-purpose, not road-damage data.

## What you get for free, already trained

| Source | Size | How it's used here |
|---|---|---|
| **COCO** | 330,000 images, 80 classes | The detector weights in `scripts/fetch_detector.py` were trained on it. You get person, bicycle, car, bus, truck, motorcycle, traffic light and stop sign detection without training anything |
| **ImageNet** | 1.28 million images | The CNN backbone in `training/train_deep_vision.py` is pretrained on it. Your fine-tuning starts from features learned across 1.28M images |

So the project *does* stand on over 1.6 million training images — through
pretrained weights, which is how every serious vision system is built. That is
a true statement you can make to judges, and it's worth making precisely.

## The plan, in order of value

**1. RDD2022 — the big one.** ~47,000 annotated images, India included.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m scripts.fetch_datasets download https://github.com/sekilab/RoadDamageDetector
```

The GitHub page lists per-country zip links; download the India set first
(smaller, most relevant), then others if you want scale. Then:

```powershell
& $py -m scripts.fetch_datasets ingest      # crops every annotated box
& $py -m scripts.fetch_datasets promote
& $py -m scripts.fetch_datasets report
```

RDD labels D00/D10/D20/D40 are already in `scripts/label_map.json`, so cracks
and potholes land in the right classes automatically.

**2. Kaggle, for the classes RDD doesn't cover.** Waterlogging, zebra
crossings, road dividers and damaged signs are your weakest classes at 14–20
images each.

```powershell
& $py -m scripts.fetch_datasets search "waterlogging road"
& $py -m scripts.fetch_datasets search "zebra crossing"
& $py -m scripts.fetch_datasets search "traffic sign damage"
& $py -m scripts.fetch_datasets search "road marking"
```

**3. Check for leakage before you trust any number.**

```powershell
& $py -m scripts.fetch_datasets report
```

The ingest step already drops near-duplicate crops by perceptual hash, and
training splits by source photograph. Both matter once you're mixing datasets:
the same public photo often appears in several Kaggle uploads.

## Training at that scale

50,000 images will not fine-tune on a laptop CPU in a useful time — that is
several days. Two realistic options:

**Your DGX H200 at GCSAIR.** Copy the repo and `datasets/` across, then:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m training.train_deep_vision --arch efficientnet_b0 --epochs 30 --batch-size 128
```

The script uses CUDA automatically when it is present. On an H200 this is
minutes per epoch at 50k images. Bring back `checkpoints/deep_vision_*.onnx`
and `deep_vision_report.json`, and the laptop serves the trained model through
ONNX Runtime without needing PyTorch at all.

**Or a subset on the laptop.** 5,000–8,000 images, `--arch mobilenet_v3_small
--img-size 160`, overnight. Lower ceiling, but it works.

## What honest numbers look like

With RDD2022 plus Kaggle for the rare classes, a fine-tuned EfficientNet-B0
typically reaches 85–92% on cracks and potholes, and less on classes with a
few hundred images. Report the per-class table, not one headline number — the
project's own model card page already does exactly that.
