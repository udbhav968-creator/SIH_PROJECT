# Real deep learning: fine-tuning a CNN on this dataset

This is the step that turns the classifier from hand-crafted features into an
actual neural network. It runs on your laptop CPU.

## 1. Install PyTorch (CPU build, ~250 MB)

```powershell
cd $env:USERPROFILE\Desktop\SIH_PROJECT
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install onnxruntime
```

`onnxruntime` is what serves the trained model afterwards — it's about 50 MB
against PyTorch's 2.5 GB installed, which matters for deployment.

## 2. Check the setup works (2 minutes)

```powershell
& $py -m training.train_deep_vision --smoke
```

One tiny epoch on 60 images. If this finishes without a traceback, the real
run will work. Paste any error to me and I'll fix it.

## 3. Get more data first

Accuracy is currently limited by having only ~163 distinct photographs, and
four classes with 10–20 each. Before a long training run, pull real datasets:

```powershell
& $py -m scripts.fetch_datasets search pothole
& $py -m scripts.fetch_datasets download <slug you picked>
& $py -m scripts.fetch_datasets ingest
& $py -m scripts.fetch_datasets promote
& $py -m scripts.fetch_datasets report
```

Aim for at least 300–500 distinct photographs per class. That single change
moves accuracy more than any model choice.

## 4. Train

```powershell
& $py -m training.train_deep_vision --arch resnet18 --epochs 12
```

On a laptop CPU expect roughly 2–6 minutes per epoch per 1000 images. Leave
it running; it prints progress every 10 batches.

Options worth knowing:

| Flag | Meaning |
|---|---|
| `--arch efficientnet_b0` | More accurate, roughly 2× slower than resnet18 |
| `--arch mobilenet_v3_small` | Fastest, for a quick pass over new data |
| `--epochs 20` | More passes; watch for the validation score flattening |
| `--img-size 160` | Smaller images, roughly 2× faster, slightly less accurate |
| `--batch-size 16` | Lower if memory is tight |

## 5. It serves itself

The API picks the CNN up automatically on next start — no code change. The
health endpoint then reports `LOADED (deep_cnn)` instead of
`LOADED (sklearn_baseline)`, and the dashboard's model card shows the CNN's
held-out test numbers alongside the old baseline so you can show the jump.

```powershell
& $py -m api.server
```

## What makes these numbers defensible

**Grouped splitting.** The augmented copies in the dataset are named after the
photograph they came from. Whole photographs go to train, validation or test —
never split across them. Without this, copies of one pothole land on both
sides and accuracy is inflated by 20–30 points. Judges who know this will ask.

**Three-way split.** Validation picks the best epoch; the test set is scored
exactly once at the end. The headline number is from the test set.

**Macro-F1, not just accuracy.** With classes ranging from 14 to 270 images, a
model that ignores the rare classes can still look accurate. Macro-F1 and the
per-class table make that visible.

**Class weighting.** Rare classes are oversampled during training and weighted
in the loss, so they aren't drowned out.

Everything is written to `checkpoints/deep_vision_report.json`: split sizes,
group counts, per-class precision/recall/F1, and both confusion matrices.

## Honest expectation

The baseline is 36.6% on 7 classes. A fine-tuned ResNet-18 on the *current*
small dataset should reach roughly 55–70%. With several thousand real images
from Kaggle, 85%+ is a reasonable target for the three well-populated classes,
with the rare ones trailing until they have more data. I'd rather you go in
with a measured number and a clear explanation than a claimed one.
