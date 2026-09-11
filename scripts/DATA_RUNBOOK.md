# Getting real training data in

Everything here runs on your own machine. Nothing downloads at serve time.

## One-time setup

```powershell
cd $env:USERPROFILE\Desktop\SIH_PROJECT
.\.venv\Scripts\python.exe -m pip install -r scripts\requirements-data.txt
```

Kaggle token (yours, keep it private):

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Set-Content "$env:USERPROFILE\.kaggle\access_token" "KGAT_your_token_here" -NoNewline
```

## The loop

```powershell
$py = ".\.venv\Scripts\python.exe"

# 1. See what's out there (slugs move; this checks what exists today)
& $py -m scripts.fetch_datasets suggest
& $py -m scripts.fetch_datasets search pothole
& $py -m scripts.fetch_datasets search "road damage"

# 2. Download the ones you want (each lands in datasets\incoming\)
& $py -m scripts.fetch_datasets download andrewmvd/pothole-detection

# 3. Turn annotations into labelled training crops
& $py -m scripts.fetch_datasets ingest

# 4. Move them into the folders the training scripts read
& $py -m scripts.fetch_datasets promote

# 5. Retrain and see the new numbers
& $py -m training.train_mega_suite
```

## What `ingest` actually does

It walks every downloaded folder and reads whichever annotation format it
finds — Pascal VOC XML, YOLO `.txt` with `data.yaml`/`classes.txt`, or COCO
JSON. For each labelled box it crops the defect with a little surrounding
road and saves it as a training example for the matching class. A detection
dataset of 600 photos typically yields 1500+ crops, each centred on a real
defect rather than on a whole road scene, which is what the classifier needs.

Images with no annotations are either labelled by their folder name (a
`normal/` folder becomes the sound-pavement class) or parked in
`datasets\patches\_unlabelled\` for you to sort by hand.

Near-duplicate crops are dropped using the perceptual hash from
`models/forensic_audit_engine.py`. This matters more than it sounds: if the
same pothole appears in both halves of a later train/validation split, the
reported accuracy is inflated and meaningless.

## When a label isn't recognised

`ingest` prints every label it couldn't map, with counts:

```
Labels no rule matched (add them to scripts/label_map.json):
  'd44': 213 boxes
```

Open `scripts/label_map.json` and add the label with the class id it belongs
to, then run `ingest` again. The class ids are listed at the top of that file.

## Checking progress

```powershell
& $py -m scripts.fetch_datasets report
```

shows how many distinct source photos and usable images each class has. The
vision model's accuracy is currently limited by the classes with only 12–20
distinct photos, so watch those numbers climb as you add datasets — that is
the single biggest lever on accuracy in this project.
