# ===== ROAD-SHIELD: train on ALL the new datasets in one go (Colab or Kaggle, T4 GPU) =====
# Paste this whole file into ONE Colab cell (or run: python scripts/colab_train_all_datasets.py).
#   1. U-Net on DNIT + CrackSeg9k + Pothole Mix + Kaggle pothole outlines   (~1-1.5 h)
#   2. YOLOv8 road-damage detector on RDD2022 India boxes                     (~1.5 h incl. a 13 GB download)
# Each model gets a 1-minute smoke run first; a failed step is reported and the next one still runs.
# Nothing here decides what is served: the laptop's end-to-end checks do that afterwards.
# Output: /content/road_shield_<models>_results.zip (downloads automatically in Colab).
# Safe to run after another training cell in the same session: it refreshes code only, never deletes results.

POTHOLE_MIX_LINK = ""     # optional: Mendeley link to pothole-mix-v1.0-20220526.zip (right-click the download arrow -> Copy link)
POTHOLE_MIX_DRIVE = ""    # or: path of that zip in Google Drive, e.g. "/content/drive/MyDrive/pothole-mix-v1.0-20220526.zip"
TRAIN_UNET = True         # multi-dataset U-Net segmenter
TRAIN_YOLO = True         # YOLOv8 road-damage boxes
UNET_EPOCHS = 30
UNET_ENCODER = "resnet34"   # deeper encoder; "resnet18" is the faster original
UNET_MIN_CROP = 0.35        # random zoom-in down to 35% of the frame, so close-ups are learned
UNET_SAMPLES = 5000         # training images drawn per epoch
YOLO_EPOCHS = 30
SAVE_TO_DRIVE = False     # True: also copy the results zip to Google Drive (Colab only)

import glob, os, subprocess, sys, time

REPO = "https://github.com/udbhav968-creator/SIH_PROJECT.git"
WORK = "/tmp/SIH_PROJECT"
OUT_DIR = "/content" if os.path.isdir("/content") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp")
T0 = time.time()
STATUS = {}


def run(cmd, keep=None):
    """Run a shell command, streaming its output (only lines containing `keep` words, if given)."""
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = []
    for line in p.stdout:
        line = line.rstrip()
        last = (last + [line])[-15:]
        if keep is None or any(k in line for k in keep):
            print(line, flush=True)
    p.wait()
    if p.returncode:
        print("  last lines:\n    " + "\n    ".join(last))
    print(f"  -> {'OK' if p.returncode == 0 else 'FAILED'} ({(time.time() - t) / 60:.1f} min)", flush=True)
    return p.returncode == 0


def step(name):
    print("\n" + "=" * 70 + f"\n{name}   [{(time.time() - T0) / 60:.0f} min elapsed]\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------- setup
step("0. code + packages")
os.chdir("/tmp")
if os.path.isdir(os.path.join(WORK, ".git")):
    # An earlier cell already trained here: refresh the CODE only, never delete its results or datasets.
    run(f"cd {WORK} && git fetch -q origin audit-2026-10-03 && "
        f"git checkout -q FETCH_HEAD -- scripts training models pipeline api")
else:
    run(f"git clone -q -b audit-2026-10-03 {REPO} {WORK}")
os.chdir(WORK)
run('pip install -q "scikit-learn>=1.8,<1.9" "opencv-python-headless>=4.8,<5" onnxruntime onnx onnxscript '
    'datasets kagglehub ultralytics', keep=["ERROR", "error:"])
run('python -c "import torch; print(\'GPU:\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'NONE - switch the runtime to a T4 GPU\')"')

# ---------------------------------------------------------------- 1. U-Net on every outline dataset
if TRAIN_UNET:
    step("1a. datasets with pixel outlines")
    pm_arg, z = "", ""
    if POTHOLE_MIX_LINK:
        z = f"{OUT_DIR}/pothole_mix.zip"
        if not (os.path.exists(z) and os.path.getsize(z) > 100_000_000):
            run(f'wget -q -O {z} "{POTHOLE_MIX_LINK}"')
    elif POTHOLE_MIX_DRIVE:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            print("Drive not mounted:", e)
        z = POTHOLE_MIX_DRIVE
    if z and os.path.exists(z) and os.path.getsize(z) > 100_000_000:
        pm_arg = f"--pothole-mix-zip {z}"
        print(f"Pothole Mix: {os.path.getsize(z) / 1e9:.2f} GB")
    else:
        print("Pothole Mix: " + ("the file is missing or not the zip (Mendeley may have sent a web page)" if z
                                 else "no link or Drive path given") + " - continuing without it")
    run("python -m scripts.fetch_cracks_potholes_dataset --limit 2235 --workers 16", keep=["[", "Error", "error"])
    if os.path.isdir("datasets/seg_multi/crackseg9k/lab"):
        # an earlier cell in this session already prepared CrackSeg9k and Kaggle: add only what is new
        if pm_arg:
            run(f"python -m scripts.fetch_seg_datasets --only pothole_mix {pm_arg}")
        else:
            print("outline datasets already prepared in this session - reusing them")
    else:
        run(f"python -m scripts.fetch_seg_datasets {pm_arg}")

    step("1b. U-Net smoke run (1 minute)")
    if run("python -m training.train_unet_multi --smoke --workers 0 --out /tmp/unet_multi_smoke",
           keep=["unet-multi", "TEST", "wrote", "Error", "error"]):
        step("1c. U-Net full training")
        STATUS["unet"] = run(f"python -m training.train_unet_multi --epochs {UNET_EPOCHS} --encoder {UNET_ENCODER} "
                             f"--min-crop-scale {UNET_MIN_CROP} --samples-per-epoch {UNET_SAMPLES}",
                             keep=["unet-multi", "encoder", "shares", "epoch", "early", "thresholds", "SELECTION", "TEST",
                                   "exported", "wrote", "Error", "error", "[!]"])
    else:
        STATUS["unet"] = False
        print("U-Net smoke run failed - full run skipped (the served models are unchanged)")

# ---------------------------------------------------------------- 2. YOLOv8 on RDD2022 India
if TRAIN_YOLO:
    step("2a. RDD2022 India boxes")
    if not os.path.isdir("datasets/rdd2022_india/images/train"):
        run("python -m scripts.fetch_rdd2022_india")
        india = subprocess.run([sys.executable, "-c", "from scripts.fetch_rdd2022_india import india_root; "
                                "print(india_root() or '')"], capture_output=True, text=True).stdout.strip()
        print("India data:", india or "NOT FOUND")
        if india:
            run(f'python -m scripts.prepare_rdd2022_voc --src "{india}" --out datasets/rdd2022_india')
            run("rm -rf datasets/_downloads/rdd2022")          # frees ~13 GB
    step("2b. YOLO smoke run (1 minute)")
    if run("python -m training.train_rdd_detector --model yolov8n.pt --epochs 1 --fraction 0.05 --out /tmp/rdd_smoke",
           keep=["rdd-detector", "TEST", "exported", "Error", "error"]):
        step("2c. YOLO full training")
        ok = run(f"python -m training.train_rdd_detector --epochs {YOLO_EPOCHS}",
                 keep=["rdd-detector", "VAL", "TEST", "exported", "Error", "error"])
        step("2d. does the exported detector file match the trained network?")
        STATUS["yolo"] = ok and run("python -m scripts.verify_rdd_detector --artefact")
    else:
        STATUS["yolo"] = False
        print("YOLO smoke run failed - full run skipped")

# ---------------------------------------------------------------- package
step("3. package the results")
# Only the files THIS run produced, so unzipping never overwrites a newer model with an older one.
wanted = []
if STATUS.get("unet"):
    wanted += ["checkpoints/defect_segmenter_unet.onnx", "checkpoints/defect_segmenter_unet.json",
               "checkpoints/segmenter_selection.json", "datasets/seg_multi/manifest.json"]
if STATUS.get("yolo") is not None and os.path.exists("checkpoints/road_damage_detector_report.json"):
    wanted += ["checkpoints/damage_rdd2022_india.onnx", "checkpoints/damage_rdd2022_india.json",
               "checkpoints/road_damage_detector_report.json"]
files = [f for f in wanted if os.path.exists(f)]
zip_path = f"{OUT_DIR}/road_shield_{'_'.join(k for k in STATUS) or 'nothing'}_results.zip"
if files:
    run(f"rm -f {zip_path} && zip -q {zip_path} " + " ".join(files) + f" && ls -la {zip_path}")
print("\nSTATUS:", STATUS, f"| total {(time.time() - T0) / 60:.0f} min")
print("files in the zip:", files)
if SAVE_TO_DRIVE and os.path.exists(zip_path):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        run(f"mkdir -p /content/drive/MyDrive/road_shield && cp {zip_path} /content/drive/MyDrive/road_shield/")
    except Exception as e:
        print("Drive copy skipped:", e)
try:
    from google.colab import files as _colab_files
    if os.path.exists(zip_path):
        _colab_files.download(zip_path)
except Exception:
    print(f"Download {zip_path} from the file browser / Output panel.")
