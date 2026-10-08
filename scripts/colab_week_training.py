# ===== ROAD-SHIELD: long training across several Colab sessions (T4 GPU, Google Drive) =====
# Paste this whole file into ONE Colab cell and run it. When Colab ends the session (or you stop for the day),
# open a new session and run the SAME cell again: it continues where it stopped. Nothing is trained twice.
#
#   1. U-Net segmenter, ResNet-34, up to 80 epochs, on DNIT + CrackSeg9k + Kaggle + Pothole Mix + YOUR photos
#   2. YOLOv8s road-damage detector on RDD2022 India, up to 100 epochs
#   3. YOLOv8s on RDD2022 India + Japan + Czech + USA + China (~4x the photographs); served only if it beats
#      the India-only detector on India's validation photographs (scripts/select_rdd_detector.py)
#   4. image classifier retrained with crops from those countries too; served only if it beats the current
#      classifier on the selection half of the Indian held-out crops (scripts/select_vision_candidate.py)
#   5. measurements, no training: privacy-redactor recall on WIDER FACE + a licence-plate set, and the
#      classifier's confidence calibration (temperature fitted on one half, reported on the other)
#
# What takes long lives in Google Drive (MyDrive/road_shield_week): the prepared datasets, the checkpoints
# (saved every 30 minutes and whenever a session pauses), and the finished results. The U-Net pauses cleanly
# before SESSION_HOURS. If Colab cuts a session off without warning, at most ~30 minutes of training is lost.
# Stages 3 and 4 only replace a served model under the rules above (fixed before scoring, test data never used
# to choose). Run the end-to-end and clean-road checks on the laptop ONCE on the final result.

OWN_PHOTOS = ""           # your outlined Indian road photos in Drive: a Roboflow/CVAT "COCO Segmentation" export zip,
                          # a LabelMe folder, or images/ + masks/  e.g. "/content/drive/MyDrive/road_photos_coco.zip"
POTHOLE_MIX_DRIVE = ""    # e.g. "/content/drive/MyDrive/pothole-mix-v1.0-20220526.zip"
POTHOLE_MIX_LINK = ""     # or a direct download link instead
RUN_NAME = "run1"         # change it to start a completely new run (new checkpoints); keep it to resume

TRAIN_UNET = True
UNET_ENCODER = "resnet34"
UNET_EPOCHS = 80          # upper limit; early stopping ends it when validation stops improving for UNET_PATIENCE
UNET_PATIENCE = 15
UNET_SAMPLES = 6000       # training images per epoch
UNET_MIN_CROP = 0.35      # random zoom-ins down to 35% of the frame: teaches close-ups (where the last U-Net missed)
UNET_BATCH = 8            # 16 on an A100 / L4

TRAIN_YOLO = True
YOLO_MODEL = "yolov8s.pt"  # "yolov8m.pt" is stronger but ~2x slower and too slow to serve on a laptop CPU
YOLO_EPOCHS = 100
YOLO_PATIENCE = 25
YOLO_BATCH = 16

TRAIN_YOLO_WORLD = True   # stage 3 (downloads the full RDD2022 release, ~13 GB, once)
WORLD_EPOCHS = 50
WORLD_PATIENCE = 12
WORLD_CACHE_TO_DRIVE = True   # ~4-5 GB in Drive; set False if your Drive is nearly full

TRAIN_CLASSIFIER = True   # stage 4
CLS_ARCHS = "mobilenet_v3_large,efficientnet_b0,efficientnet_b2"
CLS_EPOCHS = 40
CLS_MAX_PER_CLASS = 3000

MEASURE = True           # stage 5 (~410 MB download once, ~20 min)

SESSION_HOURS = 3.5       # U-Net pauses before this many hours in one session (free Colab often ends at 3-4 h)
SAVE_EVERY_MIN = 30       # how often checkpoints are copied to Drive

import glob, hashlib, json, os, shutil, subprocess, sys, threading, time

REPO = "https://github.com/udbhav968-creator/SIH_PROJECT.git"
BRANCH = "audit-2026-10-03"
WORK = "/tmp/SIH_PROJECT"
T0 = time.time()
STATUS = {}
OUTPUT = []        # every line the last command printed


def run(cmd, keep=None):
    """Run a shell command, streaming output (only lines containing a `keep` word, if given). Returns exit code."""
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = []
    OUTPUT.clear()
    for line in p.stdout:
        line = line.rstrip()
        OUTPUT.append(line)
        last = (last + [line])[-15:]
        if keep is None or any(k in line for k in keep):
            print(line, flush=True)
    p.wait()
    if p.returncode not in (0, 3):
        print("  last lines:\n    " + "\n    ".join(last))
    print(f"  -> exit {p.returncode} ({(time.time() - t) / 60:.1f} min)", flush=True)
    return p.returncode


def step(name):
    print("\n" + "=" * 72 + f"\n{name}   [{(time.time() - T0) / 60:.0f} min into this session]\n" + "=" * 72, flush=True)


def hours_left():
    return SESSION_HOURS - (time.time() - T0) / 3600


def signature(path):
    """Content fingerprint of the photos: a re-upload of the same zip is recognised as unchanged."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        return h.hexdigest()
    for f in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
        if os.path.isfile(f):
            h.update(f"{os.path.relpath(f, path)}:{os.path.getsize(f)}".encode())
    return h.hexdigest()


def load_json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def save_json(p, d):
    with open(p + ".tmp", "w") as fh:
        json.dump(d, fh, indent=2)
    os.replace(p + ".tmp", p)


def untar(tar, dest_parent):
    if os.path.exists(tar):
        print(f"  restoring {os.path.basename(tar)} ({os.path.getsize(tar) / 1e9:.2f} GB) from Drive")
        return run(f'mkdir -p "{dest_parent}" && tar -xf "{tar}" -C "{dest_parent}"') == 0
    return False


def tar_to_drive(src_dir, tar):
    if os.path.isdir(src_dir):
        tmp = tar + ".part"
        if run(f'tar -cf "{tmp}" -C "{os.path.dirname(src_dir)}" "{os.path.basename(src_dir)}" && mv "{tmp}" "{tar}"') == 0:
            print(f"  cached {os.path.basename(tar)} in Drive ({os.path.getsize(tar) / 1e9:.2f} GB)")


def manifest_counts():
    man = load_json("datasets/seg_multi/manifest.json") or {}
    out = {}
    for src, items in (man.get("items") or {}).items():
        n = sum(1 for it in items if os.path.exists(f"datasets/seg_multi/{src}/lab/{it['id']}.png"))
        if n:
            out[src] = n
    return out, man


def count_files(d):
    return len(glob.glob(os.path.join(d, "*.jpg"))) if os.path.isdir(d) else 0


def copy_run(src, dst, min_age_s=15):
    """
    Copy a YOLO run folder: changed files only, each through its own temporary name, skipping files written in
    the last `min_age_s` seconds and discarding any copy of a file that changed while it was being copied.
    """
    for f in glob.glob(os.path.join(src, "**", "*"), recursive=True):
        if not os.path.isfile(f) or f.endswith(".part"):
            continue
        before = (os.path.getsize(f), os.path.getmtime(f))
        if time.time() - before[1] < min_age_s:
            continue
        out = os.path.join(dst, os.path.relpath(f, src))
        if os.path.exists(out) and os.path.getsize(out) == before[0] and os.path.getmtime(out) >= before[1]:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = f"{out}.{os.getpid()}.{threading.get_ident()}.part"
        shutil.copy2(f, tmp)
        if (os.path.getsize(f), os.path.getmtime(f)) != before or os.path.getsize(tmp) != before[0]:
            os.remove(tmp)                        # rewritten mid-copy: the next copy picks it up
            continue
        os.replace(tmp, out)


# ---------------------------------------------------------------- 0. Drive, code, packages
step("0. Google Drive, code, packages")
if globals().get("_RS_SYNC_STOP") is not None:          # left over from an interrupted run of this cell
    _RS_SYNC_STOP.set()
    if globals().get("_RS_SYNC_THREAD") is not None:
        _RS_SYNC_THREAD.join(timeout=600)
ON_DRIVE = False
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False
if IN_COLAB:
    try:
        drive.mount("/content/drive")
    except Exception as e:
        raise SystemExit(f"Google Drive did not mount ({e}). Run the cell again and allow Drive access: without "
                         f"Drive nothing would survive the session.")
    BASE, ON_DRIVE = "/content/drive/MyDrive/road_shield_week", True
else:
    BASE = "/tmp/road_shield_week"
    print(f"(not Colab) using {BASE}: training works, but resuming needs this folder to survive")
CACHE, D = f"{BASE}/cache", f"{BASE}/{RUN_NAME}"
UNET_CK, YOLO_RUN, RESULTS = f"{D}/unet_checkpoints", f"{D}/yolo_run", f"{D}/results"
for d in (CACHE, UNET_CK, YOLO_RUN, RESULTS):
    os.makedirs(d, exist_ok=True)
print("Drive folder:", D)

os.chdir("/tmp")
if os.path.isdir(os.path.join(WORK, ".git")):
    run(f"cd {WORK} && git fetch -q origin {BRANCH} && git checkout -q FETCH_HEAD -- scripts training models pipeline api")
else:
    run(f"git clone -q -b {BRANCH} {REPO} {WORK}")
os.chdir(WORK)
run('pip install -q "scikit-learn>=1.8,<1.9" "opencv-python-headless>=4.8,<5" onnxruntime onnx onnxscript '
    'datasets kagglehub ultralytics pycocotools', keep=["ERROR", "error:"])
if run("python -m training.train_unet_multi --help", keep=["--ckpt-dir"]) != 0 or \
        not any("--ckpt-dir" in l for l in OUTPUT):
    raise SystemExit(f"The code on GitHub ({BRANCH}) is older than this cell: push the latest commit first.")
gpu = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0) "
                      "if torch.cuda.is_available() else '')"], capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "NONE")
if not gpu:
    raise SystemExit("No GPU. Runtime -> Change runtime type -> T4 GPU. If Colab says you hit the GPU usage limit, "
                     "come back later (usually within a day) and run this cell again: your checkpoints wait in Drive.")

# ---------------------------------------------------------------- 1. U-Net
unet_done = load_json(f"{RESULTS}/unet_done.json")
if TRAIN_UNET and unet_done:
    print(f"\nU-Net already finished in this run ({unet_done.get('finished')}); skipping. "
          f"Set RUN_NAME to a new name to train again.")
    STATUS["unet"] = "done earlier"
unet_go = False
if TRAIN_UNET and not unet_done:
    step("1a. pixel-labelled datasets (restored from Drive after the first session)")
    DNIT = "datasets/incoming/cracks_potholes_dnit"
    started = os.path.exists(f"{UNET_CK}/dnit_split.json")      # this run has begun: its data is frozen
    # DNIT is restored, never re-downloaded once cached: the split depends on which photographs are on disk.
    # The tar also holds the clean-road crops the download makes (cpr_*), which the U-Net uses as negatives.
    dnit_ready = untar(f"{CACHE}/dnit.tar", "datasets")
    if not dnit_ready:
        code = run("python -m scripts.fetch_cracks_potholes_dataset --limit 2235 --workers 16", keep=["[", "Error"])
        if code == 0 and count_files(DNIT) >= 2000:
            run('cd datasets && find . -name "cpr_*" -type f > /tmp/dnit_crops.txt')
            dnit_ready = run(f'cd datasets && tar -cf "{CACHE}/dnit.tar.part" incoming/cracks_potholes_dnit '
                             f'-T /tmp/dnit_crops.txt && mv "{CACHE}/dnit.tar.part" "{CACHE}/dnit.tar"') == 0
        else:
            print(f"!! DNIT download incomplete ({count_files(DNIT)} photographs)")
    print("DNIT photographs on disk:", count_files(DNIT))
    unet_go = dnit_ready
    if not dnit_ready:
        STATUS["unet"] = "waiting: DNIT not fully downloaded and cached - run the cell again"

if TRAIN_UNET and not unet_done and unet_go:
    restored = untar(f"{CACHE}/seg_multi.tar", "datasets")
    cache_meta = load_json(f"{CACHE}/seg_multi.json") or {}
    before = manifest_counts()[0] if restored else {}
    own_sig = signature(OWN_PHOTOS)
    if OWN_PHOTOS and not own_sig:
        print(f"!! OWN_PHOTOS path not found: {OWN_PHOTOS} - check the path in the Drive file browser")
    if started:
        print("This run has started: it keeps the datasets it started with (no new sources or photos).")
        if own_sig != cache_meta.get("own_photos"):
            print("   Your photos changed since it started. To train with the new ones, set RUN_NAME to a new name.")
    else:
        meta_before = dict(cache_meta)
        wanted = ["crackseg9k", "kaggle_pothole"] + (["pothole_mix"] if (POTHOLE_MIX_LINK or POTHOLE_MIX_DRIVE) else [])
        missing = [s_ for s_ in wanted if not before.get(s_)]
        pm = ""
        if "pothole_mix" in missing:
            pm = POTHOLE_MIX_DRIVE
            if POTHOLE_MIX_LINK:
                pm = "/content/pothole_mix.zip"
                if not (os.path.exists(pm) and os.path.getsize(pm) > 100_000_000):
                    run(f'wget -q -O "{pm}" "{POTHOLE_MIX_LINK}"')
            if not (os.path.exists(pm) and os.path.getsize(pm) > 100_000_000):
                print(f"!! Pothole Mix: {pm} is missing or not the real zip (Mendeley may have sent a web page)")
                missing.remove("pothole_mix")
        if missing:
            print("preparing:", missing)
            code = run(f"python -m scripts.fetch_seg_datasets --only {' '.join(missing)}"
                       + (f' --pothole-mix-zip "{pm}"' if "pothole_mix" in missing else ""),
                       keep=["leak guard", "kept", "skipped", "total", "Error"])
            if code != 0:
                print("!! dataset preparation failed for part of it; what arrived is used, the rest retried next time")
        if own_sig != cache_meta.get("own_photos") or (own_sig and not before.get("own_india")):
            if own_sig:
                print("Your photos are new or changed: preparing them")
                code = run(f'python -m scripts.fetch_seg_datasets --only own_india --own-photos "{OWN_PHOTOS}"',
                           keep=["own_india", "kept", "format", "spots", "warning", "total", "Error"])
                if code == 0 and manifest_counts()[0].get("own_india"):
                    cache_meta["own_photos"] = own_sig
                else:
                    print("!! your photos could not be read (see the lines above); training continues without them")
            else:
                shutil.rmtree("datasets/seg_multi/own_india", ignore_errors=True)
                run("python -m scripts.fetch_seg_datasets --only own_india", keep=["total"])
                cache_meta["own_photos"] = None
        if manifest_counts()[0] != before or cache_meta != meta_before:     # re-cache only when something arrived
            shutil.rmtree("datasets/seg_multi/_pothole_mix_raw", ignore_errors=True)   # raw zip contents
            shutil.rmtree("datasets/seg_multi/_own_raw", ignore_errors=True)
            tar_to_drive("datasets/seg_multi", f"{CACHE}/seg_multi.tar")
            save_json(f"{CACHE}/seg_multi.json", cache_meta)
    counts, man = manifest_counts()
    print("pixel-labelled images per source:", counts, "| total", sum(counts.values()))
    if "own_india" in counts:
        sp = {k: sum(1 for i in man["items"]["own_india"] if i["split"] == k) for k in ("train", "cal", "test")}
        print(f"your photos: {sp} (train / calibration / test, split by spot)")

    # a run that has started must keep the data it started with
    data_sig = {"sources": counts, "own_photos": cache_meta.get("own_photos")}
    saved_sig = load_json(f"{UNET_CK}/data_signature.json")
    resuming = os.path.exists(f"{UNET_CK}/unet_resume.pt")
    if started and saved_sig and saved_sig != data_sig:
        print(f"!! The datasets differ from the ones this run started with:\n   then {saved_sig}\n   now  {data_sig}\n"
              f"   Resuming would mix two experiments. Set RUN_NAME to a new name to train with the new data. "
              f"U-Net skipped.")
        STATUS["unet"] = "skipped: data changed"
    else:
        if not started:
            save_json(f"{UNET_CK}/data_signature.json", data_sig)
        ok = True
        if not started:
            step("1b. U-Net smoke run (2 minutes): trains, then resumes from its own checkpoint")
            smoke = ("python -m training.train_unet_multi --smoke --workers 0 --out /tmp/unet_multi_smoke "
                     "--ckpt-dir /tmp/unet_smoke_ck")
            shutil.rmtree("/tmp/unet_smoke_ck", ignore_errors=True)
            ok = run(smoke, keep=["unet-multi", "TEST", "wrote", "Error"]) == 0
            if ok:
                ok = run(smoke, keep=["RESUMED", "wrote", "Error"]) == 0 and any("RESUMED" in l for l in OUTPUT)
                print("  resume from a checkpoint:", "works" if ok else "DID NOT WORK - full run skipped")
        if not ok:
            STATUS["unet"] = "smoke run failed"
        else:
            step(f"1c. U-Net training ({'resuming' if resuming else 'starting'}; "
                 f"{max(hours_left(), 0):.1f} h left in this session)")
            code = run(f"python -m training.train_unet_multi --epochs {UNET_EPOCHS} --encoder {UNET_ENCODER} "
                       f"--min-crop-scale {UNET_MIN_CROP} --samples-per-epoch {UNET_SAMPLES} --batch {UNET_BATCH} "
                       f"--patience {UNET_PATIENCE} --ckpt-dir \"{UNET_CK}\" --ckpt-every-min {SAVE_EVERY_MIN} "
                       f"--time-budget-hours {max(hours_left(), 0.75):.2f}",
                       keep=["unet-multi", "RESUMED", "PAUSED", "split", "encoder", "shares", "epoch", "early",
                             "thresholds", "SELECTION", "TEST", "exported", "wrote", "Error", "[!]"])
            if code == 3:
                STATUS["unet"] = "paused"
            elif code == 0:
                for f in ("defect_segmenter_unet.onnx", "defect_segmenter_unet.json", "segmenter_selection.json",
                          "defect_segmenter_unet.pt"):
                    if os.path.exists(f"checkpoints/{f}"):
                        shutil.copy(f"checkpoints/{f}", f"{RESULTS}/{f}")
                if os.path.exists("datasets/seg_multi/manifest.json"):
                    shutil.copy("datasets/seg_multi/manifest.json", f"{RESULTS}/seg_multi_manifest.json")
                save_json(f"{RESULTS}/unet_done.json", {"finished": time.strftime("%Y-%m-%d %H:%M"),
                                                        "settings": {"encoder": UNET_ENCODER, "epochs": UNET_EPOCHS,
                                                                     "samples": UNET_SAMPLES, "min_crop": UNET_MIN_CROP}})
                STATUS["unet"] = "done"
            else:
                STATUS["unet"] = f"failed (exit {code}) - see the lines above; run the cell again to resume"

def ensure_rdd_india():
    """datasets/rdd2022_india (India's fixed 70/15/15 split), restored from Drive or built once."""
    if os.path.exists("datasets/rdd2022_india/data.yaml"):        # written last: the split is complete
        return True
    shutil.rmtree("datasets/rdd2022_india", ignore_errors=True)
    if untar(f"{CACHE}/rdd2022_india.tar", "datasets") and os.path.exists("datasets/rdd2022_india/data.yaml"):
        return True
    run("python -m scripts.fetch_rdd2022_india")
    india = subprocess.run([sys.executable, "-c", "from scripts.fetch_rdd2022_india import india_root; "
                            "print(india_root() or '')"], capture_output=True, text=True).stdout.strip()
    print("India data:", india or "NOT FOUND")
    if india and run(f'python -m scripts.prepare_rdd2022_voc --src "{india}" --out datasets/rdd2022_india') == 0:
        run("rm -rf datasets/_downloads/rdd2022")
        tar_to_drive("datasets/rdd2022_india", f"{CACHE}/rdd2022_india.tar")
        return True
    shutil.rmtree("datasets/rdd2022_india", ignore_errors=True)
    return False


def ensure_rdd_world():
    """datasets/rdd2022_world, restored from Drive or built once (downloads the full release)."""
    if os.path.exists("datasets/rdd2022_world/manifest.json"):    # written last: the set is complete
        return True
    shutil.rmtree("datasets/rdd2022_world", ignore_errors=True)
    if untar(f"{CACHE}/rdd2022_world.tar", "datasets") and os.path.exists("datasets/rdd2022_world/manifest.json"):
        return True
    if not ensure_rdd_india():
        return False
    code = run("python -m scripts.prepare_rdd2022_world", keep=["downloading", "GB", "Japan", "Czech", "United",
                                                                 "China", "Norway", "photographs", "Error", "skipped"])
    shutil.rmtree("datasets/_downloads/rdd2022_world", ignore_errors=True)
    if code != 0:
        shutil.rmtree("datasets/rdd2022_world", ignore_errors=True)
        return False
    if WORLD_CACHE_TO_DRIVE:
        tar_to_drive("datasets/rdd2022_world", f"{CACHE}/rdd2022_world.tar")
    return True


def ensure_dnit_crops():
    """The DNIT photographs and their crops (cpr_*), which the classifier corpus includes."""
    if glob.glob("datasets/*/real_images/cpr_*.jpg"):
        return True
    if untar(f"{CACHE}/dnit.tar", "datasets"):
        return True
    return run("python -m scripts.fetch_cracks_potholes_dataset --limit 2235 --workers 16", keep=["[4/4]", "Error"]) == 0


def train_yolo(tag, args, drive_run):
    """Resumable YOLO training for runs/rdd_<tag>, its folder mirrored to Drive every SAVE_EVERY_MIN."""
    local = f"runs/rdd_{tag}"
    os.makedirs(drive_run, exist_ok=True)
    if os.path.islink(local):
        os.unlink(local)
    shutil.rmtree(local, ignore_errors=True)
    if os.path.exists(f"{drive_run}/weights/last.pt"):
        print(f"restoring the {tag} YOLO run from Drive")
        copy_run(drive_run, local, min_age_s=0)
    stop = threading.Event()

    def loop():
        while not stop.wait(SAVE_EVERY_MIN * 60):
            try:
                copy_run(local, drive_run)
                print(f"  [drive] {tag} YOLO checkpoint copied to Drive at {time.strftime('%H:%M')}", flush=True)
            except Exception as e:
                print(f"  [drive] copy failed ({e}); will retry", flush=True)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    globals()["_RS_SYNC_STOP"], globals()["_RS_SYNC_THREAD"] = stop, th
    try:
        return run(f"python -m training.train_rdd_detector {args}",
                   keep=["rdd-detector", "RESUMING", "already finished", "VAL", "TEST", "exported", "Error"])
    finally:
        stop.set()
        th.join(timeout=600)
        copy_run(local, drive_run, min_age_s=0)


# ---------------------------------------------------------------- 2. YOLOv8 detector
yolo_done = load_json(f"{RESULTS}/yolo_done.json")
if TRAIN_YOLO and yolo_done:
    print(f"\nYOLO already finished in this run ({yolo_done.get('finished')}); skipping.")
    STATUS["yolo"] = "done earlier"
elif TRAIN_YOLO and STATUS.get("unet") == "paused":
    print("\nYOLO waits: the U-Net paused for time. Run this cell again in a new session.")
    STATUS["yolo"] = "waiting for the U-Net"
elif TRAIN_YOLO:
    step("2a. RDD2022 India boxes (restored from Drive after the first session)")
    ensure_rdd_india()
    LOCAL_RUN = "runs/rdd_india"
    yolo_resuming = os.path.exists(f"{YOLO_RUN}/weights/last.pt")
    ok = True
    if os.path.islink(LOCAL_RUN):
        os.unlink(LOCAL_RUN)
    shutil.rmtree(LOCAL_RUN, ignore_errors=True)
    if not yolo_resuming:
        step("2b. YOLO smoke run (1 minute)")
        ok = run("python -m training.train_rdd_detector --model yolov8n.pt --epochs 1 --fraction 0.05 "
                 "--out /tmp/rdd_smoke --fresh --run-name rdd_smoke",
                 keep=["rdd-detector", "TEST", "exported", "Error"]) == 0
        shutil.rmtree("runs/rdd_smoke", ignore_errors=True)
    else:
        print("restoring the YOLO run from Drive")
        copy_run(YOLO_RUN, LOCAL_RUN, min_age_s=0)
    if not ok:
        STATUS["yolo"] = "smoke run failed"
    else:
        # trains on local disk (fast); a background copy keeps Drive at most SAVE_EVERY_MIN behind
        _RS_SYNC_STOP = threading.Event()

        def sync_loop(stop):
            while not stop.wait(SAVE_EVERY_MIN * 60):
                try:
                    copy_run(LOCAL_RUN, YOLO_RUN)
                    print(f"  [drive] YOLO checkpoint copied to Drive at {time.strftime('%H:%M')}", flush=True)
                except Exception as e:
                    print(f"  [drive] copy failed ({e}); will retry", flush=True)

        _RS_SYNC_THREAD = threading.Thread(target=sync_loop, args=(_RS_SYNC_STOP,), daemon=True)
        _RS_SYNC_THREAD.start()
        step(f"2c. YOLO training ({'resuming' if yolo_resuming else 'starting'})")
        code = 1
        try:
            code = run(f"python -m training.train_rdd_detector --model {YOLO_MODEL} --epochs {YOLO_EPOCHS} "
                       f"--patience {YOLO_PATIENCE} --batch {YOLO_BATCH}",
                       keep=["rdd-detector", "RESUMING", "already finished", "VAL", "TEST", "exported", "Error"])
        finally:
            _RS_SYNC_STOP.set()
            _RS_SYNC_THREAD.join(timeout=600)
            copy_run(LOCAL_RUN, YOLO_RUN, min_age_s=0)
        if code == 0:
            step("2d. does the exported detector file match the trained network?")
            verified = run("python -m scripts.verify_rdd_detector --artefact", keep=None) == 0
            for f in ("damage_rdd2022_india.onnx", "damage_rdd2022_india.json", "road_damage_detector_report.json"):
                if os.path.exists(f"checkpoints/{f}"):
                    shutil.copy(f"checkpoints/{f}", f"{RESULTS}/{f}")
            save_json(f"{RESULTS}/yolo_done.json", {"finished": time.strftime("%Y-%m-%d %H:%M"),
                                                    "artefact_verified": verified, "model": YOLO_MODEL})
            STATUS["yolo"] = "done" + ("" if verified else " (ONNX check FAILED - it will not be served)")
        else:
            STATUS["yolo"] = f"stopped (exit {code}) - see the lines above; run the cell again to resume"

# ---------------------------------------------------------------- 3. YOLOv8 on several countries
SERVED_DET = ("damage_rdd2022_india.onnx", "damage_rdd2022_india.json", "road_damage_detector_report.json")
world_done = load_json(f"{RESULTS}/yolo_world_done.json")
if TRAIN_YOLO_WORLD and world_done:
    print(f"\nMulti-country YOLO already finished in this run ({world_done.get('finished')}); skipping.")
    STATUS["yolo_world"] = "done earlier"
elif TRAIN_YOLO_WORLD and (STATUS.get("unet") == "paused" or (TRAIN_YOLO and not STATUS.get("yolo", "").startswith("done"))):
    print("\nMulti-country YOLO waits for the earlier stages. Run this cell again.")
    STATUS["yolo_world"] = "waiting for earlier stages"
elif TRAIN_YOLO_WORLD:
    step("3a. multi-country RDD2022 (the first time: ~13 GB download and preparation, about an hour)")
    if not ensure_rdd_world():
        STATUS["yolo_world"] = "dataset could not be prepared - see the lines above; run the cell again"
    else:
        m = load_json("datasets/rdd2022_world/manifest.json") or {}
        print("photographs:", m.get("photographs"), "| countries:", list((m.get("countries") or {}).keys()))
        # the India-only detector this one is compared with. A copy is kept in Drive before anything can
        # replace it; a previous attempt that stopped half way is undone from that copy first.
        DET_BACKUP = f"{D}/detector_before_world"
        if os.path.isdir(DET_BACKUP) and all(os.path.exists(f"{DET_BACKUP}/{f}") for f in SERVED_DET):
            for f in SERVED_DET:
                shutil.copy(f"{DET_BACKUP}/{f}", f"checkpoints/{f}")
        else:
            for f in SERVED_DET:
                if os.path.exists(f"{RESULTS}/{f}"):
                    shutil.copy(f"{RESULTS}/{f}", f"checkpoints/{f}")
            if all(os.path.exists(f"checkpoints/{f}") for f in SERVED_DET):
                os.makedirs(DET_BACKUP, exist_ok=True)
                for f in SERVED_DET:
                    shutil.copy(f"checkpoints/{f}", f"{DET_BACKUP}/{f}")
        step("3b. multi-country YOLO training (resumes from Drive)")
        code = train_yolo("world", f"--data-dir datasets/rdd2022_world --tag world --model {YOLO_MODEL} "
                                   f"--epochs {WORLD_EPOCHS} --patience {WORLD_PATIENCE} --batch {YOLO_BATCH}",
                          f"{D}/yolo_world_run")
        if code == 0:
            step("3c. India-only or multi-country? (decided on India's validation photographs)")
            sel_code = run("python -m scripts.select_rdd_detector", keep=None)
            sel = load_json("checkpoints/rdd_detector_selection.json") or {}
            verified = None
            if sel_code == 0 and sel.get("served") == "world":
                verified = run("python -m scripts.verify_rdd_detector --artefact --run-name rdd_world "
                               "--data-dir datasets/rdd2022_world", keep=None) == 0
                if not verified and all(os.path.exists(f"{DET_BACKUP}/{f}") for f in SERVED_DET):
                    for f in SERVED_DET:                     # the India-only detector stays served
                        shutil.copy(f"{DET_BACKUP}/{f}", f"checkpoints/{f}")
                    sel.update(served="india", why=sel.get("why", "") + "; but the multi-country ONNX failed "
                               "the artefact check, so the India-only detector stays")
                    save_json("checkpoints/rdd_detector_selection.json", sel)
        if code == 0 and sel_code != 0:
            STATUS["yolo_world"] = f"selection failed (exit {sel_code}) - see the lines above; run the cell again"
        elif code == 0:
            for f in SERVED_DET + ("damage_rdd2022_world.json", "road_damage_detector_world_report.json",
                                   "rdd_detector_selection.json"):
                if os.path.exists(f"checkpoints/{f}"):
                    shutil.copy(f"checkpoints/{f}", f"{RESULTS}/{f}")
            save_json(f"{RESULTS}/yolo_world_done.json", {"finished": time.strftime("%Y-%m-%d %H:%M"),
                                                          "served": sel.get("served"), "why": sel.get("why"),
                                                          "artefact_verified": verified})
            STATUS["yolo_world"] = f"done - served: {sel.get('served')} ({sel.get('why')})"
        else:
            STATUS["yolo_world"] = f"stopped (exit {code}) - run the cell again to resume"

# ---------------------------------------------------------------- 4. image classifier with more data
cls_done = load_json(f"{RESULTS}/classifier_done.json")
if TRAIN_CLASSIFIER and cls_done:
    print(f"\nClassifier already finished in this run ({cls_done.get('finished')}); skipping.")
    STATUS["classifier"] = "done earlier"
elif TRAIN_CLASSIFIER and any(STATUS.get(k, "done").startswith(("paused", "waiting")) for k in ("unet", "yolo_world")):
    print("\nClassifier waits for the earlier stages. Run this cell again.")
    STATUS["classifier"] = "waiting for earlier stages"
elif TRAIN_CLASSIFIER:
    step("4a. classifier data: DNIT crops, RDD2022 India crops (+ held-out Indian set), other countries' crops")
    ok = ensure_dnit_crops() and ensure_rdd_india()
    ok = ok and run("python -m scripts.ingest_rdd2022_india", keep=["RDD India", "Error"]) == 0
    world_crops = "not used"
    if TRAIN_YOLO_WORLD and ensure_rdd_world():
        world_crops = ("added" if run("python -m scripts.ingest_rdd2022_world", keep=["RDD world", "Error"]) == 0
                       else "FAILED - the candidate trains without them")
    print("other countries' crops:", world_crops)
    if not ok:
        STATUS["classifier"] = "data could not be prepared - see the lines above"
    else:
        CAND = f"{D}/classifier_candidate"
        os.makedirs(CAND, exist_ok=True)
        step("4b. training the candidate classifier (each finished architecture is kept in Drive)")
        code = run(f'python -m training.train_finetune_cnn --archs {CLS_ARCHS} --epochs {CLS_EPOCHS} --batch 48 '
                   f'--max-per-class {CLS_MAX_PER_CLASS} --out "{CAND}" --resume',
                   keep=["[finetune]", "already trained", "epoch", "TEST", "VAL", "Indian", "onnx", "Error"])
        if code == 0:
            step("4c. candidate or current classifier? (decided on half of the Indian held-out crops)")
            sel_code = run(f'python -m scripts.select_vision_candidate --candidate "{CAND}"', keep=None)
            rec = load_json("checkpoints/vision_candidate_selection.json") or {}
        if code == 0 and sel_code != 0:
            STATUS["classifier"] = f"selection failed (exit {sel_code}) - see the lines above; run the cell again"
        elif code == 0:
            names = ["vision_candidate_selection.json"]
            if rec.get("winner") == "candidate":
                names += [os.path.basename(p) for p in glob.glob("checkpoints/deep_vision_*")
                          + glob.glob("checkpoints/finetune_*.json")] + ["vision_model_selection.json"]
            for f in names:
                if os.path.exists(f"checkpoints/{f}"):
                    shutil.copy(f"checkpoints/{f}", f"{RESULTS}/{f}")
            save_json(f"{RESULTS}/classifier_done.json", {"finished": time.strftime("%Y-%m-%d %H:%M"),
                                                          "winner": rec.get("winner"), "why": rec.get("why"),
                                                          "other_countries_crops": world_crops})
            STATUS["classifier"] = f"done - {rec.get('winner')} serves ({rec.get('why')})"
        else:
            STATUS["classifier"] = f"stopped (exit {code}) - run the cell again; finished architectures are kept"

# ---------------------------------------------------------------- 5. measurements
meas_done = load_json(f"{RESULTS}/measure_done.json")
if MEASURE and meas_done:
    print(f"\nMeasurements already finished in this run ({meas_done.get('finished')}); skipping.")
    STATUS["measure"] = "done earlier"
elif MEASURE and any(STATUS.get(k, "done").startswith(("paused", "waiting", "stopped")) for k in ("unet", "yolo_world", "classifier")):
    print("\nMeasurements wait for the training stages. Run this cell again.")
    STATUS["measure"] = "waiting for earlier stages"
elif MEASURE:
    step("5a. privacy redactor recall (WIDER FACE validation + licence plates)")
    run(f"{sys.executable} -m pip -q install huggingface_hub")
    red = run("python -m scripts.measure_redactor_recall", keep=["[redaction]", "Error", "WARNING"])
    step("5b. classifier confidence calibration (Indian held-out crops)")
    cal = 1
    if ensure_rdd_india() and (os.path.isdir("datasets/_eval_rdd2022_india")
                               or run("python -m scripts.ingest_rdd2022_india", keep=["RDD India", "Error"]) == 0):
        cal = run("python -m scripts.measure_calibration --apply", keep=["[calibration]", "Error"])
    names = ["privacy_redaction_report.json", "vision_calibration_report.json"]
    names += [os.path.basename(p) for p in glob.glob("checkpoints/deep_vision_*.json")]
    for f in names:
        if os.path.exists(f"checkpoints/{f}"):
            shutil.copy(f"checkpoints/{f}", f"{RESULTS}/{f}")
    if red == 0 and cal == 0:
        save_json(f"{RESULTS}/measure_done.json", {"finished": time.strftime("%Y-%m-%d %H:%M")})
        STATUS["measure"] = "done"
    else:
        STATUS["measure"] = f"failed (redaction exit {red}, calibration exit {cal}) - see above; run the cell again"

# ---------------------------------------------------------------- 6. results
step("6. results")
files = [f for f in sorted(os.listdir(RESULTS)) if not f.endswith(".pt")]
print("finished files in Drive:", files)
everything_done = all(STATUS.get(k, "done").startswith("done") for k, on in
                      (("unet", TRAIN_UNET), ("yolo", TRAIN_YOLO), ("yolo_world", TRAIN_YOLO_WORLD),
                       ("classifier", TRAIN_CLASSIFIER), ("measure", MEASURE)) if on)
zip_path = (f"/content/road_shield_{RUN_NAME}_results.zip" if os.path.isdir("/content")
            else f"/tmp/road_shield_{RUN_NAME}_results.zip")
if everything_done and files:
    # same layout as the repository, so the laptop unzips it straight into SIH_PROJECT
    stage = "/tmp/rs_zip"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(f"{stage}/checkpoints", exist_ok=True)
    os.makedirs(f"{stage}/datasets/seg_multi", exist_ok=True)
    for f in files:
        if f == "seg_multi_manifest.json":
            shutil.copy(f"{RESULTS}/{f}", f"{stage}/datasets/seg_multi/manifest.json")
        elif not f.endswith("_done.json"):
            shutil.copy(f"{RESULTS}/{f}", f"{stage}/checkpoints/{f}")
    run(f"rm -f {zip_path} && cd {stage} && zip -qr {zip_path} . && ls -la {zip_path}")
    if os.path.exists(zip_path):
        shutil.copy(zip_path, f"{D}/")
if ON_DRIVE:
    try:
        drive.flush_and_unmount()          # make sure the last checkpoint has reached Drive before the runtime ends
        print("Drive flushed: everything above is saved.")
    except Exception as e:
        print("Drive flush:", e)
print("\n" + "#" * 72)
print("STATUS:", STATUS, f"| this session {(time.time() - T0) / 60:.0f} min")
if everything_done:
    print(f"ALL TRAINING FINISHED. Results zip: {zip_path} (a copy is in Drive: {D}).")
    try:
        from google.colab import files as _f
        _f.download(zip_path)
    except Exception:
        pass
else:
    print("NOT FINISHED YET - nothing is lost. Start a new Colab session (T4 GPU) and run this same cell again.")
print("#" * 72)
