"""
Pull real road-condition datasets from Kaggle into this project's class folders.

    python -m scripts.fetch_kaggle_datasets --verify          # what's reachable, no download
    python -m scripts.fetch_kaggle_datasets --plan            # download, show the mapping, copy nothing
    python -m scripts.fetch_kaggle_datasets                   # download and ingest the catalogue
    python -m scripts.fetch_kaggle_datasets --only surface-crack pothole-mix
    python -m scripts.fetch_kaggle_datasets --search "road damage"
    python -m scripts.fetch_kaggle_datasets --from-dir some/folder --dataset-key manual
    python -m scripts.fetch_kaggle_datasets --undo            # remove everything this script added

Credentials, in the order they are tried:
    1. ~/.kaggle/kaggle.json                 username + key pair (most reliable)
    2. KAGGLE_USERNAME + KAGGLE_KEY          the same pair, from the environment
    3. KAGGLE_API_TOKEN, or ~/.kaggle/access_token
                                             the newer "KGAT_..." bearer token;
                                             only some versions of the kaggle
                                             package accept it, so this script
                                             says so plainly rather than failing
                                             with an opaque 401

Nothing here invents data. A dataset that will not download is reported as
unreachable and skipped; the run continues with the ones that did.

------------------------------------------------------------------------------
How images are mapped to classes
------------------------------------------------------------------------------
Kaggle datasets disagree about layout, so the mapping is driven by the folder
names that actually appear after extraction, not by a layout assumed in
advance. A directory called `potholes/`, `Positive/` or `good_road/` says what
its images are; FOLDER_RULES turns those words into one of this project's seven
classes. A dataset whose layout needs something more specific gets an explicit
override in the catalogue.

`--plan` prints the inferred mapping and the per-class counts and then stops,
so the mapping can be checked before a single file is copied. Use it first.

------------------------------------------------------------------------------
Duplicate control
------------------------------------------------------------------------------
Every candidate image is perceptually hashed (DCT pHash) and rejected if it is
a near-duplicate of an image already in the dataset or of one accepted earlier
in the same run. This matters more than it sounds: several Kaggle pothole
datasets are re-uploads of each other, and without this the same photograph
would land in training and test and inflate the score by tens of points.

------------------------------------------------------------------------------
Honest caveats, which belong in the model card too
------------------------------------------------------------------------------
- Surface-crack datasets are photographed on concrete at close range. They are
  real crack images, but they are not road-surface photographs, and a model
  trained on them will do better on crack texture than on road scenes. The
  catalogue marks these `domain="concrete"` and the ingest report records how
  many images came from each domain.
- Kaggle licences vary per dataset. `--verify` prints each licence. Check them
  before publishing results; this script does not decide that for you.
- REMOVED, and worth explaining: andrewmvd/road-sign-detection was in this
  catalogue mapped to the Damaged Traffic Sign class. It is a dataset of road
  signs, not of DAMAGED road signs. Training on it taught the model "a sign is
  present", while the class it was filed under claims "this sign is damaged" -
  so a photograph of a perfectly good sign came back labelled damaged, with
  high confidence. The measured F1 of 0.98 was real and measured the wrong
  thing. There is no substitute for a dataset of the condition you actually
  want to classify, and a class must not be filled with the nearest available
  images just because the folder has a similar name.
"""

import argparse
import json
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import numpy as np
from PIL import Image

from data.image_dataset import CLASS_FOLDERS, DATASETS_ROOT
from models.forensic_audit_engine import ForensicDuplicateHasher

NORMAL, CRACK, POTHOLE, WATER, ZEBRA, DIVIDER, SIGN = range(7)

CACHE_DIR = os.path.join(DATASETS_ROOT, "incoming", "kaggle")
MANIFEST = os.path.join(DATASETS_ROOT, "incoming", "kaggle_ingest_manifest.json")
PREFIX = "kag_"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MIN_SIDE = 48

# ---------------------------------------------------------------------------
# Catalogue. `slug` is the Kaggle dataset id; `folder_overrides` wins over
# FOLDER_RULES when a dataset's directory names are ambiguous.
# Run --verify before trusting any entry: Kaggle datasets get renamed and
# deleted, and this script reports what the API actually returns rather than
# assuming the catalogue is current.
# ---------------------------------------------------------------------------
CATALOG = [
    {
        "key": "surface-crack",
        "slug": "arunrk7/surface-crack-detection",
        "about": "Concrete surface crack images, 227x227, split Positive / Negative",
        "domain": "concrete",
        "cap": {CRACK: 1200, NORMAL: 1200},
        "folder_overrides": {"positive": CRACK, "negative": NORMAL},
    },
    {
        "key": "pothole-mix",
        "slug": "atulyakumar98/pothole-detection-dataset",
        "about": "Road photographs split into normal road and potholes",
        "domain": "road",
        "cap": {POTHOLE: 800, NORMAL: 800},
        "folder_overrides": {"potholes": POTHOLE, "pothole": POTHOLE, "normal": NORMAL},
    },
    {
        "key": "pothole-annotated",
        "slug": "andrewmvd/pothole-detection",
        "about": "Road images with PASCAL VOC pothole boxes (images used whole)",
        "domain": "road",
        "cap": {POTHOLE: 600},
        "folder_overrides": {"images": POTHOLE, "annotations": None},
    },
    {
        "key": "pothole-plain",
        "slug": "virenbr11/pothole-and-plain-rode-images",
        "about": "Pothole and plain road photographs",
        "domain": "road",
        "cap": {POTHOLE: 600, NORMAL: 600},
        "folder_overrides": {"pothole": POTHOLE, "plain": NORMAL, "normal": NORMAL},
    },
]

# folder name fragment -> class. Checked longest-first against the lowercased
# path of every image, so `.../train/potholes/x.jpg` resolves on "pothole".
FOLDER_RULES = [
    ("waterlog", WATER), ("flood", WATER), ("standing_water", WATER),
    ("zebra", ZEBRA), ("crosswalk", ZEBRA), ("pedestrian_crossing", ZEBRA),
    ("divider", DIVIDER), ("median", DIVIDER), ("guardrail", DIVIDER),
    ("damaged_sign", SIGN), ("traffic_sign", SIGN), ("signboard", SIGN), ("road_sign", SIGN),
    ("pothole", POTHOLE), ("potholes", POTHOLE), ("cavity", POTHOLE),
    ("alligator", CRACK), ("crack", CRACK), ("cracked", CRACK), ("fissure", CRACK),
    ("positive", CRACK),
    ("plain", NORMAL), ("normal", NORMAL), ("good", NORMAL), ("smooth", NORMAL),
    ("undamaged", NORMAL), ("negative", NORMAL), ("no_pothole", NORMAL),
]
FOLDER_RULES.sort(key=lambda kv: -len(kv[0]))


# ---------------------------------------------------------------------------
# Credentials and the Kaggle API
# ---------------------------------------------------------------------------
def kaggle_config_dir():
    return os.environ.get("KAGGLE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".kaggle")


def kaggle_credentials():
    """
    Where the credentials came from, or None.

    Never prints or returns the secret itself - only which file or variable it
    was found in. A token that ends up in a log is a token that has to be
    revoked.
    """
    cfg = kaggle_config_dir()
    path = os.path.join(cfg, "kaggle.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            if blob.get("username") and blob.get("key"):
                os.environ.setdefault("KAGGLE_USERNAME", blob["username"])
                os.environ.setdefault("KAGGLE_KEY", blob["key"])
                return f"{path} (user {blob['username']})"
        except Exception as e:
            print(f"[kaggle] {path} exists but could not be read: {e}")

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return f"environment KAGGLE_USERNAME/KAGGLE_KEY (user {os.environ['KAGGLE_USERNAME']})"

    # Newer bearer token ("KGAT_..."). Some kaggle package versions read it,
    # others only understand the username/key pair, so say which one this is.
    token_file = os.path.join(cfg, "access_token")
    if not os.environ.get("KAGGLE_API_TOKEN") and os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as fh:
                token = fh.read().strip()
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
        except Exception as e:
            print(f"[kaggle] {token_file} exists but could not be read: {e}")
    if os.environ.get("KAGGLE_API_TOKEN"):
        return f"bearer token ({token_file if os.path.exists(token_file) else 'KAGGLE_API_TOKEN'})"
    return None


def kaggle_api():
    """The authenticated Kaggle client, or None with the reason printed."""
    where = kaggle_credentials()
    if not where:
        print(f"[kaggle] No credentials found.\n"
              f"         kaggle.com -> your avatar -> Settings -> API -> Create New Token,\n"
              f"         then save the downloaded kaggle.json in {kaggle_config_dir()} and rerun.")
        return None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[kaggle] The kaggle package is not installed.  pip install kaggle")
        return None
    try:
        api = KaggleApi()
        api.authenticate()
        print(f"[kaggle] authenticated from {where}")
        return api
    except Exception as e:
        print(f"[kaggle] authentication failed: {str(e)[:200]}")
        if where.startswith("bearer token"):
            print("[kaggle] That was a KGAT_... bearer token. This version of the kaggle\n"
                  "         package wants the username/key pair instead: on the same API\n"
                  "         settings page use Create New Token, which downloads kaggle.json,\n"
                  f"         and save it as {os.path.join(kaggle_config_dir(), 'kaggle.json')}")
        return None


def verify(api, entries):
    """Ask Kaggle what each catalogue entry actually is. Downloads nothing."""
    rows = []
    for entry in entries:
        row = {"key": entry["key"], "slug": entry["slug"], "reachable": False}
        try:
            owner, name = entry["slug"].split("/", 1)
            matches = api.dataset_list(search=name, user=owner)
            hit = next((d for d in matches if str(d.ref).lower() == entry["slug"].lower()), None)
            if hit is None:
                matches = api.dataset_list(search=entry["slug"])
                hit = next((d for d in matches if str(d.ref).lower() == entry["slug"].lower()), None)
            if hit is not None:
                row.update(reachable=True,
                           title=str(getattr(hit, "title", "")),
                           size=str(getattr(hit, "size", "")),
                           licence=str(getattr(hit, "licenseName", "") or ""),
                           files=getattr(hit, "totalBytes", None))
            else:
                row["error"] = "not found via the Kaggle search API"
        except Exception as e:
            row["error"] = str(e)[:200]
        rows.append(row)
        mark = "ok " if row["reachable"] else "MISS"
        print(f"  {mark} {row['key']:20s} {row['slug']:45s} "
              f"{row.get('size', ''):>8s}  {row.get('licence', row.get('error', ''))[:40]}")
    return rows


def search(api, term, limit=20):
    print(f"[kaggle] searching for {term!r}")
    for d in api.dataset_list(search=term)[:limit]:
        print(f"  {str(d.ref):50s} {str(getattr(d, 'size', '')):>9s}  {str(getattr(d, 'title', ''))[:50]}")


def download(api, entry, force=False):
    """Download and unzip one dataset. Returns its extracted directory, or None."""
    target = os.path.join(CACHE_DIR, entry["key"])
    marker = os.path.join(target, ".complete")
    if os.path.exists(marker) and not force:
        print(f"  {entry['key']}: already downloaded")
        return target
    os.makedirs(target, exist_ok=True)
    print(f"  {entry['key']}: downloading {entry['slug']} ...", flush=True)
    t0 = time.time()
    try:
        api.dataset_download_files(entry["slug"], path=target, unzip=True, quiet=False)
    except Exception as e:
        print(f"  {entry['key']}: download failed - {str(e)[:200]}")
        return None
    # some versions leave the zip behind
    for name in os.listdir(target):
        if name.endswith(".zip"):
            try:
                with zipfile.ZipFile(os.path.join(target, name)) as zf:
                    zf.extractall(target)
                os.remove(os.path.join(target, name))
            except Exception as e:
                print(f"  {entry['key']}: could not unzip {name}: {e}")
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(str(int(time.time())))
    print(f"  {entry['key']}: done in {time.time() - t0:.0f}s")
    return target


# ---------------------------------------------------------------------------
# Mapping and ingest
# ---------------------------------------------------------------------------
def classify_path(rel_path, overrides):
    """Which class this file belongs to, from the folders above it. None = skip."""
    parts = [p.lower() for p in rel_path.replace("\\", "/").split("/")[:-1]]
    for part in reversed(parts):
        if part in overrides:
            return overrides[part]
    joined = "/".join(parts)
    for fragment, cls in FOLDER_RULES:
        if fragment in joined:
            return cls
    return None


def walk_images(root):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(IMG_EXT):
                full = os.path.join(dirpath, name)
                yield full, os.path.relpath(full, root)


def class_dir(cls):
    return os.path.join(DATASETS_ROOT, CLASS_FOLDERS[cls][0], "real_images")


class NearDuplicateIndex:
    """
    Perceptual hashes with a Hamming-distance lookup.

    An exact-match set would only catch byte-identical re-encodes. The problem
    here is re-uploads: the same photograph, resized and re-compressed, appears
    in several Kaggle pothole datasets. Those differ in a handful of hash bits,
    so membership has to be "within `threshold` bits of anything seen", which
    is a distance query, not a set lookup.

    The hashes are kept as one bool matrix so each query is a single vectorised
    XOR-and-count over every hash so far - fast enough for tens of thousands.
    """

    # Threshold measured, not guessed. Over 60 photographs from this dataset,
    # 64-bit pHash distance between an image and a copy of it was:
    #   re-encoded at JPEG q80          max  2
    #   half size, q55 (typical re-up)  max  8, 98% within 6
    #   quarter size, q45               max 12
    # while distinct photographs sat at 18 bits and above (5th percentile).
    # 8 therefore catches every half-size re-upload measured and still leaves a
    # 10-bit margin before it would start merging genuinely different photos.
    def __init__(self, hasher, threshold=8, capacity=60000):
        self.hasher = hasher
        self.threshold = int(threshold)
        self._matrix = None
        self._n = 0
        self._capacity = capacity

    def __len__(self):
        return self._n

    def hash_image(self, img):
        return self.hasher.compute_hash(img)

    def contains(self, digest):
        if self._n == 0:
            return False
        return bool((np.count_nonzero(self._matrix[:self._n] != digest, axis=1)
                     <= self.threshold).any())

    def add(self, digest):
        if self._matrix is None:
            self._matrix = np.zeros((1024, digest.size), dtype=bool)
        if self._n >= self._matrix.shape[0]:
            if self._n >= self._capacity:
                return  # index is full; later images are simply not compared against
            grow = np.zeros((min(self._matrix.shape[0], self._capacity - self._n),
                             digest.size), dtype=bool)
            self._matrix = np.concatenate([self._matrix, grow], axis=0)
        self._matrix[self._n] = digest
        self._n += 1


def index_existing(index, classes, limit_per_class=4000):
    """Hash what is already on disk, so new images can't duplicate it."""
    for cls in classes:
        folder = os.path.join(DATASETS_ROOT, CLASS_FOLDERS[cls][0])
        count = 0
        for dirpath, _d, files in os.walk(folder):
            for name in files:
                if count >= limit_per_class:
                    break
                if not name.lower().endswith(IMG_EXT):
                    continue
                try:
                    with Image.open(os.path.join(dirpath, name)) as im:
                        index.add(index.hash_image(im.convert("RGB")))
                    count += 1
                except Exception:
                    continue
    return index


def ingest(entries_dirs, plan_only=False, seed=42):
    hasher = ForensicDuplicateHasher()
    # Hash every class we might write into. With no caps declared (--from-dir),
    # that is all of them - skipping this would let a duplicate of an image
    # already in the dataset through, which is exactly the leak this guards.
    all_classes = sorted({c for _k, _d, caps in entries_dirs for c in caps}) or sorted(CLASS_FOLDERS)
    print(f"\n[ingest] hashing existing images in {len(all_classes)} class folders ...", flush=True)
    known = index_existing(NearDuplicateIndex(hasher), all_classes)
    print(f"[ingest] {len(known)} existing images hashed")

    rng = np.random.default_rng(seed)
    written = []
    stats = defaultdict(Counter)

    for key, root, caps in entries_dirs:
        entry = next((e for e in CATALOG if e["key"] == key), {})
        overrides = {k.lower(): v for k, v in (entry.get("folder_overrides") or {}).items()}
        domain = entry.get("domain", "unknown")

        candidates = defaultdict(list)
        unmapped = Counter()
        for full, rel in walk_images(root):
            cls = classify_path(rel, overrides)
            if cls is None:
                unmapped[os.path.dirname(rel).split(os.sep)[0] or "."] += 1
                continue
            candidates[cls].append((full, rel))

        print(f"\n[{key}] {entry.get('about', '')}")
        print(f"  domain: {domain}")
        for cls, items in sorted(candidates.items()):
            cap = caps.get(cls)
            print(f"  {CLASS_FOLDERS[cls][1][:40]:42s} found {len(items):6d}"
                  + (f"   cap {cap}" if cap else ""))
        for folder, n in unmapped.most_common(6):
            print(f"  (unmapped) {folder[:40]:42s} {n:6d} files")

        if plan_only:
            continue

        for cls, items in sorted(candidates.items()):
            cap = caps.get(cls)
            if cap and len(items) > cap:
                idx = sorted(rng.choice(len(items), cap, replace=False))
                items = [items[i] for i in idx]
            out_dir = class_dir(cls)
            os.makedirs(out_dir, exist_ok=True)
            kept = dup = bad = 0
            for full, rel in items:
                try:
                    with Image.open(full) as im:
                        im = im.convert("RGB")
                        if min(im.size) < MIN_SIDE:
                            bad += 1
                            continue
                        digest = known.hash_image(im)
                        if known.contains(digest):
                            dup += 1
                            continue
                        known.add(digest)
                        stem = os.path.splitext(os.path.basename(rel))[0][:40]
                        name = f"{PREFIX}{key}_{cls}_{kept:05d}_{stem}.jpg".replace(" ", "_")
                        dest = os.path.join(out_dir, name)
                        im.save(dest, "JPEG", quality=92)
                except Exception:
                    bad += 1
                    continue
                written.append({"path": dest, "class": cls, "source_key": key,
                                "source_rel": rel, "domain": domain})
                kept += 1
            stats[key][CLASS_FOLDERS[cls][1]] = kept
            print(f"  -> {CLASS_FOLDERS[cls][1][:38]:40s} kept {kept:5d}   "
                  f"duplicates skipped {dup:5d}   unreadable {bad}")

    return written, stats


def save_manifest(written, extra):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    blob = {"written_unix": int(time.time()), "count": len(written),
            "files": written, **extra}
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nmanifest -> {MANIFEST}")


def undo():
    if not os.path.exists(MANIFEST):
        sys.exit("No manifest - this script has not added anything to undo.")
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    removed = 0
    for rec in blob.get("files", []):
        path = rec["path"]
        if os.path.exists(path):
            os.remove(path)
            removed += 1
    os.remove(MANIFEST)
    print(f"Removed {removed} of {blob.get('count', 0)} ingested images.")
    print("Downloaded archives are left in datasets/incoming/kaggle/ - delete that "
          "folder by hand if you want the disk space back.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="check each catalogue entry, download nothing")
    ap.add_argument("--plan", action="store_true", help="download, print the mapping, copy nothing")
    ap.add_argument("--only", nargs="*", default=None, help="catalogue keys to use")
    ap.add_argument("--search", default=None, help="search Kaggle and exit")
    ap.add_argument("--from-dir", default=None, help="ingest an already-extracted folder instead")
    ap.add_argument("--dataset-key", default="manual", help="key to record for --from-dir")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--undo", action="store_true", help="remove every image this script added")
    ap.add_argument("--list", action="store_true", help="print the catalogue and exit")
    args = ap.parse_args()

    if args.undo:
        return undo()

    if args.list:
        for e in CATALOG:
            print(f"  {e['key']:20s} {e['slug']:45s} {e['domain']:9s} {e['about']}")
        return

    if args.from_dir:
        root = os.path.abspath(args.from_dir)
        if not os.path.isdir(root):
            sys.exit(f"Not a directory: {root}")
        written, stats = ingest([(args.dataset_key, root, {})], plan_only=args.plan)
        if not args.plan:
            save_manifest(written, {"source": "local", "dir": root})
        return

    entries = CATALOG if not args.only else [e for e in CATALOG if e["key"] in args.only]
    if args.only and not entries:
        sys.exit(f"No catalogue entry named {args.only}. Use --list.")

    api = kaggle_api()
    if api is None:
        sys.exit(1)

    if args.search:
        return search(api, args.search)

    print(f"\n[kaggle] checking {len(entries)} datasets")
    rows = verify(api, entries)
    if args.verify:
        return

    reachable = [e for e, r in zip(entries, rows) if r["reachable"]]
    if not reachable:
        sys.exit("\nNone of the catalogue datasets are reachable with these credentials. "
                 "Use --search to find current alternatives, then --from-dir to ingest one.")

    print(f"\n[kaggle] downloading {len(reachable)} datasets into {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    prepared = []
    for entry in reachable:
        path = download(api, entry, force=args.force)
        if path:
            prepared.append((entry["key"], path, entry.get("cap", {})))

    if not prepared:
        sys.exit("Nothing downloaded.")

    written, stats = ingest(prepared, plan_only=args.plan)
    if args.plan:
        print("\n--plan: nothing was copied. Rerun without --plan to ingest.")
        return

    save_manifest(written, {"source": "kaggle",
                            "datasets": {e["key"]: e["slug"] for e in reachable},
                            "per_dataset_counts": {k: dict(v) for k, v in stats.items()}})
    total = sum(sum(v.values()) for v in stats.values())
    print(f"\n{total} images added. Retrain:\n"
          f"  python -m training.train_mega_suite\n"
          f"  python -m training.train_cnn_head --compare")


if __name__ == "__main__":
    main()
