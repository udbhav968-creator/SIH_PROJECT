"""
Resolve photographs that were filed under more than one class.

An audit of datasets/ found source photographs present in two or three class
folders at once - the same picture labelled "Normal Road", "Crack" and
"Pothole" simultaneously. A classifier cannot learn from contradictory
labels, and every such photograph also pollutes whichever split it lands in.

    python -m scripts.fix_label_conflicts --dry-run     # show what would move
    python -m scripts.fix_label_conflicts               # move them
    python -m scripts.fix_label_conflicts --restore     # put everything back

Resolution rule, applied to the conflicting copies only:

  * present in "Normal Road" and in a distress class -> keep Normal Road.
    Inspection of these photographs shows intact pavement; they appear to
    have been copied into the distress folders in error.
  * present in "Crack" and "Pothole" -> keep Crack. Inspection shows surface
    cracking and patching without an open cavity.
  * any other combination -> quarantine every copy, because there is no
    defensible way to choose.

Nothing is deleted. Losing copies move to datasets/_label_conflicts/, and
--restore puts them back exactly where they were. Every decision, with file
lists, is written to datasets/_label_conflicts/manifest.json so the choice is
auditable rather than silent.
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from data.image_dataset import CLASS_FOLDERS, CLASS_NAMES, DATASETS_ROOT

AUG_RE = re.compile(r"^aug_mega_\d+_")
QUARANTINE = os.path.join(DATASETS_ROOT, "_label_conflicts")
MANIFEST = os.path.join(QUARANTINE, "manifest.json")
NORMAL_ROAD, CRACK, POTHOLE = 0, 1, 2


def source_stem(filename):
    stem = os.path.splitext(AUG_RE.sub("", filename))[0]
    for suffix in ("_RAW", "_CRACK", "_POTHOLE", "_LANE"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def index_dataset():
    """{source stem: {class id: [file paths]}}"""
    index = defaultdict(lambda: defaultdict(list))
    for cls, (folder, _name) in CLASS_FOLDERS.items():
        d = os.path.join(DATASETS_ROOT, folder, "real_images")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                continue
            index[source_stem(f)][cls].append(os.path.join(d, f))
    return index


def decide(classes):
    """Which class keeps the photograph, and why. None means quarantine all."""
    classes = set(classes)
    if NORMAL_ROAD in classes:
        return NORMAL_ROAD, "intact pavement on inspection; copies in distress folders look like a filing error"
    if classes == {CRACK, POTHOLE}:
        return CRACK, "surface cracking and patching visible, no open cavity"
    return None, "no defensible way to choose between these labels"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.restore:
        if not os.path.exists(MANIFEST):
            sys.exit("Nothing to restore - no manifest found.")
        with open(MANIFEST, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        restored = 0
        for entry in manifest["moved"]:
            src = os.path.join(ENGINE_ROOT, entry["quarantined_to"])
            dst = os.path.join(ENGINE_ROOT, entry["original_path"])
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                restored += 1
        print(f"restored {restored} files to their original folders")
        return

    index = index_dataset()
    conflicts = {stem: cls_map for stem, cls_map in index.items() if len(cls_map) > 1}
    if not conflicts:
        print("No photograph is filed under more than one class. Nothing to do.")
        return

    print(f"{len(conflicts)} source photographs are filed under more than one class:\n")
    moved, decisions = [], []
    for stem, cls_map in sorted(conflicts.items()):
        present = sorted(cls_map)
        keep, reason = decide(present)
        label = CLASS_NAMES[keep] if keep is not None else "NONE (all quarantined)"
        n_files = sum(len(v) for v in cls_map.values())
        print(f"  {stem[:52]:54s} in {[CLASS_NAMES[c].split(' ')[0] for c in present]} "
              f"-> keep {label.split(' ')[0]}  ({n_files} files)")
        decisions.append({"source_photo": stem, "found_in_classes": [CLASS_NAMES[c] for c in present],
                          "kept_class": label, "reason": reason, "files_affected": n_files})
        for cls, paths in cls_map.items():
            if cls == keep:
                continue
            for p in paths:
                rel = os.path.relpath(p, ENGINE_ROOT)
                dest = os.path.join(QUARANTINE, CLASS_FOLDERS[cls][0], os.path.basename(p))
                moved.append({"original_path": rel, "quarantined_to": os.path.relpath(dest, ENGINE_ROOT),
                              "removed_from_class": CLASS_NAMES[cls]})

    print(f"\nfiles to move out of their wrong folders: {len(moved)}")
    if args.dry_run:
        print("dry run - nothing moved. Re-run without --dry-run to apply.")
        return

    for entry in moved:
        src = os.path.join(ENGINE_ROOT, entry["original_path"])
        dst = os.path.join(ENGINE_ROOT, entry["quarantined_to"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            shutil.move(src, dst)

    os.makedirs(QUARANTINE, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump({"decisions": decisions, "moved": moved,
                   "note": "Run scripts/fix_label_conflicts.py --restore to undo."}, fh, indent=2)
    print(f"moved {len(moved)} files to {os.path.relpath(QUARANTINE, ENGINE_ROOT)}")
    print(f"manifest -> {os.path.relpath(MANIFEST, ENGINE_ROOT)}")
    print("\nNow retrain:  python -m training.train_mega_suite")


if __name__ == "__main__":
    main()
