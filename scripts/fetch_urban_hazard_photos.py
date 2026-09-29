"""
Expands the four rare urban-safety road classes (09_waterlogging_hazard,
10_missing_zebra_crossing, 11_missing_road_divider, 12_damaged_traffic_signs)
with real field photographs from Wikimedia Commons via the MediaWiki API,
deduplicated against the entire existing corpus using ForensicDuplicateHasher
(64-bit DCT perceptual hash).

New photographs are saved as `WM_<class>_<id>_RAW.png` inside each class's
`real_images/` directory so they are automatically included in
`data/image_dataset.py`, `training/train_cnn_head.py`, and
`training/train_vision.py`, while keeping the `.jpg` segmenter evaluation
fingerprint in `pipeline/corpus_fingerprint.py` stable.

Run directly:
    python -m scripts.fetch_urban_hazard_photos
"""

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.image_dataset import CLASS_FOLDERS, DATASETS_ROOT, _list_photos
from models.forensic_audit_engine import ForensicDuplicateHasher

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ROAD-SHIELD-ResearchBot/1.0 (https://github.com/road-shield; academic road safety dataset)"

CLASS_QUERIES = {
    3: (
        "09_waterlogging_hazard",
        "WM_09_waterlogging",
        [
            "flooded road India",
            "waterlogged street road",
            "flooded street cars road",
            "water on road flooding",
            "monsoon flooded road",
            "standing water road puddle street",
        ],
    ),
    4: (
        "10_missing_zebra_crossing",
        "WM_10_zebra",
        [
            "pedestrian crossing road asphalt",
            "zebra crossing street road",
            "faded pedestrian crossing road",
            "crosswalk road street view",
            "zebra crossing India road",
            "unmarked pedestrian crossing street",
        ],
    ),
    5: (
        "11_missing_road_divider",
        "WM_11_divider",
        [
            "road median barrier highway",
            "concrete road divider street",
            "damaged guardrail road",
            "highway median divider India",
            "central reservation road barrier",
            "road jersey barrier street",
        ],
    ),
    6: (
        "12_damaged_traffic_signs",
        "WM_12_sign",
        [
            "damaged road sign",
            "bent street sign road",
            "vandalized traffic sign",
            "rusty road sign street",
            "broken traffic sign post",
            "old weathered road sign",
        ],
    ),
}


def build_existing_hash_index(hasher):
    """Computes perceptual hashes of all existing photos across all 7 classes."""
    hashes = []
    for cls_id, (folder, _name) in CLASS_FOLDERS.items():
        for path in _list_photos(folder, dedupe_augmented=True):
            try:
                with Image.open(path) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
                hashes.append(hasher.compute_hash(arr))
            except Exception:
                continue
    return hashes


def is_near_duplicate(hasher, candidate_hash, existing_hashes, max_hamming=8):
    for h in existing_hashes:
        if hasher.hamming_distance(candidate_hash, h) <= max_hamming:
            return True
    return False


def search_commons(query, limit=25):
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|dimensions",
        "iiurlwidth": "640",
        "format": "json",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    pages = (data.get("query") or {}).get("pages") or {}
    results = []
    for page_id, page in pages.items():
        title = page.get("title", "")
        lower_t = title.lower()
        if any(bad in lower_t for bad in (".svg", ".gif", ".tif", "icon", "diagram", "map", "logo", "chart", "cartoon")):
            continue
        ii = (page.get("imageinfo") or [{}])[0]
        thumb = ii.get("thumburl") or ii.get("url")
        w = int(ii.get("thumbwidth") or ii.get("width") or 0)
        h = int(ii.get("thumbheight") or ii.get("height") or 0)
        if thumb and w >= 240 and h >= 180:
            results.append((title, thumb))
    return results


def download_valid_rgb(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    with Image.open(io.BytesIO(raw)) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if w < 220 or h < 160:
            return None
        if max(w, h) > 640:
            scale = 640.0 / float(max(w, h))
            rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.uint8)
        # Reject near-uniform graphics/blank frames (must have natural scene texture)
        if float(arr.std()) < 22.0:
            return None
        return rgb, arr


def expand_classes(target_new_per_class=35):
    hasher = ForensicDuplicateHasher(hash_size=8)
    print("[Urban Hazard Fetch] Indexing existing dataset photographs with 64-bit pHash ...")
    existing_hashes = build_existing_hash_index(hasher)
    print(f"  indexed {len(existing_hashes)} distinct existing photographs.")

    summary = {}
    for cls_id, (folder, prefix, queries) in sorted(CLASS_QUERIES.items()):
        out_dir = os.path.join(DATASETS_ROOT, folder, "real_images")
        os.makedirs(out_dir, exist_ok=True)
        existing_in_folder = len(_list_photos(folder, dedupe_augmented=True))
        added = 0
        seen_urls = set()

        # Count already-downloaded WM_ files so re-runs are idempotent
        already_wm = [f for f in os.listdir(out_dir) if f.startswith(prefix) and f.endswith(".png")]
        if len(already_wm) >= target_new_per_class:
            print(f"  [{folder}] already has {len(already_wm)} Wikimedia photos; skipping.")
            summary[folder] = len(_list_photos(folder, dedupe_augmented=True))
            continue

        next_idx = len(already_wm) + 1
        print(f"  [{folder}] starting with {existing_in_folder} photos; fetching up to {target_new_per_class} new ...")

        for q in queries:
            if added >= target_new_per_class:
                break
            try:
                candidates = search_commons(q, limit=30)
            except Exception as e:
                print(f"    search error for '{q}': {e}")
                continue

            for title, url in candidates:
                if added >= target_new_per_class:
                    break
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                try:
                    res = download_valid_rgb(url)
                    if res is None:
                        continue
                    pil_img, arr = res
                    ph = hasher.compute_hash(arr)
                    if is_near_duplicate(hasher, ph, existing_hashes, max_hamming=8):
                        continue
                    fname = f"{prefix}_{next_idx:03d}_RAW.png"
                    save_path = os.path.join(out_dir, fname)
                    pil_img.save(save_path, format="PNG")
                    existing_hashes.append(ph)
                    added += 1
                    next_idx += 1
                except Exception:
                    continue
                time.sleep(0.08)

        total_now = len(_list_photos(folder, dedupe_augmented=True))
        summary[folder] = total_now
        print(f"  [{folder}] +{added} new real photos -> {total_now} distinct photos total.")

    return summary


if __name__ == "__main__":
    expand_classes(target_new_per_class=35)
