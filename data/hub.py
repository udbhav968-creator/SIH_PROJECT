"""
Every dataset this project uses or can use, and one way to fetch any of them through its provider's API.

    python -m data.hub list                                   # the catalogue: task, licence, provider, key needed
    python -m data.hub search hf pothole                      # Hugging Face Hub dataset search
    python -m data.hub fetch crackseg9k                       # a catalogue entry
    python -m data.hub fetch hf:keremberke/pothole-segmentation
    python -m data.hub fetch zenodo:<record id>
    python -m data.hub fetch roboflow:<workspace>/<project>/<version> [--format coco-segmentation]
    python -m data.hub fetch git:https://github.com/<owner>/<repo>.git
    python -m data.hub fetch kaggle:<owner>/<dataset>
    python -m data.hub mapillary --bbox 77.58,12.96,77.61,12.99 --limit 200 [--queue]
    python -m data.hub verify datasets/incoming/<folder>      # files still match their manifest?

Where things go
  datasets/incoming/<provider>__<name>/   files as the provider ships them, plus MANIFEST.json:
      source, reference, URL, licence (read from the provider's own metadata where it has any), retrieval
      time, commit or version, SHA-256 and size of every file, and the leakage check below
  `python -m scripts.fetch_datasets ingest` then crops labelled boxes into the training folders, and
  `python -m scripts.fetch_seg_datasets` handles pixel-labelled sets; this module only fetches and records.

Leakage check (every fetch)
  Each image is perceptually hashed (64-bit dHash) and compared with the photographs this project measures
  its models on (scripts/fetch_seg_datasets.EVAL_FOLDERS). Near-duplicates (Hamming <= 6) are listed in the
  manifest and moved to _quarantine/ unless --keep-leaks, so an outside dataset that happens to contain a
  test photograph cannot inflate a score.

Keys come only from the environment (HF_TOKEN, ROBOFLOW_API_KEY, MAPILLARY_TOKEN, Kaggle's own files) and
are never written to a manifest or printed. Licences differ (several are non-commercial research only); the
manifest keeps them next to the files so nobody has to guess later.

Mapillary
  Street-level photographs (CC BY-SA 4.0, faces and plates already blurred by Mapillary) inside a bounding
  box: unlabelled, so they are not training data by themselves. With --queue each photograph is run through
  the pipeline and the informative ones go to the labelling queue on the MLOps page (mlops/active_learning.py,
  which blurs people and plates again before storing) - the loop by which real Indian street images become
  labelled training data.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCOMING = os.path.join(ROOT, "datasets", "incoming")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HF_API = "https://huggingface.co/api/datasets"
HF_FILE = "https://huggingface.co/datasets/{id}/resolve/{rev}/{path}"
ZENODO_API = "https://zenodo.org/api/records/{id}"
ROBOFLOW_API = "https://api.roboflow.com"
MAPILLARY_API = "https://graph.mapillary.com/images"

# What this project uses, and where each one comes from. "how" is the command that fetches it.
CATALOGUE = [
    {"name": "rdd2022_india", "task": "road-damage boxes D00/D10/D20/D40 (India)", "licence": "CC BY-SA 4.0",
     "provider": "CRDDC 2022 release", "how": "python -m scripts.fetch_rdd2022_india", "key": None,
     "used_by": "YOLOv8 / RT-DETR detector, classifier crops, Indian held-out test"},
    {"name": "dnit_cracks_potholes", "task": "crack and pothole polygons (Brazil, 2,235 photographs)",
     "licence": "see repository", "provider": "git",
     "ref": "https://github.com/andrijdavid/Cracks-and-Potholes-in-Road-Images-Dataset.git",
     "how": "python -m scripts.fetch_cracks_potholes_dataset", "key": None,
     "used_by": "pixel segmenter, U-Net, classifier crops"},
    {"name": "crackseg9k", "task": "9,159 crack masks from 10 datasets", "licence": "see dataset card",
     "provider": "hf", "ref": "rimvydasrub/crackseg9k", "how": "python -m scripts.fetch_seg_datasets --only crackseg9k",
     "key": None, "used_by": "U-Net (multi-source)"},
    {"name": "deepcrack", "task": "537 crack masks", "licence": "non-commercial research and education",
     "provider": "git", "ref": "https://github.com/yhlleo/DeepCrack.git",
     "how": "python -m scripts.fetch_seg_datasets --only deepcrack", "key": None,
     "used_by": "crack verifier, U-Net"},
    {"name": "crackforest", "task": "118 urban road crack masks (CFD)", "licence": "non-commercial research",
     "provider": "git", "ref": "https://github.com/cuilimeng/CrackForest-dataset.git",
     "how": "python -m scripts.fetch_seg_datasets --only crackforest", "key": None,
     "used_by": "crack verifier, U-Net"},
    {"name": "kaggle_pothole_segmentation", "task": "~780 pothole polygons", "licence": "see Kaggle page",
     "provider": "kaggle", "ref": "farzadnekouei/pothole-image-segmentation-dataset",
     "how": "python -m scripts.fetch_seg_datasets --only kaggle_pothole", "key": "Kaggle token",
     "used_by": "U-Net"},
    {"name": "kaggle_pothole_detection", "task": "pothole boxes", "licence": "see Kaggle page", "provider": "kaggle",
     "ref": "andrewmvd/pothole-detection", "how": "python -m scripts.fetch_datasets download andrewmvd/pothole-detection",
     "key": "Kaggle token", "used_by": "classifier crops"},
    {"name": "pothole_mix", "task": "4,340 pothole + crack masks", "licence": "see the Mendeley Data page",
     "provider": "manual", "ref": "https://data.mendeley.com/datasets/kfth5g2xk3/2",
     "how": "download in a browser, then python -m scripts.fetch_seg_datasets --pothole-mix-zip <zip>",
     "key": None, "used_by": "U-Net"},
    {"name": "pothole_segmentation_hf", "task": "pothole polygons (Roboflow export)", "licence": "see dataset card",
     "provider": "hf", "ref": "keremberke/pothole-segmentation", "how": "python -m data.hub fetch pothole_segmentation_hf",
     "key": None, "used_by": "candidate: U-Net / SegFormer"},
    {"name": "wider_face", "task": "face boxes", "licence": "see dataset card", "provider": "hf", "ref": "CUHK-CSE/wider_face",
     "how": "python -m scripts.measure_redactor_recall", "key": None, "used_by": "privacy-redactor recall"},
    {"name": "licence_plates", "task": "number-plate boxes", "licence": "see dataset card", "provider": "hf",
     "ref": "keremberke/license-plate-object-detection", "how": "python -m scripts.measure_redactor_recall",
     "key": None, "used_by": "privacy-redactor recall"},
    {"name": "imu_drive_logs", "task": "accelerometer drive logs with potholes (100 Hz)", "licence": "see repository",
     "provider": "git", "ref": "https://github.com/VishalSingh25/Pothole-Project.git",
     "how": "python -m scripts.fetch_real_imu_dataset", "key": None, "used_by": "IMU shock classifier"},
    {"name": "imagenet_samples", "task": "one everyday photograph per ImageNet class", "licence": "ImageNet terms",
     "provider": "git", "ref": "https://github.com/EliSchwartz/imagenet-sample-images.git",
     "how": "git clone --depth 1 https://github.com/EliSchwartz/imagenet-sample-images", "key": None,
     "used_by": "input guard (not-a-road examples)"},
    {"name": "mapillary_streets", "task": "unlabelled street-level photographs in a bounding box",
     "licence": "CC BY-SA 4.0", "provider": "mapillary", "ref": "graph.mapillary.com/images",
     "how": "python -m data.hub mapillary --bbox <minlon,minlat,maxlon,maxlat> --queue", "key": "MAPILLARY_TOKEN",
     "used_by": "labelling queue (active learning), monitoring reference"},
    {"name": "open_meteo_rain", "task": "daily rainfall history at a location", "licence": "CC BY 4.0",
     "provider": "api", "ref": "archive-api.open-meteo.com", "how": "live, services/road_context.py", "key": None,
     "used_by": "deterioration forecast input"},
    {"name": "osm_roads", "task": "road class, name, surface, lanes", "licence": "ODbL", "provider": "api",
     "ref": "overpass-api.de", "how": "live, services/road_context.py", "key": None, "used_by": "work-order context"},
]


class HubError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- http
def _ua():
    return f"ROAD-SHIELD/3.0 ({os.environ.get('ROAD_SHIELD_CONTACT', 'SIH 2026 student project')})"


class _SameHostAuthRedirect(urllib.request.HTTPRedirectHandler):
    """urllib copies every header, Authorization included, to wherever a redirect points. Hugging Face
    redirects file downloads to its CDN or to presigned storage: the token must not follow."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            for h in ("Authorization", "authorization"):
                new.headers.pop(h, None)
                new.unredirected_hdrs.pop(h, None)
        return new


_OPENER = urllib.request.build_opener(_SameHostAuthRedirect())


def _open(url, headers=None, timeout=30):
    return _OPENER.open(urllib.request.Request(url, headers={"User-Agent": _ua(), **(headers or {})}), timeout=timeout)


def http_json(url, headers=None, timeout=30):
    with _open(url, headers, timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_json_pages(url, headers=None, timeout=30, max_pages=200):
    """A paginated JSON list (Link: <...>; rel="next", as the Hugging Face Hub API pages long listings)."""
    out = []
    for _ in range(max_pages):
        with _open(url, headers, timeout) as r:
            out += json.loads(r.read().decode("utf-8"))
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        if not m:
            return out
        url = m.group(1)
    raise HubError("listing longer than expected; pass --pattern to narrow it")


def http_download(url, dest, headers=None, timeout=120):
    """Stream url to dest (atomically); returns (sha256, bytes)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h, n = hashlib.sha256(), 0
    tmp = dest + ".part"
    with _open(url, headers, timeout) as r, open(tmp, "wb") as fh:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            fh.write(chunk)
            h.update(chunk)
            n += len(chunk)
    os.replace(tmp, dest)
    return h.hexdigest(), n


# --------------------------------------------------------------------------------------------- files
def safe_rel(path):
    """A provider-supplied relative path, refused if it could leave the destination folder."""
    p = str(path).replace("\\", "/")
    norm = os.path.normpath(p).replace("\\", "/")
    if not p or norm.startswith("../") or norm == ".." or os.path.isabs(p) or re.match(r"^[A-Za-z]:", p):
        raise HubError(f"refused path from provider: {path!r}")
    return norm


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")[:80] or "dataset"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zpath, dest):
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            rel = safe_rel(m.filename)
            out = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)


def file_records(folder):
    out = []
    for r, _, fs in os.walk(folder):
        if "/_quarantine" in r.replace("\\", "/") or "/.git" in r.replace("\\", "/"):
            continue
        for f in sorted(fs):
            if f == "MANIFEST.json" or f.endswith(".part"):
                continue
            p = os.path.join(r, f)
            out.append({"path": os.path.relpath(p, folder).replace("\\", "/"), "sha256": sha256_file(p),
                        "bytes": os.path.getsize(p)})
    return sorted(out, key=lambda x: x["path"])


def leakage_check(folder, keep=False, guard=None):
    """Near-duplicates of the measurement photographs; moved to _quarantine/ unless keep."""
    import numpy as np
    from PIL import Image
    if guard is None:
        from scripts.fetch_seg_datasets import LeakGuard
        guard = LeakGuard()
    leaks, checked = [], 0
    for r, _, fs in os.walk(folder):
        if "_quarantine" in r or "/.git" in r.replace("\\", "/"):
            continue
        for f in fs:
            if not f.lower().endswith(IMG_EXT):
                continue
            p = os.path.join(r, f)
            try:
                img = np.asarray(Image.open(p).convert("RGB"))
            except Exception:
                continue
            checked += 1
            if guard.is_leak(img):
                leaks.append(os.path.relpath(p, folder).replace("\\", "/"))
    if leaks and not keep:
        for rel in leaks:
            dst = os.path.join(folder, "_quarantine", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.join(folder, rel), dst)
    return {"images_checked": checked, "near_duplicates_of_measurement_photographs": len(leaks),
            "examples": leaks[:10], "action": "kept (--keep-leaks)" if keep else "moved to _quarantine/",
            "hamming_threshold": getattr(guard, "threshold", 6)}


def write_manifest(folder, record, keep_leaks=False, guard=None):
    record = dict(record)
    record["leakage"] = leakage_check(folder, keep=keep_leaks, guard=guard)
    record["files"] = file_records(folder)
    record["retrieved_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record["total_bytes"] = sum(f["bytes"] for f in record["files"])
    with open(os.path.join(folder, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    return record


def verify(folder):
    """Do the files still match MANIFEST.json? Returns {ok, changed, missing, extra}."""
    with open(os.path.join(folder, "MANIFEST.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    want = {f["path"]: f["sha256"] for f in man.get("files", [])}
    have = {f["path"]: f["sha256"] for f in file_records(folder)}
    changed = sorted(p for p in want if p in have and have[p] != want[p])
    missing = sorted(p for p in want if p not in have)
    extra = sorted(p for p in have if p not in want)
    return {"ok": not (changed or missing or extra), "changed": changed, "missing": missing, "extra": extra}


# --------------------------------------------------------------------------------------------- providers
def _licence_from_tags(tags):
    for t in tags or []:
        if str(t).startswith("license:"):
            return str(t).split(":", 1)[1]
    return None


def hf_search(query, limit=20, get=http_json):
    q = urllib.parse.urlencode({"search": query, "limit": int(limit), "full": "true"})
    rows = get(f"{HF_API}?{q}", headers=_hf_headers())
    return [{"id": r.get("id"), "downloads": r.get("downloads"), "likes": r.get("likes"),
             "licence": _licence_from_tags(r.get("tags")), "updated": r.get("lastModified")} for r in rows]


def _hf_headers():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def fetch_hf(repo_id, dest, pattern=None, max_files=2000, get=http_json, download=http_download,
             get_pages=http_json_pages):
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo_id):
        raise HubError("Hugging Face reference is <owner>/<dataset>")
    info = get(f"{HF_API}/{repo_id}", headers=_hf_headers())
    rev = info.get("sha") or "main"                           # files from the very commit the manifest names
    tree = get_pages(f"{HF_API}/{repo_id}/tree/{rev}?recursive=true", headers=_hf_headers())
    files = [t for t in tree if t.get("type") == "file"]
    if pattern:
        import fnmatch
        files = [t for t in files if fnmatch.fnmatch(t["path"], pattern)]
    if len(files) > max_files:
        raise HubError(f"{repo_id} has {len(files)} files; pass --pattern to choose, or --max-files")
    for t in files:
        rel = safe_rel(t["path"])
        url = HF_FILE.format(id=repo_id, rev=rev, path=urllib.parse.quote(t["path"]))
        sha, _ = download(url, os.path.join(dest, rel), headers=_hf_headers())
        lfs = (t.get("lfs") or {}).get("oid")
        if lfs and len(lfs) == 64 and lfs != sha:
            raise HubError(f"{rel}: checksum differs from the Hub's ({sha[:12]} vs {lfs[:12]})")
    card = info.get("cardData") or {}
    return {"provider": "huggingface", "reference": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}",
            "version": info.get("sha"), "licence": card.get("license") or _licence_from_tags(info.get("tags")),
            "files_requested": len(files)}


def fetch_zenodo(record_id, dest, get=http_json, download=http_download):
    rec = get(ZENODO_API.format(id=int(record_id)))
    for f in rec.get("files", []):
        rel = safe_rel(f["key"])
        out = os.path.join(dest, rel)
        download(f["links"]["self"], out)
        algo, _, digest = str(f.get("checksum", "")).partition(":")
        if algo == "md5":
            h = hashlib.md5()
            with open(out, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                raise HubError(f"{rel}: md5 differs from Zenodo's record")
        if rel.lower().endswith(".zip"):
            safe_extract(out, os.path.join(dest, os.path.splitext(rel)[0]))
    meta = rec.get("metadata") or {}
    lic = meta.get("license")
    return {"provider": "zenodo", "reference": str(record_id), "url": (rec.get("links") or {}).get("html"),
            "version": meta.get("version") or rec.get("revision"), "doi": rec.get("doi") or meta.get("doi"),
            "licence": lic.get("id") if isinstance(lic, dict) else lic, "title": meta.get("title")}


def fetch_roboflow(ref, dest, fmt="coco-segmentation", get=http_json, download=http_download):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise HubError("set ROBOFLOW_API_KEY in this shell (Roboflow > Settings > API keys); it is never saved")
    parts = ref.strip("/").split("/")
    if len(parts) != 3 or not parts[2].isdigit():
        raise HubError("roboflow reference is <workspace>/<project>/<version number>")
    ws, proj, ver = (urllib.parse.quote(p) for p in parts)
    meta = get(f"{ROBOFLOW_API}/{ws}/{proj}?api_key={urllib.parse.quote(key)}")
    exp = get(f"{ROBOFLOW_API}/{ws}/{proj}/{ver}/{urllib.parse.quote(fmt)}?api_key={urllib.parse.quote(key)}")
    link = (exp.get("export") or {}).get("link")
    if not link:
        raise HubError("Roboflow did not return an export link (is the version generated?)")
    zpath = os.path.join(dest, "_export.zip")
    download(link, zpath)
    safe_extract(zpath, dest)
    os.remove(zpath)
    project = meta.get("project") or {}
    return {"provider": "roboflow", "reference": ref, "url": f"https://universe.roboflow.com/{parts[0]}/{parts[1]}",
            "version": parts[2], "format": fmt, "licence": project.get("license"),
            "classes": project.get("classes")}


def fetch_git(url, dest, run=subprocess.run):
    if not re.match(r"^https://[A-Za-z0-9.-]+/[\w.-]+/[\w.-]+(\.git)?$", url):
        raise HubError("git source must be an https URL of a repository")
    if not os.path.isdir(os.path.join(dest, ".git")):
        # symlinks off: a link in someone else's repository must not point this tool at files on the machine
        run(["git", "-c", "core.symlinks=false", "clone", "-q", "--depth", "1", url, dest], check=True, timeout=1800)
    commit = run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    lic = None
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "README.md", "README"):
        p = os.path.join(dest, name)
        if os.path.exists(p):
            text = open(p, encoding="utf-8", errors="replace").read(4000)
            m = re.search(r"(non-commercial[^\n.]*|CC[- ]BY[-A-Z0-9 .]*|MIT License|Apache License[^\n]*)", text, re.I)
            if m:
                lic = m.group(1).strip()[:120]
                break
    return {"provider": "git", "reference": url, "url": url, "version": commit,
            "licence": lic}


def fetch_kaggle(ref, dest):
    if not re.match(r"^[\w.-]+/[\w.-]+$", ref):
        raise HubError("kaggle reference is <owner>/<dataset>")
    from scripts import fetch_datasets as fd
    api = fd.kaggle_api()                       # Kaggle's own credential files; never read or printed here
    api.dataset_download_files(ref, path=dest, unzip=True, quiet=True)
    return {"provider": "kaggle", "reference": ref, "url": f"https://www.kaggle.com/datasets/{ref}",
            "licence": "see the Kaggle page"}


def mapillary_images(bbox, limit=200, get=http_json):
    tok = os.environ.get("MAPILLARY_TOKEN")
    if not tok:
        raise HubError("set MAPILLARY_TOKEN (a client token from mapillary.com/dashboard/developers); never saved")
    b = [float(x) for x in str(bbox).split(",")]
    if len(b) != 4 or not (b[0] < b[2] and b[1] < b[3]):
        raise HubError("--bbox is min_lon,min_lat,max_lon,max_lat")
    if (b[2] - b[0]) * (b[3] - b[1]) > 0.01:
        raise HubError("bounding box larger than 0.01 square degrees (~10 km x 10 km): Mapillary refuses those")
    q = urllib.parse.urlencode({"access_token": tok, "bbox": ",".join(f"{x:.6f}" for x in b),
                                "fields": "id,thumb_1024_url,captured_at,compass_angle,computed_geometry,creator",
                                "limit": int(min(limit, 2000))})
    return (get(f"{MAPILLARY_API}?{q}") or {}).get("data") or []


def fetch_mapillary(bbox, dest, limit=200, get=http_json, download=http_download):
    rows = mapillary_images(bbox, limit, get=get)
    kept = []
    for r in rows[:limit]:
        url = r.get("thumb_1024_url")
        if not url or not str(r.get("id", "")).isdigit():
            continue
        p = os.path.join(dest, "images", f"mly_{r['id']}.jpg")
        if not os.path.exists(p):
            download(url, p)
        g = (r.get("computed_geometry") or {}).get("coordinates") or [None, None]
        kept.append({"id": r["id"], "file": os.path.relpath(p, dest).replace("\\", "/"), "lon": g[0], "lat": g[1],
                     "captured_at": r.get("captured_at"), "compass_angle": r.get("compass_angle"),
                     "creator": (r.get("creator") or {}).get("username")})
    with open(os.path.join(dest, "images.json"), "w", encoding="utf-8") as fh:
        json.dump(kept, fh, indent=1)
    return {"provider": "mapillary", "reference": f"bbox {bbox}", "url": "https://www.mapillary.com",
            "licence": "CC BY-SA 4.0 (attribute the creators listed in images.json)", "images": len(kept)}


def queue_for_labelling(folder, writable_dir=None, limit=None):
    """Run each photograph through the pipeline; the active-learning queue keeps the informative ones."""
    import numpy as np
    from PIL import Image
    from mlops.active_learning import ActiveLearningQueue
    from pipeline.deep_inference_pipeline import DeepInferencePipeline
    wd = writable_dir or os.environ.get("ROAD_SHIELD_WRITABLE_DIR") or os.path.join(ROOT, "checkpoints")
    al = ActiveLearningQueue(os.path.join(wd, "road_shield.db"), os.path.join(wd, "al_images"))
    pipe = DeepInferencePipeline()
    meta = {}
    p_json = os.path.join(folder, "images.json")
    if os.path.exists(p_json):
        meta = {m["file"]: m for m in json.load(open(p_json, encoding="utf-8"))}
    queued, seen = [], 0
    for p in sorted(glob.glob(os.path.join(folder, "images", "*.jpg")))[: limit or None]:
        img = np.asarray(Image.open(p).convert("RGB"))
        m = meta.get(os.path.relpath(p, folder).replace("\\", "/"), {})
        res = pipe.audit_image(img, latitude=m.get("lat"), longitude=m.get("lon"))
        seen += 1
        item = al.consider(img, res, "mapillary")
        if item:
            queued.append(item)
    return {"photographs_analysed": seen, "queued_for_labelling": len(queued), "items": queued[:20]}


# --------------------------------------------------------------------------------------------- CLI
def _target(spec):
    """'crackseg9k' | 'hf:<id>' | 'git:<url>' | 'zenodo:<id>' | 'roboflow:<ws>/<p>/<v>' | 'kaggle:<o>/<d>'."""
    if ":" in spec and not spec.startswith("http"):
        kind, ref = spec.split(":", 1)
        return kind, ref, None
    entry = next((c for c in CATALOGUE if c["name"] == spec), None)
    if entry is None:
        raise HubError(f"unknown dataset {spec!r}: see `python -m data.hub list`")
    return entry["provider"], entry.get("ref"), entry


def fetch(spec, fmt="coco-segmentation", pattern=None, keep_leaks=False, max_files=2000):
    kind, ref, entry = _target(spec)
    if entry is not None and kind not in ("hf", "git", "zenodo", "roboflow", "kaggle"):
        return {"how": entry["how"], "note": "this source has its own script, which also prepares the data"}
    if entry is not None and entry["how"].startswith("python -m scripts"):
        return {"how": entry["how"], "note": "this source has its own script, which also prepares the data"}
    name = re.sub(r"^https?://[^/]+/", "", ref)[:-4] if kind == "git" and ref.endswith(".git") else ref
    dest = os.path.join(INCOMING, f"{slug(kind)}__{slug(name.replace('/', '__'))}")
    os.makedirs(dest, exist_ok=True)
    if kind == "hf":
        rec = fetch_hf(ref, dest, pattern=pattern, max_files=max_files)
    elif kind == "zenodo":
        rec = fetch_zenodo(ref, dest)
    elif kind == "roboflow":
        rec = fetch_roboflow(ref, dest, fmt=fmt)
    elif kind == "git":
        rec = fetch_git(ref, dest)
    elif kind == "kaggle":
        rec = fetch_kaggle(ref, dest)
    else:
        raise HubError(f"provider {kind!r} is not fetched by this module")
    if entry is not None:
        rec.setdefault("catalogue_entry", entry["name"])
        if not rec.get("licence"):
            rec["licence"] = entry["licence"]
    rec["licence"] = rec.get("licence") or "not stated by the provider: check before use"
    rec = write_manifest(dest, rec, keep_leaks=keep_leaks)
    return {"folder": os.path.relpath(dest, ROOT), **{k: v for k, v in rec.items() if k != "files"},
            "files": len(rec["files"])}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m data.hub", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("search"); p.add_argument("provider", choices=["hf"]); p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("fetch"); p.add_argument("spec"); p.add_argument("--format", default="coco-segmentation")
    p.add_argument("--pattern"); p.add_argument("--keep-leaks", action="store_true")
    p.add_argument("--max-files", type=int, default=2000)
    p = sub.add_parser("mapillary"); p.add_argument("--bbox", required=True); p.add_argument("--limit", type=int, default=200)
    p.add_argument("--queue", action="store_true", help="send informative photographs to the labelling queue")
    p.add_argument("--keep-leaks", action="store_true")
    p = sub.add_parser("verify"); p.add_argument("folder")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "list":
            for c in CATALOGUE:
                print(f"{c['name']:<28} {c['provider']:<10} {('key: ' + c['key']) if c['key'] else 'no key':<22} "
                      f"{c['licence'][:34]:<36} {c['task'][:60]}")
                print(f"{'':28} used by: {c['used_by']}\n{'':28} {c['how']}")
        elif a.cmd == "search":
            for r in hf_search(a.query, a.limit):
                print(f"{r['id']:<60} downloads {r['downloads'] or 0:>8}  licence {r['licence'] or '?'}")
        elif a.cmd == "fetch":
            print(json.dumps(fetch(a.spec, fmt=a.format, pattern=a.pattern, keep_leaks=a.keep_leaks,
                                   max_files=a.max_files), indent=1))
        elif a.cmd == "mapillary":
            dest = os.path.join(INCOMING, "mapillary__" + slug(a.bbox))
            os.makedirs(dest, exist_ok=True)
            rec = write_manifest(dest, fetch_mapillary(a.bbox, dest, a.limit), keep_leaks=a.keep_leaks)
            print(json.dumps({k: v for k, v in rec.items() if k != "files"}, indent=1))
            if a.queue:
                print(json.dumps(queue_for_labelling(dest), indent=1))
        elif a.cmd == "verify":
            r = verify(a.folder)
            print(json.dumps(r, indent=1))
            return 0 if r["ok"] else 1
    except HubError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
