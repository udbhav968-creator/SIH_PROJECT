"""
Download RDD2022 India and leave it unpacked at datasets/_downloads/rdd2022/India.

The per-country archive on the CRDDC S3 bucket (bigdatacup.s3...) now answers
403, which is why the first Colab run got an HTML error page instead of a zip.
Sources, tried in order:

  1. --local-zip PATH       a copy you already have (e.g. uploaded to Colab)
  2. the CRDDC S3 per-country archive (~500 MB), if it ever comes back
  3. the official figshare release of all of RDD2022 (~13.3 GB) - only the India
     part is extracted. Colab downloads it in a few minutes.

Every archive is checked with zipfile before use, so an error page can never be
mistaken for data again; the script exits non-zero if no source works.

    python -m scripts.fetch_rdd2022_india
    python -m scripts.fetch_rdd2022_india --local-zip /content/RDD2022_India.zip
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "datasets", "_downloads", "rdd2022")
S3_INDIA = ("https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/"
            "Country_Specific_Data_CRDDC2022/RDD2022_India.zip")
FIGSHARE_ALL = "https://ndownloader.figshare.com/files/38030910"   # RDD2022_released_through_CRDDC2022.zip


def download(url, dest, min_bytes=1_000_000):
    print(f"  downloading {url}", flush=True)
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "road-shield/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            got, step = 0, 0
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if got // (500 << 20) > step:
                    step = got // (500 << 20)
                    print(f"    {got / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)
    except Exception as e:
        print(f"  failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    if os.path.getsize(tmp) < min_bytes or not zipfile.is_zipfile(tmp):
        print("  not a zip archive (probably an error page) - ignored")
        os.remove(tmp)
        return None
    os.replace(tmp, dest)
    return dest


def india_root():
    for base, dirs, _files in os.walk(OUT):
        if os.path.basename(base) == "India" and "train" in dirs:
            return base
    return None


def extract_india(zpath):
    """Extract the India part of an archive. Handles a per-country zip, a folder
    inside the full release, or a country zip nested inside the full release."""
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        nested = [n for n in names if n.lower().endswith(".zip") and "india" in n.lower()]
        if nested:
            print(f"  nested archive: {nested[0]}")
            inner = os.path.join(OUT, os.path.basename(nested[0]))
            with z.open(nested[0]) as src, open(inner, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            with zipfile.ZipFile(inner) as zi:
                zi.extractall(OUT)
            os.remove(inner)
            return
        members = [n for n in names if "/india/" in ("/" + n.lower())]
        if not members:
            raise SystemExit(f"no India data inside {zpath}")
        print(f"  extracting {len(members)} India files")
        z.extractall(OUT, members=members)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-zip", default=None)
    ap.add_argument("--keep-archive", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if india_root():
        print(f"RDD2022 India already present: {india_root()}")
        return india_root()

    candidates = []
    if a.local_zip:
        if zipfile.is_zipfile(a.local_zip):
            candidates.append(a.local_zip)
        else:
            print(f"--local-zip {a.local_zip} is not a zip archive")
    for url, name in ((S3_INDIA, "RDD2022_India.zip"), (FIGSHARE_ALL, "RDD2022_all.zip")):
        if candidates:
            break
        got = download(url, os.path.join(OUT, name))
        if got:
            candidates.append(got)
    if not candidates:
        sys.exit("RDD2022 India could not be downloaded from any source. Upload RDD2022_India.zip "
                 "to Colab and rerun with --local-zip /content/RDD2022_India.zip")

    extract_india(candidates[0])
    if not a.keep_archive and candidates[0] != a.local_zip:
        os.remove(candidates[0])
    root = india_root()
    if not root:
        sys.exit("archive extracted but no India/train folder was found")
    print(f"RDD2022 India ready: {root}")
    return root


if __name__ == "__main__":
    main()
