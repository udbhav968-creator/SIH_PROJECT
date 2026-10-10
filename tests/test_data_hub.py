"""Dataset hub (data/hub.py): provider adapters with a fake network, manifests, leakage, path safety."""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import hub  # noqa: E402


def _jpeg(seed, size=(64, 96)):
    from PIL import Image
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


class FakeNet:
    def __init__(self, files):
        self.files, self.calls = files, []

    def get(self, url, headers=None, timeout=30):
        self.calls.append(url)
        return self.files[url.split("?")[0]] if url.split("?")[0] in self.files else self.files[url]

    def download(self, url, dest, headers=None, timeout=120):
        data = self.files[url]
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)
        return hashlib.sha256(data).hexdigest(), len(data)


class NoLeaks:
    threshold = 6

    def __init__(self, leaky=()):
        self.leaky = {hashlib.sha1(np.asarray(x).tobytes()).hexdigest() for x in leaky}

    def is_leak(self, img):
        return hashlib.sha1(np.asarray(img).tobytes()).hexdigest() in self.leaky


class HubTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_hf_files_checksums_and_licence(self):
        a = _jpeg(1)
        sha = hashlib.sha256(a).hexdigest()
        net = FakeNet({
            f"{hub.HF_API}/me/potholes": {"sha": "abc123", "cardData": {"license": "cc-by-4.0"}},
            f"{hub.HF_API}/me/potholes/tree/abc123": [
                {"type": "file", "path": "train/a.jpg", "lfs": {"oid": sha}},
                {"type": "directory", "path": "train"}],
            hub.HF_FILE.format(id="me/potholes", rev="abc123", path="train/a.jpg"): a})
        rec = hub.fetch_hf("me/potholes", self.d, get=net.get, download=net.download, get_pages=net.get)
        self.assertEqual((rec["licence"], rec["version"]), ("cc-by-4.0", "abc123"))
        self.assertTrue(any("/tree/abc123" in c for c in net.calls), "files listed at the commit the manifest names")
        bad = FakeNet(dict(net.files))
        bad.files[f"{hub.HF_API}/me/potholes/tree/abc123"] = [{"type": "file", "path": "a.jpg", "lfs": {"oid": "0" * 64}}]
        bad.files[hub.HF_FILE.format(id="me/potholes", rev="abc123", path="a.jpg")] = a
        with self.assertRaises(hub.HubError):
            hub.fetch_hf("me/potholes", self.d, get=bad.get, download=bad.download, get_pages=bad.get)
        evil = FakeNet(dict(net.files))
        evil.files[f"{hub.HF_API}/me/potholes/tree/abc123"] = [{"type": "file", "path": "../../etc/x.jpg"}]
        with self.assertRaises(hub.HubError):
            hub.fetch_hf("me/potholes", self.d, get=evil.get, download=evil.download, get_pages=evil.get)

    def test_token_does_not_follow_a_redirect_to_another_host(self):
        import urllib.request
        h = hub._SameHostAuthRedirect()
        req = urllib.request.Request("https://huggingface.co/datasets/a/b/resolve/x/f.zip",
                                     headers={"Authorization": "Bearer secret"})
        same = h.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/other")
        other = h.redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.hf.co/abc?sig=1")
        self.assertEqual(same.get_header("Authorization"), "Bearer secret")
        self.assertIsNone(other.get_header("Authorization"))

    def test_pagination_follows_the_link_header(self):
        pages = {"https://h/1": ([1, 2], '<https://h/2>; rel="next"'), "https://h/2": ([3], "")}

        class R:
            def __init__(self, url):
                self.body, link = pages[url]
                self.headers = {"Link": link}

            def read(self):
                return json.dumps(self.body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        with mock.patch.object(hub, "_open", lambda url, headers=None, timeout=30: R(url)):
            self.assertEqual(hub.http_json_pages("https://h/1"), [1, 2, 3])

    def test_zenodo_md5_and_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("imgs/x.jpg", _jpeg(2))
        data = buf.getvalue()
        net = FakeNet({hub.ZENODO_API.format(id=42): {
            "doi": "10.5281/zenodo.42", "links": {"html": "https://zenodo.org/records/42"},
            "metadata": {"title": "Potholes", "license": {"id": "cc-by-4.0"}},
            "files": [{"key": "set.zip", "checksum": "md5:" + hashlib.md5(data).hexdigest(),
                       "links": {"self": "https://zenodo.org/f/set.zip"}}]},
            "https://zenodo.org/f/set.zip": data})
        rec = hub.fetch_zenodo(42, self.d, get=net.get, download=net.download)
        self.assertEqual(rec["licence"], "cc-by-4.0")
        self.assertTrue(os.path.exists(os.path.join(self.d, "set", "imgs", "x.jpg")))
        net.files[hub.ZENODO_API.format(id=42)]["files"][0]["checksum"] = "md5:" + "0" * 32
        with self.assertRaises(hub.HubError):
            hub.fetch_zenodo(42, tempfile.mkdtemp(dir=self.d), get=net.get, download=net.download)

    def test_roboflow_needs_key_and_unzips_safely(self):
        with mock.patch.dict(os.environ, {"ROBOFLOW_API_KEY": ""}):
            with self.assertRaises(hub.HubError):
                hub.fetch_roboflow("ws/proj/3", self.d)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("train/_annotations.coco.json", "{}")
            z.writestr("train/a.jpg", _jpeg(3))
        net = FakeNet({f"{hub.ROBOFLOW_API}/ws/proj": {"project": {"license": "CC BY 4.0", "classes": {"pothole": 9}}},
                       f"{hub.ROBOFLOW_API}/ws/proj/3/coco-segmentation": {"export": {"link": "https://rf/x.zip"}},
                       "https://rf/x.zip": buf.getvalue()})
        with mock.patch.dict(os.environ, {"ROBOFLOW_API_KEY": "secret-key"}):
            rec = hub.fetch_roboflow("ws/proj/3", self.d, get=net.get, download=net.download)
        self.assertEqual(rec["licence"], "CC BY 4.0")
        self.assertNotIn("secret-key", json.dumps(rec), "the key never lands in a record")
        self.assertTrue(os.path.exists(os.path.join(self.d, "train", "a.jpg")))
        evil = io.BytesIO()
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("../escape.txt", "x")
        with self.assertRaises(hub.HubError):
            p = os.path.join(self.d, "e.zip")
            open(p, "wb").write(evil.getvalue())
            hub.safe_extract(p, os.path.join(self.d, "out"))

    def test_mapillary_bbox_rules_and_download(self):
        with mock.patch.dict(os.environ, {"MAPILLARY_TOKEN": "tok"}):
            with self.assertRaises(hub.HubError):
                hub.mapillary_images("77.0,12.0,78.0,13.0")          # far too large
            net = FakeNet({hub.MAPILLARY_API: {"data": [
                {"id": "111", "thumb_1024_url": "https://m/1.jpg", "computed_geometry": {"coordinates": [77.6, 12.97]},
                 "captured_at": 1, "creator": {"username": "someone"}},
                {"id": "../x", "thumb_1024_url": "https://m/2.jpg"}]},
                "https://m/1.jpg": _jpeg(4)})
            rec = hub.fetch_mapillary("77.59,12.96,77.60,12.98", self.d, get=net.get, download=net.download)
        self.assertEqual(rec["images"], 1, "an id that is not a number is skipped")
        self.assertIn("CC BY-SA", rec["licence"])
        self.assertNotIn("tok", json.dumps(rec))
        meta = json.load(open(os.path.join(self.d, "images.json")))
        self.assertEqual(meta[0]["lat"], 12.97)

    def test_manifest_leakage_and_verify(self):
        from PIL import Image
        imgs = [_jpeg(5), _jpeg(6)]
        for i, b in enumerate(imgs):
            os.makedirs(os.path.join(self.d, "images"), exist_ok=True)
            open(os.path.join(self.d, "images", f"{i}.jpg"), "wb").write(b)
        leaky = np.asarray(Image.open(io.BytesIO(imgs[1])).convert("RGB"))
        rec = hub.write_manifest(self.d, {"provider": "test"}, guard=NoLeaks([leaky]))
        self.assertEqual(rec["leakage"]["near_duplicates_of_measurement_photographs"], 1)
        self.assertTrue(os.path.exists(os.path.join(self.d, "_quarantine", "images", "1.jpg")))
        self.assertEqual([f["path"] for f in rec["files"]], ["images/0.jpg"])
        self.assertTrue(hub.verify(self.d)["ok"])
        open(os.path.join(self.d, "images", "0.jpg"), "ab").write(b"x")
        self.assertEqual(hub.verify(self.d)["changed"], ["images/0.jpg"])

    def test_catalogue_and_targets(self):
        names = [c["name"] for c in hub.CATALOGUE]
        self.assertEqual(len(names), len(set(names)))
        for c in hub.CATALOGUE:
            self.assertTrue(c["licence"] and c["how"] and c["used_by"], c["name"])
        self.assertEqual(hub._target("hf:a/b")[:2], ("hf", "a/b"))
        self.assertIn("how", hub.fetch("deepcrack"), "sources with their own script say how instead")
        with self.assertRaises(hub.HubError):
            hub._target("nope")
        with self.assertRaises(hub.HubError):
            hub.fetch_git("file:///etc", self.d)
        self.assertEqual(hub.safe_rel("a/./b.jpg"), "a/b.jpg")
        for bad in ("../a", "/abs", "C:\\\\x", "a/../../b"):
            with self.assertRaises(hub.HubError):
                hub.safe_rel(bad)


if __name__ == "__main__":
    unittest.main()
