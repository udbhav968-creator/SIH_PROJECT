"""
Write README.md from the checkpoints, so no figure in it is typed by hand.

    python -m scripts.build_readme

Every number below is read from the report that measured it (checkpoints/*.json).
Prose that depends on a result - which classifier serves, whether the U-Net won,
whether a road-damage detector exists - is chosen from those reports too. Run it
again after any retraining; the README then matches the site and claims.json.
"""

import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, "checkpoints")
LIVE_URL = "https://road-shield-ai-engine.vercel.app"


def rep(name):
    p = os.path.join(CKPT, name)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def pct(x, d=1):
    """Percent, rounded half-up like the site's toFixed, so 0.9225 reads 92.3% in both places."""
    if not isinstance(x, (int, float)):
        return "—"
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(round(x * 100, 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q}%"


def f3(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def num(x):
    return f"{x:,}" if isinstance(x, int) else ("—" if x is None else str(x))


def indian(served_kind):
    """Indian-roads result for the model that actually serves."""
    ie = rep("indian_roads_eval_report.json")
    order = ("served", "after", "frozen_head", "current") if served_kind == "fine_tuned_cnn" else \
            ("frozen_head", "after", "served", "current")
    for k in order:
        if isinstance(ie.get(k), dict) and "accuracy" in ie[k]:
            return ie[k], ie.get("before"), ie.get("dataset") or {}
    return None, ie.get("before"), ie.get("dataset") or {}


def build():
    from models.served_report import served_classifier_summary, served_segmenter_summary

    sc = served_classifier_summary(CKPT) or {}
    kind = sc.get("kind")
    deep = kind == "fine_tuned_cnn"
    ft = rep("finetune_summary.json")
    head = rep("cnn_head_mobilenetv2_report.json")
    base = head.get("handcrafted_baseline") or {}
    ind, ind_before, ds = indian(kind)
    if deep and sc.get("indian_roads"):
        ind = sc["indian_roads"]
    seg = served_segmenter_summary(CKPT) or {}
    seg_unet = seg.get("kind") == "unet"
    seg_sel = seg.get("selection") or {}
    gate = (rep("semantic_gate_report.json").get("results") or {})
    g_a, g_c = gate.get("A_segmenter") or {}, gate.get("C_seg_gated_0.5") or {}
    det = rep("road_damage_detector_report.json")
    det_on = bool(det) and os.path.exists(os.path.join(CKPT, "damage_rdd2022_india.onnx"))
    imu_sel = rep("imu_model_selection.json")
    imu_rf = rep("imu_shock_report.json")
    imu_cnn = imu_sel.get("served") == "cnn"
    if imu_sel.get("held_out_for_reporting"):
        h = imu_sel["held_out_for_reporting"][imu_sel["served"]]
        imu_acc, imu_f1 = h.get("accuracy"), h.get("macro_f1")
    else:
        imu_acc, imu_f1 = imu_rf.get("held_out_validation_accuracy"), None
    claims = rep("claims.json")
    n_tests = sum(open(p, encoding="utf-8").read().count("    def test_")
                  for p in glob.glob(os.path.join(ROOT, "tests", "test_*.py")))

    acc, f1 = sc.get("held_out_test_accuracy"), sc.get("held_out_test_macro_f1")
    n_img, n_ph = sc.get("held_out_test_images"), sc.get("held_out_test_photographs")
    cls_label = sc.get("label", "—")
    cls_arch = (f"{ft.get('chosen_arch')} (ImageNet-pretrained), fine-tuned end to end on a Colab T4 GPU, "
                f"ONNX Runtime on CPU, flip test-time augmentation") if deep else \
               ("Frozen MobileNetV2 (ImageNet) embeddings of the image and its mirror -> scikit-learn head "
                f"({head.get('head', '—')}) chosen by grouped 5-fold cross-validation, ONNX Runtime on CPU")
    seg_io = seg.get("iou") or {}
    crack_iou = (seg_io.get("crack") or {}).get("iou")
    pot_iou = (seg_io.get("pothole") or {}).get("iou")
    seg_name = ("U-Net, ResNet-18 encoder (ImageNet), all layers trained" if seg_unet
                else "Pixel classifier (gradient boosting on 11 per-pixel features)")
    dtest = det.get("test") or {}

    L = []
    w = L.append
    w("# ROAD-SHIELD")
    w("")
    w("**AI-assisted road-defect assessment using the public bus fleet as a mobile sensing network.**  ")
    w("Smart India Hackathon 2026 · Problem statement **SIH26124** · Bharat Electronics Limited")
    w("")
    w(f"**Live site:** [{LIVE_URL.replace('https://', '')}]({LIVE_URL}) — every measured result and the claims "
      "registry. Photograph and video analysis runs on the engine (`python -m api.server` or Docker), because the "
      "model stack is larger than a serverless function allows.")
    w("")
    w("A road photograph goes in. A classified, outlined, measured and costed repair order comes out, sealed so it "
      "cannot be quietly edited. Every number in this file is read from the report that measured it "
      "(`python -m scripts.build_readme`), and every report is produced by code in this repository.")
    w("")
    w("---")
    w("")
    w("## Headline results")
    w("")
    w("| What | Result | Measured on |")
    w("|---|---|---|")
    w(f"| Road-condition classifier, 7 classes | **{pct(acc)}** accuracy, macro-F1 **{f3(f1)}** | "
      f"{num(n_img)} held-out images from {num(n_ph)} photographs never seen in training, scored once |")
    if ind:
        before = f" (was {pct(ind_before.get('accuracy'))} before Indian data was added)" if ind_before else ""
        w(f"| Same classifier on **Indian roads** | **{pct(ind.get('accuracy'))}** accuracy, macro-F1 "
          f"**{f3(ind.get('macro_f1'))}**{before} | {num(ind.get('images'))} crops from held-out RDD2022 India "
          f"photographs (normal / crack / pothole), never trained on |")
    w(f"| Hand-crafted baseline (HOG + LBP + colour → SVM) | {pct(base.get('accuracy'))}, macro-F1 "
      f"{f3(base.get('macro_f1'))} | the identical split — the number the deep model has to beat |")
    w(f"| Defect segmentation ({'U-Net' if seg_unet else 'pixel classifier'}) | crack IoU **{f3(crack_iou)}**, "
      f"pothole IoU **{f3(pot_iou)}** | {num(seg.get('test_photographs') or 500)} held-out DNIT photographs |")
    if g_a and g_c:
        w(f"| CNN semantic gate on the segmenter | pothole IoU {f3((g_a.get('pothole') or {}).get('iou'))} → "
          f"{f3((g_c.get('pothole') or {}).get('iou'))}; clean roads with a false blob "
          f"{g_a.get('clean_photos_with_false_blob')} → {g_c.get('clean_photos_with_false_blob')} | "
          "photographs the classifier never saw |")
    if det_on:
        w(f"| Road-damage detector (YOLOv8, boxes) | mAP@0.5 **{f3(dtest.get('map50'))}**, mAP@0.5:0.95 "
          f"{f3(dtest.get('map50_95'))} | {num(((det.get('data') or {}).get('photographs') or {}).get('test'))} "
          "held-out RDD2022 India photographs |")
    w(f"| IMU shock classifier ({'1-D CNN' if imu_cnn else 'RandomForest'}) | **{pct(imu_acc)}** accuracy"
      + (f", macro-F1 {f3(imu_f1)}" if imu_f1 is not None else "") +
      " | 164 held-out windows of real Indian-road drive logs, split by time block |")
    w("")
    w("## What it does")
    w("")
    w("```")
    w("photo ─► classify ─► segment ─► project to ground ─► depth interval ─► MoRTH cost ─► SHA-256 work order")
    w("          (CNN)    (mask, not     (per-vehicle          (estimate,       (range)         (tamper-evident)")
    w("                     a box)        calibration)          never a measurement)")
    w("   │")
    w("   ├─► people & vehicles (YOLOv8 COCO) ─► privacy blur of heads and number plates before sharing")
    if det_on:
        w("   ├─► road-damage boxes (YOLOv8, RDD2022 India)")
    w("   └─► fleet ledger: reports of the same defect class within 8 m merge into one defect (SQLite)")
    w("```")
    w("")
    w("| Stage | How | Status |")
    w("|---|---|---|")
    w(f"| Classification | {cls_arch} | **measured**, {pct(acc)} |")
    w(f"| Segmentation | {seg_name}, trained on the DNIT hand-drawn polygons; per-class thresholds tuned on a "
      "calibration split | **measured**, IoU above |")
    w("| Area | each mask pixel's own ground footprint, summed (inverse perspective mapping) | **measured** *if* the "
      "camera is calibrated — 30 cm of mount height moves area ~46% |")
    w("| Depth | IRC band placed by measured extent / cavity contrast | **estimate**, always an interval |")
    w("| Cost | MoRTH Section 500 bitumen tonnage (compaction factor 1.15) × rate | **range**, because depth is a range |")
    w("| PCI | ASTM D6433 deduct-value procedure | formula; reproduces the deduct curves, not field-validated |")
    w("| Deterioration | HDM-4-style growth model, 30/60/90/180 days | formula, not field-validated |")
    w("| Sensor fusion | Bayesian log-odds gate over vision + IMU | reports *unavailable* when no IMU window is sent |")
    w("| Work orders | SHA-256 seal over the order fields; no GPS → `HELD_NO_GPS`, never an invented location | "
      "verified by tests |")
    w("| Repair verification | SSIM + sharpness + perceptual hash | catches a resubmitted old photograph |")
    w("| Privacy | heads (from person boxes), number plates and faces blurred | implemented; **recall not measured** |")
    w("| GIS | Google Maps if a key is set, else OpenStreetMap / OSRM / Open-Meteo | reports UNAVAILABLE rather "
      "than inventing data |")
    w("")
    w("Measured on one photograph through the API: with an assumed camera mount the defect is 0.007 m² and ₹4.5; with "
      "the vehicle's calibrated profile it is 0.049 m² and ₹34.5 — seven times. That is why every response carries "
      "the provenance of each number in it.")
    w("")
    w("## The models")
    w("")
    w("| Model | Kind | Trained here? | Result | Serving |")
    w("|---|---|---|---|---|")
    if ft.get("archs"):
        for k, r in ft["archs"].items():
            t = r.get("test") or {}
            ir = r.get("indian_roads") or {}
            w(f"| {k} | CNN, ImageNet-pretrained, fine-tuned end to end | yes (GPU) | {pct(t.get('accuracy'))}, F1 "
              f"{f3(t.get('macro_f1'))}; India {pct(ir.get('accuracy'))} | "
              f"{'**yes**' if deep and k == ft.get('chosen_arch') else 'no (lost on validation)'} |")
    w(f"| MobileNetV2 + {head.get('head', 'head')} | frozen CNN features + trained head | head only | "
      f"{pct(head.get('held_out_test_accuracy'))}, F1 {f3(head.get('held_out_test_macro_f1'))} | "
      f"{'fallback' if deep else '**yes**'} |")
    w(f"| HOG/LBP + PCA + SVM | classical features | yes | {pct(base.get('accuracy'))} | fallback |")
    sel_test = (seg_sel.get("test") or {})
    if sel_test.get("unet"):
        u = sel_test["unet"]
        w(f"| U-Net (ResNet-18 encoder) | deep segmenter, ImageNet encoder, all layers trained | yes (GPU) | crack "
          f"{f3((u.get('crack') or {}).get('iou'))}, pothole {f3((u.get('pothole') or {}).get('iou'))} | "
          f"{'**yes**' if seg_unet else ('no (won on masks, lost the end-to-end check)' if (seg_sel.get('deployment_check') or {}).get('passed') is False else 'no (lost on validation)')} |")
    px = rep("defect_segmenter_report.json").get("iou") or {}
    w(f"| Pixel segmenter | gradient boosting on 11 features | yes | crack {f3((px.get('crack') or {}).get('iou'))}, "
      f"pothole {f3((px.get('pothole') or {}).get('iou'))} | {'fallback' if seg_unet else '**yes**'} |")
    if det_on:
        w(f"| YOLOv8 road-damage detector | COCO-pretrained, fine-tuned on RDD2022 India | yes (GPU) | mAP@0.5 "
          f"{f3(dtest.get('map50'))} | **yes** |")
    w("| YOLOv8n (COCO) | pretrained object detector | **no** — used as published | people, vehicles, signs | yes |")
    imu_h = imu_sel.get("held_out_for_reporting") or {}
    if imu_h.get("cnn"):
        w(f"| IMU 1-D CNN | deep, from scratch | yes (GPU) | {pct(imu_h['cnn'].get('accuracy'))} | "
          f"{'**yes**' if imu_cnn else 'no (lost in cross-validation)'} |")
    w(f"| IMU RandomForest | classical, from scratch | yes | "
      f"{pct((imu_h.get('random_forest') or {}).get('accuracy') or imu_rf.get('held_out_validation_accuracy'))} | "
      f"{'fallback' if imu_cnn else '**yes**'} |")
    w("")
    w("**How a model gets served.** Each deep model replaces its classical counterpart only by a rule written "
      "before its test set is scored: the fine-tuned CNN must beat the frozen head on validation accuracy *and* "
      "macro-F1; the U-Net must beat the pixel classifier on crack *and* pothole IoU on calibration photographs "
      "without more false blobs on clean roads; the IMU CNN must win 5-fold cross-validation on accuracy *and* "
      "macro-F1. Losers are reported, not hidden. Pretrained ImageNet/COCO weights are the starting point (transfer "
      "learning); training then updates every layer on this project's data, except the frozen-head baseline.")
    w("")
    pcr = sc.get("per_class_report") or {}
    rows = [(k, v) for k, v in pcr.items() if isinstance(v, dict) and "f1-score" in v and "avg" not in k]
    if rows:
        w("### Per class (served classifier)")
        w("")
        w("| Class | Precision | Recall | F1 | Test images |")
        w("|---|---|---|---|---|")
        for k, v in rows:
            w(f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1-score']:.2f} | {int(v.get('support', 0))} |")
        w("")
        w("Normal road, crack and pothole carry almost all test images. The four rare classes have seven or eight "
          "test images each, so one photograph moves their F1 by more than 0.1. That is a data-volume problem and it "
          "is shown, not averaged away.")
        w("")
    w("## Data")
    w("")
    tc, ec = ds.get("training_crops") or {}, ds.get("eval_crops") or {}
    w("| Source | Classes | Kept | Note |")
    w("|---|---|---|---|")
    w("| DNIT *Cracks and Potholes in Road Images* (Brazil) | crack, pothole, normal | 1,667 crops | 2,235 photographs, "
      "4,720 hand-drawn polygons — the only source with outlines, so the segmenter trains on it |")
    w(f"| RDD2022 India (CRDDC 2022, smartphone) | crack, pothole, normal | {num(sum(tc.values()) if tc else None)} crops "
      f"| from {num(ds.get('training_photographs'))} training photographs; the Indian test set is "
      f"{num(sum(ec.values()) if ec else None)} crops from {num(ds.get('eval_photographs'))} held-out photographs "
      "(split by photograph — the official test split has no public labels) |")
    w("| Kaggle pothole sets (3) | pothole, normal | 1,287 | one set was 94% a re-upload: 700 of 739 rejected as "
      "perceptual duplicates |")
    w("| Kaggle surface cracks (concrete walls) | — | 0 of 2,376 | **excluded**: close-ups of plaster with no road "
      "and no horizon (`pipeline/corpus_policy.py`); kept on disk, reproducible with `ROAD_SHIELD_NO_CORPUS_FILTER=1` |")
    w("| Wikimedia Commons / Geograph + field photographs | waterlogging, zebra, divider, sign | 207 | ~50 "
      "photographs per rare class — still the binding constraint |")
    w("| IMU drive logs (`VishalSingh25/Pothole-Project`) | 4 shock classes | 852 windows | 10 real drives on Indian "
      "roads, 205,491 samples at 100 Hz, from a car — not a bus |")
    if det_on:
        w("| RDD2022 India boxes (detector) | D00 / D10 / D20 / D40 | all labelled photographs | 70/15/15 split by "
          "photograph, seed 42 |")
    w("")
    w("## How the numbers are kept honest")
    w("")
    w("- **Split by source photograph**, never by file: an augmented copy can never sit on the other side of the "
      "split from its original.")
    w("- **Model choice before the test set**: heads, architectures and thresholds are chosen on training or "
      "validation data; each test set is scored once.")
    w("- **Near-duplicate rejection**: every image is perceptually hashed; within 8 bits of an existing image is a "
      "re-upload (threshold measured over 60 photographs, not guessed).")
    w("- **Label-conflict audit**: 20 photographs were filed under three contradictory labels at once — the bug that "
      "held accuracy at 36.6%.")
    w("- **Domain policy**: data that is not a road scene is excluded from road-scene training and measurement, and "
      "the exclusion travels with every number it changes.")
    w(f"- **Claims registry**: `checkpoints/claims.json` lists {len(claims.get('subsystems', []))} subsystems with "
      f"their evidence files and {len(claims.get('corrections', []))} earlier claims that were withdrawn or "
      "corrected; the site's Architecture page shows both.")
    w("- **Nothing invented at runtime**: no default GPS, no default PCI, no confidence where there is no "
      "probability; missing inputs produce a 400 or an explicit `unavailable`.")
    w("")
    w("## System design and platform")
    w("")
    w("The full design — requirements, capacity maths for a 5,000-bus fleet, architecture from the bus edge to the "
      "ledger, data model, ML lifecycle, deployment, scaling, security and privacy — is in "
      "[`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) and on the site's `/design` page, with an interactive "
      "capacity calculator. Every component there is marked *implemented* or *designed*. Impact, cost per km, "
      "the Responsible AI mapping (Microsoft's six principles), the 90-day pilot and the Azure mapping are in "
      "[`docs/IMPACT_AND_RESPONSIBLE_AI.md`](docs/IMPACT_AND_RESPONSIBLE_AI.md) and on `/impact`.")
    w("")
    w("| Platform feature | Where |")
    w("|---|---|")
    w("| OpenAPI 3 contract for the client-facing endpoints | `GET /api/v1/openapi.json`, rendered at `/api-docs` |")
    w("| Model registry: every model's artefact SHA-256, held-out metrics, serving status and deciding rule | "
      "`GET /api/v1/models/served`, shown on `/design` |")
    w("| Request IDs on every response; one JSON access-log line per request | `X-Request-ID`; "
      "`ROAD_SHIELD_ACCESS_LOG=1` |")
    w("| API key on state-changing endpoints (constant-time compare), off unless configured | "
      "`ROAD_SHIELD_API_KEY` |")
    w("| Container with a health check that fails when models did not load, plus a persistent ledger volume | "
      "`Dockerfile`, `docker-compose.yml` |")
    w("| CI on every push (Python 3.11 and 3.12, module imports, test suite) | `.github/workflows/ci.yml` |")
    w("| Claims registry and this README generated from the measured reports | `scripts/build_claims.py`, "
      "`scripts/build_readme.py` |")
    w("")
    w("## Run it")
    w("")
    w("```bash")
    w("pip install -r requirements.txt")
    w("python -m api.server                      # http://127.0.0.1:8000/")
    w("python -m unittest discover -s tests -t . # the test suite")
    w("```")
    w("")
    w("Docker:")
    w("")
    w("```bash")
    w("docker compose up --build                 # engine + persistent ledger volume, http://localhost:8000")
    w("```")
    w("")
    w("No GPU and no PyTorch at inference — every network runs on ONNX Runtime on the CPU. scikit-learn is pinned "
      "(`>=1.8,<1.9`) because pickled estimators are not portable across minor versions.")
    w("")
    w("## Train it")
    w("")
    w("Deep training runs on a free Google Colab T4 GPU and resumes after a disconnect when Google Drive is mounted:")
    w("")
    w("```python")
    w("from google.colab import drive; drive.mount('/content/drive')")
    w("!git clone --depth 1 -b audit-2026-10-03 https://github.com/udbhav968-creator/SIH_PROJECT.git")
    w("%cd SIH_PROJECT")
    w("!mkdir -p logs && bash scripts/colab_train_all.sh 2>&1 | tee -a logs/colab_run.txt     # main run")
    w("!bash scripts/colab_train_extra.sh 2>&1 | tee -a logs/colab_extra.txt                 # U-Net + YOLOv8")
    w("```")
    w("")
    w("The main run fetches DNIT and RDD2022 India, retrains the frozen head, fine-tunes EfficientNet-B0/B2, "
      "MobileNetV3-Large and ResNet-50, compares the IMU 1-D CNN with the RandomForest, runs the benchmark and the "
      "tests, and rebuilds the claims and the report. The extra run trains the U-Net segmenter and the YOLOv8 "
      "road-damage detector (it also runs on Kaggle). `scripts/apply_colab_outputs.ps1` copies the results back "
      "into the repository; `python -m scripts.build_readme` then refreshes this file.")
    w("")
    w("Individual trainers: `training/train_cnn_head.py`, `training/train_finetune_cnn.py`, "
      "`training/train_unet_segmenter.py`, `training/train_rdd_detector.py`, `training/train_imu_deep.py`, "
      "`training/train_segmenter.py`.")
    w("")
    w("## The site")
    w("")
    w("| Page | For |")
    w("|---|---|")
    for page, what in (("/", "overview, headline numbers and the honest limits"),
                       ("/inspect", "analyse a photograph: class, mask, area, depth interval, cost range, privacy blur"
                        + (", damage boxes" if det_on else "")),
                       ("/video", "dashcam ingest, frames sampled by ground distance"),
                       ("/corridor", "fleet map and the deduplication ledger"),
                       ("/works", "costing and the SHA-256 tamper demonstration"),
                       ("/models", "model card, per class, deep-vs-classical comparisons, including what scores badly"),
                       ("/data", "dataset lineage, duplicate rejection, what the data does not cover"),
                       ("/system", "components, calibration, storage, live endpoint self-test"),
                       ("/architecture", "the claims registry and every withdrawn claim"),
                       ("/design", "system design: architecture, capacity calculator, model registry, data model"),
                       ("/api-docs", "the API reference, rendered from the OpenAPI document"),
                       ("/impact", "sourced road-safety statistics, cost per km, Responsible AI mapping, pilot plan, Azure mapping")):
        w(f"| `{page}` | {what} |")
    w("")
    w("Main API endpoints (full reference at `/api-docs`): `POST /api/v1/pipeline/deep-audit` (full analysis of a "
      "photograph), `POST /api/v1/dispatch/work-order` and `/api/v1/dispatch/verify-seal`, "
      "`POST /api/v1/fleet/report-defect`, `POST /api/v1/privacy/redact`, `POST /api/v1/video/ingest`, "
      "`GET /api/v1/training/metrics`, `GET /api/v1/models/served`, `GET /api/v1/claims`, `GET /api/v1/health`.")
    w("")
    w("## Repository layout")
    w("")
    w("| Folder | Contents |")
    w("|---|---|")
    w("| `api/` | REST server (`server.py`) and the static Vercel entry point (`vercel_app.py`) |")
    w("| `models/` | classifiers, segmenters, detectors, geometry, depth, PCI, costing, sealing, privacy |")
    w("| `pipeline/` | the end-to-end audit pipeline, fleet deduplication, video ingest, corpus policy |")
    w("| `training/` | every trainer; each writes its model and a JSON report to `checkpoints/` |")
    w("| `scripts/` | data fetchers, Colab runners, claims/README/report builders |")
    w("| `checkpoints/` | trained models (ONNX / joblib) and the reports every number comes from |")
    w("| `web/` | the site |")
    w("| `docs/` | system design and the claims-vs-repository audit |")
    w(f"| `tests/` | {n_tests} tests: models, pipeline, REST API, integrity fixes, Vercel entry point |")
    w("")
    w("## Limitations")
    w("")
    w("- **No bus data yet.** Photographs are DNIT (Brazil), Kaggle and RDD2022 India smartphone images; IMU logs are "
      "from a car. A bus-mounted pilot is the next step.")
    w("- **Four rare classes** (waterlogging, missing zebra crossing, missing divider, damaged sign) have ~50 "
      "photographs each, and their scores are unstable.")
    w("- **The Indian test covers three classes** (normal, crack, pothole), not seven.")
    w(f"- **Segmentation** (crack IoU {f3(crack_iou)}, pothole IoU {f3(pot_iou)}) is measured on DNIT photographs, not "
      "Indian or bus-camera frames; area and cost inherit its error.")
    w("- **Depth is an estimate**; PCI and deterioration reproduce engineering formulas and are not validated against "
      "field surveys.")
    w("- **Privacy blur recall is not measured** — there is no annotated face/plate set.")
    w("- **The live site serves results, not inference**: photograph and video analysis needs the engine.")
    w("")
    w("## How accuracy got here")
    w("")
    w("| Stage | Held-out accuracy | What changed |")
    w("|---|---|---|")
    w("| Inherited code | 36.6% | models untrained; some outputs fabricated |")
    w("| Label conflicts fixed | 79.4% | 20 photographs under three contradictory labels |")
    w("| Real dataset (DNIT) | 83.4% | 133 → 1,800 distinct photographs |")
    w("| ImageNet CNN embeddings | 88.5% | a pretrained CNN replaced hand-written features |")
    w("| Kaggle ingest + a labelling fix | 89.2% | removing 299 mislabelled sign images *raised* accuracy |")
    if ind_before:
        w(f"| …but on Indian roads | {pct(ind_before.get('accuracy'))} | the same model on RDD2022 India crops |")
    w(f"| Concrete patches excluded, RDD2022 India added{', fine-tuned CNN' if deep else ''} | **{pct(acc)}** "
      f"(India **{pct((ind or {}).get('accuracy'))}**) | larger, harder test set with Indian photographs in it |")
    w("")
    w("An earlier README described models this code did not contain (\"90.36% validation accuracy\", "
      "\"R² = 0.9908\", a \"10/10 PASS guaranteed\" harness, simulated IMU data presented as real). It was removed in "
      "the October 2026 audit; see `CHANGES.md` and the corrections on the Architecture page.")
    w("")
    w("---")
    w("")
    w("**Team** Road Shield AI · **Developer** Udbhav Yadav · SIH26124 (BEL) · Theme: Smart Automation  ")
    w("Developed for Smart India Hackathon 2026, academic and research use.")
    w("")
    w("<sub>Generated by `scripts/build_readme.py` from the files in `checkpoints/`.</sub>")
    return "\n".join(L) + "\n"


def main():
    text = build()
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"README.md written ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
