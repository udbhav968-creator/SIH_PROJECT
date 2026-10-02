# ROAD-SHIELD AI Engine — rewrite notes

## Latest: RDD2022 India - training data and an Indian-roads test

`scripts/ingest_rdd2022_india.py` reads the YOLO-format RDD2022 India set
(images/labels split into train/valid/test). Crack (D00/D10/D20) and pothole
(D40) boxes from train+valid become `rddin_*` training crops; photographs with
no label supply capped normal-road crops. The RDD **test** split goes only to
`datasets/_eval_rdd2022_india/`, which no training run reads, and
`scripts/eval_indian_roads.py` scores the served classifier on it - the first
accuracy figure measured on Indian roads. Crops from one photograph share a
group key, so they never straddle a split. Tested on a synthetic RDD layout,
including the guarantee that no test-split photograph reaches a training folder.

## Previous: semantic gate - pothole precision 0.12 -> 0.53

The pixel segmenter judges pixels from 11 local features and marks rough
foreground asphalt and dark crack lines as "pothole". The CNN classifier now
scores overlapping windows, and segmenter pothole pixels it does not see as
pothole are dropped (`models/semantic_gate.py`). On 110 annotated photographs
with no crop in the classifier's training data and 50 clean roads from its test
split: pothole IoU 0.102 -> 0.271, precision 0.12 -> 0.53, recall 0.39 -> 0.36,
crack IoU unchanged, clean roads falsely flagged 8/50 -> 2/50
(`checkpoints/semantic_gate_report.json`). It cannot add pixels the segmenter
missed, so water-filled cavities remain under-detected; that needs a stronger
segmenter or pixel labels for such cavities. Cost: about 1.5 s per image.

## Previous: website brought in line with the models

- Homepage, architecture, data-lineage and model-card pages described the
  previous configuration (ResNet-50 + logistic head, simulated IMU, 2-3 rare-class
  test images). They now describe what is served: MobileNetV2 with flip
  averaging and a cross-validated soft-voting ensemble, real IMU drive logs,
  7-8 rare-class test images, and the segmenter's clean-road false-alarm rate.
- The stale ResNet-50 head and its report are removed. They came from an older
  data snapshot and a scikit-learn 1.9 pickle, and because neither head report
  is flagged active, the homepage could headline the old 89.2% instead of the
  served 88.8%. Retrain it after fetching its backbone if you want that path.
- `.vercelignore` excludes model binaries and local run outputs; the Vercel
  handler reads only the JSON reports. `.gitignore` excludes local datasets,
  YOLO run folders and weights, and `.env*.local`.

## Previous: reproducible retrain, thirteen fixes, honest model selection, API under test

Every shipped checkpoint was retrained under scikit-learn 1.8.0 from a clean
data fetch, the test suite grew from 78 to 140 tests, and the REST API went
from 0% to tested. Numbers below are what the training scripts printed.

### Models retrained

| Model | Held-out result | Notes |
|---|---|---|
| MobileNetV2 + flip TTA -> soft-voting ensemble | **88.8% accuracy, macro-F1 0.747** | chosen by grouped CV over 13 candidates; test scored once |
| same split, hand-crafted baseline | 79.9%, macro-F1 0.520 | |
| HOG/LBP/colour -> SVM (own split) | 86.2% | 807 held-out photographs |
| IMU shock classifier | 87.2% | real Indian-road drive logs, time-block split |
| Defect segmenter | crack IoU 0.231, pothole IoU 0.144; clean-road false blobs 23.3% | shipped: 0.228 / 0.126; pothole precision 0.186 -> 0.242 |
| PCI / deterioration / depth | R² 0.94 / 0.98 / 1.00 | **surrogate fits to engineering formulas, not field accuracy** |

**Classifier: selected honestly, not a measured gain.** The trainer used to fit
three heads and keep whichever scored best on the test set. It now selects by
5-fold cross-validation grouped by photograph inside the training split. The
winner - a soft-voting ensemble of SVC, logistic regression and an MLP on
mirror-averaged embeddings (CV 84.9%, macro-F1 0.769) - beat every single head
in CV and was then scored on the test set once. Its 88.8% sits within two
images of the earlier 89.2%, so this is a more trustworthy number rather than a
higher one. The previous committed head (90.1% on a different split) cannot be
compared fairly: scored on this split its rare-class recall of 96.9% shows most
of those test photographs were in its training data. Flip TTA doubles the
embedding cost to about 30 ms.

**Segmenter: two calibration bugs fixed, false alarms back under the gate.**
The first retrain failed two regression tests (28.3% of clean photographs with a
false blob against a 25% gate; 3 of 5 zebra crossings reported). Swapping models
one at a time showed neither new model failed alone - only the pair. The cause
was in threshold calibration (fixes 11 and 12 below). After both fixes the
pothole threshold is 0.30 rather than 0.20, false blobs fall to
23.3%, pothole IoU and precision both rise, and every guard passes.

The ResNet-50 head was not retrained (its backbone was unreachable from the
build machine) and still carries a scikit-learn 1.9 pickle; it stays dormant
unless the backbone is fetched, in which case retrain it first. DAN-DAG was not
retrained (needs PyTorch, now in `requirements-train.txt`); it stores plain
NumPy arrays and is unaffected by the version issue.

**Surrogate models.** The depth regressor's targets are computed from its own
inputs, so R² = 1.00 means it reproduced a formula. PCI and deterioration are
likewise fitted to ASTM D6433 / HDM-4 curves. Never quote these R² as accuracy.

### Bugs fixed

1. **`train_cnn_head --compare` always crashed** before saving. Training sub-crops
   made the label vector longer than the baseline's feature matrix (4,006 vs
   2,953). The documented reproduce command and stage 5 of `run_full_pipeline`
   could never produce a model. Labels for the baseline now come from the items.
2. **Work-order IDs collided** — five orders on one corridor in one second all
   got the same ID. IDs now carry a random suffix.
3. **`/dispatch/verify-seal` crashed** on a non-object `work_order`; now 400.
4. **Incident reports invented a location** (Delhi) when no GPS was supplied;
   now `gps_coordinates: null`.
5. **`/incidents/alpr` fabricated an incident** from an empty request (demo
   track + Bengaluru GPS, then sealed it). A real track is now required.
6. **The video tracker counted every non-pothole hazard as a crack.** Waterlogging,
   markings and signs now go to `total_unique_other_hazards_counted`.
7. **A speed reading was labelled "reckless lane cutting"**; it is now
   `OVERSPEEDING`. Lane cutting is only inferred from a track.
8. **Fresh edge export crashed**: the exporter read `PCI.SEVERITY_CAP`, removed
   when the PCI engine moved to deduct curves. It now exports the curves.
   `/models/registry` hid this by reusing old files already on disk.

9. **The claims registry described a model that was not running.** M1 was always
   filled from the ResNet-50 report, but the server serves MobileNetV2 whenever
   the ResNet-50 backbone is absent - which it is in this repository. M1 now
   follows the head the loader would actually pick. Its PCA-variance figure was
   also hand-carried (45.1%) while the retrained model keeps 41.5%; it is now
   measured from the fitted model on every refresh.
10. **The robustness check's level was misleading.** It samples up to 12 images
    per class, so the rare classes are half the set and the figure (47.6%) is
    class-balanced accuracy, not comparable to headline accuracy. The output
    and report now say so; read the deltas under degradation.

11. **Segmenter calibration never saw a clean road.** It was passed 240 defect
    photographs followed by the clean ones and truncated to the first 60, so
    the clean-road photographs that exist to stop zebra-crossing false alarms
    never influenced the operating point. It now samples 60 defect and 30 clean.
12. **Threshold near-ties went to the loosest option.** The pothole IoU curve is
    nearly flat (0.070 at both 0.20 and 0.30), and the first maximum on an
    ascending grid always won. Within 3% of the peak, the strictest threshold
    now wins: when the objective cannot tell them apart, fewer false defects is
    the better answer.
13. **The classifier head was chosen on the test set.** Now chosen by grouped
    cross-validation on the training split; the test set is scored once.

### Dependencies

- `scikit-learn>=1.8.0,<1.9` — the checkpoints had been pickled under 1.9.0
  while this file capped the version below 1.9, so every load crossed a
  version. All retrained checkpoints now load with zero version warnings.
- `onnxruntime` added. The CNN head was trained on ONNX Runtime embeddings;
  without it the embedder silently falls back to cv2.dnn.
- `requirements-train.txt` for the optional PyTorch trainers.

### Tests

| File | Tests | Covers |
|---|---|---|
| `tests/test_api_server.py` (new) | 34 | every page and frontend endpoint, tamper seal, path traversal, CORS, malformed input, video probe + ingest, edge export, Vercel entrypoint |
| `tests/test_untested_modules.py` (new) | 28 | ALPR, video tracker, augmentation, late fusion, ADAS policy, CAN encoding, edge exporter |
| existing three files | 78 | unchanged |

The API tests run the real server on an ephemeral port against a temporary
ledger. `ROAD_SHIELD_WRITABLE_DIR` (new) points the server's writable state at
any directory; the tests use it so they never touch `checkpoints/road_shield.db`.

---

## Previous: deep CNN embeddings replace hand-engineered features

The classifier's input used to be HOG gradients, local binary patterns and
colour histograms — 4,419 numbers written by hand. It is now the output of a
ResNet-50 trained on ImageNet's 1.28 million images, run frozen through ONNX
Runtime, with a class-balanced logistic head learning the mapping onto the seven
road classes. This is ordinary transfer learning; the only unusual choice is
ONNX rather than PyTorch, made because PyTorch's DLLs will not load on the
demo laptop and ONNX Runtime installs everywhere.

Measured on the identical grouped split (243 images from 205 photographs the
model never saw, scored once):

| Features | Accuracy | macro-F1 |
|---|---|---|
| ResNet-50 embeddings → logistic | **88.5%** | 0.846 |
| ResNet-50 embeddings → SVC-RBF | 90.5% | 0.813 |
| MobileNetV2 embeddings → logistic | 83.5% | 0.868 |
| HOG + LBP + colour → SVM (previous) | 87.7% | 0.840 |

Selection is by the mean of accuracy and macro-F1. SVC-RBF has the highest raw
accuracy but loses two of the rare classes entirely; macro-F1 alone swings by
0.1 on a single image when four classes have two or three test examples. The
mean is the compromise, and it is stated here rather than buried, because either
metric alone could have been quoted to flatter a different model.

Per class, the change fixed exactly what was broken: Damaged Traffic Sign F1
0.00 → 1.00, Missing Road Divider 0.44 → 0.80, Normal Road 0.80 → 0.98.

New files: `models/cnn_embedder.py`, `training/train_cnn_head.py`,
`scripts/fetch_cnn_backbone.py`. `models/deep_vision_net.py` gained
`CNNHeadClassifier`, and `load_best_vision_model` now prefers it over the
baseline when a backbone is on disk.

A head is only ever paired with the backbone it was trained on. ResNet-50 and
MobileNetV2 embeddings are unrelated vector spaces, so feeding one head the
other's vectors would produce confident nonsense with no error; the loader skips
any head whose backbone file is absent rather than substituting what it finds.

The ResNet-50 file is 98 MB and is not in version control — GitHub rejects files
that size, and weights do not belong in git. `python -m scripts.fetch_cnn_backbone`
downloads it from the official ONNX Model Zoo. MobileNetV2 (14 MB) is committed,
so a fresh clone has a working CNN path with no download at all.

The rewrite covers `models/`, `pipeline/`, `api/`, `services/`,
`training/`, `data/`, plus the trained models in `checkpoints/`,
`requirements.txt` and the Vercel config.
`tests/`, `datasets/*.py` scripts and `README.md` were intentionally left
untouched. The frontend was only changed to stop hard-coding the local
API address (see "Vercel deployment").

## What changed and why

The previous version of this codebase had several places where the
"AI models" were either untrained NumPy arrays of random weights, or
where a "sensor fusion" step was silently synthesizing one sensor's
reading from another model's own output, or where a demo path
returned data made up on the spot instead of the model's real
output. This pass replaces every one of those with something that
actually computes what it claims to compute, and — where a claimed
capability genuinely doesn't exist yet (e.g. a real dashcam video
dataset) — reports that honestly instead of faking it.

**models/**
- `forensic_audit_engine.py` — the old "deep metric embedder" duplicate
  detector (random untrained weights) is replaced by a real DCT-based
  perceptual hash (`ForensicDuplicateHasher`). SSIM/Laplacian-variance
  texture checks now prefer `skimage`/`opencv`'s real implementations,
  with a manual fallback if those aren't installed.
- `multimodal_transformer_fusion.py` — the old "5-modality cross-attention
  transformer" fabricated 3 of its 5 claimed sensor inputs (no LiDAR,
  CAN-bus, or environment-vector data exists in this project). Replaced
  with an honest weighted late-fusion of the two modalities this project
  actually has trained models for: vision + IMU.
- `edge_model_exporter.py` — used to dump the untrained fake network's
  random weight arrays as a "neural spec". Now introspects the real
  fitted scikit-learn pipelines (PCA/SVC/RandomForest) and only
  hand-ports the genuinely deterministic formula engines (PCI,
  deterioration) to C — it does not pretend to export an SVM to C.
- `alpr_incident_tracker.py` — license-plate text used to be randomly
  generated. Now does real classical-CV plate localization (Canny +
  contour filtering) and real Tesseract OCR, honestly reporting
  `detected: False` when there's no image or no plate found.
- `morth_dispatch_agent.py`, `realworld_video_tracker.py` — logic was
  already real; only docstrings were cleaned up.

**pipeline/**
- `deep_inference_pipeline.py` — rewritten end-to-end. The most
  important fix: when no real IMU telemetry is supplied for a frame,
  the pipeline now honestly reports `available: False` and falls back
  to a neutral prior, instead of the old behavior of fabricating an
  IMU reading from the vision model's own prediction (which made the
  "dual-sensor Bayesian fusion" claim circular and meaningless).
- `fleet_deduplication_engine.py` — fabricated fallback address/
  elevation values replaced with honest `None` + a `geocode_source`
  field; added a real `get_deduplication_stats()` computed from actual
  ingested/registered counts instead of a hardcoded percentage.
- Removed `pipeline/google_maps_service.py`, a byte-identical unused
  duplicate of `services/google_maps_service.py`.

**data/**
- `dataset_generator.py`, `benchmark_dataset_hub.py`,
  `realworld_media_engine.py` — these used to fabricate entire
  datasets (Gaussian noise dressed up as "RDD2022", "Kaggle
  Pothole-600", etc., with invented sample counts) and a curated video
  catalog that doesn't exist on disk. Now they report real counts read
  straight from `datasets/`, and honestly report that no dashcam video
  files are present (`datasets/08_dashcam_video_streams` is empty)
  instead of returning fabricated clip metadata.

**training/**
- `mega_pipeline.py` — used to run a fake NumPy training loop over the
  fabricated datasets above and write out an invented "model zoo"
  with made-up accuracy numbers. Now it calls this project's two real
  training scripts (`train_vision.py`, `train_imu.py`) and reports
  their genuine held-out accuracy, confusion matrix, and per-class
  precision/recall/F1.
- Removed five training scripts that were fully superseded and, after
  the `data/` rewrite, broken (`ImportError` on load): `train_all.py`,
  `train_deep_suite.py`, `train_deep_vision_suite.py`,
  `train_forensic_embedder.py`, `train_realworld_vision.py`. They
  claimed to train on "100,000+ samples" across benchmarks that were
  never actually in this repo.
- `train_mega_suite.py` — rewritten as a small, working CLI entrypoint
  (`python3 -m training.train_mega_suite`) that runs the real training
  suite and prints/saves the genuine results.

**api/ and services/**
- `server.py` — rewritten endpoint-by-endpoint. Removed the
  `preferred_class` parameter that let a caller force the vision
  classifier's answer (the "classifier-rigging" cheat), random-noise
  image generation standing in for real forensic photos, a fake IMU
  generator, and several hardcoded fake payloads (ledger list, fleet
  stats, model registry). All endpoints now call into the real,
  rewritten models above. Every endpoint path is unchanged, for
  frontend and `tests/test_api_server.py` compatibility.
- `google_maps_service.py` — rewritten. An earlier version of these
  notes called this file's fallbacks genuine; that was wrong. When no
  Google key was set it invented elevations from a sine/cosine formula,
  returned the same four made-up "nearby facilities" (hospital, depot,
  police post) at fixed offsets from any point, answered unknown searches
  with Silk Board, drew fake curved routes with invented turn
  instructions, and labelled every address outside its table as
  "Bengaluru". Now each lookup uses a real source (Google → OpenStreetMap
  Nominatim / OSRM / Overpass → Open-Meteo elevation) and reports
  `UNAVAILABLE`, or a clearly labelled straight-line / city-level
  estimate, when none answers. Pothole avoidance now picks, among the
  router's real alternative routes, the one passing the fewest known
  defects instead of nudging line coordinates off the road. Drainage risk
  is a documented local-relief heuristic (point vs. ring 250 m away).

## Vercel deployment

- `vercel.json`: removed the 15 MB `maxLambdaSize` cap, which the real
  scikit-learn / SciPy / scikit-image / OpenCV stack cannot fit under.
  Python functions on Vercel may be up to 500 MB uncompressed; this stack
  is roughly 400 MB, so it should fit, but it is close to the limit.
- `.vercelignore`: training datasets are still excluded, except the small
  IMU validation split so `/api/v1/telemetry/imu` can serve real windows.
- On Vercel the deployed files are read-only, so the server writes runtime
  files (feedback log, exported specs) to `/tmp`, `/api/v1/training/launch`
  returns a clear 503 (train locally, commit `checkpoints/`, redeploy), and
  endpoints that need the photo datasets return 503 instead of crashing.
- Startup no longer calls external geocoding services (demo defects are
  geocoded lazily on first map load), so cold starts don't wait on them.
- Frontend: 9 API calls that were hard-coded to `http://127.0.0.1:8000`
  now use the page's own origin, so the dashboard talks to the Vercel
  backend when served from Vercel (and still to the local server locally).

## Honest limitations, stated plainly

- The vision classifier's held-out accuracy is currently ~37% on 7
  classes (vs. a 14% random-guess baseline). That's real, and it's
  low mainly because several classes have very few distinct source
  photos (see the `dataset_inventory` returned by
  `/api/v1/datasets/benchmarks`) — a data problem, not a hidden one.
  More labeled photos per class is the fix, not more code.
- The IMU shock classifier scores 100% on its held-out split, but its
  data is the simulated 100Hz accelerometer windows shipped in this
  repo, not field-collected vehicle logs — treat it as validated
  against the simulator until retrained on real sensor data.
- The ASTM D6433 PCI "deduct value" curve is a documented analytic
  stand-in shaped like the real published curves, not a pixel-accurate
  digitization of them (the real curves are charts, not equations).
  The surrounding iterative-correction algorithm is the real ASTM
  procedure.

## What will change in `tests/`

`tests/` was left untouched per scope, but several test files
exercise the exact fabricated behaviors this pass removed (e.g. the
old `preferred_class` cheat, `generate_forensic_triplets`,
`data.massive_dataset_generator`, `PCIRegressorNet`'s old fake-weight
interface). Those specific tests will now fail or error on import —
that's expected, since they were testing behavior that no longer
exists on purpose. Everything under `tests/test_api_server.py` that
hits real endpoint paths with real payloads should still pass, since
every endpoint path was preserved.
