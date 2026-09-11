# ROAD-SHIELD AI Engine — rewrite notes

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
