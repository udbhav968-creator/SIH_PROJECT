# ROAD-SHIELD datasets — what is actually here

This file used to describe "7 canonical benchmarks, 73,060+ samples" with
tensors such as `rdd2022_train.npz` (16,000 x 64), `kaggle_potholes.npz`
(10,000), 15,000 IMU sequences and "NHAI Monsoon Degradation Logs". None of
those files exist and none of those counts were ever true. It has been
rewritten from the folders on disk (October 2026 audit).

## Folder names are historical — read the source column

The class folders keep their original names so existing paths keep working,
but several names do not describe their contents:

| Folder | Class it trains | What is really inside (tracked in git) | Regenerated locally (not in git) |
|---|---|---|---|
| `05_morth_civil_hard_negatives/` | 0 Normal road | 349 + 23 Kaggle pothole-dataset "plain road" images, 240 `aug_mega_*` augmented copies, 1,193 Kaggle concrete-wall patches (excluded, see below) | `cpr_*` DNIT lane-only crops, `rddin_*` RDD2022 India clean-road crops |
| `03_crack500_fatigue/` | 1 Crack | 30 DNIT road photographs, 1,183 Kaggle concrete-crack patches (excluded) — **no CRACK500 images** | `cpr_*` DNIT crack crops (1,200), `rddin_*` RDD2022 India crack crops |
| `02_kaggle_pothole_600/` | 2 Pothole | 572 + 327 + 16 Kaggle pothole images (annotated crops, mixed, plain), 24 DNIT photographs | `cpr_*` DNIT pothole crops, `rddin_*` RDD2022 India pothole crops |
| `09_waterlogging_hazard/` | 3 Waterlogging | 14 field photographs (`real_*`) + 35 Wikimedia Commons photographs (`WM_*`) | — |
| `10_missing_zebra_crossing/` | 4 | 19 field + 35 Wikimedia | — |
| `11_missing_road_divider/` | 5 | 20 field + 35 Wikimedia | — |
| `12_damaged_traffic_signs/` | 6 | 14 field + 35 Wikimedia (299 intact-sign images removed as mislabelled) | — |
| `01_rdd2022_india/` | none (hold-out folder) | 15 DNIT (Brazilian) photographs + 240 `aug_*` augmented copies. **Not RDD2022 data** - real RDD2022 India crops are the `rddin_*` files in the class folders. | — |
| `06_astm_d6433_pci_benchmark/`, `07_monsoon_pavement_deterioration/` | none | 240 `aug_*` augmented road photographs each. **No ASTM or monsoon data.** The PCI and deterioration models are fitted to engineering formulas in `training/train_civil_models.py`, not to these folders. | — |
| `08_dashcam_video_streams/` | none | 10 dashcam frames, used by the video tests | — |
| `13_urban_traffic_vehicles/`, `14_pedestrian_safety/` | none | 4 and 2 photographs, used by detector smoke tests | — |
| `04_mobile_imu_telemetry_100hz/` | IMU model | 10 real drive logs (205,501 rows at 100 Hz) from [VishalSingh25/Pothole-Project](https://github.com/VishalSingh25/Pothole-Project) cut into 688 train + 164 held-out 1-second windows, time-block split | — |

**Excluded from road-scene training:** the 2,376 `kag_surface-crack_*` files are
227x227 close-ups of concrete and plaster from a wall-crack dataset.
`pipeline/corpus_policy.py` excludes them because they have no road, horizon or
camera geometry. Set `ROAD_SHIELD_NO_CORPUS_FILTER=1` to reproduce unfiltered numbers.

## The real external sources

| Source | Script | What it produces |
|---|---|---|
| DNIT "Cracks and Potholes in Road Images" (Brazil, 2,235 photographs, COCO polygons) — [GitHub mirror](https://github.com/andrijdavid/Cracks-and-Potholes-in-Road-Images-Dataset) | `python -m scripts.fetch_cracks_potholes_dataset --limit 2235` | `cpr_*` crack / pothole / clean-lane crops, max 1,200 per class, perceptual-hash de-duplicated |
| RDD2022 India (CRDDC 2022, official archive, Pascal VOC) — [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector) | `python -m scripts.prepare_rdd2022_voc --src <unzipped>/India` then `python -m scripts.ingest_rdd2022_india` | `rddin_*` training crops from the train+valid photographs; the test photographs go only to `datasets/_eval_rdd2022_india/`, which no training run reads |
| Kaggle pothole / plain-road collections | `python -m scripts.fetch_kaggle_datasets` | `kag_pothole-*` files (pHash de-duplicated, Hamming <= 8) |
| Wikimedia Commons / field photographs | `python -m scripts.fetch_urban_hazard_photos` | `WM_*` files for the four municipal classes |
| IMU drive logs | `python -m scripts.fetch_real_imu_dataset` | the two `.npz` window files |

`scripts/colab_train_all.sh` runs all of the regeneration steps, prints an
inventory (`logs/corpus_inventory.json`) and retrains every model on the result.

## What this data cannot support

- No image here was taken from a bus. Indian-road evidence comes from RDD2022
  India (smartphone, car-mounted) only.
- The four municipal classes have about 50 photographs each, so their test
  scores rest on 7–8 images and move a lot with a single mistake.
- There are no pixel-level masks for water-filled potholes, no depth ground
  truth and no field PCI surveys.
