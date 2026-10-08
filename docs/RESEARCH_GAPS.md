# Research gaps this project works on

Each row names a gap in the work this project's report cites, what ROAD-SHIELD does about it, and the
evidence that exists today. "Open" means the code is ready but the measurement that would close the
gap has not been made yet. Nothing here is claimed beyond what the repository can show.

| Gap | What ROAD-SHIELD does | Evidence today | Open |
|---|---|---|---|
| **Indian roads in road-damage benchmarks.** RDD2022 [1] pools six countries; results are usually reported on the pooled set. | A fixed India-only test split, never used for selection. A multi-country detector is served only if it beats the India-only one on India's validation photographs. | `scripts/prepare_rdd2022_world.py`, `scripts/select_rdd_detector.py`, `checkpoints/rdd_detector_selection.json` after training | The multi-country run (Colab stage 3) |
| **Vision-only vs vibration-only sensing.** Accelerometer systems [2, 3] feel a pothole only after driving into it; cameras are fooled by shadows and paint. | Bayesian fusion of both; when the camera cannot see (night, rain, blur) the IMU alone reports, labelled as IMU-only, never as a camera confirmation. | `models/bayesian_fusion_gate.py`, `edge/bus_agent.py`, `edge/frame_quality.py` | A labelled bus drive with both sensors, to measure how many false alarms fusion removes |
| **Confidence that means something.** Road-damage classifiers are usually reported by accuracy alone. A probability that feeds a fusion rule has to be calibrated. | Expected calibration error, NLL and a reliability table on held-out Indian crops; temperature fitted on one half, reported on the other. | `scripts/measure_calibration.py` | The numbers (Colab stage 5) |
| **Accountable crowd-sensed reports.** Crowd-sensing systems [4] aggregate reports, but repairs are paid from them. | Every raw sighting is kept with the defect it merged into. Work orders carry a keyed HMAC seal that cannot be re-computed by whoever edits them. Bus packets are authenticated, so sightings cannot be forged or replayed. | `pipeline/defect_store.py`, `models/morth_dispatch_agent.py`, `edge/crypto.py`, tests | An audit with a municipality's real billing data |
| **Repair order by need, not influence.** | A Priority Index with stated scales and weights, no location or "VIP" input, and a sensitivity check (Kendall τ under weight changes) published with every ranking. | `models/priority_index.py`, `/api/v1/priority/ranking` | Weights agreed with a municipality |
| **Privacy measured, not asserted.** Systems that blur faces and plates rarely report how many they miss. | Recall measured on public annotated sets, by object size, with the share of each photo blurred as the cost. | `scripts/measure_redactor_recall.py` | The numbers (Colab stage 5), then the same on bus frames |
| **Edge claims without device numbers.** | One command that times every model on the device and checks an INT8 copy against FP32. | `scripts/benchmark_edge.py` | Running it on a Raspberry Pi 5 / Jetson |
| **Area and depth from one dashcam frame.** | Per-pixel ground projection from a calibrated camera; depth reported as an interval, never as a measurement. | `models/ipm_homography_engine.py`, `models/depth_estimator.py` | Field measurements to check against |

References are the report's: [1] Arya et al., RDD2022, Geoscience Data Journal 2024; [2] Mednis et al.,
MobiSensor 2011; [3] Eriksson et al., The Pothole Patrol, MobiSys 2008; [4] Zhong et al., Proc. ACM
IMWUT 2019.
