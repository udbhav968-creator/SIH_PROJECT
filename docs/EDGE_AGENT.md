# The bus agent: from a camera on a bus to the live map

`edge/bus_agent.py` is the program the Raspberry Pi in each bus runs. It is the missing half of the
SIH design: until now the engine could analyse a photograph you uploaded, but nothing ran on a bus.

```
camera ─┐                       usable frame ─> full pipeline + IMU window ─> confirmed defect ─┐
MPU-6050 ┼─> frame quality check                                                                   ├─> sealed packet ─> SQLite queue ─> server
Neo-6M ─┘                       unusable ─────> IMU only ─> strong shock ─> IMU-only sighting ─────┘        (encrypted at rest)    /api/v1/fleet/ingest-sealed
                                position every 10 s ───────────────────────────────────────────────┘                                    │
                                                                                                                       ledger + Priority Index + live map
```

## What it does, and the rules it follows

| Step | Module | Rule |
|---|---|---|
| Frame quality | `edge/frame_quality.py` | Sharpness, brightness, clipping, contrast on the road part of the frame. Thresholds near the 1st percentile of 600 project road photographs: 6 of those 600 are rejected. Re-tune on real bus footage with `tune()`. |
| Vision + IMU | `pipeline/deep_inference_pipeline.py` | The full pipeline with the last second of real accelerometer data, so the Bayesian gate fuses both sensors. A visual detection the gate rejects as an optical false alarm is not reported. |
| IMU precedence | `edge/bus_agent.py` | When the camera cannot be used (dark, blurred, glare), a shock the IMU classifier rates ≥ 0.85 pothole, or a jolt above 28 m/s² (beyond 99% of plain-road seconds in the drive logs), is reported as an **IMU-only sighting**, kept apart from the ledger (no area or depth to price). |
| No GPS, no defect | `edge/bus_agent.py` | A defect without a location is never queued; the tick is counted under `skipped_no_gps`. |
| Packet | `edge/crypto.py` | Compact JSON under 1 KB, AES-256-GCM. Bus id and sequence number are authenticated, so a packet cannot be relabelled or replayed. |
| Offline queue | `edge/store_forward.py` | SQLite, sealed envelopes only (encrypted at rest). Sent oldest first; stops at the first failure; backs off 5 s → 5 min. Bounded: drops position heartbeats before events and counts every drop. A packet the server refuses 3 times (wrong key, corrupted row) moves to a dead-letter table so the rest keep flowing; a network outage never dead-letters anything. Each queue database has a random epoch, so a bus with a new SD card is not mistaken for a replay. |
| Server | `pipeline/edge_ingest.py` | Opens packets, refuses any sequence number already seen for that bus and epoch (acknowledged as duplicate, not applied twice), converts every field to its proper type, rejects out-of-range PCI, applies defects to the ledger and publishes everything to the live map. |
| Privacy | `models/privacy_redactor.py` | Photographs never leave the bus. With `--evidence-dir`, the frame behind each defect is kept on the device with people and plates blurred. |

## Try it without a bus (laptop, 2 minutes)

PowerShell, two windows, from the project folder:

```powershell
# window 1 - the engine, with a fleet key
$env:ROAD_SHIELD_FLEET_KEY = "pick-any-long-secret"
python -m api.server 8001
```

```powershell
# window 2 - one bus, replaying project photographs, a real pothole drive log and a 2 km route
$env:ROAD_SHIELD_FLEET_KEY = "pick-any-long-secret"
python -m edge.bus_agent --bus DEMO-1 --server http://127.0.0.1:8001 `
  --replay-frames datasets/02_kaggle_pothole_600 `
  --replay-imu datasets/04_mobile_imu_telemetry_100hz/raw_logs/potholes_1.csv `
  --replay-gps edge/samples/demo_route.csv --interval 1 --duration 120
```

Open http://127.0.0.1:8001/corridor: the bus moves, defects appear and are ranked by Priority Index,
and the live feed lists every event as it arrives. Run a second agent with `--bus DEMO-2` to see two
buses confirm the same defects (deduplication). Stop window 1 for a minute while the agent runs: the
queue grows on the "bus" and empties when the engine is back.

`edge/samples/demo_route.csv` is a demonstration route along a 2 km stretch in central Bengaluru at
25 km/h. It is not a recorded bus trip, and the photographs replayed along it were not taken there.

## The whole city loop in one command

`scripts/demo_city.py` runs three replayed buses on the demo route against a running engine and takes one
repair from order to fleet verification while you watch the Road map and Works pages:

```powershell
# window 1
$env:ROAD_SHIELD_FLEET_KEY = "demo-fleet-key"; $env:ROAD_SHIELD_VERIFY_MIN_HOURS = "0"; $env:ROAD_SHIELD_VERIFY_PASSES = "2"
python -m api.server 8001
# window 2
$env:ROAD_SHIELD_FLEET_KEY = "demo-fleet-key"
python -m scripts.demo_city --server http://127.0.0.1:8001 --buses 3 --minutes 8 --repair-demo
```

Measured here: 3 buses, 8 minutes, 24 defects in the ledger; the top defect's order went issued →
assigned → in progress → repaired, and two bus passes plus the settle window verified it about 3 minutes
after the repair. In another run a bus reported the same defect again after the repair and the order
reopened instead; both are the system working. The two environment variables shorten verification for
the demonstration (the default is 3 passes over 24 hours).

## On the Raspberry Pi

Wiring:

| Part | Pi pins |
|---|---|
| MPU-6050 | VCC → 3.3 V (pin 1), GND → pin 6, SDA → GPIO2 (pin 3), SCL → GPIO3 (pin 5) |
| Neo-6M GPS | VCC → 5 V (pin 2), GND → pin 14, TX → GPIO15/RXD (pin 10) |
| Dashcam | USB (any camera OpenCV opens) |

Enable I²C and the serial port (`sudo raspi-config` → Interface Options), then:

```bash
git clone -b audit-2026-10-03 https://github.com/udbhav968-creator/SIH_PROJECT.git && cd SIH_PROJECT
pip install -r requirements.txt -r edge/requirements-pi.txt
export ROAD_SHIELD_FLEET_KEY=<the same key as the server>
python -m edge.bus_agent --bus BMTC-KA01-1234 --server https://<your engine> --evidence-dir evidence
```

Mount the IMU rigidly to the chassis, not to a panel that rattles, with its z axis vertical. The
calibration profile for the camera (`scripts/calibrate_camera.py`) is what turns pixels into square
metres; without one, areas are labelled as estimates from an assumed mount.

## What has and has not been tested

* Tested here: every module by unit test (`tests/test_edge_and_priority.py`), the server endpoints
  (`tests/test_api_server.py`), and a full replay run against the real server (45 s: 43 frames, 4
  defects, 1 IMU-only shock, 9 positions, every packet delivered in order).
* Not tested: real hardware. The MPU-6050 and Neo-6M readers follow the parts' datasheets and the
  NMEA 0183 standard, but have not run on a Pi with the parts attached. The first bus trial should
  start with `--duration 600` and a look at the counters it prints.
