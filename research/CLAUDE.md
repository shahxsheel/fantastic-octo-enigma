# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time driver monitoring system (DMS) for Raspberry Pi 4/5. Detects drowsiness, distraction, and speeding using USB camera + YOLO Nano (NCNN) + MediaPipe. CPU-only inference, targeting 6–20 FPS on Pi hardware.

## Setup & Run

```bash
./setup.sh        # One-time: system libs, .venv, downloads models
./run.sh          # Run the app (activates .venv, sets env vars)
HEADLESS=1 ./run.sh  # SSH/headless mode (no GUI)
```

## Testing

```bash
python -m pytest tests/ -v
# or
python -m unittest discover tests -v
```

Test files are in `tests/` and cover: risk engine, driver lock, face/eye cache, camera scheduling, and inference scheduling.

## Configuration

All tuning is done via environment variables (no config file). Key ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `HEADLESS` | `0` | `1` = no GUI |
| `CAMERA_INDEX` | `0` | USB camera `/dev/videoN` |
| `INFER_WIDTH`/`INFER_HEIGHT` | `256`/`256` | Inference resolution (lower = faster) |
| `YOLO_FILTER` | `person,cell phone,bottle,cup` | Whitelisted COCO classes |
| `LOW_LIGHT` | `1` | Gamma correction for night |
| `SUPABASE_URL`/`SUPABASE_KEY` | — | Optional cloud telemetry |
| `VEHICLE_ID` | `test-vehicle-1` | Supabase telemetry ID |

## Architecture

### Threading Model (`src/run_single_usb.py`)

Four concurrent threads communicate via shared state protected by a single lock:

1. **Camera thread** — USB camera producer; uses double-buffer pattern (writes to `_back`, swaps pointers with `_front` under lock — zero allocation per frame)
2. **Inference thread** — Runs MediaPipe face detection every frame; YOLO every 10 frames (reuses last detections between frames); updates `_shared_results`
3. **Telemetry thread** — Drains a bounded queue (`maxsize=50`) to Supabase asynchronously
4. **Speed limit thread** — Queries OSM Overpass API every 10s for local speed limit
5. **Main thread** — Reads `_shared_results`, renders GUI at 30 FPS, prints CLI stats at 0.5s intervals, fires buzzer alerts

### Key Design Decisions

- **YOLO interleaving:** YOLO runs every 10 frames to save ~90% of its compute cost
- **256×256 inference:** YOLO Nano at 256 input is ~1.6× faster than 320 and is the default on Pi4/Pi5
- **Hardware fallback:** `buzzer.py` and `gps.py` both fall back to simulated/mock mode when GPIO/serial is unavailable (safe for non-Pi dev)
- **Risk engine** (`src/infer/risk_engine.py`): State machine (FOCUSED → WARN → ALERT) using PERCLOS over a 20s sliding window, head pose thresholds, phone detection, and face visibility

### Pipeline

```
USBCameraSource → FrameBundle → inference_thread_fn
                                  ├── FaceEyeEstimatorMediaPipeSync (every frame)
                                  ├── YoloDetector (every 10 frames)
                                  └── RiskEngine → _shared_results
                                                        └── main thread → GUI + buzzer + telemetry
```

### Model Files

- `face_landmarker.task` — MediaPipe face model (downloaded by `setup.sh`)
- `yolo26n_ncnn_model/` — YOLO Nano NCNN weights (downloaded by `setup.sh`)

These are runtime-required; the app will not start without them.
