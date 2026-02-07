# Infineon - Raspberry Pi 5 Driver Monitoring System

Real-time camera preview on Raspberry Pi 5 with:
- **YOLOv8s object detection** (NCNN-optimized for CPU)
- **Eye open/close detection** using MediaPipe FaceLandmarker blendshapes
- **Headless-friendly terminal logs** (objects + eye state)

## Architecture

Two processes communicate over ZMQ on localhost:

| Process | Venv | Python | Role |
|---------|------|--------|------|
| Camera | `.venv-cam` | 3.13 (system) | PiCamera2/USB capture, GUI preview, publishes frames |
| Inference | `.venv-infer` | 3.12.8 (uv) | YOLO + MediaPipe eye detection, publishes results |

- Frames: `tcp://127.0.0.1:5555`
- Results: `tcp://127.0.0.1:5556`

## Prerequisites

- Raspberry Pi 5 (tested on 8GB) with Pi Camera Module 2 or USB webcam
- Raspberry Pi OS Bookworm 64-bit with PiCamera2 installed
- `python3.13` available (ships with Bookworm)
- [`uv`](https://docs.astral.sh/uv/) installed:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Setup

```bash
git clone <your-repo-url>
cd infineon
./scripts/setup_split_envs.sh
```

This will:
1. Download `face_landmarker.task` (MediaPipe) and `yolov8s.pt` (YOLO weights)
2. Create `.venv-cam` (Python 3.13 + PiCamera2 + OpenCV)
3. Create `.venv-infer` (Python 3.12.8 + MediaPipe + YOLO + NCNN)
4. Export `yolov8s.pt` to NCNN format (`yolov8s_ncnn_model/`)

## Run

```bash
./scripts/run_split.sh
```

Or equivalently:

```bash
python main.py
```

Press `q` in the preview window to quit. Press `b` to toggle channel swap if colors look wrong.

### Headless (no display / SSH)

```bash
HEADLESS=1 ./scripts/run_split.sh
```

## What You'll See

**Preview window** (when not headless):
- FPS counter
- YOLO bounding boxes (filtered to: person, cell phone, bottle, cup)
- Face bounding box
- Eye state + openness percentage (L/R)

**Terminal logs** (always, works headless):
```
[infer] objects=[person:0.86, cell phone:0.41]
[infer] eyes=L OPEN 85% | R OPEN 90%
[infer] eye_state_change  L CLOSED 12%  R CLOSED 8%
```

## Configuration

All configuration is via environment variables. Defaults work out of the box.

### Camera

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCE_CAMERA` | `auto` | `auto`, `usb`, or `pi` |
| `CAMERA_INDEX` | `0` | USB camera index hint |
| `CAPTURE_WIDTH` | `640` | Preview resolution width |
| `CAPTURE_HEIGHT` | `480` | Preview resolution height |
| `INFER_WIDTH` | `416` | Inference frame width |
| `INFER_HEIGHT` | `312` | Inference frame height |
| `CAMERA_FPS` | `30` | Target FPS |
| `SWAP_RB` | `0` | Swap B/R channels (fix blue tint) |

### YOLO

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8s_ncnn_model` | Model path |
| `YOLO_CONF` | `0.25` | Confidence threshold |
| `YOLO_FILTER` | `person,cell phone,bottle,cup` | Classes to detect |
| `YOLO_EVERY_N` | `4` | Run YOLO every N frames |

### Eye Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `EYE_CLOSED_THRESHOLD` | `0.5` | Blink score above this = CLOSED |
| `FACE_EVERY_N` | `1` | Run face detection every N frames |
| `EYE_TIMING` | `0` | Set to `1` for per-frame timing logs |

### Logging / IPC

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_EVERY_SEC` | `1.0` | Periodic log interval |
| `YOLO_LOG` | `1` | Enable YOLO object logs |
| `EYE_LOG` | `1` | Enable eye state logs |
| `FRAMES_ADDR` | `tcp://127.0.0.1:5555` | ZMQ frames address |
| `RESULTS_ADDR` | `tcp://127.0.0.1:5556` | ZMQ results address |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Colors look blue/wrong | Set `SWAP_RB=1` or press `b` in preview |
| `Failed to acquire Pi camera (device busy)` | Close other libcamera apps and retry |
| `Results port already in use` | `run_split.sh` auto-kills old processes; otherwise change `RESULTS_ADDR` |
| `could not connect to display` | Use `HEADLESS=1` or run from Pi desktop |

## Scripts

### `scripts/setup_split_envs.sh`

One-time setup. Run this after cloning the repo.

```bash
./scripts/setup_split_envs.sh
```

What it does:
1. Downloads `face_landmarker.task` (MediaPipe FaceLandmarker model, 3.6MB) from Google
2. Downloads `yolov8s.pt` (YOLOv8s weights, 22MB) from Ultralytics GitHub (skipped if already present)
3. Creates `.venv-cam` using `python3.13 -m venv --system-site-packages` and installs `requirements-camera.txt` (OpenCV, numpy, pyzmq)
4. Creates `.venv-infer` using `uv venv --python 3.12.8` and installs `requirements-infer.txt` (MediaPipe, Ultralytics YOLO, NCNN, OpenCV, numpy, pyzmq)
5. Exports `yolov8s.pt` to NCNN format (`yolov8s_ncnn_model/`) for fast CPU inference (skipped if already present)

### `scripts/run_split.sh`

Runs the full system. Requires venvs to already exist (exits with an error if they don't).

```bash
./scripts/run_split.sh
```

What it does:
1. Checks that `.venv-infer` and `.venv-cam` exist
2. Auto-detects `HEADLESS` mode (on if no `$DISPLAY`)
3. Kills any previous stuck instances to free ZMQ ports
4. Starts the **inference process** in the background (`.venv-infer/bin/python -m src.infer.run_infer`)
5. Starts the **camera process** in the foreground (`.venv-cam/bin/python -m src.camera.run_camera`)
6. On exit (Ctrl-C or `q` in preview), cleans up both processes

Environment variable overrides:
```bash
# Force Pi camera, headless, fix colors
FORCE_CAMERA=pi HEADLESS=1 SWAP_RB=1 ./scripts/run_split.sh

# Custom resolution
CAPTURE_WIDTH=1280 CAPTURE_HEIGHT=960 ./scripts/run_split.sh
```

## Repo Layout

```
main.py                     Launcher (starts both processes)
scripts/
  setup_split_envs.sh       One-time setup (venvs + model download)
  run_split.sh              Run both processes
src/
  camera/
    camera_source.py         PiCamera2 / USB camera abstraction
    run_camera.py            Camera process (capture + preview + overlays)
  infer/
    face_eye_mediapipe.py    Eye detection (FaceLandmarker blendshapes)
    yolo_detector.py         YOLOv8s object detection
    run_infer.py             Inference process (eyes + YOLO)
  ipc/
    zmq_frames.py            ZMQ frame pub/sub
    zmq_results.py           ZMQ results pub/sub
yolov8s_ncnn_model/          NCNN model (downloaded + exported by setup script)
yolov8s.pt                   YOLOv8s weights (downloaded by setup script)
requirements-camera.txt      Camera venv dependencies
requirements-infer.txt       Inference venv dependencies
```
