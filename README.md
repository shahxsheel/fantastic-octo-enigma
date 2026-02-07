# Infineon - Raspberry Pi Driver Monitoring System

Real-time camera preview on Raspberry Pi with:
- **YOLOv8s object detection** (NCNN-optimized for CPU)
- **Eye open/close detection** using MediaPipe FaceLandmarker blendshapes
- **Headless-friendly terminal logs** (objects + eye state)

Supports **Raspberry Pi 5** and **Raspberry Pi 4B**.

## Architecture

Two processes communicate over ZMQ on localhost:

| Process | Venv | Role |
|---------|------|------|
| Camera | `.venv-cam` | PiCamera2/USB capture, GUI preview, publishes frames |
| Inference | `.venv-infer` | YOLO + MediaPipe eye detection, publishes results |

- Frames: `tcp://127.0.0.1:5555`
- Results: `tcp://127.0.0.1:5556`

| | Pi 5 | Pi 4B |
|---|------|-------|
| Camera Python | 3.13 (system) | system `python3` |
| Inference Python | 3.12.8 (via `uv`) | 3.11 (piwheels-compatible) |
| torch source | PyPI | piwheels (Cortex-A72 safe) |

## Prerequisites

- Raspberry Pi 5 or Pi 4B with Pi Camera Module 2 or USB webcam
- Raspberry Pi OS Bookworm 64-bit with PiCamera2 installed
- **Pi 5 only:** [`uv`](https://docs.astral.sh/uv/) installed:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Setup

The setup script **auto-detects your Pi model**:

```bash
git clone https://github.com/shahxsheel/fantastic-octo-enigma.git
cd fantastic-octo-enigma
./scripts/setup_split_envs.sh
```

Or run the model-specific script directly:

```bash
./scripts/setup_pi5.sh     # Raspberry Pi 5
./scripts/setup_pi4.sh     # Raspberry Pi 4B
```

This will:
1. Download `face_landmarker.task` (MediaPipe) and `yolov8s.pt` (YOLO weights)
2. Create `.venv-cam` (system Python + PiCamera2 + OpenCV)
3. Create `.venv-infer` (MediaPipe + YOLO + NCNN + torch)
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
| `Illegal instruction` on Pi 4B | Use `./scripts/setup_pi4.sh` (installs Pi 4B-compatible torch from piwheels) |
| `imshow` not implemented | System OpenCV missing GTK support. Use `HEADLESS=1` or install `libgtk-3-dev` and reinstall OpenCV |

## Scripts

### `scripts/setup_split_envs.sh`

One-time setup. Run this after cloning the repo. **Auto-detects Pi model** and runs the right script.

```bash
./scripts/setup_split_envs.sh
```

### `scripts/setup_pi5.sh`

Pi 5 specific setup. Uses `uv` + Python 3.12.8 for inference, Python 3.13 for camera.

What it does (7 steps):
1. Downloads `face_landmarker.task` (3.6 MB) from Google
2. Downloads `yolov8s.pt` (22 MB) from Ultralytics GitHub
3. Creates `.venv-cam` (Python 3.13, system-site-packages)
4. Installs camera deps (numpy, pyzmq — OpenCV inherited from system)
5. Creates `.venv-infer` (Python 3.12.8 via `uv`)
6. Installs inference deps (mediapipe, ultralytics, ncnn, opencv, torch)
7. Exports `yolov8s.pt` → NCNN format

### `scripts/setup_pi4.sh`

Pi 4B specific setup. Uses Python 3.11 + piwheels torch (Cortex-A72 compatible).

What it does (8 steps):
1. Checks/installs Python 3.11 (needed for Cortex-A72 compatible torch wheels)
2. Downloads `face_landmarker.task` (3.6 MB)
3. Downloads `yolov8s.pt` (22 MB)
4. Creates `.venv-cam` (system Python, system-site-packages)
5. Installs camera deps (numpy, pyzmq)
6. Creates `.venv-infer` (Python 3.11)
7. Installs inference deps — torch from piwheels first, then rest of packages
8. Exports `yolov8s.pt` → NCNN format

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
  setup_split_envs.sh       Auto-detect Pi model → run right setup
  setup_pi5.sh              Pi 5 setup (uv + Python 3.12.8)
  setup_pi4.sh              Pi 4B setup (Python 3.11 + piwheels)
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
