# Infineon - Raspberry Pi Driver Monitoring System

Real-time camera preview on Raspberry Pi with:
- **YOLO Nano object detection** (NCNN, CPU-only, no torch at runtime)
- **Eye open/close detection** using MediaPipe FaceLandmarker blendshapes
- **Headless-friendly** terminal logs (objects + eye state)

Single Python app, **USB camera only**, same workflow on **Raspberry Pi 4 and Pi 5**.

## Prerequisites

- Raspberry Pi 4 or Pi 5
- USB webcam
- Raspberry Pi OS (or Debian/Ubuntu) with Python 3

## Setup

```bash
git clone https://github.com/shahxsheel/fantastic-octo-enigma.git
cd fantastic-octo-enigma
./setup.sh
```

This installs system libraries (OpenCV/GStreamer), creates a `.venv`, installs Python dependencies, and downloads:
- **face_landmarker.task** (MediaPipe, ~3.6 MB)
- **yolov8n_ncnn_model** (YOLO Nano NCNN; override URL with `YOLO_NCNN_MODEL_URL` if needed)

## Run

```bash
./run.sh
```

Press `q` in the preview window to quit.

### Headless (no display / SSH)

```bash
HEADLESS=1 ./run.sh
```

## What You'll See

**Preview window** (when not headless):
- FPS counter
- YOLO bounding boxes (person + objects)
- Face bounding box + eye state (L/R open %)

**Terminal:** `[single-usb] FPS=...` logs.

## Configuration

Environment variables (set before `./run.sh` or in `run.sh`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HEADLESS` | `0` | Set to `1` for no GUI (e.g. SSH) |
| `CAMERA_INDEX` | `0` | USB camera index (`/dev/videoN`) |
| `CAPTURE_WIDTH` | `640` | Capture width |
| `CAPTURE_HEIGHT` | `480` | Capture height |
| `INFER_WIDTH` | `640` | Inference frame width |
| `INFER_HEIGHT` | `480` | Inference frame height |
| `CAMERA_USE_GSTREAMER` | `1` | Use GStreamer for USB (set by run.sh) |
| `YOLO_MODEL` | `yolov8n_ncnn_model` | NCNN model directory (set by run.sh) |
| `NCNN_THREADS` | `4` | CPU threads for NCNN (set by run.sh) |
| `OMP_NUM_THREADS` | `4` | OpenMP threads (set by run.sh) |
| `SWAP_RB` | `0` | Set to `1` if colors look wrong (e.g. blue tint) |
| `YOLO_CONF` | `0.25` | Detection confidence threshold |
| `YOLO_NMS` | `0.45` | NMS IoU threshold |

Example: `CAPTURE_WIDTH=1280 CAPTURE_HEIGHT=720 ./run.sh`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Run ./setup.sh first` | Run `./setup.sh` in the repo root |
| No USB camera found | Ensure a webcam is connected; check `ls /dev/video*`. Set `CAMERA_INDEX` if needed |
| Colors look blue/wrong | Set `SWAP_RB=1` before `./run.sh` |
| No display / SSH | Use `HEADLESS=1 ./run.sh` |
| `yolov8n_ncnn_model` missing after setup | Set `YOLO_NCNN_MODEL_URL` to a valid tar.gz URL, or add the release asset to the repo and re-run `./setup.sh` |
| GStreamer errors | `./setup.sh` installs GStreamer deps; on non-Debian systems install equivalent libs for OpenCV |

## Repo Layout

```
setup.sh                 One-time setup (apt, .venv, models)
run.sh                   Run app (activates .venv, runs single-USB pipeline)
requirements.txt         Python dependencies
face_landmarker.task     MediaPipe model (downloaded by setup.sh)
yolov8n_ncnn_model/      YOLO Nano NCNN model (downloaded by setup.sh)
src/
  run_single_usb.py      App entry (threaded camera + inference + display)
  camera/
    camera_source.py     USB camera (scans /dev/video*)
  infer/
    face_eye_mediapipe.py  Eye detection
    yolo_detector.py       YOLO NCNN detector
```
