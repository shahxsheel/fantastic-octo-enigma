#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Setup script for RASPBERRY PI 5
#  Uses: uv + Python 3.12.8 (inference), Python 3.13 (camera)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL_STEPS=7
step=0

banner() {
  step=$((step + 1))
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  [$step/$TOTAL_STEPS] $1"
  echo "════════════════════════════════════════════════════════════════"
}

# ── 1. Download face_landmarker.task ─────────────────────────────
banner "Downloading face_landmarker.task (MediaPipe, ~3.6 MB)"
if [ -f face_landmarker.task ]; then
  echo "  → Already exists, skipping"
else
  wget -O face_landmarker.task --progress=bar:force \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  echo "  → Done"
fi

# ── 2. Download yolov8s.pt ───────────────────────────────────────
banner "Downloading yolov8s.pt (YOLO weights, ~22 MB)"
if [ -f yolov8s.pt ]; then
  echo "  → Already exists, skipping"
else
  wget -O yolov8s.pt --progress=bar:force \
    https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
  echo "  → Done"
fi

# ── 3. Create camera venv ────────────────────────────────────────
banner "Creating .venv-cam (Python 3.13, system-site-packages)"
python3.13 -m venv --system-site-packages .venv-cam
echo "  → Venv created"

# ── 4. Install camera dependencies ──────────────────────────────
banner "Installing camera dependencies (numpy, pyzmq)"
echo "  → OpenCV is inherited from system-site-packages"
.venv-cam/bin/python -m pip install --upgrade pip setuptools wheel
.venv-cam/bin/python -m pip install -r requirements-camera.txt
echo "  → .venv-cam ready"

# ── 5. Create inference venv ─────────────────────────────────────
banner "Creating .venv-infer (Python 3.12.8 via uv)"
if ! command -v uv &>/dev/null; then
  echo "  → uv not found, installing ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12.8 .venv-infer
echo "  → Venv created"

# ── 6. Install inference dependencies ────────────────────────────
banner "Installing inference dependencies (mediapipe, ultralytics, ncnn, opencv, numpy, pyzmq)"
echo "  → This is the heaviest step — may take several minutes"
.venv-infer/bin/python -m ensurepip --upgrade
.venv-infer/bin/python -m pip install --upgrade pip setuptools wheel
.venv-infer/bin/python -m pip install -r requirements-infer.txt
echo "  → .venv-infer ready"

# ── 7. Export YOLOv8s to NCNN ────────────────────────────────────
banner "Exporting yolov8s.pt → NCNN format"
if [ -d yolov8s_ncnn_model ]; then
  echo "  → yolov8s_ncnn_model/ already exists, skipping"
else
  echo "  → Running ultralytics export (this may take a minute) ..."
  .venv-infer/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='ncnn')"
  echo "  → yolov8s_ncnn_model/ ready"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL DONE (Pi 5)! Run with: ./scripts/run_split.sh"
echo "════════════════════════════════════════════════════════════════"
