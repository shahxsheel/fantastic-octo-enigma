#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Setup script for RASPBERRY PI 4B
#  Uses: Python 3.11 via uv (inference), system Python (camera)
#  NO torch / ultralytics — uses raw ncnn for YOLO inference
#  Downloads pre-exported NCNN model from GitHub release
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NCNN_URL="https://github.com/shahxsheel/fantastic-octo-enigma/releases/download/v1.0.0/yolo26s_ncnn_model.tar.gz"

TOTAL_STEPS=7
step=0

banner() {
  step=$((step + 1))
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  [$step/$TOTAL_STEPS] $1"
  echo "════════════════════════════════════════════════════════════════"
}

# ── 1. Ensure uv is available ────────────────────────────────────
banner "Checking for uv (needed to install Python 3.11)"
if command -v uv &>/dev/null; then
  echo "  → Found: $(uv --version)"
else
  echo "  → uv not found, installing ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "  → Installed: $(uv --version)"
fi

# ── 2. Download models ───────────────────────────────────────────
banner "Downloading models"

echo "  → face_landmarker.task (MediaPipe, ~3.6 MB) ..."
if [ -f face_landmarker.task ]; then
  echo "    Already exists, skipping"
else
  wget -O face_landmarker.task --progress=bar:force \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  echo "    Done"
fi

echo "  → yolo26s_ncnn_model (pre-exported NCNN, ~32 MB) ..."
if [ -d yolo26s_ncnn_model ]; then
  echo "    Already exists, skipping"
else
  wget -O yolo26s_ncnn_model.tar.gz --progress=bar:force "$NCNN_URL"
  tar xzf yolo26s_ncnn_model.tar.gz
  rm -f yolo26s_ncnn_model.tar.gz
  echo "    Done"
fi

# ── 3. Create camera venv ────────────────────────────────────────
banner "Creating .venv-cam (system Python, system-site-packages)"
python3 -m venv --system-site-packages .venv-cam
echo "  → Venv created with $(python3 --version)"

# ── 4. Install camera dependencies ──────────────────────────────
banner "Installing camera dependencies (numpy, pyzmq)"
echo "  → OpenCV is inherited from system-site-packages"
.venv-cam/bin/python -m pip install --upgrade pip setuptools wheel
.venv-cam/bin/python -m pip install -r requirements-camera.txt
echo "  → .venv-cam ready"

# ── 5. Create inference venv (Python 3.11 via uv) ───────────────
banner "Creating .venv-infer (Python 3.11 via uv)"
uv venv --python 3.11 .venv-infer --clear
echo "  → Venv created"

# ── 6. Install inference dependencies ────────────────────────────
banner "Installing inference dependencies (mediapipe, ncnn, opencv — NO torch)"
echo "  → Pi 4B uses raw ncnn for YOLO — no torch or ultralytics needed"
.venv-infer/bin/python -m ensurepip --upgrade
.venv-infer/bin/python -m pip install --upgrade pip setuptools wheel
.venv-infer/bin/python -m pip install -r requirements-infer-pi4.txt
echo "  → .venv-infer ready"

# ── 7. Verify setup ─────────────────────────────────────────────
banner "Verifying installation"

echo "  → Checking ncnn import ..."
if .venv-infer/bin/python -c "import ncnn; print(f'    ncnn OK')" 2>/dev/null; then
  echo "  → ncnn verified"
else
  echo "  ✗ ncnn import failed"
  exit 1
fi

echo "  → Checking mediapipe import ..."
if .venv-infer/bin/python -c "import mediapipe; print(f'    mediapipe {mediapipe.__version__} OK')" 2>/dev/null; then
  echo "  → mediapipe verified"
else
  echo "  ✗ mediapipe import failed"
  exit 1
fi

echo "  → Checking NCNN model files ..."
if [ -f yolo26s_ncnn_model/model.ncnn.param ] && [ -f yolo26s_ncnn_model/model.ncnn.bin ]; then
  echo "  → NCNN model files present"
else
  echo "  ✗ NCNN model files missing"
  exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL DONE (Pi 4B)! Run with: ./scripts/run_split.sh"
echo ""
echo "  Key difference from Pi 5:"
echo "    - No torch/ultralytics installed (saves ~150 MB)"
echo "    - YOLO runs via raw ncnn (auto-detected at startup)"
echo "════════════════════════════════════════════════════════════════"
