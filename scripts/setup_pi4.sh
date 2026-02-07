#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Setup script for RASPBERRY PI 4B
#  Uses: system Python 3.11 (inference), system Python (camera)
#  Installs torch from piwheels (Cortex-A72 compatible)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL_STEPS=8
step=0

banner() {
  step=$((step + 1))
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  [$step/$TOTAL_STEPS] $1"
  echo "════════════════════════════════════════════════════════════════"
}

# ── 1. Ensure Python 3.11 is available ──────────────────────────
banner "Checking for Python 3.11 (needed for Pi 4B torch compatibility)"
if command -v python3.11 &>/dev/null; then
  PY_INFER="python3.11"
  echo "  → Found: $(python3.11 --version)"
else
  echo "  → python3.11 not found, installing via apt ..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
  if command -v python3.11 &>/dev/null; then
    PY_INFER="python3.11"
    echo "  → Installed: $(python3.11 --version)"
  else
    echo "  ✗ ERROR: Could not install python3.11."
    echo "    On Raspberry Pi OS Bookworm, try: sudo apt install python3.11"
    echo "    Alternatively, use deadsnakes PPA."
    exit 1
  fi
fi

# ── 2. Download face_landmarker.task ─────────────────────────────
banner "Downloading face_landmarker.task (MediaPipe, ~3.6 MB)"
if [ -f face_landmarker.task ]; then
  echo "  → Already exists, skipping"
else
  wget -O face_landmarker.task --progress=bar:force \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  echo "  → Done"
fi

# ── 3. Download yolov8s.pt ───────────────────────────────────────
banner "Downloading yolov8s.pt (YOLO weights, ~22 MB)"
if [ -f yolov8s.pt ]; then
  echo "  → Already exists, skipping"
else
  wget -O yolov8s.pt --progress=bar:force \
    https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
  echo "  → Done"
fi

# ── 4. Create camera venv ────────────────────────────────────────
banner "Creating .venv-cam (system Python, system-site-packages)"
# Use whatever system python3 is available (for PiCamera2 access)
python3 -m venv --system-site-packages .venv-cam
echo "  → Venv created with $(python3 --version)"

# ── 5. Install camera dependencies ──────────────────────────────
banner "Installing camera dependencies (numpy, pyzmq)"
echo "  → OpenCV is inherited from system-site-packages"
.venv-cam/bin/python -m pip install --upgrade pip setuptools wheel
.venv-cam/bin/python -m pip install -r requirements-camera.txt
echo "  → .venv-cam ready"

# ── 6. Create inference venv ─────────────────────────────────────
banner "Creating .venv-infer (Python 3.11)"
$PY_INFER -m venv .venv-infer
echo "  → Venv created with $($PY_INFER --version)"

# ── 7. Install inference dependencies ────────────────────────────
banner "Installing inference dependencies (torch from piwheels, mediapipe, ultralytics, ncnn)"
echo "  → This is the heaviest step — may take several minutes on Pi 4B"
.venv-infer/bin/python -m pip install --upgrade pip setuptools wheel

echo ""
echo "  → Step 7a: Installing PyTorch from piwheels (Cortex-A72 compatible) ..."
.venv-infer/bin/python -m pip install torch --extra-index-url https://www.piwheels.org/simple

# Verify torch works before continuing
echo "  → Verifying torch import ..."
if .venv-infer/bin/python -c "import torch; print(f'    torch {torch.__version__} OK')" 2>/dev/null; then
  echo "  → torch verified"
else
  echo ""
  echo "  ✗ ERROR: torch gives 'Illegal instruction' on this Pi."
  echo "    This means no compatible torch wheel exists for your Python + CPU combo."
  echo "    Try: sudo apt install python3.11 python3.11-venv"
  echo "    Then delete .venv-infer and re-run this script."
  exit 1
fi

echo ""
echo "  → Step 7b: Installing remaining packages (mediapipe, ultralytics, ncnn, opencv) ..."
.venv-infer/bin/python -m pip install -r requirements-infer.txt
echo "  → .venv-infer ready"

# ── 8. Export YOLOv8s to NCNN ────────────────────────────────────
banner "Exporting yolov8s.pt → NCNN format"
if [ -d yolov8s_ncnn_model ]; then
  echo "  → yolov8s_ncnn_model/ already exists, skipping"
else
  echo "  → Running ultralytics export (this may take a few minutes on Pi 4B) ..."
  .venv-infer/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='ncnn')"
  echo "  → yolov8s_ncnn_model/ ready"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL DONE (Pi 4B)! Run with: ./scripts/run_split.sh"
echo "════════════════════════════════════════════════════════════════"
