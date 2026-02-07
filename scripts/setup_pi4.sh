#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Setup script for RASPBERRY PI 4B
#  Uses: Python 3.11 via uv (inference), system Python (camera)
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

# ── 1. Ensure uv is available (to install Python 3.11) ──────────
banner "Checking for uv (needed to install Python 3.11)"
if command -v uv &>/dev/null; then
  echo "  → Found: $(uv --version)"
else
  echo "  → uv not found, installing ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "  → Installed: $(uv --version)"
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
python3 -m venv --system-site-packages .venv-cam
echo "  → Venv created with $(python3 --version)"

# ── 5. Install camera dependencies ──────────────────────────────
banner "Installing camera dependencies (numpy, pyzmq)"
echo "  → OpenCV is inherited from system-site-packages"
.venv-cam/bin/python -m pip install --upgrade pip setuptools wheel
.venv-cam/bin/python -m pip install -r requirements-camera.txt
echo "  → .venv-cam ready"

# ── 6. Create inference venv (Python 3.11 via uv) ───────────────
banner "Creating .venv-infer (Python 3.11 via uv — needed for piwheels torch)"
echo "  → Pi 4B Cortex-A72 needs torch from piwheels, which requires Python 3.11"
uv venv --python 3.11 .venv-infer
echo "  → Venv created"

# ── 7. Install inference dependencies ────────────────────────────
banner "Installing inference dependencies (torch from piwheels, mediapipe, ultralytics, ncnn)"
echo "  → This is the heaviest step — may take several minutes on Pi 4B"
.venv-infer/bin/python -m ensurepip --upgrade
.venv-infer/bin/python -m pip install --upgrade pip setuptools wheel

echo ""
echo "  → Step 7a: Installing PyTorch from piwheels (Cortex-A72 compatible) ..."
echo "    (piwheels builds wheels ON Raspberry Pi hardware, so they're always compatible)"
.venv-infer/bin/python -m pip install torch \
  --index-url https://www.piwheels.org/simple \
  --extra-index-url https://pypi.org/simple

# Verify torch works before continuing
echo "  → Verifying torch import ..."
if .venv-infer/bin/python -c "import torch; print(f'    torch {torch.__version__} OK')" 2>/dev/null; then
  echo "  → torch verified — no Illegal Instruction"
else
  echo ""
  echo "  ✗ ERROR: torch still gives 'Illegal instruction' on this Pi."
  echo "    The piwheels torch wheel did not get picked up."
  echo "    Try manually: .venv-infer/bin/python -m pip install torch -i https://www.piwheels.org/simple --extra-index-url https://pypi.org/simple"
  echo "    Then re-run this script."
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
