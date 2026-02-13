#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Unified setup for Raspberry Pi 4 and Pi 5
#  USB camera + NCNN (CPU) inference. Single .venv, no split processes.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Default URL for YOLO Nano NCNN model (override with YOLO_NCNN_MODEL_URL).
# Use Nano only — Small model is too slow for real-time Pi inference.
YOLO_NCNN_URL="${YOLO_NCNN_MODEL_URL:-https://github.com/shahxsheel/fantastic-octo-enigma/releases/download/v1.1.0/yolov8n_ncnn_model.tar.gz}"
FACE_MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

TOTAL_STEPS=5
step=0

banner() {
  step=$((step + 1))
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  [$step/$TOTAL_STEPS] $1"
  echo "════════════════════════════════════════════════════════════════"
}

# ── 1. System dependencies (OpenCV / GStreamer) ─────────────────────
banner "Installing system dependencies"
if command -v apt-get &>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good
  echo "  → System packages installed"
else
  echo "  → Skipping apt (not Debian/Ubuntu); ensure OpenCV/GStreamer libs are available"
fi

# ── 2. Python venv + requirements ───────────────────────────────────
banner "Creating .venv and installing dependencies"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  echo "  → .venv created with $(python3 --version)"
else
  echo "  → .venv already exists"
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "  → requirements.txt installed"

# ── 3. Face model ──────────────────────────────────────────────────
banner "Downloading face_landmarker.task (MediaPipe, ~3.6 MB)"
if [[ -f face_landmarker.task ]]; then
  echo "  → Already exists, skipping"
else
  if command -v wget &>/dev/null; then
    wget -O face_landmarker.task --progress=bar:force "$FACE_MODEL_URL"
  else
    curl -sSL -o face_landmarker.task "$FACE_MODEL_URL"
  fi
  echo "  → Done"
fi

# ── 4. YOLO Nano NCNN model ────────────────────────────────────────
banner "Downloading yolov8n_ncnn_model (Nano only — not Small)"
if [[ -f yolov8n_ncnn_model/model.ncnn.param ]] && [[ -f yolov8n_ncnn_model/model.ncnn.bin ]]; then
  echo "  → Already exists, skipping"
else
  TMP_ARCHIVE="yolov8n_ncnn_model.tar.gz"
  if command -v wget &>/dev/null; then
    wget -O "$TMP_ARCHIVE" --progress=bar:force "$YOLO_NCNN_URL"
  else
    curl -sSL -o "$TMP_ARCHIVE" "$YOLO_NCNN_URL"
  fi
  tar xzf "$TMP_ARCHIVE"
  rm -f "$TMP_ARCHIVE"
  echo "  → Done"
fi

# ── 5. Verify ─────────────────────────────────────────────────────
banner "Verifying installation"
if [[ ! -f yolov8n_ncnn_model/model.ncnn.param ]] || [[ ! -f yolov8n_ncnn_model/model.ncnn.bin ]]; then
  echo "  ✗ yolov8n_ncnn_model missing (model.ncnn.param / model.ncnn.bin)."
  echo "    Set YOLO_NCNN_MODEL_URL to a valid tar.gz URL or add the release asset to the repo."
  exit 1
fi
echo "  → yolov8n_ncnn_model OK"
if ! .venv/bin/python -c "import ncnn; import mediapipe; import cv2" 2>/dev/null; then
  echo "  ✗ Python imports failed (ncnn, mediapipe, cv2)"
  exit 1
fi
echo "  → Python imports OK"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete. Run: ./run.sh"
echo "════════════════════════════════════════════════════════════════"
