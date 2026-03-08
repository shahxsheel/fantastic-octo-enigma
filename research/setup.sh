#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Unified setup for Raspberry Pi 4 and Pi 5
#  USB camera + NCNN (CPU) inference. Single .venv, no split processes.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REQUIRED_PYTHON_VERSION="3.12.8"

# Default URL for YOLO Nano NCNN model (override with YOLO_NCNN_MODEL_URL).
# Use Nano only — Small model is too slow for real-time Pi inference.
YOLO_NCNN_URL="${YOLO_NCNN_MODEL_URL:-https://github.com/shahxsheel/fantastic-octo-enigma/releases/download/v1.1.0/yolov8n_ncnn_model.tar.gz}"

TOTAL_STEPS=4
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

if command -v python3.12 &>/dev/null; then
  PYTHON_BIN="python3.12"
else
  echo "  ✗ python3.12 not found."
  echo "    Install Python ${REQUIRED_PYTHON_VERSION} and retry."
  exit 1
fi

PY_VERSION="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if [[ "$PY_VERSION" != "$REQUIRED_PYTHON_VERSION" ]]; then
  echo "  ✗ Found ${PYTHON_BIN} version ${PY_VERSION}, expected ${REQUIRED_PYTHON_VERSION}."
  echo "    Install Python ${REQUIRED_PYTHON_VERSION} and retry."
  exit 1
fi

if [[ -d .venv ]] && [[ -f .venv/bin/python ]]; then
  VENV_VERSION="$(.venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
  if [[ "$VENV_VERSION" != "$REQUIRED_PYTHON_VERSION" ]]; then
    echo "  → Existing .venv uses Python ${VENV_VERSION}; recreating with ${REQUIRED_PYTHON_VERSION}"
    rm -rf .venv
  fi
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
  echo "  → .venv created with $("$PYTHON_BIN" --version)"
else
  echo "  → .venv already exists (Python ${REQUIRED_PYTHON_VERSION})"
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "  → requirements.txt installed"

# ── 3. YOLO Nano NCNN model ────────────────────────────────────────
banner "Installing yolov8n_ncnn_model (Nano only — not Small)"
if [[ -f yolov8n_ncnn_model/model.ncnn.param ]] && [[ -f yolov8n_ncnn_model/model.ncnn.bin ]]; then
  echo "  → Already exists, skipping"
else
  TARBALL="yolov8n_ncnn_model.tar.gz"
  USE_LOCAL=false
  if [[ -f "$TARBALL" ]]; then
    if gzip -t "$TARBALL" 2>/dev/null; then
      USE_LOCAL=true
    else
      echo "  → Included tarball corrupt or incomplete, will download"
      rm -f "$TARBALL"
    fi
  fi
  if [[ "$USE_LOCAL" == true ]]; then
    echo "  → Using included tarball"
    tar xzf "$TARBALL"
  else
    echo "  → Downloading from $YOLO_NCNN_URL"
    if command -v wget &>/dev/null; then
      wget -O "$TARBALL" --progress=bar:force "$YOLO_NCNN_URL" 2>/dev/null || true
    else
      curl -sSL -o "$TARBALL" "$YOLO_NCNN_URL" 2>/dev/null || true
    fi
    if [[ ! -s "$TARBALL" ]] || ! gzip -t "$TARBALL" 2>/dev/null; then
      rm -f "$TARBALL"
      echo ""
      echo "  ✗ Download failed (404 or invalid). The v1.1.0 release may not exist yet."
      echo ""
      echo "  To get the YOLO Nano model:"
      echo "  1. On a machine with Python and internet, run: ./scripts/build_yolov8n_ncnn_archive.sh"
      echo "     Then copy yolov8n_ncnn_model.tar.gz to this Pi and run ./setup.sh again."
      echo "  2. Or create a GitHub release with yolov8n_ncnn_model.tar.gz and set:"
      echo "     YOLO_NCNN_MODEL_URL=https://github.com/OWNER/REPO/releases/download/TAG/yolov8n_ncnn_model.tar.gz"
      echo ""
      exit 1
    fi
    tar xzf "$TARBALL"
    rm -f "$TARBALL"
  fi
  echo "  → Done"
fi

# ── 4. Verify ─────────────────────────────────────────────────────
banner "Verifying installation"
if [[ ! -f yolov8n_ncnn_model/model.ncnn.param ]] || [[ ! -f yolov8n_ncnn_model/model.ncnn.bin ]]; then
  echo "  ✗ yolov8n_ncnn_model missing (model.ncnn.param / model.ncnn.bin)."
  echo "    Set YOLO_NCNN_MODEL_URL to a valid tar.gz URL or add the release asset to the repo."
  exit 1
fi
echo "  → yolov8n_ncnn_model OK"
if ! .venv/bin/python -c "import ncnn; import cv2" 2>/dev/null; then
  echo "  ✗ Python imports failed (ncnn, cv2)"
  exit 1
fi
echo "  → Python imports OK"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete. Run: ./run.sh"
echo "════════════════════════════════════════════════════════════════"
