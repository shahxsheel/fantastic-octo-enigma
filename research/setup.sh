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
    gstreamer1.0-plugins-good \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    libffi-dev \
    liblzma-dev \
    curl \
    git
  echo "  → System packages installed"
else
  echo "  → Skipping apt (not Debian/Ubuntu); ensure OpenCV/GStreamer libs are available"
fi

# ── 2. Python venv + requirements ───────────────────────────────────
banner "Creating .venv and installing dependencies"

# Activate pyenv if already installed but not yet on PATH.
export PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
if [[ -d "$PYENV_ROOT/bin" ]]; then
  export PATH="${PYENV_ROOT}/bin:${PATH}"
  eval "$(pyenv init -)" 2>/dev/null || true
fi

# Check whether the exact required version is already available.
PYTHON_BIN=""
if command -v python3.12 &>/dev/null; then
  _ver="$(python3.12 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  if [[ "$_ver" == "$REQUIRED_PYTHON_VERSION" ]]; then
    PYTHON_BIN="python3.12"
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "  → Python ${REQUIRED_PYTHON_VERSION} not found; installing via pyenv"

  if ! command -v pyenv &>/dev/null; then
    echo "  → Installing pyenv..."
    curl -fsSL https://pyenv.run | bash
    export PATH="${PYENV_ROOT}/bin:${PATH}"
    eval "$(pyenv init -)"
    # Persist pyenv initialisation for future shell sessions.
    SHELL_RC="${HOME}/.bashrc"
    if [[ -n "${ZSH_VERSION:-}" ]]; then SHELL_RC="${HOME}/.zshrc"; fi
    if ! grep -q 'pyenv init' "$SHELL_RC" 2>/dev/null; then
      {
        echo ''
        echo '# pyenv (added by ADA setup.sh)'
        echo 'export PYENV_ROOT="$HOME/.pyenv"'
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
        echo 'eval "$(pyenv init -)"'
      } >> "$SHELL_RC"
      echo "  → pyenv init added to ${SHELL_RC}"
    fi
  fi

  if ! pyenv versions --bare 2>/dev/null | grep -qx "${REQUIRED_PYTHON_VERSION}"; then
    echo "  → Building Python ${REQUIRED_PYTHON_VERSION} (this takes a few minutes on Pi)..."
    pyenv install "${REQUIRED_PYTHON_VERSION}"
  else
    echo "  → Python ${REQUIRED_PYTHON_VERSION} already installed in pyenv"
  fi

  PYTHON_BIN="${PYENV_ROOT}/versions/${REQUIRED_PYTHON_VERSION}/bin/python3"
fi

# Final sanity check.
_final_ver="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if [[ "$_final_ver" != "$REQUIRED_PYTHON_VERSION" ]]; then
  echo "  ✗ Expected Python ${REQUIRED_PYTHON_VERSION}, got ${_final_ver}. Aborting."
  exit 1
fi
echo "  → Using Python ${_final_ver} (${PYTHON_BIN})"

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
