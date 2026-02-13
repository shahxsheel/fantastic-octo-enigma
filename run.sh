#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Run the driver monitoring app (single-process USB pipeline).
#  Requires: ./setup.sh has been run first.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]] || [[ ! -f .venv/bin/python ]]; then
  echo "Run ./setup.sh first."
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

export PYTHONPATH=.
export CAMERA_USE_GSTREAMER=1
export YOLO_MODEL=yolov8n_ncnn_model
export OMP_NUM_THREADS=4
export NCNN_THREADS=4

exec python src/run_single_usb.py "$@"
