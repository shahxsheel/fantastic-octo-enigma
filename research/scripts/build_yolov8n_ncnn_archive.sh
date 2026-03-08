#!/usr/bin/env bash
# Build yolov8n_ncnn_model.tar.gz for inclusion in the repo (offline setup).
# Run once, then commit the tarball: git add yolov8n_ncnn_model.tar.gz && git commit -m "Add YOLO Nano NCNN tarball"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Run ./setup.sh first (need .venv)."
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate
pip install -q ultralytics

echo "Downloading yolov8n.pt and exporting to NCNN..."
python -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.export(format='ncnn')
"

if [[ ! -d yolov8n_ncnn_model ]] || [[ ! -f yolov8n_ncnn_model/model.ncnn.param ]]; then
  echo "Export failed (yolov8n_ncnn_model/ not found)."
  exit 1
fi

echo "Creating yolov8n_ncnn_model.tar.gz..."
tar czf yolov8n_ncnn_model.tar.gz yolov8n_ncnn_model
echo "Done. Add and commit: git add yolov8n_ncnn_model.tar.gz && git commit -m 'Add YOLO Nano NCNN tarball'"
