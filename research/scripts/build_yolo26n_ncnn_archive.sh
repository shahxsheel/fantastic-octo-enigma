#!/usr/bin/env bash
# Build yolo26n_ncnn_model.tar.gz for inclusion in the repo (offline Pi setup).
# Run once, then commit the tarball:
#   git add research/yolo26n_ncnn_model.tar.gz
#   git rm  research/yolov8n_ncnn_model.tar.gz
#   git commit -m "Switch YOLO: v8n → v26n NCNN (2x faster, +3.6% mAP)"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]] || [[ ! -f .venv/bin/python ]]; then
  echo "Run ./setup.sh first (need .venv)."
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

# YOLO26 requires ultralytics ≥ 8.3.0
pip install -q "ultralytics>=8.3.0"

echo "Downloading yolo26n.pt and exporting to NCNN..."
python -c "
from ultralytics import YOLO
m = YOLO('yolo26n.pt')
# Export with runtime-matching input size to avoid unstable 640-default artifacts
# on constrained edge CPUs. Runtime default is YOLO_INPUT_SIZE=256.
m.export(format='ncnn', imgsz=256)
"

if [[ ! -f yolo26n_ncnn_model/model.ncnn.param ]] || [[ ! -f yolo26n_ncnn_model/model.ncnn.bin ]]; then
  echo "Export failed — yolo26n_ncnn_model/ not found or incomplete."
  exit 1
fi

# Embed a tiny marker so setup.sh can detect stale artifacts and re-install.
cat > yolo26n_ncnn_model/build_info.txt <<'EOF'
model=yolo26n
format=ncnn
imgsz=256
EOF

# Keep archive deterministic and runtime-focused.
find yolo26n_ncnn_model -type d -name "__pycache__" -prune -exec rm -r {} +
find yolo26n_ncnn_model -type f -name "*.pyc" -delete
rm -f yolo26n.pt

echo "Creating yolo26n_ncnn_model.tar.gz..."
tar czf yolo26n_ncnn_model.tar.gz yolo26n_ncnn_model/
echo ""
echo "Done. Next steps:"
echo "  git add research/yolo26n_ncnn_model.tar.gz"
echo "  git rm  research/yolov8n_ncnn_model.tar.gz"
echo "  git commit -m 'Switch YOLO: v8n → v26n NCNN (2x faster, +3.6% mAP)'"
