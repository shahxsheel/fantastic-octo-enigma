#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Download models ──────────────────────────────────────────────
echo "[setup] downloading face_landmarker.task ..."
wget -O face_landmarker.task -q \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "[setup] face_landmarker.task downloaded"

if [ ! -f yolov8s.pt ]; then
  echo "[setup] downloading yolov8s.pt ..."
  wget -O yolov8s.pt -q \
    https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
  echo "[setup] yolov8s.pt downloaded"
else
  echo "[setup] yolov8s.pt already exists, skipping"
fi

# ── Camera venv (Python 3.13, system-site-packages for PiCamera2) ─
echo "[setup] creating .venv-cam (python3.13, system-site-packages) ..."
python3.13 -m venv --system-site-packages .venv-cam
.venv-cam/bin/python -m pip install --upgrade pip setuptools wheel -q
.venv-cam/bin/python -m pip install -r requirements-camera.txt -q
echo "[setup] .venv-cam ready"

# ── Inference venv (Python 3.12.8 via uv) ─────────────────────────
echo "[setup] creating .venv-infer (uv python 3.12.8) ..."
uv venv --python 3.12.8 .venv-infer
.venv-infer/bin/python -m ensurepip --upgrade
.venv-infer/bin/python -m pip install --upgrade pip setuptools wheel -q
.venv-infer/bin/python -m pip install -r requirements-infer.txt -q
echo "[setup] .venv-infer ready"

# ── Export YOLOv8s to NCNN (needs ultralytics from .venv-infer) ──
if [ ! -d yolov8s_ncnn_model ]; then
  echo "[setup] exporting yolov8s.pt to NCNN format ..."
  .venv-infer/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='ncnn')"
  echo "[setup] yolov8s_ncnn_model ready"
else
  echo "[setup] yolov8s_ncnn_model already exists, skipping export"
fi

echo "[setup] done. Run with: ./scripts/run_split.sh"
