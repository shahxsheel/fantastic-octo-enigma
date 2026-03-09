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

# Load .env if present (parse as KEY=VALUE data, do not execute it as shell).
if [[ -f .env ]]; then
  while IFS= read -r raw_line || [[ -n "${raw_line:-}" ]]; do
    line="${raw_line%$'\r'}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"

      value="${value#"${value%%[![:space:]]*}"}"

      if [[ "$value" =~ ^\"(.*)\"$ ]]; then
        value="${BASH_REMATCH[1]}"
      elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
        value="${BASH_REMATCH[1]}"
      fi

      export "$key=$value"
    else
      echo "[run] Warning: skipping invalid .env line: $line"
    fi
  done < .env
fi

# Default to headless when running without an active display server.
if [[ -z "${HEADLESS:-}" ]] && [[ -z "${DISPLAY:-}" ]] && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
  export HEADLESS=1
  echo "[run] No display detected; defaulting to HEADLESS=1"
fi

export PYTHONPATH=.
export CAMERA_USE_GSTREAMER="${CAMERA_USE_GSTREAMER:-1}"
export YOLO_MODEL="${YOLO_MODEL:-yolo26n_ncnn_model}"
# 256×256 is the real-time sweet spot for YOLO26n Nano on Raspberry Pi.
export YOLO_INPUT_SIZE="${YOLO_INPUT_SIZE:-256}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCNN_THREADS="${NCNN_THREADS:-4}"

# Set CPU governor to performance to prevent throttling during sustained inference.
# Silently skipped if cpufreq-set is not available (non-Pi hosts, VMs).
if command -v cpufreq-set &>/dev/null; then
  cpufreq-set -g performance 2>/dev/null || true
fi
export USE_FAKE_GPS="${USE_FAKE_GPS:-0}"
export USE_FAKE_GYRO="${USE_FAKE_GYRO:-0}"
export ENABLE_HEAD_DIRECTION="${ENABLE_HEAD_DIRECTION:-1}"
export HEAD_DIRECTION_EVERY_N="${HEAD_DIRECTION_EVERY_N:-4}"

# Suppress noisy startup messages from TFLite / MediaPipe / abseil.
# These are cosmetic: "All log messages before absl::InitializeLog", W0000 xnnpack
# messages, and "prctl(PR_SVE_GET_VL) failed" from cpuinfo on non-SVE ARM cores.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export GLOG_minloglevel="${GLOG_minloglevel:-3}"
export ABSL_MIN_LOG_LEVEL="${ABSL_MIN_LOG_LEVEL:-4}"

exec python src/run_single_usb.py "$@"
