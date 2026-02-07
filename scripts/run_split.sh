#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Check venvs exist ────────────────────────────────────────────
if [[ ! -f .venv-infer/bin/python ]] || [[ ! -f .venv-cam/bin/python ]]; then
  echo "[run] ERROR: venvs not found. Run ./scripts/setup_split_envs.sh first."
  exit 1
fi

# ── Environment ──────────────────────────────────────────────────
FRAMES_ADDR="${FRAMES_ADDR:-tcp://127.0.0.1:5555}"
RESULTS_ADDR="${RESULTS_ADDR:-tcp://127.0.0.1:5556}"

if [[ -z "${HEADLESS+x}" ]]; then
  if [[ -n "${DISPLAY:-}" ]]; then
    HEADLESS=0
  else
    HEADLESS=1
  fi
fi

# ── Cleanup ──────────────────────────────────────────────────────
cleanup() {
  pkill -f "\\.venv-infer/bin/python -m src\\.infer\\.run_infer" 2>/dev/null || true
  pkill -f "\\.venv-cam/bin/python -m src\\.camera\\.run_camera" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# Kill any previous stuck instance so ports are free.
cleanup

# ── Start inference ──────────────────────────────────────────────
echo "[run] starting inference (py3.12) ..."
FRAMES_ADDR="$FRAMES_ADDR" RESULTS_ADDR="$RESULTS_ADDR" \
  .venv-infer/bin/python -m src.infer.run_infer &
INFER_PID=$!

sleep 0.3

# ── Start camera (foreground — Ctrl-C stops everything) ──────────
echo "[run] starting camera (py3.13) ..."
FRAMES_ADDR="$FRAMES_ADDR" RESULTS_ADDR="$RESULTS_ADDR" HEADLESS="$HEADLESS" \
  .venv-cam/bin/python -m src.camera.run_camera

echo "[run] stopping inference ..."
kill "$INFER_PID" 2>/dev/null || true
