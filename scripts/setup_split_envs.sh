#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Auto-detect Raspberry Pi model and run the right setup script
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Infineon Driver Monitoring — Setup"
echo "════════════════════════════════════════════════════════════════"

# Read Pi model from device tree
PI_MODEL=""
if [ -f /proc/device-tree/model ]; then
  PI_MODEL="$(tr -d '\0' < /proc/device-tree/model)"
  echo "  Detected: $PI_MODEL"
fi

# Determine which setup script to run
if echo "$PI_MODEL" | grep -qi "Pi 5"; then
  echo "  → Using Pi 5 setup"
  exec bash "$SCRIPT_DIR/setup_pi5.sh"
elif echo "$PI_MODEL" | grep -qi "Pi 4"; then
  echo "  → Using Pi 4 setup"
  exec bash "$SCRIPT_DIR/setup_pi4.sh"
elif [ -n "$PI_MODEL" ]; then
  echo "  → Unknown Pi model, defaulting to Pi 4 setup (safer)"
  exec bash "$SCRIPT_DIR/setup_pi4.sh"
else
  echo ""
  echo "  Could not auto-detect Pi model."
  echo "  Please run one of these directly:"
  echo ""
  echo "    ./scripts/setup_pi5.sh    # Raspberry Pi 5"
  echo "    ./scripts/setup_pi4.sh    # Raspberry Pi 4B"
  echo ""
  exit 1
fi
