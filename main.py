#!/usr/bin/env python3
"""
Split-pipeline launcher.

This repo runs best on Raspberry Pi with two processes:
- Camera: Python 3.13 + system picamera2/libcamera
- Inference: Python 3.12.8 (uv) + YOLO/face+eye estimator + terminal logs

See README.md for details.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _bin(path: str) -> str:
    return str((ROOT / path).resolve())


def run_both(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env.setdefault("FRAMES_ADDR", "tcp://127.0.0.1:5555")
    env.setdefault("RESULTS_ADDR", "tcp://127.0.0.1:5556")

    infer_py = _bin(".venv-infer/bin/python")
    cam_py = _bin(".venv-cam/bin/python")

    if not Path(infer_py).exists() or not Path(cam_py).exists():
        print("Missing venvs. Run `./scripts/setup_split_envs.sh` first.", file=sys.stderr)
        return 2

    infer_cmd = [infer_py, "-m", "src.infer.run_infer"]
    cam_cmd = [cam_py, "-m", "src.camera.run_camera"]

    print("[main] starting inference:", " ".join(infer_cmd))
    infer_p = subprocess.Popen(infer_cmd, env=env)
    time.sleep(0.3)

    try:
        print("[main] starting camera:", " ".join(cam_cmd))
        cam_rc = subprocess.call(cam_cmd, env=env)
        return cam_rc
    finally:
        print("[main] stopping inference ...")
        try:
            infer_p.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            infer_p.wait(timeout=2)
        except Exception:
            try:
                infer_p.kill()
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--both",
        action="store_true",
        help="Run inference + camera (default).",
    )
    p.add_argument(
        "--infer",
        action="store_true",
        help="Run inference only.",
    )
    p.add_argument(
        "--camera",
        action="store_true",
        help="Run camera only.",
    )
    args = p.parse_args()

    if args.infer and args.camera:
        args.both = True

    if not (args.both or args.infer or args.camera):
        args.both = True

    if args.both:
        return run_both(args)

    env = os.environ.copy()
    env.setdefault("FRAMES_ADDR", "tcp://127.0.0.1:5555")
    env.setdefault("RESULTS_ADDR", "tcp://127.0.0.1:5556")

    if args.infer:
        infer_py = _bin(".venv-infer/bin/python")
        if not Path(infer_py).exists():
            print("Missing .venv-infer. Run `./scripts/setup_split_envs.sh`.", file=sys.stderr)
            return 2
        return subprocess.call([infer_py, "-m", "src.infer.run_infer"], env=env)

    if args.camera:
        cam_py = _bin(".venv-cam/bin/python")
        if not Path(cam_py).exists():
            print("Missing .venv-cam. Run `./scripts/setup_split_envs.sh`.", file=sys.stderr)
            return 2
        return subprocess.call([cam_py, "-m", "src.camera.run_camera"], env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

