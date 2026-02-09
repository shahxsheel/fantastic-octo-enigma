#!/usr/bin/env python3
"""Print basic camera info for given V4L2 devices using OpenCV.

By default checks /dev/video0 (rpicam) and /dev/video8 (USB).
Uses the OpenCV build (GStreamer enabled) that ships with the repo's venvs.
"""

import os
import sys
from typing import Iterable

import cv2


def fourcc_str(code: float) -> str:
    try:
        code_int = int(code)
        return "".join(chr((code_int >> 8 * i) & 0xFF) for i in range(4))
    except Exception:
        return "????"


def probe(devs: Iterable[str]) -> None:
    for dev in devs:
        print(f"\n=== {dev} ===")
        # OpenCV accepts either numeric index or device path; try path first.
        cap = cv2.VideoCapture(dev, cv2.CAP_ANY)
        if not cap.isOpened():
            # fallback: numeric index if provided as /dev/videoX
            try:
                idx = int(str(dev).replace("/dev/video", ""))
                cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            except Exception:
                pass
        if not cap.isOpened():
            print("could not open")
            continue

        backend = getattr(cap, "getBackendName", lambda: "unknown")()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
        fmt = cap.get(cv2.CAP_PROP_FORMAT)
        buff = cap.get(cv2.CAP_PROP_BUFFERSIZE)

        print(f"backend       : {backend}")
        print(f"size          : {int(width)}x{int(height)}")
        print(f"fps           : {fps:.2f}")
        print(f"fourcc        : {fourcc}")
        print(f"cv2 format id : {fmt}")
        print(f"buffer size   : {buff}")

        cap.release()


def main() -> None:
    if len(sys.argv) > 1:
        devs = sys.argv[1:]
    else:
        devs = ["/dev/video0", "/dev/video8"]

    # Ensure devices exist; skip missing ones.
    devs = [d for d in devs if os.path.exists(d) or d.isdigit()]
    if not devs:
        print("No devices to probe (none of the provided paths exist).", file=sys.stderr)
        sys.exit(1)

    probe(devs)


if __name__ == "__main__":
    main()
