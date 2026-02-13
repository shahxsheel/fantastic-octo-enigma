import glob
import os
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


@dataclass
class FrameBundle:
    frame_id: int
    ts_ms: int
    main_bgr: np.ndarray
    infer_bgr: np.ndarray


class CameraSource:
    def read(self) -> FrameBundle:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class USBCameraSource(CameraSource):
    def __init__(
        self,
        index: int,
        capture_size: Tuple[int, int],
        infer_size: Tuple[int, int],
        headless: bool = False,
    ):
        self.index = index
        self.capture_w, self.capture_h = capture_size
        self.infer_w, self.infer_h = infer_size
        use_gst = os.environ.get("CAMERA_USE_GSTREAMER", "1") == "1"
        fourcc_env = os.environ.get("CAMERA_FOURCC")
        fourcc = fourcc_env if fourcc_env else "MJPG"

        if use_gst:
            pipeline = (
                f"v4l2src device=/dev/video{index} ! "
                f"image/jpeg,width={self.capture_w},height={self.capture_h},framerate={_env_int('CAMERA_FPS', 30)}/1 ! "
                "jpegdec ! videoconvert ! appsink drop=true sync=false max-buffers=1"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = None

        # Fallback to V4L2 if GStreamer failed or disabled
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_FPS, _env_int("CAMERA_FPS", 30))
        self.frame_id = 0

    def read(self) -> FrameBundle:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("USB camera read failed")
        h, w = frame.shape[:2]
        if w != self.capture_w or h != self.capture_h:
            frame = cv2.resize(frame, (self.capture_w, self.capture_h), interpolation=cv2.INTER_AREA)

        if os.environ.get("SWAP_RB", "0") == "1":
            frame = frame[:, :, ::-1].copy()

        # Skip second resize when capture and infer sizes match (sweet spot for Pi 4 FPS).
        if self.infer_w == self.capture_w and self.infer_h == self.capture_h:
            infer = frame
        else:
            infer = cv2.resize(frame, (self.infer_w, self.infer_h), interpolation=cv2.INTER_AREA)
        self.frame_id += 1
        ts_ms = int(time.time() * 1000)
        return FrameBundle(frame_id=self.frame_id, ts_ms=ts_ms, main_bgr=frame, infer_bgr=infer)

    def release(self) -> None:
        self.cap.release()


def _usb_indices() -> list[int]:
    devs = sorted(glob.glob("/dev/video*"))
    indices: list[int] = []
    for d in devs:
        try:
            indices.append(int(d.replace("/dev/video", "")))
        except Exception:
            continue
    return sorted(set(indices))


def open_camera(headless: bool = False) -> tuple[CameraSource, str]:
    """
    Open first available USB camera by scanning /dev/video*.
    Standardized on USBCameraSource for Pi 4 and Pi 5.
    Default 640x480 (VGA) for inference speed vs. accuracy sweet spot.
    """
    # Default VGA: sweet spot for Pi 4 inference FPS.
    default_w = _env_int("CAPTURE_WIDTH", 640)
    default_h = _env_int("CAPTURE_HEIGHT", 480)
    capture_size = (default_w, default_h)
    infer_size = (_env_int("INFER_WIDTH", default_w), _env_int("INFER_HEIGHT", default_h))

    hint = os.environ.get("CAMERA_INDEX")
    candidates: list[int] = []
    if hint is not None:
        try:
            candidates.append(int(hint))
        except Exception:
            pass

    indices = _usb_indices()
    probe_max = 4
    probe_env = os.environ.get("USB_PROBE_MAX")
    if probe_env is not None:
        try:
            probe_max = int(probe_env)
        except Exception:
            pass
    for i in range(min(probe_max, max(1, len(indices)))):
        if i not in candidates:
            candidates.append(i)
    for idx in indices:
        if idx not in candidates:
            candidates.append(idx)
    candidates = sorted(set(candidates))

    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        cap.release()
        if ok:
            return USBCameraSource(idx, capture_size, infer_size, headless=headless), f"usb:{idx}"

    raise RuntimeError(
        "No USB camera found. Scan /dev/video* or set CAMERA_INDEX. "
        "PiCamera2 is no longer supported; use USBCameraSource only."
    )
