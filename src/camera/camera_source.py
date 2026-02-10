import glob
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


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
        frame = cv2.resize(frame, (self.capture_w, self.capture_h), interpolation=cv2.INTER_AREA)

        if os.environ.get("SWAP_RB", "0") == "1":
            frame = frame[:, :, ::-1].copy()

        infer = cv2.resize(frame, (self.infer_w, self.infer_h), interpolation=cv2.INTER_AREA)
        self.frame_id += 1
        ts_ms = int(time.time() * 1000)
        return FrameBundle(frame_id=self.frame_id, ts_ms=ts_ms, main_bgr=frame, infer_bgr=infer)

    def release(self) -> None:
        self.cap.release()


class PiCamera2Source(CameraSource):
    def __init__(
        self,
        capture_size: Tuple[int, int],
        infer_size: Tuple[int, int],
        headless: bool = False,
    ):
        # Camera process must run on a Python that can import picamera2/libcamera (usually system Python on Pi OS)
        from picamera2 import Picamera2

        try:
            from libcamera import ColorSpace
        except Exception:
            ColorSpace = None

        self.capture_w, self.capture_h = capture_size
        self.infer_w, self.infer_h = infer_size
        self.headless = headless
        self.drop_main = headless and (os.environ.get("DROP_MAIN_WHEN_HEADLESS", "1") == "1")
        self.lores_format = _env_str("CAM_LORES_FORMAT", "YUV420")

        # Camera can be "busy" if another libcamera app is running (e.g. rpicam-hello).
        last_err = None
        for attempt in range(1, 6):
            try:
                self.picam2 = Picamera2()
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.2)
        if last_err is not None:
            raise RuntimeError(
                "Failed to acquire Pi camera (device busy). "
                "Close any other libcamera apps (e.g. rpicam-hello, libcamera-vid) "
                "and try again."
            ) from last_err
        # Use preview configuration for better color handling (matches common OpenCV + PiCam examples).
        # If headless + drop_main, still request a main stream but keep it small to lower ISP load.
        main_size = (self.capture_w, self.capture_h)
        if self.drop_main:
            main_size = (max(320, self.infer_w), max(240, self.infer_h))

        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": main_size},
            lores={"format": self.lores_format, "size": (self.infer_w, self.infer_h)},
        )

        # Optional ColorSpace overrides (default: leave preview defaults)
        cs = _env_str("CAMERA_COLORSPACE", "default").lower()
        if ColorSpace is not None and cs != "default":
            if cs == "srgb" and hasattr(ColorSpace, "Srgb"):
                config["colour_space"] = ColorSpace.Srgb()
            elif cs == "rec709" and hasattr(ColorSpace, "Rec709"):
                config["colour_space"] = ColorSpace.Rec709()
            elif cs == "smpte170m" and hasattr(ColorSpace, "Smpte170m"):
                config["colour_space"] = ColorSpace.Smpte170m()

        self.picam2.configure(config)
        # Slightly higher buffer_count reduces dropped frames under load.
        try:
            self.picam2.set_controls({"BufferCount": 6})
        except Exception:
            pass

        # Try to keep FPS high. Low-light may get darker.
        if os.environ.get("FORCE_FPS", "1") == "1":
            fps = _env_int("CAMERA_FPS", 30)
            # Frame duration in microseconds
            frame_us = int(1_000_000 / max(1, fps))
            self.picam2.set_controls({"FrameDurationLimits": (frame_us, frame_us)})

        self.picam2.start()
        time.sleep(0.2)
        self.frame_id = 0

    def read(self) -> FrameBundle:
        # capture_arrays() returns (list_of_arrays, metadata) on many versions.
        main_rgb = None
        lores_rgb = None
        try:
            res = self.picam2.capture_arrays(["main", "lores"])
            if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], list):
                arrs = res[0]
                if len(arrs) >= 1:
                    main_rgb = arrs[0]
                if len(arrs) >= 2:
                    lores_rgb = arrs[1]
        except Exception:
            res = None

        if not isinstance(main_rgb, np.ndarray):
            main_rgb = self.picam2.capture_array("main")
        if not isinstance(lores_rgb, np.ndarray):
            lores_rgb = self.picam2.capture_array("lores")

        # Convert camera data -> OpenCV BGR
        infer_bgr = cv2.cvtColor(lores_rgb, cv2.COLOR_RGB2BGR)

        if self.drop_main:
            main_bgr = infer_bgr
        else:
            main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)

        if os.environ.get("SWAP_RB", "0") == "1":
            main_bgr = main_bgr[:, :, ::-1].copy()
            infer_bgr = infer_bgr[:, :, ::-1].copy()

        self.frame_id += 1
        ts_ms = int(time.time() * 1000)
        return FrameBundle(
            frame_id=self.frame_id,
            ts_ms=ts_ms,
            main_bgr=main_bgr,
            infer_bgr=infer_bgr,
        )

    def release(self) -> None:
        self.picam2.stop()


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
    mode = _env_str("FORCE_CAMERA", "auto").lower()
    # Tolerate common boolean-like values.
    # Users sometimes set FORCE_CAMERA=1 meaning "use Pi camera".
    if mode in ("1", "true", "yes", "on", "picam", "pi_cam", "picamera", "picamera2"):
        mode = "pi"
    if mode in ("0", "false", "no", "off"):
        mode = "auto"
    def _sizes(is_usb: bool) -> tuple[tuple[int, int], tuple[int, int]]:
        if is_usb:
            cap_w = _env_int("CAPTURE_WIDTH", 1280)
            cap_h = _env_int("CAPTURE_HEIGHT", 720)
            inf_w = _env_int("INFER_WIDTH", 1280)
            inf_h = _env_int("INFER_HEIGHT", 720)
        else:
            cap_w = _env_int("CAPTURE_WIDTH", 1280)
            cap_h = _env_int("CAPTURE_HEIGHT", 720)
            inf_w = _env_int("INFER_WIDTH", 1280)
            inf_h = _env_int("INFER_HEIGHT", 720)
        return (cap_w, cap_h), (inf_w, inf_h)

    if mode == "usb":
        capture_size, infer_size = _sizes(is_usb=True)
        idx = _env_int("CAMERA_INDEX", 0)
        return USBCameraSource(idx, capture_size, infer_size, headless=headless), f"usb:{idx}"
    if mode == "pi":
        capture_size, infer_size = _sizes(is_usb=False)
        return PiCamera2Source(capture_size, infer_size, headless=headless), "pi"

    # auto: try USB devices first (but avoid long hangs probing non-capture nodes)
    hint = os.environ.get("CAMERA_INDEX")
    candidates: list[int] = []
    if hint is not None:
        try:
            candidates.append(int(hint))
        except Exception:
            pass

    # Default probe set keeps startup fast/noisy-free on Pi (many /dev/video* are not cameras).
    probe_env = os.environ.get("USB_PROBE_MAX")
    if probe_env is not None:
        try:
            probe_max = int(probe_env)
        except Exception:
            probe_max = 4
    else:
        probe_max = 4

    candidates.extend([i for i in range(0, max(0, probe_max))])
    candidates = [c for c in candidates if c is not None]
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        cap.release()
        if ok:
            capture_size, infer_size = _sizes(is_usb=True)
            return USBCameraSource(idx, capture_size, infer_size, headless=headless), f"usb:{idx}"

    capture_size, infer_size = _sizes(is_usb=False)
    return PiCamera2Source(capture_size, infer_size, headless=headless), "pi"
