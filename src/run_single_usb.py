"""
Threaded USB pipeline for maximum inference FPS on Raspberry Pi 4.

- Thread 1 (Camera): continuously grabs frames, stores latest in a thread-safe variable.
- Thread 2 (Inference): pulls latest frame, runs face_eye.detect() + yolo.detect(), updates latest_detections.
- Main (Display): reads latest frame and latest_detections, draws overlays, displays at smooth rate.

Video feed stays smooth (~30 FPS); boxes update at inference rate (e.g. 15–20 FPS) without stutter.
"""

import os
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from src.camera.camera_source import open_camera
from src.infer.face_eye_mediapipe import FaceEyeEstimatorMediaPipeSync
from src.infer.yolo_detector import YoloDetector


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


# Fast font for Pi 4 (2x faster than SIMPLEX).
_FONT = cv2.FONT_HERSHEY_PLAIN

# Dashboard: sidebar width (px).
SIDEBAR_WIDTH = 300
SIDEBAR_BG = (30, 30, 30)
SIDEBAR_BG_DISTRACTED = (0, 0, 255)  # Red when distracted
SIDEBAR_TEXT_X = 20

# Driver distraction: head pose thresholds (degrees).
DISTRACT_YAW_THRESH = 20
DISTRACT_PITCH_THRESH = 15


def _draw_graphics_on_frame(
    frame: np.ndarray,
    face_bbox: Optional[list],
    eyes: Optional[Any],
    objects: list[dict],
) -> str:
    """
    Draw only graphics on the video (face box, eye polylines, YOLO boxes).
    Returns driver state: 'FOCUSED', 'DISTRACTED', or 'DROWSY'.
    """
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        name = str(obj.get("name", "?"))
        conf = float(obj.get("conf", 0.0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            _FONT,
            1.0,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    if not eyes:
        return "FOCUSED"

    # Cyan polylines around eyes (on video only).
    cyan_bgr = (255, 255, 0)
    for pts in (getattr(eyes, "left_pts", None), getattr(eyes, "right_pts", None)):
        if not pts or len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [arr], isClosed=True, color=cyan_bgr, thickness=1)

    left_state = getattr(eyes, "left_state", "?")
    right_state = getattr(eyes, "right_state", "?")
    yaw = float(getattr(eyes, "yaw_deg", 0.0))
    pitch = float(getattr(eyes, "pitch_deg", 0.0))

    # DROWSY: both eyes closed.
    if left_state == "CLOSED" and right_state == "CLOSED":
        return "DROWSY"
    if abs(yaw) > DISTRACT_YAW_THRESH or abs(pitch) > DISTRACT_PITCH_THRESH:
        return "DISTRACTED"
    return "FOCUSED"


def _draw_sidebar_stats(
    sidebar: np.ndarray,
    fps: float,
    status: str,
    eyes: Optional[Any],
) -> None:
    """Draw all text and metrics on the sidebar. Sidebar is (H x SIDEBAR_WIDTH)."""
    h_sb = sidebar.shape[0]
    x = SIDEBAR_TEXT_X
    line_height = 28
    y = 30

    white = (255, 255, 255)
    cv2.putText(
        sidebar,
        "SYSTEM STATUS",
        (x, y),
        _FONT,
        1.2,
        white,
        2,
        cv2.LINE_AA,
    )
    y += line_height

    cv2.putText(
        sidebar,
        f"FPS: {fps:.1f}",
        (x, y),
        _FONT,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += line_height + 10

    # Driver state: FOCUSED (green), DISTRACTED (red), DROWSY (orange).
    if status == "FOCUSED":
        color = (0, 255, 0)
    elif status == "DISTRACTED":
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)  # Orange
    cv2.putText(
        sidebar,
        status,
        (x, y),
        _FONT,
        2.0,
        color,
        2,
        cv2.LINE_AA,
    )
    y += line_height * 2

    # Head pose telemetry.
    if eyes is not None:
        yaw = float(getattr(eyes, "yaw_deg", 0.0))
        pitch = float(getattr(eyes, "pitch_deg", 0.0))
        cv2.putText(
            sidebar,
            "Head Pose",
            (x, y),
            _FONT,
            1.0,
            white,
            2,
            cv2.LINE_AA,
        )
        y += line_height
        cv2.putText(
            sidebar,
            f"Yaw: {yaw:.1f}\u00b0",
            (x, y),
            _FONT,
            1.0,
            white,
            2,
            cv2.LINE_AA,
        )
        y += line_height
        cv2.putText(
            sidebar,
            f"Pitch: {pitch:.1f}\u00b0",
            (x, y),
            _FONT,
            1.0,
            white,
            2,
            cv2.LINE_AA,
        )
        y += line_height + 10

        left_pct = float(getattr(eyes, "left_pct", 0.0))
        right_pct = float(getattr(eyes, "right_pct", 0.0))
        cv2.putText(
            sidebar,
            "Eye Openness",
            (x, y),
            _FONT,
            1.0,
            white,
            2,
            cv2.LINE_AA,
        )
        y += line_height
        cv2.putText(
            sidebar,
            f"L: {left_pct:.0f}%  R: {right_pct:.0f}%",
            (x, y),
            _FONT,
            1.0,
            white,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    os.environ.setdefault("FORCE_CAMERA", "usb")

    headless = _env_bool("HEADLESS", False)
    cam, cam_name = open_camera(headless=headless)
    print(f"[single-usb] using camera={cam_name}", flush=True)

    bundle = cam.read()
    infer_h, infer_w = bundle.infer_bgr.shape[:2]
    infer_size = (infer_w, infer_h)
    print(f"[single-usb] infer_size={infer_w}x{infer_h}", flush=True)

    # Tighter MediaPipe confidence for less jitter and fewer false gaze triggers.
    os.environ.setdefault("MP_MIN_DET_CONF", "0.6")
    os.environ.setdefault("MP_MIN_TRACK_CONF", "0.6")
    os.environ.setdefault("MP_MIN_PRESENCE_CONF", "0.6")

    face_eye = FaceEyeEstimatorMediaPipeSync(input_size=infer_size)
    yolo = YoloDetector()

    # Thread-safe shared state: latest frame and latest detections.
    _lock = threading.Lock()
    _latest_frame: Optional[np.ndarray] = None
    _latest_detections: dict = {
        "face_bbox": None,
        "eyes": None,
        "objects": [],
    }
    _camera_stop = threading.Event()
    _inference_stop = threading.Event()

    def camera_thread_fn() -> None:
        nonlocal _latest_frame
        while not _camera_stop.is_set():
            try:
                bundle = cam.read()
                with _lock:
                    _latest_frame = bundle.infer_bgr.copy()
            except Exception:
                if _camera_stop.is_set():
                    break
                time.sleep(0.001)

    def inference_thread_fn() -> None:
        while not _inference_stop.is_set():
            with _lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
            if frame is None:
                time.sleep(0.001)
                continue
            face_bbox, eyes = face_eye.detect(frame)
            objects = yolo.detect(frame)
            with _lock:
                _latest_detections["face_bbox"] = face_bbox
                _latest_detections["eyes"] = eyes
                _latest_detections["objects"] = objects

    cam_thread = threading.Thread(target=camera_thread_fn, daemon=True)
    inf_thread = threading.Thread(target=inference_thread_fn, daemon=True)
    cam_thread.start()
    inf_thread.start()

    window_name = "Single-Process USB (YOLO + Face/Eye per-frame)"
    if not headless:
        try:
            cv2.startWindowThread()
            gui_normal = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | gui_normal)
        except Exception:
            pass

    fps = 0.0
    fps_count = 0
    t0 = time.time()

    # Pre-allocate dashboard buffer once (zero-allocation visualization).
    vis_buffer: Optional[np.ndarray] = None
    if not headless:
        vis_h, vis_w = infer_h, infer_w + SIDEBAR_WIDTH
        vis_buffer = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        vis_buffer[:, infer_w:] = SIDEBAR_BG  # Pre-fill sidebar background
        print(f"[single-usb] pre-allocated dashboard buffer: {vis_h}x{vis_w}", flush=True)

    try:
        while True:
            fps_count += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = fps_count / (now - t0)
                print(f"[single-usb] FPS={fps:.1f}", flush=True)
                fps_count = 0
                t0 = now

            if not headless:
                with _lock:
                    frame = _latest_frame.copy() if _latest_frame is not None else None
                    face_bbox = _latest_detections.get("face_bbox")
                    eyes = _latest_detections.get("eyes")
                    objects = _latest_detections.get("objects", [])

                if frame is None:
                    time.sleep(0.001)
                    continue

                # Zero-allocation: copy frame into pre-allocated buffer.
                h, w = frame.shape[:2]
                vis_buffer[:, :w] = frame

                status = _draw_graphics_on_frame(
                    vis_buffer[:, :w], face_bbox, eyes, objects
                )

                # Clear sidebar text area with filled rectangle (faster than reassigning).
                sidebar = vis_buffer[:, w:]
                if status == "DISTRACTED":
                    sidebar[:] = SIDEBAR_BG_DISTRACTED
                else:
                    sidebar[:] = SIDEBAR_BG
                _draw_sidebar_stats(sidebar, fps, status, eyes)

                try:
                    cv2.imshow(window_name, vis_buffer)
                except cv2.error:
                    headless = True
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
            else:
                # Headless: skip all drawing and frame copy, just update FPS counter.
                time.sleep(0.001)
    finally:
        _camera_stop.set()
        _inference_stop.set()
        cam_thread.join(timeout=1.0)
        inf_thread.join(timeout=1.0)
        cam.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            face_eye.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
