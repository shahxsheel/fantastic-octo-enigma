"""
Threaded USB pipeline for maximum inference FPS on Raspberry Pi 4.

- Thread 1 (Camera): continuously grabs frames, stores latest in a thread-safe variable.
- Main Thread: pulls latest frame, runs inference (face_eye + yolo), draws (GUI only), updates CLI.

Headless: Pure inference speed (no drawing). GUI: Minimal graphics (polylines, boxes, border).
"""

import os
import sys
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


# Driver distraction: head pose thresholds (degrees).
DISTRACT_YAW_THRESH = 20
DISTRACT_PITCH_THRESH = 15


def _compute_status(eyes: Optional[Any]) -> str:
    """Compute driver status from eye data. Returns 'FOCUSED', 'DISTRACTED', or 'DROWSY'."""
    if not eyes:
        return "FOCUSED"
    left_state = getattr(eyes, "left_state", "?")
    right_state = getattr(eyes, "right_state", "?")
    yaw = float(getattr(eyes, "yaw_deg", 0.0))
    pitch = float(getattr(eyes, "pitch_deg", 0.0))
    if left_state == "CLOSED" and right_state == "CLOSED":
        return "DROWSY"
    if abs(yaw) > DISTRACT_YAW_THRESH or abs(pitch) > DISTRACT_PITCH_THRESH:
        return "DISTRACTED"
    return "FOCUSED"


def _draw_minimal_graphics(
    frame: np.ndarray,
    face_bbox: Optional[list],
    eyes: Optional[Any],
    objects: list[dict],
    status: str,
) -> None:
    """
    Draw minimal "Iron Man" graphics on video frame (GUI mode only):
    - Cyan polylines around eyes
    - Green bounding box around face
    - Red border if DISTRACTED
    - YOLO boxes with labels
    """
    h, w = frame.shape[:2]

    # Face bounding box (green).
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Cyan polylines around eyes.
    if eyes:
        cyan_bgr = (255, 255, 0)
        for pts in (getattr(eyes, "left_pts", None), getattr(eyes, "right_pts", None)):
            if not pts or len(pts) < 2:
                continue
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [arr], isClosed=True, color=cyan_bgr, thickness=1)

    # YOLO boxes with labels.
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
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # Red border if distracted (thickness=5).
    if status == "DISTRACTED":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)


def _print_cli_stats(
    fps: float,
    latency_ms: float,
    status: str,
    eyes: Optional[Any],
    objects: list[dict],
) -> None:
    """Print CLI dashboard on a single line (overwrites with \\r)."""
    parts = [f"FPS: {fps:.1f}", f"LAT: {latency_ms:.0f}ms"]
    parts.append(f"STATUS: {status}")

    if eyes:
        yaw = float(getattr(eyes, "yaw_deg", 0.0))
        pitch = float(getattr(eyes, "pitch_deg", 0.0))
        left_pct = float(getattr(eyes, "left_pct", 0.0))
        right_pct = float(getattr(eyes, "right_pct", 0.0))
        parts.append(f"HEAD: {yaw:.1f}\u00b0 / {pitch:.1f}\u00b0")
        parts.append(f"EYES: L {left_pct:.0f}% R {right_pct:.0f}%")

    if objects:
        obj_names = [str(obj.get("name", "?")) for obj in objects[:3]]
        parts.append(f"OBJS: {', '.join(obj_names)}")

    line = " | ".join(parts)
    sys.stdout.write(f"\r{line}")
    sys.stdout.flush()


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

    # Thread-safe shared state: latest frame.
    _lock = threading.Lock()
    _latest_frame: Optional[np.ndarray] = None
    _camera_stop = threading.Event()

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

    cam_thread = threading.Thread(target=camera_thread_fn, daemon=True)
    cam_thread.start()

    window_name = "Driver Monitoring System"
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

    try:
        while True:
            with _lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None

            if frame is None:
                time.sleep(0.001)
                continue

            # Measure inference latency (face + YOLO).
            inference_start = time.time()
            face_bbox, eyes = face_eye.detect(frame)
            objects = yolo.detect(frame)
            inference_end = time.time()
            latency_ms = (inference_end - inference_start) * 1000.0

            status = _compute_status(eyes)

            fps_count += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = fps_count / (now - t0)
                fps_count = 0
                t0 = now

            if not headless:
                # GUI mode: minimal drawing on frame.
                vis = frame.copy()
                _draw_minimal_graphics(vis, face_bbox, eyes, objects, status)
                try:
                    cv2.imshow(window_name, vis)
                except cv2.error:
                    headless = True
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            # CLI dashboard (both modes).
            _print_cli_stats(fps, latency_ms, status, eyes, objects)

            if headless:
                time.sleep(0.001)
    finally:
        _camera_stop.set()
        cam_thread.join(timeout=1.0)
        cam.release()
        sys.stdout.write("\n")
        sys.stdout.flush()
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
