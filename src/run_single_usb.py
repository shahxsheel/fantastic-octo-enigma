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


def _draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 24),
        _FONT,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_face_and_eyes(
    frame: np.ndarray,
    face_bbox: Optional[list],
    eyes: Optional[Any],
) -> None:
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not eyes:
        return

    left_state = getattr(eyes, "left_state", "?")
    right_state = getattr(eyes, "right_state", "?")
    left_pct = float(getattr(eyes, "left_pct", 0.0))
    right_pct = float(getattr(eyes, "right_pct", 0.0))

    text = f"L {left_state} {left_pct:.0f}%  |  R {right_state} {right_pct:.0f}%"
    y = 40 if face_bbox is None else max(40, face_bbox[1] - 10)
    cv2.putText(
        frame,
        text,
        (10, y),
        _FONT,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _draw_yolo_objects(frame: np.ndarray, objects: list[dict]) -> None:
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


def main() -> None:
    os.environ.setdefault("FORCE_CAMERA", "usb")

    headless = _env_bool("HEADLESS", False)
    cam, cam_name = open_camera(headless=headless)
    print(f"[single-usb] using camera={cam_name}", flush=True)

    bundle = cam.read()
    infer_h, infer_w = bundle.infer_bgr.shape[:2]
    infer_size = (infer_w, infer_h)
    print(f"[single-usb] infer_size={infer_w}x{infer_h}", flush=True)

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

    try:
        while True:
            with _lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
                face_bbox = _latest_detections.get("face_bbox")
                eyes = _latest_detections.get("eyes")
                objects = _latest_detections.get("objects", [])

            if frame is None:
                time.sleep(0.001)
                continue

            vis = frame.copy()
            _draw_yolo_objects(vis, objects)
            _draw_face_and_eyes(vis, face_bbox, eyes)

            fps_count += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = fps_count / (now - t0)
                print(f"[single-usb] FPS={fps:.1f}", flush=True)
                fps_count = 0
                t0 = now

            _draw_fps(vis, fps)

            if not headless:
                try:
                    cv2.imshow(window_name, vis)
                except cv2.error:
                    headless = True
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
            else:
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
