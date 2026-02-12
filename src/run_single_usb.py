import os
import time
from typing import Optional

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


def _draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_face_and_eyes(
    frame: np.ndarray,
    face_bbox: Optional[list],
    eyes: Optional[object],
) -> None:
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not eyes:
        return

    # Eye percentages / states
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
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
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
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    """
    Single-process USB pipeline:

    - Captures frames from USB camera.
    - Runs YOLO + MediaPipe face/eye on **every frame**.
    - Draws overlays directly and reports FPS.
    """
    # Force USB camera path unless explicitly overridden.
    os.environ.setdefault("FORCE_CAMERA", "usb")

    headless = _env_bool("HEADLESS", False)
    cam, cam_name = open_camera(headless=headless)
    print(f"[single-usb] using camera={cam_name}", flush=True)

    # First frame to determine inference size.
    bundle = cam.read()
    infer_h, infer_w = bundle.infer_bgr.shape[:2]
    infer_size = (infer_w, infer_h)
    print(f"[single-usb] infer_size={infer_w}x{infer_h}", flush=True)

    # Create detectors.
    face_eye = FaceEyeEstimatorMediaPipeSync(input_size=infer_size)
    yolo = YoloDetector()

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
            bundle = cam.read()
            frame = bundle.infer_bgr  # Work at inference resolution to keep cost low.

            # Per-frame inference: YOLO and MediaPipe every frame.
            yolo_start = time.time()
            objects = yolo.detect(frame)
            yolo_ms = (time.time() - yolo_start) * 1000.0

            face_start = time.time()
            face_bbox, eyes = face_eye.detect(frame)
            face_ms = (time.time() - face_start) * 1000.0

            # Simple debug timing (optional).
            if _env_bool("SINGLE_USB_TIMING", False):
                print(
                    f"[single-usb] frame={bundle.frame_id} "
                    f"yolo={yolo_ms:.1f}ms face={face_ms:.1f}ms",
                    flush=True,
                )

            vis = frame.copy()
            _draw_yolo_objects(vis, objects)
            _draw_face_and_eyes(vis, face_bbox, eyes)

            # FPS computation.
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
                # Small sleep to avoid busy looping in true headless mode.
                time.sleep(0.001)
    finally:
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

