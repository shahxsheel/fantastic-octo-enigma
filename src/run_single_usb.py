"""
Threaded USB pipeline for maximum inference FPS on Raspberry Pi 4.

- Thread 1 (Camera): continuously grabs frames, stores latest in a thread-safe variable.
- Thread 2 (Inference): continuously runs face_eye + yolo on latest frame, updates shared results.
- Main Thread (Display/CLI): reads latest results, draws (GUI only), updates CLI (rate-limited).

True decoupling: Display FPS (100+) independent of Inference FPS (~6-20fps).
"""

import os
import sys
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

# Force NCNN maximum resources (set before any imports that use NCNN).
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NCNN_THREADS"] = "4"

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

# CLI update rate (seconds).
CLI_UPDATE_INTERVAL = 0.5


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
    display_fps: float,
    infer_fps: float,
    latency_ms: float,
    status: str,
    eyes: Optional[Any],
    objects: list[dict],
) -> None:
    """Print CLI dashboard on a single line (overwrites with \\r, padded with spaces)."""
    parts = [
        f"[DISPLAY: {display_fps:.0f}fps]",
        f"[INFER: {infer_fps:.1f}fps ({latency_ms:.0f}ms)]",
        f"STATUS: {status}",
    ]

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
    # Pad with spaces to overwrite any previous longer line.
    padded = line.ljust(120)
    sys.stdout.write(f"\r{padded}")
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
    # Software filter: keep only relevant YOLO classes for DMS.
    os.environ.setdefault("YOLO_FILTER", "person,cell phone,bottle,cup")
    yolo = YoloDetector()

    # Thread-safe shared state.
    _lock = threading.Lock()
    _latest_frame: Optional[np.ndarray] = None
    _shared_results: dict = {
        "face_bbox": None,
        "eyes": None,
        "objects": [],
        "status": "FOCUSED",
        "latency_ms": 0.0,
        "infer_fps": 0.0,
        "infer_count": 0,
        "infer_t0": time.time(),
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
        """Background inference thread: continuously runs face_eye + yolo on latest frame."""
        inf_frame_idx = 0
        last_objects: list[dict] = []
        while not _inference_stop.is_set():
            with _lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
            if frame is None:
                time.sleep(0.001)
                continue

            # Always run face detection (fast).
            inference_start = time.time()
            face_bbox, eyes = face_eye.detect(frame)
            
            # Interleaving: ONLY run YOLO every 5 inference loops (saves 80% of YOLO compute).
            if inf_frame_idx % 5 == 0:
                objects = yolo.detect(frame)
                last_objects = objects
            else:
                objects = last_objects  # Re-use last known objects
            
            inference_end = time.time()
            latency_ms = (inference_end - inference_start) * 1000.0
            inf_frame_idx += 1

            status = _compute_status(eyes)

            # Update shared results (protected by lock).
            with _lock:
                _shared_results["face_bbox"] = face_bbox
                _shared_results["eyes"] = eyes
                _shared_results["objects"] = objects
                _shared_results["status"] = status
                _shared_results["latency_ms"] = latency_ms

                # Calculate inference FPS.
                _shared_results["infer_count"] += 1
                now = time.time()
                if now - _shared_results["infer_t0"] >= 1.0:
                    _shared_results["infer_fps"] = (
                        _shared_results["infer_count"] / (now - _shared_results["infer_t0"])
                    )
                    _shared_results["infer_count"] = 0
                    _shared_results["infer_t0"] = now

    cam_thread = threading.Thread(target=camera_thread_fn, daemon=True)
    inf_thread = threading.Thread(target=inference_thread_fn, daemon=True)
    cam_thread.start()
    inf_thread.start()

    window_name = "Driver Monitoring System"
    if not headless:
        try:
            cv2.startWindowThread()
            gui_normal = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | gui_normal)
        except Exception:
            pass

    # Display FPS tracking (main loop).
    display_fps = 0.0
    display_count = 0
    display_t0 = time.time()

    # CLI update rate limiting.
    last_cli_update = 0.0

    # Print header once.
    print("\n" + "=" * 120)
    print("Driver Monitoring System - Real-time Dashboard")
    print("=" * 120)

    try:
        while True:
            with _lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
                face_bbox = _shared_results.get("face_bbox")
                eyes = _shared_results.get("eyes")
                objects = _shared_results.get("objects", [])
                status = _shared_results.get("status", "FOCUSED")
                latency_ms = _shared_results.get("latency_ms", 0.0)
                infer_fps = _shared_results.get("infer_fps", 0.0)

            if frame is None:
                time.sleep(0.001)
                continue

            # Display FPS (main loop speed).
            display_count += 1
            now = time.time()
            if now - display_t0 >= 1.0:
                display_fps = display_count / (now - display_t0)
                display_count = 0
                display_t0 = now

            if not headless:
                # GUI mode: minimal drawing on frame (non-blocking).
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

            # Rate-limited CLI update (every 0.5s).
            if now - last_cli_update >= CLI_UPDATE_INTERVAL:
                _print_cli_stats(
                    display_fps, infer_fps, latency_ms, status, eyes, objects
                )
                last_cli_update = now

            # Throttle main loop to ~30 FPS to prevent CPU starvation of inference thread.
            time.sleep(1.0 / 30.0)
    finally:
        _camera_stop.set()
        _inference_stop.set()
        cam_thread.join(timeout=1.0)
        inf_thread.join(timeout=1.0)
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
