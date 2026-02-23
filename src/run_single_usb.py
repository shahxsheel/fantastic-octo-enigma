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
import queue

# Pi 5 ARM SVE: set before imports that use MediaPipe/XNNPACK.
os.environ.setdefault("XNNPACK_FORCE_QUIRK_FOR_ARM_SVE", "1")
# Force NCNN maximum resources (set before any imports that use NCNN).
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NCNN_THREADS"] = "4"

from supabase import Client, create_client

from src.camera.camera_source import open_camera
from src.components.buzzer import BuzzerController
from src.components.gps import GPSReader
from src.components.speed_limit import SpeedLimitChecker
from src.infer.face_eye_mediapipe import FaceEyeEstimatorMediaPipeSync
from src.infer.risk_engine import RiskEngine
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


# CLI update rate (seconds).
CLI_UPDATE_INTERVAL = 0.5

# Camera thread target FPS (avoids pulling 60 FPS when display is capped at 30).
CAMERA_TARGET_FPS = 35.0


def create_gamma_lut(gamma: float = 1.5) -> np.ndarray:
    """Build a 256-entry LUT for fast gamma correction (brightens low-light)."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return table


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

    # Red border if RiskEngine WARN/ALERT (thickness=5).
    if status in ("WARN", "ALERT"):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)


def _print_cli_stats(
    display_fps: float,
    infer_fps: float,
    latency_ms: float,
    status: str,
    eyes: Optional[Any],
    objects: list[dict],
    night_mode: bool = False,
    head_direction: Optional[str] = None,
    speed_limit: Optional[int] = None,
) -> None:
    """Print CLI dashboard on a single line (overwrites with \\r, padded with spaces)."""
    parts = [
        f"[DISPLAY: {display_fps:.0f}fps]",
        f"[INFER: {infer_fps:.1f}fps ({latency_ms:.0f}ms)]",
        f"STATUS: {status}",
    ]
    if head_direction is not None:
        parts.append(f"DIR: {head_direction}")
    if speed_limit is not None:
        parts.append(f"LIMIT: {speed_limit}mph")
    if night_mode:
        parts.append("[NIGHT_MODE: ON]")

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
    # Smaller infer size (e.g. 256x256) reduces memory bandwidth for Pi 5.
    os.environ.setdefault("INFER_WIDTH", "256")
    os.environ.setdefault("INFER_HEIGHT", "256")

    headless = _env_bool("HEADLESS", False)
    # Low-light: gamma correction (default ON for safety).
    night_mode = _env_bool("LOW_LIGHT", True)
    gamma_table: Optional[np.ndarray] = None
    if night_mode:
        gamma_table = create_gamma_lut(gamma=1.5)
        print("[single-usb] night mode ON (gamma=1.5)", flush=True)

    cam, cam_name = open_camera(headless=headless)
    print(f"[single-usb] using camera={cam_name}", flush=True)
    if hasattr(cam, "configure_low_light"):
        try:
            cam.configure_low_light()
        except Exception:
            pass

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
    risk_engine = RiskEngine()
    buzzer = BuzzerController(pin=18)
    buzzer.start()

    # Supabase client (optional; only if env vars provided).
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    supabase_client: Optional[Client] = None
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("[single-usb] Supabase client initialized", flush=True)
        except Exception as e:
            print(f"[single-usb] Supabase init failed: {e}", flush=True)

    # GPS reader (background thread inside GPSReader).
    gps = GPSReader()
    gps.start()

    # Telemetry queue for background Supabase writes.
    telemetry_queue: "queue.Queue[dict]" = queue.Queue(maxsize=50)

    # Speed limit checker (OSM Overpass); background thread updates _shared_results.
    speed_checker = SpeedLimitChecker()

    # Thread-safe shared state.
    _lock = threading.Lock()
    # Double buffer: camera writes to _back, readers use _front; swap under lock (minimal hold).
    _buf0: Optional[np.ndarray] = None
    _buf1: Optional[np.ndarray] = None
    _front: Optional[np.ndarray] = None
    _back: Optional[np.ndarray] = None
    _shared_results: dict = {
        "face_bbox": None,
        "eyes": None,
        "objects": [],
        "status": "FOCUSED",
        "head_direction": "CENTER",
        "speed_limit": None,
        "latency_ms": 0.0,
        "infer_fps": 0.0,
        "infer_count": 0,
        "infer_t0": time.time(),
    }
    _camera_stop = threading.Event()
    _inference_stop = threading.Event()

    # One-time read to get frame shape; allocate double buffer and prime _front.
    try:
        bundle = cam.read()
        first_frame = bundle.infer_bgr.copy()
        if night_mode and gamma_table is not None:
            first_frame = cv2.LUT(first_frame, gamma_table)
    except Exception as e:
        print(f"[single-usb] Initial camera read failed: {e}", flush=True)
        return
    h, w = first_frame.shape[:2]
    _buf0 = np.empty((h, w, 3), dtype=np.uint8)
    _buf1 = np.empty((h, w, 3), dtype=np.uint8)
    np.copyto(_buf0, first_frame)
    _front = _buf0
    _back = _buf1

    def camera_thread_fn() -> None:
        nonlocal _front, _back
        while not _camera_stop.is_set():
            cam_loop_start = time.time()
            try:
                bundle = cam.read()
                np.copyto(_back, bundle.infer_bgr)
                if night_mode and gamma_table is not None:
                    np.copyto(_back, cv2.LUT(_back, gamma_table))
                with _lock:
                    _front, _back = _back, _front
            except Exception:
                if _camera_stop.is_set():
                    break
                time.sleep(0.001)
            else:
                # Throttle camera to ~35 FPS to save CPU/thermal when display is 30 FPS.
                elapsed = time.time() - cam_loop_start
                sleep_time = (1.0 / CAMERA_TARGET_FPS) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def inference_thread_fn() -> None:
        """Background inference thread: continuously runs face_eye + yolo on latest frame."""
        inf_frame_idx = 0
        last_objects: list[dict] = []
        calib_frames = 0
        yaw_offset = 0.0
        pitch_offset = 0.0
        smooth_yaw = 0.0
        smooth_pitch = 0.0
        alpha = 0.3  # EMA smoothing factor
        infer_prealloc: Optional[np.ndarray] = None

        while not _inference_stop.is_set():
            with _lock:
                ref = _front
            if ref is None:
                time.sleep(0.001)
                continue
            if infer_prealloc is None:
                infer_prealloc = np.empty_like(ref)
            np.copyto(infer_prealloc, ref)

            # Always run face detection (fast).
            inference_start = time.time()
            face_bbox, eyes = face_eye.detect(infer_prealloc)

            if eyes:
                raw_yaw = float(getattr(eyes, "yaw_deg", 0.0))
                raw_pitch = float(getattr(eyes, "pitch_deg", 0.0))

                # 1. Calibration (first 60 frames)
                if calib_frames < 60:
                    yaw_offset += raw_yaw
                    pitch_offset += raw_pitch
                    calib_frames += 1
                    if calib_frames == 60:
                        yaw_offset /= 60.0
                        pitch_offset /= 60.0
                        print(f"[System] Calibrated Center: Yaw={yaw_offset:.1f}, Pitch={pitch_offset:.1f}", flush=True)

                # 2. Apply offset (after calibration complete)
                corr_yaw = raw_yaw - (yaw_offset if calib_frames >= 60 else 0)
                corr_pitch = raw_pitch - (pitch_offset if calib_frames >= 60 else 0)

                # 3. Apply EMA smoothing
                smooth_yaw = (alpha * corr_yaw) + ((1 - alpha) * smooth_yaw)
                smooth_pitch = (alpha * corr_pitch) + ((1 - alpha) * smooth_pitch)

                # Write smoothed values back to eyes for RiskEngine
                eyes.yaw_deg = smooth_yaw
                eyes.pitch_deg = smooth_pitch

            # Interleaving: ONLY run YOLO every 10 inference loops (saves ~90% of YOLO compute).
            if inf_frame_idx % 10 == 0:
                objects = yolo.detect(infer_prealloc)
                last_objects = objects
            else:
                objects = last_objects  # Re-use last known objects

            inference_end = time.time()
            latency_ms = (inference_end - inference_start) * 1000.0
            inf_frame_idx += 1

            driver_locked = face_bbox is not None
            risk_out = risk_engine.update(
                int(time.time() * 1000),
                driver_locked,
                face_bbox,
                eyes,
                objects,
                gps.speed_mph,
            )
            status = risk_out.risk["state"]
            head_direction = risk_out.attention.get("head_direction", "CENTER")

            # Update shared results every frame so display reflects latest status/direction.
            with _lock:
                _shared_results["face_bbox"] = face_bbox
                _shared_results["eyes"] = eyes
                _shared_results["objects"] = objects
                _shared_results["status"] = status
                _shared_results["head_direction"] = head_direction
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

    def telemetry_thread_fn() -> None:
        """Background Supabase telemetry thread."""
        while True:
            payload = telemetry_queue.get()
            if not supabase_client:
                continue
            try:
                payload_type = payload.get("type", "face_detection")
                if payload_type == "face_detection":
                    del payload["type"]
                    supabase_client.table("face_detections").insert(payload).execute()
                elif payload_type == "vehicle_realtime":
                    del payload["type"]
                    supabase_client.table("vehicle_realtime").insert(payload).execute()
            except Exception:
                # Swallow errors to avoid killing the thread on network issues.
                pass

    def speed_limit_thread_fn() -> None:
        """Background thread: fetch OSM speed limit every 10s and update shared state."""
        while True:
            try:
                lat = gps.latitude
                lon = gps.longitude
                if lat != 0.0 or lon != 0.0:
                    limit = speed_checker.get_speed_limit(lat, lon)
                    if limit is not None:
                        with _lock:
                            _shared_results["speed_limit"] = limit
            except Exception:
                pass
            time.sleep(10)

    cam_thread = threading.Thread(target=camera_thread_fn, daemon=True)
    inf_thread = threading.Thread(target=inference_thread_fn, daemon=True)
    telemetry_thread = threading.Thread(target=telemetry_thread_fn, daemon=True)
    speed_thread = threading.Thread(target=speed_limit_thread_fn, daemon=True)
    cam_thread.start()
    inf_thread.start()
    telemetry_thread.start()
    speed_thread.start()

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

    # Telemetry alert rate limiting.
    ALERT_COOLDOWN_SEC = 3.0
    last_alert_time = 0.0
    last_alert_state: Optional[str] = None

    # Speeding alert rate limiting (once per 10s).
    last_speed_alert_time = 0.0

    # Reusable display buffer (zero allocation per frame in GUI mode).
    vis_buffer: Optional[np.ndarray] = None

    # Print header once.
    print("\n" + "=" * 120)
    print("Driver Monitoring System - Real-time Dashboard")
    print("=" * 120)

    try:
        while True:
            loop_start = time.time()

            with _lock:
                ref = _front
                face_bbox = _shared_results.get("face_bbox")
                eyes = _shared_results.get("eyes")
                objects = _shared_results.get("objects", [])
                status = _shared_results.get("status", "FOCUSED")
                head_direction = _shared_results.get("head_direction", "CENTER")
                speed_limit = _shared_results.get("speed_limit")
                latency_ms = _shared_results.get("latency_ms", 0.0)
                infer_fps = _shared_results.get("infer_fps", 0.0)

            if ref is None:
                time.sleep(0.001)
                continue

            # Copy into reusable display buffer (no per-frame allocation).
            if vis_buffer is None:
                vis_buffer = np.empty_like(ref)
            np.copyto(vis_buffer, ref)

            # Display FPS (main loop speed).
            display_count += 1
            now = time.time()
            if now - display_t0 >= 1.0:
                display_fps = display_count / (now - display_t0)
                display_count = 0
                display_t0 = now

            if not headless:
                # GUI mode: draw on reusable buffer (no extra copy).
                _draw_minimal_graphics(vis_buffer, face_bbox, eyes, objects, status)
                try:
                    cv2.imshow(window_name, vis_buffer)
                except cv2.error:
                    headless = True
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            # Buzzer alerts based on RiskEngine state and attention.
            if status == "ALERT":
                if head_direction != "CENTER" or any(o.get("cls") == 67 for o in objects):
                    buzzer.play_distraction_alert()
                else:
                    buzzer.play_drowsy_alert()
            elif status == "WARN":
                if head_direction != "CENTER":
                    buzzer.play_distraction_alert()

            # Telemetry alerts (Supabase) with cooldown.
            if status == "ALERT":
                alert_type = "drowsy" if head_direction == "CENTER" else "distracted"
                now_ts = time.time()
                if (now_ts - last_alert_time) >= ALERT_COOLDOWN_SEC or last_alert_state != alert_type:
                    payload = {
                        "type": "face_detection",
                        "vehicle_id": os.environ.get("VEHICLE_ID", "test-vehicle-1"),
                        "is_drowsy": True if head_direction == "CENTER" else False,
                        "intoxication_score": 0,
                        "latitude": gps.latitude,
                        "longitude": gps.longitude,
                        "speed": gps.speed_mph,
                    }
                    try:
                        telemetry_queue.put_nowait(payload)
                    except queue.Full:
                        pass  # Drop payload, keep video feed alive
                    last_alert_time = now_ts
                    last_alert_state = alert_type

            # Speeding alerts (5 MPH tolerance); buzzer + telemetry, 10s cooldown.
            if speed_limit is not None and gps.speed_mph > (speed_limit + 5):
                now_ts = time.time()
                if (now_ts - last_speed_alert_time) >= 10.0:
                    buzzer.play_speeding_alert()
                    speed_payload = {
                        "type": "vehicle_realtime",
                        "vehicle_id": os.environ.get("VEHICLE_ID", "test-vehicle-1"),
                        "is_speeding": True,
                        "speed": gps.speed_mph,
                        "speed_limit": speed_limit,
                        "latitude": gps.latitude,
                        "longitude": gps.longitude,
                    }
                    try:
                        telemetry_queue.put_nowait(speed_payload)
                    except queue.Full:
                        pass  # Drop payload, keep video feed alive
                    last_speed_alert_time = now_ts

            # Rate-limited CLI update (every 0.5s).
            if now - last_cli_update >= CLI_UPDATE_INTERVAL:
                _print_cli_stats(
                    display_fps, infer_fps, latency_ms, status, eyes, objects,
                    night_mode=night_mode,
                    head_direction=head_direction,
                    speed_limit=speed_limit,
                )
                last_cli_update = now

            # Smart sleep: lock display loop to 30 FPS.
            process_time = time.time() - loop_start
            sleep_time = (1.0 / 30.0) - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)
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
        try:
            buzzer.stop()
        except Exception:
            pass
        try:
            gps.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
