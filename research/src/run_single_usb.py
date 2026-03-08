"""
Threaded USB pipeline optimized for Raspberry Pi.

- Thread 1 (Camera): continuously grabs frames and swaps a double buffer.
- Thread 2 (Inference): runs YOLO only (no face model), updates shared state.
- Main thread: handles GUI/CLI, buzzer alerts, and telemetry scheduling.
"""

import os
import queue
import random
import string
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
from supabase import Client, create_client

from src.camera.camera_source import open_camera
from src.components.bluetooth_peripheral import BluetoothPeripheral
from src.components.buzzer import BuzzerController
from src.components.gps import GPSReader
from src.components.gyro import GyroReader
from src.components.speed_limit import SpeedLimitChecker
from src.infer.face_eye_estimator import FaceEyeEstimator
from src.infer.yolo_detector import YoloDetector

# Pi 5 ARM SVE + NCNN threading knobs.
os.environ.setdefault("XNNPACK_FORCE_QUIRK_FOR_ARM_SVE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NCNN_THREADS", "4")

# CLI update rate (seconds).
CLI_UPDATE_INTERVAL = 0.5
# Camera thread target FPS.
CAMERA_TARGET_FPS = 35.0
ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def create_gamma_lut(gamma: float = 1.5) -> np.ndarray:
    """Build a 256-entry LUT for fast gamma correction (brightens low-light)."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return table


def _colorize(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{ANSI_RESET}"
    return text


def _largest_person_bbox(objects: list[dict]) -> Optional[list[int]]:
    best_bbox: Optional[list[int]] = None
    best_area = -1
    for obj in objects:
        if obj.get("name") != "person":
            continue
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best_bbox = bbox
    return best_bbox


def _scale_bbox(
    bbox: list[int],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[int]:
    if src_w <= 0 or src_h <= 0:
        return bbox

    x_scale = dst_w / float(src_w)
    y_scale = dst_h / float(src_h)
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * x_scale)),
        int(round(y1 * y_scale)),
        int(round(x2 * x_scale)),
        int(round(y2 * y_scale)),
    ]


def _object_name(obj: dict) -> str:
    return str(obj.get("name", "")).strip().lower()


def _object_cls(obj: dict) -> Optional[int]:
    try:
        cls_val = obj.get("cls")
        if cls_val is None:
            return None
        return int(cls_val)
    except Exception:
        return None


def _is_phone_object(obj: dict) -> bool:
    name = _object_name(obj)
    cls_id = _object_cls(obj)
    return name in {"cell phone", "phone", "mobile phone", "smartphone"} or cls_id == 67


def _is_drinking_object(obj: dict) -> bool:
    name = _object_name(obj)
    cls_id = _object_cls(obj)
    return name in {"bottle", "cup"} or cls_id in {39, 41}


def _compute_driver_state(objects: list[dict]) -> tuple[str, str, bool, bool, int]:
    """
    Returns:
      alert_state: FOCUSED/WARN/ALERT (for local buzzer/overlay)
      driver_status: value persisted to vehicle_realtime.driver_status
      is_phone_detected
      is_drinking_detected
      intoxication_score (coarse score for existing UI contract)
    """
    phone_detected = any(_is_phone_object(o) for o in objects)
    drinking_detected = any(_is_drinking_object(o) for o in objects)

    if phone_detected:
        return "ALERT", "distracted_phone", True, drinking_detected, 4
    if drinking_detected:
        return "WARN", "distracted_drinking", False, True, 2
    return "FOCUSED", "alert", False, False, 0


class SideLookTracker:
    """Tracks continuous side-looking duration and emits warning when threshold is reached."""

    def __init__(self, threshold_seconds: float = 2.0):
        self.threshold_seconds = max(0.1, threshold_seconds)
        self._side_look_start_ts: Optional[float] = None

    def update(self, head_direction: str, now_ts: float) -> bool:
        direction = head_direction.upper().strip()
        if direction in ("LEFT", "RIGHT"):
            if self._side_look_start_ts is None:
                self._side_look_start_ts = now_ts
            return (now_ts - self._side_look_start_ts) >= self.threshold_seconds

        if direction == "CENTER":
            self._side_look_start_ts = None

        return False


def _resolve_driver_state(
    objects: list[dict], sideways_warning_active: bool, is_drowsy: bool = False
) -> tuple[str, str, bool, bool, int]:
    """
    Resolves final driver state with precedence:
      1) phone alert (4/6)
      2) drowsiness alert (5/6)
      3) side-look warning (2/6)
      4) drinking warning (2/6)
      5) focused (0/6)
    """
    base_alert, base_status, phone_detected, drinking_detected, base_score = _compute_driver_state(objects)
    if phone_detected:
        return base_alert, base_status, phone_detected, drinking_detected, base_score

    if is_drowsy:
        return "ALERT", "drowsy", False, drinking_detected, 5

    if sideways_warning_active:
        return "WARN", "distracted_side_look", False, drinking_detected, 2

    return base_alert, base_status, phone_detected, drinking_detected, base_score


def _generate_invite_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _draw_minimal_graphics(
    frame: np.ndarray, objects: list[dict], alert_state: str, head_direction: str
) -> None:
    """Draw YOLO boxes and an alert border in GUI mode."""
    h, w = frame.shape[:2]

    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        name = str(obj.get("name", "?"))
        conf = float(obj.get("conf", 0.0))

        is_phone = _is_phone_object(obj)
        is_drink = _is_drinking_object(obj)
        if is_phone:
            color = (0, 0, 255)  # red
        elif is_drink:
            color = (0, 165, 255)  # orange
        else:
            color = (0, 255, 255)  # yellow

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            color,
            1,
            cv2.LINE_AA,
        )

    if alert_state in ("WARN", "ALERT"):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)

    cv2.putText(
        frame,
        f"DIR: {head_direction}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _print_cli_stats(
    infer_fps: float,
    latency_ms: float,
    alert_state: str,
    driver_status: str,
    head_direction: str,
    objects: list[dict],
    speed_mph: int,
    speed_limit: Optional[int],
    heading: float,
    night_mode: bool = False,
    left_ear: float = 1.0,
    right_ear: float = 1.0,
    is_drowsy: bool = False,
) -> None:
    """Print single-line dashboard (overwrite with \r)."""
    parts = [
        f"[INFER: {infer_fps:.1f}fps ({latency_ms:.0f}ms)]",
        f"STATE: {alert_state}",
        f"DRIVER: {driver_status}",
        f"LOOK: {head_direction}",
        f"L:{min(100, int(left_ear / 0.35 * 100))}% R:{min(100, int(right_ear / 0.35 * 100))}%",
        f"SPEED: {speed_mph}mph",
        f"HDG: {int(round(heading)) % 360}",
    ]

    if is_drowsy:
        parts.append(_colorize("[DROWSY!]", ANSI_RED))
    if speed_limit is not None:
        parts.append(f"LIMIT: {speed_limit}mph")
    if night_mode:
        parts.append("[NIGHT_MODE: ON]")
    if objects:
        obj_names = [str(obj.get("name", "?")) for obj in objects[:3]]
        parts.append(f"OBJS: {', '.join(obj_names)}")

    line = " | ".join(parts)
    padded = line.ljust(140)
    sys.stdout.write(f"\r{padded}")
    sys.stdout.flush()


def main() -> None:
    os.environ.setdefault("FORCE_CAMERA", "usb")
    os.environ.setdefault("INFER_WIDTH", "256")
    os.environ.setdefault("INFER_HEIGHT", "256")
    # Keep YOLO focused on distraction classes.
    os.environ.setdefault("YOLO_FILTER", "person,cell phone,bottle,cup")

    vehicle_id = os.environ.get("VEHICLE_ID", "test-vehicle-1")

    headless = _env_bool("HEADLESS", False)
    if not headless and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        headless = True
        print("[single-usb] no display found; switching to HEADLESS=1", flush=True)

    use_fake_gps = _env_bool("USE_FAKE_GPS", False)
    use_fake_gyro = _env_bool("USE_FAKE_GYRO", False)

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

    try:
        bundle = cam.read()
        infer_h, infer_w = bundle.infer_bgr.shape[:2]
        print(f"[single-usb] infer_size={infer_w}x{infer_h}", flush=True)
    except Exception as e:
        print(f"[single-usb] Initial camera read failed: {e}", flush=True)
        return

    yolo = YoloDetector()
    face_eye_estimator = FaceEyeEstimator(
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face_landmarker.task")
    )
    side_look_tracker = SideLookTracker(threshold_seconds=2.0)
    buzzer = BuzzerController(pin=18)
    buzzer.start()

    supabase_client: Optional[Client] = None
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase_configured = bool(supabase_url and supabase_key)
    supabase_placeholder = (
        not supabase_configured
        or "your-project-id" in (supabase_url or "")
        or "your-supabase-anon-key" in (supabase_key or "")
    )

    if supabase_configured and not supabase_placeholder:
        try:
            supabase_client = create_client(supabase_url, supabase_key)
        except Exception as e:
            print(_colorize(f"[single-usb] SUPABASE: INACTIVE ({e})", ANSI_RED), flush=True)
    else:
        reason = "missing config" if not supabase_configured else "placeholder config"
        print(_colorize(f"[single-usb] SUPABASE: INACTIVE ({reason})", ANSI_RED), flush=True)

    if supabase_client is not None:
        print(_colorize("[single-usb] SUPABASE: ACTIVE", ANSI_GREEN), flush=True)
        # Register this vehicle in the vehicles table so the iOS app can discover it.
        # Uses upsert so repeated startups are idempotent and don't overwrite existing metadata.
        try:
            vehicle_name = os.environ.get("VEHICLE_NAME", "").strip() or None
            vehicle_row = {"id": vehicle_id}
            if vehicle_name is not None:
                vehicle_row["name"] = vehicle_name
            supabase_client.table("vehicles").upsert(
                vehicle_row,
                on_conflict="id",
                ignore_duplicates=False,
            ).execute()
            print(f"[single-usb] Vehicle registered: {vehicle_id}", flush=True)
        except Exception as e:
            error_text = str(e)
            if "invite_code" in error_text and "null value" in error_text:
                try:
                    legacy_row = dict(vehicle_row)
                    legacy_row["invite_code"] = _generate_invite_code()
                    supabase_client.table("vehicles").upsert(
                        legacy_row,
                        on_conflict="id",
                        ignore_duplicates=False,
                    ).execute()
                    print(
                        _colorize(
                            f"[single-usb] Vehicle registered with legacy invite_code fallback: {vehicle_id}",
                            ANSI_YELLOW,
                        ),
                        flush=True,
                    )
                except Exception as fallback_error:
                    print(
                        _colorize(
                            f"[single-usb] Vehicle registration failed (fallback): {fallback_error}",
                            ANSI_YELLOW,
                        ),
                        flush=True,
                    )
            else:
                print(_colorize(f"[single-usb] Vehicle registration failed: {e}", ANSI_YELLOW), flush=True)

    if face_eye_estimator.enabled:
        print(_colorize("[single-usb] FACE/EYE: ACTIVE (MediaPipe FaceLandmarker)", ANSI_GREEN), flush=True)
    else:
        print(_colorize(f"[single-usb] FACE/EYE: INACTIVE ({face_eye_estimator.reason})", ANSI_YELLOW), flush=True)

    gps = GPSReader(force_fake=use_fake_gps)
    gps.start()
    if gps.is_fake:
        print("[single-usb] GPS source: fake data", flush=True)

    gyro: Optional[GyroReader] = GyroReader(
        force_fake=use_fake_gyro,
        allow_fake_fallback=True,
    )
    try:
        gyro.start()
        if gyro.is_fake:
            print("[single-usb] Gyro source: fake data", flush=True)
        else:
            print("[single-usb] Gyro reader started", flush=True)
    except Exception as e:
        print(f"[single-usb] Gyro unavailable (simulated): {e}", flush=True)
        gyro = None

    speed_checker = SpeedLimitChecker()

    telemetry_queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=100)

    _lock = threading.Lock()
    _infer_buf0: Optional[np.ndarray] = None
    _infer_buf1: Optional[np.ndarray] = None
    _infer_front: Optional[np.ndarray] = None
    _infer_back: Optional[np.ndarray] = None
    _main_buf0: Optional[np.ndarray] = None
    _main_buf1: Optional[np.ndarray] = None
    _main_front: Optional[np.ndarray] = None
    _main_back: Optional[np.ndarray] = None

    _shared_results: dict = {
        "objects": [],
        "alert_state": "FOCUSED",
        "driver_status": "alert",
        "is_phone_detected": False,
        "is_drinking_detected": False,
        "intoxication_score": 0,
        "head_direction": "UNKNOWN",
        "left_ear": 1.0,
        "right_ear": 1.0,
        "is_drowsy": False,
        "speed_limit": None,
        "latency_ms": 0.0,
        "infer_fps": 0.0,
        "infer_count": 0,
        "infer_t0": time.time(),
        "gyro": None,
        # GPS fields — updated each main-loop tick for BLE peripheral
        "speed_mph": 0,
        "heading_degrees": 0,
        "compass_direction": "N",
        "latitude": 0.0,
        "longitude": 0.0,
        "satellites": 0,
        "is_speeding": False,
    }

    # ── BLE peripheral ──────────────────────────────────────────────────────
    def _on_ble_settings(data: dict) -> None:
        """iOS wrote new feature-toggle settings; apply them to the runtime."""
        # These are advisory — the Pi honours them on a best-effort basis.
        print(f"[BLE] Settings update from app: {data}", flush=True)

    def _on_ble_buzzer(data: dict) -> None:
        """iOS wrote a buzzer command; fire the local buzzer immediately."""
        if data.get("active"):
            buzzer.play_distraction_alert()
        else:
            buzzer.stop_alert() if hasattr(buzzer, "stop_alert") else None

    ble = BluetoothPeripheral(
        vehicle_id=vehicle_id,
        shared_results=_shared_results,
        lock=_lock,
        on_settings=_on_ble_settings,
        on_buzzer=_on_ble_buzzer,
    )
    ble.start()
    if ble.enabled:
        print(_colorize("[single-usb] BLE: ACTIVE", ANSI_GREEN), flush=True)
    else:
        print(_colorize(f"[single-usb] BLE: INACTIVE ({ble.reason})", ANSI_YELLOW), flush=True)

    _camera_stop = threading.Event()
    _inference_stop = threading.Event()
    _telemetry_stop = threading.Event()
    _speed_stop = threading.Event()

    try:
        first_infer_frame = bundle.infer_bgr.copy()
        if night_mode and gamma_table is not None:
            first_infer_frame = cv2.LUT(first_infer_frame, gamma_table)
        if not headless:
            first_main_frame = bundle.main_bgr.copy()
            if night_mode and gamma_table is not None:
                first_main_frame = cv2.LUT(first_main_frame, gamma_table)
    except Exception as e:
        print(f"[single-usb] Failed to prime buffers: {e}", flush=True)
        return

    infer_h, infer_w = first_infer_frame.shape[:2]
    _infer_buf0 = np.empty((infer_h, infer_w, 3), dtype=np.uint8)
    _infer_buf1 = np.empty((infer_h, infer_w, 3), dtype=np.uint8)
    np.copyto(_infer_buf0, first_infer_frame)
    _infer_front = _infer_buf0
    _infer_back = _infer_buf1

    if not headless:
        main_h, main_w = first_main_frame.shape[:2]
        _main_buf0 = np.empty((main_h, main_w, 3), dtype=np.uint8)
        _main_buf1 = np.empty((main_h, main_w, 3), dtype=np.uint8)
        np.copyto(_main_buf0, first_main_frame)
        _main_front = _main_buf0
        _main_back = _main_buf1

    def camera_thread_fn() -> None:
        nonlocal _infer_front, _infer_back, _main_front, _main_back
        while not _camera_stop.is_set():
            cam_loop_start = time.time()
            try:
                local_bundle = cam.read()
                np.copyto(_infer_back, local_bundle.infer_bgr)
                if night_mode and gamma_table is not None:
                    np.copyto(_infer_back, cv2.LUT(_infer_back, gamma_table))
                if not headless:
                    np.copyto(_main_back, local_bundle.main_bgr)
                    if night_mode and gamma_table is not None:
                        np.copyto(_main_back, cv2.LUT(_main_back, gamma_table))
                with _lock:
                    _infer_front, _infer_back = _infer_back, _infer_front
                    if not headless:
                        _main_front, _main_back = _main_back, _main_front
            except Exception:
                if _camera_stop.is_set():
                    break
                time.sleep(0.001)
            else:
                if not getattr(cam, 'using_gstreamer', False):
                    elapsed = time.time() - cam_loop_start
                    sleep_time = (1.0 / CAMERA_TARGET_FPS) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

    def inference_thread_fn() -> None:
        inf_frame_idx = 0
        last_objects: list[dict] = []
        infer_prealloc: Optional[np.ndarray] = None
        main_prealloc: Optional[np.ndarray] = None

        while not _inference_stop.is_set():
            with _lock:
                infer_ref = _infer_front
                main_ref = _main_front
            if infer_ref is None:
                time.sleep(0.001)
                continue

            if infer_prealloc is None:
                infer_prealloc = np.empty_like(infer_ref)
            np.copyto(infer_prealloc, infer_ref)

            if main_ref is not None:
                if main_prealloc is None:
                    main_prealloc = np.empty_like(main_ref)
                np.copyto(main_prealloc, main_ref)

            inference_start = time.time()

            # Run full YOLO every 3 loops; reuse last objects between loops.
            if inf_frame_idx % 3 == 0:
                objects = yolo.detect(infer_prealloc)
                last_objects = objects
            else:
                objects = last_objects

            src_frame = main_prealloc if main_prealloc is not None else infer_prealloc
            head_direction, left_ear, right_ear, is_drowsy = face_eye_estimator.estimate(src_frame, inf_frame_idx)

            sideways_warning_active = side_look_tracker.update(
                head_direction=head_direction,
                now_ts=time.time(),
            )
            alert_state, driver_status, phone_detected, drinking_detected, intox_score = _resolve_driver_state(
                objects,
                sideways_warning_active=sideways_warning_active,
                is_drowsy=is_drowsy,
            )

            latency_ms = (time.time() - inference_start) * 1000.0
            inf_frame_idx += 1

            gyro_reading = gyro.get_latest() if gyro is not None else None
            with _lock:
                _shared_results["objects"] = objects
                _shared_results["alert_state"] = alert_state
                _shared_results["driver_status"] = driver_status
                _shared_results["is_phone_detected"] = phone_detected
                _shared_results["is_drinking_detected"] = drinking_detected
                _shared_results["intoxication_score"] = intox_score
                _shared_results["head_direction"] = head_direction
                _shared_results["left_ear"] = left_ear
                _shared_results["right_ear"] = right_ear
                _shared_results["is_drowsy"] = is_drowsy
                _shared_results["latency_ms"] = latency_ms
                _shared_results["gyro"] = gyro_reading

                _shared_results["infer_count"] += 1
                now = time.time()
                if now - _shared_results["infer_t0"] >= 1.0:
                    _shared_results["infer_fps"] = (
                        _shared_results["infer_count"] / (now - _shared_results["infer_t0"])
                    )
                    _shared_results["infer_count"] = 0
                    _shared_results["infer_t0"] = now

    def telemetry_thread_fn() -> None:
        while not _telemetry_stop.is_set():
            try:
                payload = telemetry_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if payload is None:
                break

            if not supabase_client:
                continue

            try:
                payload_type = payload.pop("type", "vehicle_realtime")
                if payload_type == "vehicle_realtime":
                    supabase_client.table("vehicle_realtime").upsert(payload).execute()
                elif payload_type == "vehicle_trips":
                    supabase_client.table("vehicle_trips").upsert(payload).execute()
            except Exception:
                # Keep loop alive through transient network failures.
                pass

    def speed_limit_thread_fn() -> None:
        while not _speed_stop.is_set():
            try:
                lat = gps.latitude
                lon = gps.longitude
                if lat != 0.0 or lon != 0.0:
                    limit = speed_checker.get_speed_limit(lat, lon)
                    if limit is not None:
                        with _lock:
                            _shared_results["speed_limit"] = int(limit)
            except Exception:
                pass
            _speed_stop.wait(10.0)

    cam_thread = threading.Thread(target=camera_thread_fn, daemon=True)
    inf_thread = threading.Thread(target=inference_thread_fn, daemon=True)
    telemetry_thread = threading.Thread(target=telemetry_thread_fn, daemon=True)
    speed_thread = threading.Thread(target=speed_limit_thread_fn, daemon=True)
    cam_thread.start()
    inf_thread.start()
    telemetry_thread.start()
    speed_thread.start()

    window_name = "Driver Monitoring System"
    if headless:
        print(_colorize("[single-usb] HEADLESS MODE: UI rendering disabled", ANSI_YELLOW), flush=True)
    else:
        try:
            cv2.startWindowThread()
            gui_normal = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | gui_normal)
        except Exception as e:
            headless = True
            print(
                _colorize(
                    f"[single-usb] GUI unavailable; switching to HEADLESS=1 ({e})",
                    ANSI_YELLOW,
                ),
                flush=True,
            )

    last_cli_update = 0.0

    last_distraction_buzzer_time = 0.0
    distraction_buzzer_cooldown = 2.0
    last_speed_buzzer_time = 0.0

    telemetry_interval_sec = 1.0
    last_telemetry_time = 0.0

    vis_buffer: Optional[np.ndarray] = None if not headless else None

    print("\n" + "=" * 120)
    print("Driver Monitoring System - YOLO / GPS / Gyro / Buzzer Dashboard")
    print("=" * 120)

    try:
        while True:
            loop_start = time.time()

            with _lock:
                ref = _infer_front
                objects = _shared_results.get("objects", [])
                alert_state = _shared_results.get("alert_state", "FOCUSED")
                driver_status = _shared_results.get("driver_status", "alert")
                phone_detected = bool(_shared_results.get("is_phone_detected", False))
                drinking_detected = bool(_shared_results.get("is_drinking_detected", False))
                intoxication_score = int(_shared_results.get("intoxication_score", 0))
                head_direction = _shared_results.get("head_direction", "UNKNOWN")
                left_ear = float(_shared_results.get("left_ear", 1.0))
                right_ear = float(_shared_results.get("right_ear", 1.0))
                is_drowsy = bool(_shared_results.get("is_drowsy", False))
                speed_limit = _shared_results.get("speed_limit")
                latency_ms = float(_shared_results.get("latency_ms", 0.0))
                infer_fps = float(_shared_results.get("infer_fps", 0.0))
                gyro_snap = _shared_results.get("gyro")

            if ref is None:
                time.sleep(0.001)
                continue

            now = time.time()

            if not headless:
                if vis_buffer is None:
                    vis_buffer = np.empty_like(ref)
                np.copyto(vis_buffer, ref)
                _draw_minimal_graphics(vis_buffer, objects, alert_state, head_direction)
                try:
                    cv2.imshow(window_name, vis_buffer)
                except cv2.error as e:
                    headless = True
                    vis_buffer = None
                    print(
                        _colorize(
                            f"[single-usb] GUI render failed; continuing headless ({e})",
                            ANSI_YELLOW,
                        ),
                        flush=True,
                    )
                else:
                    wait_ms = max(1, 33 - int((time.time() - loop_start) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord("q"):
                        break

            speed_mph = int(round(gps.speed_mph))
            heading_degrees = int(round(gps.heading)) % 360
            speed_limit_mph = int(speed_limit) if speed_limit is not None else 0
            is_speeding = speed_limit is not None and speed_mph > (speed_limit_mph + 5)

            # Keep GPS fields in shared state so the BLE peripheral can read them.
            with _lock:
                _shared_results["speed_mph"] = speed_mph
                _shared_results["heading_degrees"] = heading_degrees
                _shared_results["compass_direction"] = gps.compass_direction
                _shared_results["latitude"] = gps.latitude
                _shared_results["longitude"] = gps.longitude
                _shared_results["satellites"] = gps.satellites
                _shared_results["is_speeding"] = is_speeding

            # Buzzer: distraction alerts from YOLO state.
            if (phone_detected or drinking_detected) and (now - last_distraction_buzzer_time) >= distraction_buzzer_cooldown:
                buzzer.play_distraction_alert()
                last_distraction_buzzer_time = now

            # Buzzer: speeding alerts every 10s while speeding.
            if is_speeding and (now - last_speed_buzzer_time) >= 10.0:
                buzzer.play_speeding_alert()
                last_speed_buzzer_time = now

            # Heartbeat telemetry to keep app realtime view fresh.
            if supabase_client and (now - last_telemetry_time) >= telemetry_interval_sec:
                payload = {
                    "type": "vehicle_realtime",
                    "vehicle_id": vehicle_id,
                    "latitude": gps.latitude,
                    "longitude": gps.longitude,
                    "speed_mph": speed_mph,
                    "speed_limit_mph": speed_limit_mph,
                    "heading_degrees": heading_degrees,
                    "compass_direction": gps.compass_direction,
                    "is_speeding": is_speeding,
                    "is_moving": speed_mph > 0,
                    "driver_status": driver_status,
                    "intoxication_score": intoxication_score,
                    "satellites": gps.satellites,
                    "is_phone_detected": phone_detected,
                    "is_drinking_detected": drinking_detected,
                    "is_drowsy": is_drowsy,
                }

                # Keep gyro values in logs only unless matching DB columns are added.
                _ = gyro_snap

                try:
                    telemetry_queue.put_nowait(payload)
                except queue.Full:
                    pass

                last_telemetry_time = now

            if now - last_cli_update >= CLI_UPDATE_INTERVAL:
                _print_cli_stats(
                    infer_fps=infer_fps,
                    latency_ms=latency_ms,
                    alert_state=alert_state,
                    driver_status=driver_status,
                    head_direction=head_direction,
                    objects=objects,
                    speed_mph=speed_mph,
                    speed_limit=speed_limit_mph if speed_limit is not None else None,
                    heading=float(heading_degrees),
                    night_mode=night_mode,
                    left_ear=left_ear,
                    right_ear=right_ear,
                    is_drowsy=is_drowsy,
                )
                last_cli_update = now

            if headless:
                process_time = time.time() - loop_start
                sleep_time = (1.0 / 30.0) - process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        _camera_stop.set()
        _inference_stop.set()
        _speed_stop.set()
        _telemetry_stop.set()

        try:
            telemetry_queue.put_nowait(None)
        except queue.Full:
            pass

        cam_thread.join(timeout=1.0)
        inf_thread.join(timeout=1.0)
        speed_thread.join(timeout=1.0)
        telemetry_thread.join(timeout=1.0)

        cam.release()
        sys.stdout.write("\n")
        sys.stdout.flush()

        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        try:
            face_eye_estimator.close()
        except Exception:
            pass

        try:
            ble.stop()
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

        if gyro is not None:
            try:
                gyro.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
