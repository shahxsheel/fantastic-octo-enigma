import os
import time
from typing import Optional

import cv2
import numpy as np

from src.camera.camera_source import open_camera
from src.ipc.zmq_frames import FramePublisher
from src.ipc.zmq_results import ResultSubscriber


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _overlay_objects(result: dict) -> list[dict]:
    """Keep overlays readable by capping duplicate class boxes."""
    objs = list(result.get("objects", []))
    max_total = _env_int("UI_MAX_OBJECTS", 4)
    max_person = _env_int("UI_MAX_PERSON", 1)

    if not objs:
        return []

    out: list[dict] = []
    person_count = 0
    for obj in objs:
        if len(out) >= max_total:
            break
        if str(obj.get("name", "")) == "person":
            if max_person > 0 and person_count >= max_person:
                continue
            person_count += 1
        out.append(obj)
    return out


def _risk_color(state: str) -> tuple[int, int, int]:
    s = (state or "").upper()
    if s == "ALERT":
        return (0, 0, 255)
    if s == "WARN":
        return (0, 255, 255)
    return (0, 255, 120)


def draw_results(frame: np.ndarray, result: dict, scale_x: float, scale_y: float) -> None:
    # objects
    for obj in _overlay_objects(result):
        bbox = obj.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        conf = obj.get("conf", 0.0)
        name = obj.get("name", str(obj.get("cls", "")))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
        )

    # face
    fb = result.get("face_bbox")
    if fb:
        x1, y1, x2, y2 = fb
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # eyes as dots
    eyes_pts = result.get("eyes_points") or {}
    for side, color in (("left", (0, 255, 255)), ("right", (0, 165, 255))):
        pts = eyes_pts.get(side) or []
        for (ex, ey) in pts:
            cv2.circle(frame, (int(ex * scale_x), int(ey * scale_y)), 3, color, thickness=-1)


def _build_sidebar_rows(
    fps: float, latency_sec: Optional[float], result: Optional[dict]
) -> list[tuple[str, tuple[int, int, int], float, int, int]]:
    rows: list[tuple[str, tuple[int, int, int], float, int, int]] = []
    rows.append((f"FPS: {fps:.1f}", (0, 255, 255), 0.6, 2, 24))
    if latency_sec is not None:
        rows.append((f"Latency: {latency_sec*1000:.0f} ms", (0, 255, 255), 0.55, 2, 24))
    else:
        rows.append(("Latency: --", (0, 255, 255), 0.55, 2, 24))

    if not result:
        return rows

    objs = _overlay_objects(result)
    rows.append(("Objects:", (0, 200, 255), 0.55, 2, 20))
    for o in objs:
        rows.append(
            (f"{o.get('name', '?')[:12]} {o.get('conf', 0):.2f}", (0, 200, 255), 0.5, 1, 18)
        )

    eyes = result.get("eyes") or {}
    if eyes:
        rows.append(("Eyes:", (0, 255, 180), 0.55, 2, 20))
        rows.append(
            (f"L {eyes.get('left_state', '?')} {eyes.get('left_pct', 0):.0f}%", (0, 255, 180), 0.5, 1, 18)
        )
        rows.append(
            (f"R {eyes.get('right_state', '?')} {eyes.get('right_pct', 0):.0f}%", (0, 255, 180), 0.5, 1, 18)
        )

    driver = result.get("driver") or {}
    risk = result.get("risk") or {}
    alerts = result.get("alerts") or {}
    attention = result.get("attention") or {}
    locked = "Y" if driver.get("locked") else "N"
    conf = float(driver.get("lock_conf", 0.0))
    rows.append((f"Driver lock: {locked} {conf:.2f}", (255, 220, 120), 0.5, 1, 18))

    state = str(risk.get("state", "NORMAL"))
    score = float(risk.get("score", 0.0))
    rows.append((f"Risk: {state} {score:.0f}", _risk_color(state), 0.55, 2, 20))

    reason_codes = risk.get("reason_codes") or []
    if reason_codes:
        rows.append((f"Reason: {str(reason_codes[0])[:22]}", (0, 180, 255), 0.45, 1, 16))

    if attention:
        rows.append((f"Yaw: {attention.get('head_yaw', 0):.1f}", (180, 220, 255), 0.45, 1, 16))
        rows.append((f"PERCLOS: {attention.get('perclos', 0):.2f}", (180, 220, 255), 0.45, 1, 16))

    if alerts.get("alert"):
        rows.append(("ALERT: ATTEND ROAD", (0, 0, 255), 0.55, 2, 22))
    elif alerts.get("warn"):
        rows.append(("WARN: CHECK FOCUS", (0, 255, 255), 0.55, 2, 22))

    return rows


def build_sidebar_panel(
    frame_shape: tuple[int, int, int],
    fps: float,
    latency_sec: Optional[float],
    result: Optional[dict],
) -> tuple[int, np.ndarray]:
    h, w = frame_shape[:2]
    bar_w = max(200, min(w, _env_int("SIDEBAR_WIDTH", 260)))
    x0 = w - bar_w

    # Render only the sidebar region to avoid full-frame blending overhead.
    panel = np.zeros((h, bar_w, 3), dtype=np.uint8)
    panel[:, :] = (18, 18, 18)

    y = 22
    for text, color, scale, thickness, step in _build_sidebar_rows(fps, latency_sec, result):
        if y > h - 8:
            break
        cv2.putText(
            panel,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )
        y += step
    return x0, panel


def main() -> None:
    headless = os.environ.get("HEADLESS", "0") == "1"
    # If there's no display (common over SSH), force headless to avoid Qt/xcb crashes.
    if not os.environ.get("DISPLAY"):
        headless = True
    frames_addr = os.environ.get("FRAMES_ADDR", "tcp://127.0.0.1:5555")
    results_addr = os.environ.get("RESULTS_ADDR", "tcp://127.0.0.1:5556")
    ui_every_n = _env_int("UI_EVERY_N", 2)
    sidebar_every_n = max(1, _env_int("SIDEBAR_EVERY_N", 2))
    verbose = _env_bool("VERBOSE", True)

    cam, cam_name = open_camera(headless=headless)
    print(f"[camera] using camera={cam_name}")
    window_name = "Camera Preview (Split Pipeline) - Q to quit"

    if not headless:
        # Some OpenCV builds behave better with an explicit window thread.
        try:
            cv2.startWindowThread()
            # Use GUI_NORMAL to hide the Qt toolbar/status UI.
            gui_normal = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | gui_normal)
        except Exception:
            pass

    # ZMQ: publish frames; subscribe to results
    pub = FramePublisher(frames_addr)
    sub = ResultSubscriber(results_addr, conflate=True)

    last_result = None
    last_alert_state = "NORMAL"
    fps = 0.0
    fps_count = 0
    fps_t0 = time.time()
    sidebar_cache: Optional[tuple[int, np.ndarray]] = None

    try:
        while True:
            bundle = cam.read()
            main_bgr = bundle.main_bgr
            infer_bgr = bundle.infer_bgr

            # Send inference-sized frame
            pub.send(infer_bgr, frame_id=bundle.frame_id, ts_ms=bundle.ts_ms)

            # Non-blocking receive latest result
            latest = sub.recv_latest(timeout_ms=0)
            if latest is not None:
                last_result = latest.to_dict()

            now = time.time()
            now_ms = int(now * 1000)
            # Result age = how old is the overlay we're drawing (capture → now)
            latency_sec = None
            if last_result and "ts_ms" in last_result:
                latency_ms = now_ms - last_result["ts_ms"]
                latency_sec = max(0, latency_ms) / 1000.0

            if last_result:
                risk = last_result.get("risk") or {}
                alerts = last_result.get("alerts") or {}
                cur_alert_state = str(risk.get("state", "NORMAL"))
                if cur_alert_state != last_alert_state:
                    last_alert_state = cur_alert_state
                    if verbose:
                        print(
                            f"[camera] risk transition -> {cur_alert_state} "
                            f"alerts={alerts}",
                            flush=True,
                        )
                    if alerts.get("alert") and _env_bool("ALERT_BELL", False):
                        print("\a", end="", flush=True)

            infer_h, infer_w = infer_bgr.shape[:2]
            main_h, main_w = main_bgr.shape[:2]
            scale_x = main_w / max(1, infer_w)
            scale_y = main_h / max(1, infer_h)

            fps_count += 1
            if now - fps_t0 >= 1.0:
                fps = fps_count / (now - fps_t0)
                if verbose:
                    if latency_sec is not None:
                        print(f"[camera] FPS: {fps:.1f}  result latency: {latency_sec:.2f}s", flush=True)
                    else:
                        print(f"[camera] FPS: {fps:.1f}  (no result yet)", flush=True)
                fps_count = 0
                fps_t0 = now

            if not headless:
                if last_result and (bundle.frame_id % ui_every_n == 0):
                    draw_results(main_bgr, last_result, scale_x, scale_y)
                if (
                    sidebar_cache is None
                    or bundle.frame_id % sidebar_every_n == 0
                    or sidebar_cache[1].shape[0] != main_bgr.shape[0]
                ):
                    sidebar_cache = build_sidebar_panel(
                        main_bgr.shape, fps, latency_sec, last_result
                    )
                if sidebar_cache is not None:
                    x0, panel = sidebar_cache
                    main_bgr[:, x0:] = panel
                try:
                    cv2.imshow(window_name, main_bgr)
                except cv2.error as e:
                    # Headless OpenCV builds don't implement imshow; fall back to headless mode.
                    print(f"[camera] imshow unavailable ({e}). Switching to HEADLESS=1 behavior.")
                    headless = True
                    continue
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    # Toggle channel swap at runtime (persists for this process)
                    cur = os.environ.get("SWAP_RB", "0")
                    os.environ["SWAP_RB"] = "0" if cur == "1" else "1"
                    print(f"[camera] SWAP_RB={os.environ['SWAP_RB']}")
            else:
                time.sleep(0.001)
    finally:
        cam.release()
        pub.close()
        sub.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
