import os
import time

import cv2
import numpy as np
import zmq

from src.camera.camera_source import open_camera
from src.ipc.zmq_frames import FramePublisher
from src.ipc.zmq_results import ResultSubscriber


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def draw_results(frame: np.ndarray, result: dict, scale_x: float, scale_y: float) -> None:
    # objects
    for obj in result.get("objects", [])[:20]:
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

    # face + eyes
    fb = result.get("face_bbox")
    if fb:
        x1, y1, x2, y2 = fb
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        eyes = result.get("eyes") or {}
        if eyes:
            lpct = eyes.get("left_pct", eyes.get("left_score", 0.0))
            rpct = eyes.get("right_pct", eyes.get("right_score", 0.0))
            txt = (
                f"L:{eyes.get('left_state','?')} {float(lpct):.0f}%  "
                f"R:{eyes.get('right_state','?')} {float(rpct):.0f}%"
            )
            cv2.putText(
                frame,
                txt,
                (x1, min(frame.shape[0] - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )


def main() -> None:
    headless = os.environ.get("HEADLESS", "0") == "1"
    # If there's no display (common over SSH), force headless to avoid Qt/xcb crashes.
    if not os.environ.get("DISPLAY"):
        headless = True
    frames_addr = os.environ.get("FRAMES_ADDR", "tcp://127.0.0.1:5555")
    results_addr = os.environ.get("RESULTS_ADDR", "tcp://127.0.0.1:5556")
    ui_every_n = _env_int("UI_EVERY_N", 2)

    cam, cam_name = open_camera()
    print(f"[camera] using camera={cam_name}")

    if not headless:
        # Some OpenCV builds behave better with an explicit window thread.
        try:
            cv2.startWindowThread()
        except Exception:
            pass

    # ZMQ: publish frames; subscribe to results
    pub = FramePublisher(frames_addr)
    sub = ResultSubscriber(results_addr, conflate=True)

    last_result = None
    fps = 0.0
    fps_count = 0
    fps_t0 = time.time()

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

            infer_h, infer_w = infer_bgr.shape[:2]
            main_h, main_w = main_bgr.shape[:2]
            scale_x = main_w / max(1, infer_w)
            scale_y = main_h / max(1, infer_h)

            fps_count += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_count / (now - fps_t0)
                fps_count = 0
                fps_t0 = now

            if not headless:
                if last_result and (bundle.frame_id % ui_every_n == 0):
                    draw_results(main_bgr, last_result, scale_x, scale_y)
                cv2.putText(
                    main_bgr,
                    f"FPS: {fps:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                try:
                    cv2.imshow("Camera Preview (Split Pipeline) - Q to quit", main_bgr)
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
