import os
import time
from typing import Optional

from src.infer.driver_lock import DriverLock
from src.infer.face_eye_mediapipe import FaceEyeEstimatorMediaPipe
from src.infer.risk_engine import RiskEngine
from src.infer.yolo_detector import YoloDetector
from src.ipc.zmq_alerts import AlertPublisher
from src.ipc.zmq_frames import FrameSubscriber
from src.ipc.zmq_results import InferResult, ResultPublisher


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _clip_bbox(b: list[int], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def _crop_frame(frame, bbox: list[int]):
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def _map_bbox(bbox: Optional[list[int]], ox: int, oy: int) -> Optional[list[int]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]


def _map_points(points, ox: int, oy: int):
    if not points:
        return None
    return [(int(x + ox), int(y + oy)) for x, y in points]


def _map_objects(objs: list[dict], ox: int, oy: int, w: int, h: int) -> list[dict]:
    out: list[dict] = []
    for o in objs:
        bbox = o.get("bbox")
        if not bbox:
            continue
        b = _clip_bbox(
            [int(bbox[0] + ox), int(bbox[1] + oy), int(bbox[2] + ox), int(bbox[3] + oy)],
            w,
            h,
        )
        mapped = dict(o)
        mapped["bbox"] = b
        out.append(mapped)
    return out


def _is_n_frame(frame_count: int, every_n: int) -> bool:
    return every_n > 0 and frame_count % every_n == 0


def _effective_phone_every_n(base_every_n: int, lock_stable: bool, risk_state: str) -> int:
    if base_every_n <= 0:
        return 0
    if (not lock_stable) or risk_state in ("WARN", "ALERT"):
        return 2
    return base_every_n


def _should_run_roi_phone(
    has_driver_roi: bool,
    run_full_yolo: bool,
    frame_count: int,
    phone_every_n: int,
) -> bool:
    return has_driver_roi and (not run_full_yolo) and _is_n_frame(frame_count, phone_every_n)


def _merge_objects(
    full_objects: list[dict],
    roi_phone_objects: list[dict],
    driver_bbox: Optional[list[int]],
    lock_conf: float,
    track_id: int,
) -> list[dict]:
    merged: list[dict] = []
    for obj in full_objects:
        if obj.get("name") == "person":
            continue
        merged.append(obj)
    if driver_bbox is not None:
        merged.append(
            {
                "cls": 0,
                "name": "person",
                "conf": float(lock_conf),
                "bbox": driver_bbox,
                "track_id": track_id,
            }
        )
    # Prefer ROI phone detections for better relevance.
    merged = [o for o in merged if "phone" not in str(o.get("name", "")).lower()]
    merged.extend(roi_phone_objects)
    merged.sort(key=lambda x: float(x.get("conf", 0.0)), reverse=True)
    return merged


def main() -> None:
    frames_addr = os.environ.get("FRAMES_ADDR", "tcp://127.0.0.1:5555")
    results_addr = os.environ.get("RESULTS_ADDR", "tcp://127.0.0.1:5556")
    alerts_addr = os.environ.get("ALERTS_ADDR", "tcp://127.0.0.1:5557")
    alerts_pub_enabled = _env_bool("ALERTS_PUB", False)

    yolo_every_n = _env_int("YOLO_EVERY_N", 4)
    face_every_n = _env_int("FACE_EVERY_N", 1)
    phone_every_n = max(0, _env_int("PHONE_EVERY_N", 4))
    full_reacquire_every_n = max(0, _env_int("FULL_REACQUIRE_EVERY_N", 0))
    phone_hold_ms = max(0, _env_int("PHONE_HOLD_MS", 250))
    eyes_stale_ms = max(0, _env_int("EYES_STALE_MS", 500))
    log_every = float(os.environ.get("LOG_EVERY_SEC", "1.0"))
    yolo_log = os.environ.get("YOLO_LOG", "1") == "1"
    eye_log = os.environ.get("EYE_LOG", "1") == "1"

    roi_margin = _env_float("DRIVER_ROI_MARGIN", 1.35)
    roi_min_w = _env_int("DRIVER_ROI_MIN_W", 220)
    roi_min_h = _env_int("DRIVER_ROI_MIN_H", 220)

    keep_all_frames = _env_bool("KEEP_ALL_FRAMES", False)
    drop_old_frames = _env_bool("DROP_OLD_FRAMES", True)
    process_all = keep_all_frames or not drop_old_frames

    sub = FrameSubscriber(frames_addr, conflate=False)
    try:
        pub = ResultPublisher(results_addr)
    except Exception as e:
        if "Address already in use" in str(e):
            raise RuntimeError(
                f"Results port already in use ({results_addr}). "
                "Kill the previous infer process or change RESULTS_ADDR."
            ) from e
        raise

    alert_pub = AlertPublisher(alerts_addr) if alerts_pub_enabled else None

    print(f"[infer] waiting for frames on {frames_addr} ...", flush=True)
    header, frame = sub.recv()
    infer_size = (header.w, header.h)
    print(f"[infer] got first frame: {infer_size[0]}x{infer_size[1]}", flush=True)

    face_eye = FaceEyeEstimatorMediaPipe(input_size=infer_size)
    yolo = YoloDetector()
    driver_lock = DriverLock(frame.shape)
    risk_engine = RiskEngine()

    print(
        f"[infer] ready (face_every_n={face_every_n}, yolo_every_n={yolo_every_n}, "
        f"phone_every_n={phone_every_n}, full_reacquire_every_n={full_reacquire_every_n}, "
        f"log_every={log_every})",
        flush=True,
    )

    frame_count = 0
    last_objects: list[dict] = []
    last_full_objects: list[dict] = []
    last_roi_phone_objects: list[dict] = []
    last_face_bbox = None
    last_eyes = None
    last_eyes_ts_ms = 0
    last_eye_states = (None, None)
    next_log_t = time.time() + log_every
    last_risk_state = "NORMAL"
    last_phone_ts_ms = 0

    try:
        while True:
            if process_all:
                header, frame = sub.recv()
            else:
                latest = sub.recv_latest(timeout_ms=50)
                if latest is None:
                    continue
                header, frame = latest

            frame_count += 1
            ts_ms = int(header.ts_ms)
            h, w = frame.shape[:2]

            driver_lock.update_tracking(frame, ts_ms)

            run_face = (face_every_n <= 1 or (frame_count % face_every_n == 0))
            face_seen_this_frame: Optional[list[int]] = None
            if run_face:
                roi_bbox = driver_lock.expanded_roi(roi_margin, roi_min_w, roi_min_h)
                if roi_bbox is not None:
                    x1, y1, x2, y2 = roi_bbox
                    face_bbox_raw, eyes_raw = face_eye.detect(_crop_frame(frame, roi_bbox))
                    face_bbox_mapped = _map_bbox(face_bbox_raw, x1, y1)
                    if face_bbox_mapped is not None:
                        face_bbox_mapped = _clip_bbox(face_bbox_mapped, w, h)
                    last_face_bbox = face_bbox_mapped
                    face_seen_this_frame = face_bbox_mapped
                    if eyes_raw is not None:
                        last_eyes_ts_ms = ts_ms
                        last_eyes = {
                            "left_pct": float(getattr(eyes_raw, "left_pct", 0.0)),
                            "right_pct": float(getattr(eyes_raw, "right_pct", 0.0)),
                            "left_state": str(getattr(eyes_raw, "left_state", "?")),
                            "right_state": str(getattr(eyes_raw, "right_state", "?")),
                            "left_ear": float(getattr(eyes_raw, "left_ear", 0.0)),
                            "right_ear": float(getattr(eyes_raw, "right_ear", 0.0)),
                            "left_pts": _map_points(getattr(eyes_raw, "left_pts", None), x1, y1),
                            "right_pts": _map_points(getattr(eyes_raw, "right_pts", None), x1, y1),
                            "yaw_deg": float(getattr(eyes_raw, "yaw_deg", 0.0)),
                            "pitch_deg": float(getattr(eyes_raw, "pitch_deg", 0.0)),
                            "roll_deg": float(getattr(eyes_raw, "roll_deg", 0.0)),
                        }
                else:
                    face_bbox_raw, eyes_raw = face_eye.detect(frame)
                    last_face_bbox = face_bbox_raw
                    face_seen_this_frame = face_bbox_raw
                    if eyes_raw is not None:
                        last_eyes_ts_ms = ts_ms
                        last_eyes = {
                            "left_pct": float(getattr(eyes_raw, "left_pct", 0.0)),
                            "right_pct": float(getattr(eyes_raw, "right_pct", 0.0)),
                            "left_state": str(getattr(eyes_raw, "left_state", "?")),
                            "right_state": str(getattr(eyes_raw, "right_state", "?")),
                            "left_ear": float(getattr(eyes_raw, "left_ear", 0.0)),
                            "right_ear": float(getattr(eyes_raw, "right_ear", 0.0)),
                            "left_pts": getattr(eyes_raw, "left_pts", None),
                            "right_pts": getattr(eyes_raw, "right_pts", None),
                            "yaw_deg": float(getattr(eyes_raw, "yaw_deg", 0.0)),
                            "pitch_deg": float(getattr(eyes_raw, "pitch_deg", 0.0)),
                            "roll_deg": float(getattr(eyes_raw, "roll_deg", 0.0)),
                        }
            driver_lock.report_face(face_seen_this_frame, ts_ms)
            if face_seen_this_frame is None and last_eyes is not None:
                if ts_ms - last_eyes_ts_ms > eyes_stale_ms:
                    last_eyes = None
            health = driver_lock.lock_health(ts_ms)
            state = driver_lock.state()

            run_full_yolo = not health["stable"]
            if health["stable"] and _is_n_frame(frame_count, full_reacquire_every_n):
                run_full_yolo = True
            if (not state.locked) and yolo_every_n > 1 and not _is_n_frame(frame_count, yolo_every_n):
                run_full_yolo = False
            if run_full_yolo:
                last_full_objects = yolo.detect(frame)
                person_boxes = [o["bbox"] for o in last_full_objects if o.get("name") == "person"]
                driver_lock.update_from_detections(frame, ts_ms, person_boxes, last_face_bbox)

            state = driver_lock.state()
            health = driver_lock.lock_health(ts_ms)
            effective_phone_every_n = _effective_phone_every_n(
                phone_every_n,
                bool(health["stable"]),
                last_risk_state,
            )
            driver_roi = driver_lock.expanded_roi(roi_margin, roi_min_w, roi_min_h)
            run_roi_phone = _should_run_roi_phone(
                has_driver_roi=driver_roi is not None,
                run_full_yolo=run_full_yolo,
                frame_count=frame_count,
                phone_every_n=effective_phone_every_n,
            )
            if run_roi_phone:
                x1, y1, x2, y2 = driver_roi
                roi_objects = yolo.detect(_crop_frame(frame, driver_roi))
                roi_objects = [o for o in roi_objects if "phone" in str(o.get("name", "")).lower()]
                last_roi_phone_objects = _map_objects(roi_objects, x1, y1, w, h)
                last_phone_ts_ms = ts_ms
            elif ts_ms - last_phone_ts_ms > phone_hold_ms:
                last_roi_phone_objects = []

            driver_bbox = state.bbox
            last_objects = _merge_objects(
                last_full_objects,
                last_roi_phone_objects,
                driver_bbox,
                state.lock_conf,
                state.track_id,
            )

            risk_out = risk_engine.update(
                ts_ms=ts_ms,
                driver_locked=state.locked,
                face_bbox=last_face_bbox,
                eyes=last_eyes,
                objects=last_objects,
            )

            res = InferResult(
                frame_id=int(header.frame_id),
                ts_ms=ts_ms,
                objects=last_objects,
                face_bbox=last_face_bbox,
                eyes=last_eyes,
                eyes_points=(
                    {"left": last_eyes.get("left_pts"), "right": last_eyes.get("right_pts")}
                    if last_eyes
                    else None
                ),
                driver={
                    "track_id": state.track_id,
                    "locked": state.locked,
                    "lock_conf": state.lock_conf,
                    "bbox": state.bbox,
                    "lost_frames": state.lost_frames,
                },
                attention=risk_out.attention,
                risk=risk_out.risk,
                alerts=risk_out.alerts,
            )
            pub.send(res)
            last_risk_state = str(risk_out.risk.get("state", "NORMAL"))

            if alert_pub is not None and risk_out.alerts.get("warn"):
                alert_pub.send(
                    {
                        "frame_id": int(header.frame_id),
                        "ts_ms": ts_ms,
                        "driver": res.driver,
                        "risk": res.risk,
                        "alerts": res.alerts,
                    }
                )

            now = time.time()
            if eye_log and last_eyes:
                states = (last_eyes.get("left_state"), last_eyes.get("right_state"))
                if states != last_eye_states:
                    last_eye_states = states
                    print(
                        "[infer] eye_state_change  "
                        f"L {last_eyes.get('left_state')} {last_eyes.get('left_pct', 0):.0f}%  "
                        f"R {last_eyes.get('right_state')} {last_eyes.get('right_pct', 0):.0f}%",
                        flush=True,
                    )

            if now >= next_log_t:
                next_log_t = now + log_every
                parts: list[str] = []
                parts.append(
                    f"driver={'locked' if state.locked else 'unlock'} "
                    f"id={state.track_id} conf={state.lock_conf:.2f}"
                )
                if yolo_log:
                    objs = ", ".join([f"{o['name']}:{o['conf']:.2f}" for o in last_objects[:4]])
                    parts.append(f"objects=[{objs}]" if objs else "objects=[]")
                if eye_log and last_eyes:
                    parts.append(
                        f"eyes=L {last_eyes.get('left_state')} {last_eyes.get('left_pct', 0):.0f}% | "
                        f"R {last_eyes.get('right_state')} {last_eyes.get('right_pct', 0):.0f}%"
                    )
                parts.append(
                    f"risk={risk_out.risk.get('state')} score={risk_out.risk.get('score'):.1f}"
                )
                print("[infer] " + "  ".join(parts), flush=True)
    finally:
        sub.close()
        pub.close()
        if alert_pub is not None:
            alert_pub.close()
        try:
            face_eye.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
