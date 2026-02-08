import os
import time

import zmq

from src.infer.face_eye_mediapipe import FaceEyeEstimatorMediaPipe
from src.infer.yolo_detector import YoloDetector
from src.ipc.zmq_frames import FrameSubscriber
from src.ipc.zmq_results import InferResult, ResultPublisher


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def main() -> None:
    frames_addr = os.environ.get("FRAMES_ADDR", "tcp://127.0.0.1:5555")
    results_addr = os.environ.get("RESULTS_ADDR", "tcp://127.0.0.1:5556")

    yolo_every_n = _env_int("YOLO_EVERY_N", 4)
    face_every_n = _env_int("FACE_EVERY_N", 1)
    log_every = float(os.environ.get("LOG_EVERY_SEC", "1.0"))
    yolo_log = os.environ.get("YOLO_LOG", "1") == "1"
    eye_log = os.environ.get("EYE_LOG", "1") == "1"

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

    # Prime one frame to know input size
    print(f"[infer] waiting for frames on {frames_addr} ...", flush=True)
    header, frame = sub.recv()
    infer_size = (header.w, header.h)
    print(f"[infer] got first frame: {infer_size[0]}x{infer_size[1]}", flush=True)

    face_eye = FaceEyeEstimatorMediaPipe(input_size=infer_size)
    print("[infer] eye backend: mediapipe (FaceLandmarker blendshapes)", flush=True)

    yolo = YoloDetector()
    print(
        f"[infer] ready (face_every_n={face_every_n}, yolo_every_n={yolo_every_n}, log_every={log_every})",
        flush=True,
    )

    frame_count = 0
    last_objects = []
    last_face_bbox = None
    last_eyes = None
    last_eye_states = (None, None)

    next_log_t = time.time() + log_every

    try:
        while True:
            latest = sub.recv_latest(timeout_ms=50)
            if latest is None:
                continue
            header, frame = latest

            frame_count += 1
            ts_ms = header.ts_ms
            if frame_count <= 3:
                print(f"[infer] loop frame_count={frame_count} frame_id={header.frame_id}", flush=True)

            run_yolo = (frame_count % yolo_every_n == 0)
            run_face = (face_every_n <= 1 or (frame_count % face_every_n == 0))

            # Face + eye on cadence
            if run_face:
                face_bbox, eyes = face_eye.detect(frame)
                if face_bbox is not None:
                    last_face_bbox = face_bbox
                if eyes is not None:
                    last_eyes = {
                        "left_pct": float(getattr(eyes, "left_pct", 0.0)),
                        "right_pct": float(getattr(eyes, "right_pct", 0.0)),
                        "left_state": str(getattr(eyes, "left_state", "?")),
                        "right_state": str(getattr(eyes, "right_state", "?")),
                        "left_ear": float(getattr(eyes, "left_ear", 0.0)),
                        "right_ear": float(getattr(eyes, "right_ear", 0.0)),
                    }

            # YOLO on cadence
            if run_yolo:
                last_objects = yolo.detect(frame)

            res = InferResult(
                frame_id=header.frame_id,
                ts_ms=ts_ms,
                objects=last_objects,
                face_bbox=last_face_bbox,
                eyes=last_eyes,
            )
            pub.send(res)

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
                parts = []
                if yolo_log:
                    objs = ", ".join(
                        [f"{o['name']}:{o['conf']:.2f}" for o in last_objects[:5]]
                    )
                    parts.append(f"objects=[{objs}]" if objs else "objects=[]")
                if eye_log:
                    if last_eyes:
                        parts.append(
                            f"eyes=L {last_eyes.get('left_state')} {last_eyes.get('left_pct', 0):.0f}% | "
                            f"R {last_eyes.get('right_state')} {last_eyes.get('right_pct', 0):.0f}%"
                        )
                    else:
                        parts.append("eyes=none")
                print("[infer] " + "  ".join(parts), flush=True)
    finally:
        sub.close()
        pub.close()
        try:
            face_eye.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
