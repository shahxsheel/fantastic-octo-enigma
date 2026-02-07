import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class EyeInfo:
    left_pct: float
    right_pct: float
    left_state: str
    right_state: str
    left_ear: float   # raw blink score (0.0 = open, 1.0 = closed)
    right_ear: float   # raw blink score (0.0 = open, 1.0 = closed)


class FaceEyeEstimatorMediaPipe:
    """
    Eye openness via MediaPipe FaceLandmarker blendshapes.

    Uses the Tasks-API FaceLandmarker with ``output_face_blendshapes=True``.
    The model returns ML-predicted ``eyeBlinkLeft`` / ``eyeBlinkRight`` scores
    (0 = fully open, 1 = fully closed), which are far more accurate than
    hand-crafted EAR geometry and require no calibration phase.
    """

    def __init__(self, input_size: Tuple[int, int]):
        # Explicit submodule imports -- matches the official Google Pi example.
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._input_size = input_size

        # --- Model path ---
        model_path = os.environ.get(
            "FACE_LANDMARKER_MODEL", "face_landmarker.task"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {model_path}. "
                "Download it from https://storage.googleapis.com/mediapipe-models"
                "/face_landmarker/face_landmarker/float16/latest/face_landmarker.task "
                "or set FACE_LANDMARKER_MODEL."
            )

        # --- Configuration ---
        self.max_faces = int(os.environ.get("MP_MAX_FACES", "1"))
        self.min_det_conf = float(os.environ.get("MP_MIN_DET_CONF", "0.5"))
        self.min_presence_conf = float(os.environ.get("MP_MIN_PRESENCE_CONF", "0.5"))
        self.min_track_conf = float(os.environ.get("MP_MIN_TRACK_CONF", "0.5"))
        self.closed_threshold = float(os.environ.get("EYE_CLOSED_THRESHOLD", "0.5"))
        self.swap_eyes = os.environ.get("SWAP_EYES", "0") == "1"

        # --- Create FaceLandmarker (Tasks API, VIDEO mode) ---
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=self.max_faces,
            min_face_detection_confidence=self.min_det_conf,
            min_face_presence_confidence=self.min_presence_conf,
            min_tracking_confidence=self.min_track_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(
            options
        )

        # Monotonic timestamp tracking (VIDEO mode requires increasing ts)
        self._last_ts_ms: int = 0

        # --- Optional per-frame timing (EYE_TIMING=1) ---
        self._timing_enabled = os.environ.get("EYE_TIMING", "0") == "1"
        self._timing_count: int = 0
        self._timing_total: float = 0.0
        self._timing_interval: int = 100

        print(
            f"[infer] FaceLandmarker ready  model={model_path}  "
            f"closed_threshold={self.closed_threshold}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def detect(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[list], Optional[EyeInfo]]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame_rgb
        )

        # Ensure monotonically increasing timestamps for VIDEO mode
        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        # --- Run FaceLandmarker ---
        if self._timing_enabled:
            t0 = time.time()

        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if self._timing_enabled:
            elapsed_ms = (time.time() - t0) * 1000.0
            self._timing_total += elapsed_ms
            self._timing_count += 1
            if self._timing_count % self._timing_interval == 0:
                avg_ms = self._timing_total / self._timing_count
                print(
                    f"[infer] EYE_TIMING: FaceLandmarker avg={avg_ms:.1f}ms "
                    f"over {self._timing_count} frames",
                    flush=True,
                )

        if not result.face_landmarks:
            return None, None

        # --- Face bounding box from landmarks ---
        landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1 = int(max(0, min(xs) * w))
        y1 = int(max(0, min(ys) * h))
        x2 = int(min(w - 1, max(xs) * w))
        y2 = int(min(h - 1, max(ys) * h))
        face_bbox = [x1, y1, x2, y2]

        # --- Extract eyeBlink blendshape scores ---
        left_blink = 0.0
        right_blink = 0.0

        if result.face_blendshapes:
            for bs in result.face_blendshapes[0]:
                if bs.category_name == "eyeBlinkLeft":
                    left_blink = float(bs.score)
                elif bs.category_name == "eyeBlinkRight":
                    right_blink = float(bs.score)

        if self.swap_eyes:
            left_blink, right_blink = right_blink, left_blink

        # Blink score: 0.0 = open, 1.0 = closed
        # Convert to openness percentage (0–100)
        left_pct = (1.0 - left_blink) * 100.0
        right_pct = (1.0 - right_blink) * 100.0

        left_state = "CLOSED" if left_blink > self.closed_threshold else "OPEN"
        right_state = "CLOSED" if right_blink > self.closed_threshold else "OPEN"

        eyes = EyeInfo(
            left_pct=left_pct,
            right_pct=right_pct,
            left_state=left_state,
            right_state=right_state,
            left_ear=left_blink,
            right_ear=right_blink,
        )
        return face_bbox, eyes
