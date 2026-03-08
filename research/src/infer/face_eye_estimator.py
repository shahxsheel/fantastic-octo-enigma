"""
MediaPipe FaceLandmarker-based estimator.

Provides:
  - Eye Aspect Ratio (EAR) drowsiness detection
  - Head direction (LEFT/RIGHT/CENTER/UP/DOWN) via facial transformation matrix

Graceful degradation: if mediapipe is not installed or the model file is
missing, self.enabled = False and estimate() returns ("UNKNOWN", 1.0, 1.0, False)
without raising.
"""

import math
import os
from typing import Optional

import numpy as np

_mp_available = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    _mp_available = True
except ImportError:
    pass


# MediaPipe canonical landmark indices for EAR computation.
# (P1, P2, P3, P4, P5, P6) matching the standard 6-point eye model.
_EYE_RIGHT = (33, 160, 158, 133, 153, 144)
_EYE_LEFT  = (362, 385, 387, 263, 373, 380)


def _ear(landmarks, eye_indices: tuple) -> float:
    """Eye Aspect Ratio from 6 landmark indices."""
    p = [landmarks[i] for i in eye_indices]
    # Vertical distances
    v1 = math.dist((p[1].x, p[1].y), (p[5].x, p[5].y))
    v2 = math.dist((p[2].x, p[2].y), (p[4].x, p[4].y))
    # Horizontal distance
    h = math.dist((p[0].x, p[0].y), (p[3].x, p[3].y))
    if h < 1e-6:
        return 1.0
    return (v1 + v2) / (2.0 * h)


class FaceEyeEstimator:
    """
    MediaPipe FaceLandmarker wrapper for head direction + eye drowsiness.

    Environment variables (all optional):
      HEAD_DIRECTION_EVERY_N  int   default 4   — run every N inference frames
      EAR_THRESHOLD           float default 0.22 — EAR below = eye closed
      DROWSY_FRAMES           int   default 15  — consecutive closed frames → drowsy
      HEAD_YAW_THRESHOLD      float default 20.0 — degrees yaw for LEFT/RIGHT
      HEAD_PITCH_THRESHOLD    float default 20.0 — degrees pitch for UP/DOWN
    """

    def __init__(self, model_path: str = "face_landmarker.task") -> None:
        self.enabled = False
        self.reason = "not initialised"

        self.every_n = max(1, int(os.environ.get("HEAD_DIRECTION_EVERY_N", "4")))
        self._ear_threshold = float(os.environ.get("EAR_THRESHOLD", "0.22"))
        self._drowsy_frames = max(1, int(os.environ.get("DROWSY_FRAMES", "15")))
        self._yaw_threshold = float(os.environ.get("HEAD_YAW_THRESHOLD", "20.0"))
        self._pitch_threshold = float(os.environ.get("HEAD_PITCH_THRESHOLD", "20.0"))

        # Per-frame state
        self._closed_frames: int = 0
        self._last_direction: str = "UNKNOWN"
        self._last_left_ear: float = 1.0
        self._last_right_ear: float = 1.0
        self._last_is_drowsy: bool = False

        if not _mp_available:
            self.reason = "mediapipe not installed"
            return

        model_path = os.path.abspath(model_path)
        if not os.path.isfile(model_path):
            self.reason = f"model file not found: {model_path}"
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
            self._detector = mp_vision.FaceLandmarker.create_from_options(options)
            self.enabled = True
            self.reason = "ok"
        except Exception as exc:
            self.reason = str(exc)
            self._detector = None

    # ------------------------------------------------------------------
    def estimate(
        self, frame_bgr: np.ndarray, frame_idx: int
    ) -> tuple[str, float, float, bool]:
        """
        Returns (head_direction, left_ear, right_ear, is_drowsy).

        When disabled or frame_idx % every_n != 0, returns cached values.
        """
        if not self.enabled:
            return "UNKNOWN", 1.0, 1.0, False

        if frame_idx % self.every_n != 0:
            return self._last_direction, self._last_left_ear, self._last_right_ear, self._last_is_drowsy

        # BGR → RGB for MediaPipe
        frame_rgb = frame_bgr[:, :, ::-1].astype(np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            result = self._detector.detect(mp_image)
        except Exception:
            return self._last_direction, self._last_left_ear, self._last_right_ear, self._last_is_drowsy

        if not result.face_landmarks:
            # No face — decay closed_frames slowly
            self._closed_frames = max(0, self._closed_frames - 1)
            self._last_direction = "UNKNOWN"
            return "UNKNOWN", self._last_left_ear, self._last_right_ear, self._last_is_drowsy

        landmarks = result.face_landmarks[0]

        # ── EAR ──────────────────────────────────────────────────────
        left_ear = _ear(landmarks, _EYE_LEFT)
        right_ear = _ear(landmarks, _EYE_RIGHT)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < self._ear_threshold:
            self._closed_frames += 1
        else:
            self._closed_frames = max(0, self._closed_frames - 1)

        is_drowsy = self._closed_frames >= self._drowsy_frames

        # ── Head direction ────────────────────────────────────────────
        direction = self._compute_head_direction(result, landmarks)

        self._last_direction = direction
        self._last_left_ear = left_ear
        self._last_right_ear = right_ear
        self._last_is_drowsy = is_drowsy

        return direction, left_ear, right_ear, is_drowsy

    def _compute_head_direction(self, result, landmarks) -> str:
        # Prefer transformation matrix (more accurate).
        if (
            result.facial_transformation_matrixes
            and len(result.facial_transformation_matrixes) > 0
        ):
            m = result.facial_transformation_matrixes[0]
            # m is a 4×4 numpy-like matrix
            try:
                sy    = math.sqrt(float(m[0][0]) ** 2 + float(m[1][0]) ** 2)
                yaw   = math.degrees(math.atan2(-float(m[2][0]), sy))
                pitch = math.degrees(math.atan2(float(m[2][1]), float(m[2][2])))
                if abs(yaw) > self._yaw_threshold:
                    return "LEFT" if yaw < 0 else "RIGHT"
                if pitch > self._pitch_threshold:
                    return "DOWN"
                if pitch < -self._pitch_threshold:
                    return "UP"
                return "CENTER"
            except Exception:
                pass  # fall through to landmark fallback

        # Fallback: nose tip vs cheek midpoint (rough left/centre/right only).
        try:
            nose_x = landmarks[1].x
            left_cheek_x = landmarks[234].x
            right_cheek_x = landmarks[454].x
            mid_x = (left_cheek_x + right_cheek_x) / 2.0
            diff = nose_x - mid_x
            if diff < -0.05:
                return "LEFT"
            if diff > 0.05:
                return "RIGHT"
            return "CENTER"
        except Exception:
            return "UNKNOWN"

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.enabled and self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
