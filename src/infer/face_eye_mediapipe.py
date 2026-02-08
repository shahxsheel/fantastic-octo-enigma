import os
import time
from collections import deque
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
    left_pts: Optional[list[tuple[int, int]]] = None
    right_pts: Optional[list[tuple[int, int]]] = None
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


class FaceEyeEstimatorMediaPipe:
    """
    Eye openness via MediaPipe FaceLandmarker blendshapes.

    LIVE_STREAM mode drops frames while busy but keeps latency low.
    ROI reuse avoids re-processing the full frame while keeping accuracy.
    """

    def __init__(self, input_size: Tuple[int, int]):
        # Explicit submodule imports -- matches the official Google Pi example.
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._input_size = input_size

        # --- Model path ---
        model_path = os.environ.get("FACE_LANDMARKER_MODEL", "face_landmarker.task")
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

        # ROI settings: reuse last face region to cut work without changing accuracy.
        self._roi_enabled = os.environ.get("FACE_ROI_REUSE", "1") == "1"
        self._roi_margin = float(os.environ.get("FACE_ROI_MARGIN", "1.4"))
        self._roi_downscale = float(os.environ.get("FACE_ROI_DOWNSCALE", "0.6"))
        self._roi_min = int(os.environ.get("FACE_ROI_MIN", "160"))

        # --- Create FaceLandmarker (Tasks API, LIVE_STREAM mode) ---
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_faces=self.max_faces,
            min_face_detection_confidence=self.min_det_conf,
            min_face_presence_confidence=self.min_presence_conf,
            min_tracking_confidence=self.min_track_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=self._on_result,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Monotonic timestamp tracking (LIVE_STREAM requires increasing ts)
        self._last_ts_ms: int = 0
        self._last_face_bbox: Optional[list] = None
        self._last_eyes: Optional[EyeInfo] = None
        self._last_roi_info: Optional[dict] = None
        self._roi_by_ts: dict[int, dict] = {}
        self._roi_ts_queue: deque[int] = deque()
        try:
            self._pose_every_n = max(1, int(os.environ.get("POSE_EVERY_N", "2")))
        except Exception:
            self._pose_every_n = 2
        self._pose_counter = 0
        self._last_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)

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
        """Submit a frame; return the latest available result (may be from previous frame)."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Optional ROI reuse to shrink work
        mp_image = None
        roi_info: Optional[dict] = None
        if self._roi_enabled and self._last_face_bbox:
            lx1, ly1, lx2, ly2 = self._last_face_bbox
            bw = max(1, lx2 - lx1)
            bh = max(1, ly2 - ly1)
            cx = lx1 + bw / 2.0
            cy = ly1 + bh / 2.0
            roi_w = min(w, int(bw * self._roi_margin))
            roi_h = min(h, int(bh * self._roi_margin))
            rx1 = max(0, int(cx - roi_w / 2))
            ry1 = max(0, int(cy - roi_h / 2))
            rx2 = min(w, rx1 + roi_w)
            ry2 = min(h, ry1 + roi_h)
            roi_w = max(1, rx2 - rx1)
            roi_h = max(1, ry2 - ry1)
            if roi_w >= self._roi_min and roi_h >= self._roi_min:
                crop = frame_rgb[ry1:ry2, rx1:rx2]
                if self._roi_downscale < 0.999:
                    new_w = max(32, int(roi_w * self._roi_downscale))
                    new_h = max(32, int(roi_h * self._roi_downscale))
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mp_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB, data=crop
                )
                roi_info = {
                    "offset_x": rx1,
                    "offset_y": ry1,
                    "crop_w": roi_w,
                    "crop_h": roi_h,
                    "frame_w": w,
                    "frame_h": h,
                }

        if mp_image is None:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=frame_rgb
            )
            roi_info = {
                "offset_x": 0,
                "offset_y": 0,
                "crop_w": w,
                "crop_h": h,
                "frame_w": w,
                "frame_h": h,
            }

        # Ensure monotonically increasing timestamps for LIVE_STREAM
        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        self._remember_roi(ts_ms, roi_info)

        if self._timing_enabled:
            t0 = time.time()

        self._landmarker.detect_async(mp_image, ts_ms)

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

        # Return latest available results (may be from a previous frame).
        return self._last_face_bbox, self._last_eyes

    # ------------------------------------------------------------------
    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        if not result.face_landmarks:
            self._last_face_bbox = None
            self._last_eyes = None
            return

        roi = self._roi_by_ts.pop(timestamp_ms, None) or self._last_roi_info or {
            "offset_x": 0,
            "offset_y": 0,
            "crop_w": output_image.width,
            "crop_h": output_image.height,
            "frame_w": output_image.width,
            "frame_h": output_image.height,
        }

        cw = max(1, int(roi.get("crop_w", output_image.width)))
        ch = max(1, int(roi.get("crop_h", output_image.height)))
        ox = int(roi.get("offset_x", 0))
        oy = int(roi.get("offset_y", 0))
        frame_w = int(roi.get("frame_w", cw))
        frame_h = int(roi.get("frame_h", ch))

        landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1 = int(max(0, min(xs) * cw)) + ox
        y1 = int(max(0, min(ys) * ch)) + oy
        x2 = int(min(frame_w - 1, ox + max(xs) * cw))
        y2 = int(min(frame_h - 1, oy + max(ys) * ch))
        self._last_face_bbox = [x1, y1, x2, y2]

        # Tight eyelid/corner subsets to keep points on the eyes (not brows).
        left_idx = [33, 133, 159, 145, 158, 153]
        right_idx = [362, 263, 386, 374, 385, 380]
        left_pts: list[tuple[int, int]] = []
        right_pts: list[tuple[int, int]] = []
        for i in left_idx:
            lm = landmarks[i]
            left_pts.append((int(lm.x * cw) + ox, int(lm.y * ch) + oy))
        for i in right_idx:
            lm = landmarks[i]
            right_pts.append((int(lm.x * cw) + ox, int(lm.y * ch) + oy))

        # Extract eyeBlink blendshape scores
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

        left_pct = (1.0 - left_blink) * 100.0
        right_pct = (1.0 - right_blink) * 100.0
        left_state = "CLOSED" if left_blink > self.closed_threshold else "OPEN"
        right_state = "CLOSED" if right_blink > self.closed_threshold else "OPEN"
        self._pose_counter += 1
        if self._pose_counter == 1 or self._pose_counter % self._pose_every_n == 0:
            self._last_pose = self._extract_head_pose(result)
        yaw_deg, pitch_deg, roll_deg = self._last_pose

        self._last_eyes = EyeInfo(
            left_pct=left_pct,
            right_pct=right_pct,
            left_state=left_state,
            right_state=right_state,
            left_ear=left_blink,
            right_ear=right_blink,
            left_pts=left_pts,
            right_pts=right_pts,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )

    @staticmethod
    def _extract_head_pose(result) -> tuple[float, float, float]:
        mats = getattr(result, "facial_transformation_matrixes", None)
        if not mats:
            return 0.0, 0.0, 0.0
        raw = mats[0]
        arr: Optional[np.ndarray] = None
        try:
            if hasattr(raw, "data"):
                arr = np.array(raw.data, dtype=np.float32)
            else:
                arr = np.array(raw, dtype=np.float32)
            if arr.size >= 16:
                arr = arr.reshape(4, 4)
            else:
                return 0.0, 0.0, 0.0
            r = arr[:3, :3]
            yaw = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
            pitch = float(
                np.degrees(
                    np.arctan2(
                        -r[2, 0],
                        np.sqrt(r[2, 1] * r[2, 1] + r[2, 2] * r[2, 2]),
                    )
                )
            )
            roll = float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))
            return yaw, pitch, roll
        except Exception:
            return 0.0, 0.0, 0.0

    def _remember_roi(self, ts_ms: int, roi_info: dict) -> None:
        self._last_roi_info = roi_info
        self._roi_by_ts[ts_ms] = roi_info
        self._roi_ts_queue.append(ts_ms)
        while len(self._roi_ts_queue) > 64:
            old_ts = self._roi_ts_queue.popleft()
            self._roi_by_ts.pop(old_ts, None)
