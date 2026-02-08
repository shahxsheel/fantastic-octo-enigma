import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def _clip_bbox(b: list[int], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


@dataclass
class DriverState:
    track_id: int
    locked: bool
    lock_conf: float
    bbox: Optional[list[int]]
    lost_frames: int


class DriverLock:
    """Lightweight driver lock-on manager with tracker + reacquire scoring."""

    def __init__(self, frame_shape: tuple[int, int, int]):
        self.h, self.w = frame_shape[:2]
        self.driver_side = os.environ.get("DRIVER_SIDE", "LHD").upper()
        self.tracker_type = os.environ.get("TRACKER_TYPE", "MOSSE").upper()
        self.lock_min_frames = _env_int("LOCK_MIN_FRAMES", 6)
        self.lock_lost_frames = _env_int("LOCK_LOST_FRAMES", 8)
        self.lock_hist_min = _env_float("LOCK_HIST_MIN", 0.45)
        self.face_missing_ms = _env_int("FACE_MISSING_MS", 1200)
        self.zone_weight = _env_float("LOCK_ZONE_WEIGHT", 0.35)
        self.face_weight = _env_float("LOCK_FACE_WEIGHT", 0.35)
        self.inertia_weight = _env_float("LOCK_INERTIA_WEIGHT", 0.20)
        self.hist_weight = _env_float("LOCK_HIST_WEIGHT", 0.10)

        self.track_id = 0
        self.locked = False
        self.lock_conf = 0.0
        self.locked_bbox: Optional[list[int]] = None
        self.pending_bbox: Optional[list[int]] = None
        self.pending_hits = 0
        self.lost_frames = 0
        self.last_face_ts = 0
        self.last_seen_ts = 0
        self.hist_ref: Optional[np.ndarray] = None
        self._tracker = None

    def state(self) -> DriverState:
        return DriverState(
            track_id=self.track_id,
            locked=self.locked,
            lock_conf=float(self.lock_conf),
            bbox=list(self.locked_bbox) if self.locked_bbox else None,
            lost_frames=int(self.lost_frames),
        )

    def lock_health(self, ts_ms: int) -> dict:
        face_age_ms = -1 if self.last_face_ts <= 0 else max(0, int(ts_ms) - self.last_face_ts)
        stable = (
            self.locked
            and self.locked_bbox is not None
            and self.lost_frames == 0
            and (face_age_ms < 0 or face_age_ms <= self.face_missing_ms)
        )
        return {
            "stable": bool(stable),
            "face_age_ms": int(face_age_ms),
            "lost_frames": int(self.lost_frames),
        }

    def in_seat_zone(self, bbox: list[int]) -> float:
        x1, _, x2, _ = bbox
        bxw = max(1, x2 - x1)
        if self.driver_side == "RHD":
            zx1, zx2 = int(0.35 * self.w), self.w
        else:
            # LHD + AUTO default
            zx1, zx2 = 0, int(0.65 * self.w)
        ix = max(0, min(x2, zx2) - max(x1, zx1))
        return float(ix) / float(bxw)

    def _create_tracker(self):
        # OpenCV tracker factories are split across cv2 and cv2.legacy depending on build.
        creators = []
        if self.tracker_type == "KCF":
            creators = ["TrackerKCF_create"]
        else:
            creators = ["TrackerMOSSE_create", "TrackerKCF_create"]

        for name in creators:
            fn = getattr(cv2, name, None)
            if callable(fn):
                return fn()
            legacy = getattr(cv2, "legacy", None)
            if legacy is not None:
                fn_legacy = getattr(legacy, name, None)
                if callable(fn_legacy):
                    return fn_legacy()
        return None

    def _bbox_hist(self, frame_bgr: np.ndarray, bbox: list[int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape[1], frame_bgr.shape[0])
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _hist_similarity(self, hist_cur: Optional[np.ndarray]) -> float:
        if self.hist_ref is None or hist_cur is None:
            return 1.0
        sim = cv2.compareHist(self.hist_ref, hist_cur, cv2.HISTCMP_CORREL)
        return float(max(0.0, min(1.0, sim)))

    def _candidate_score(
        self,
        frame_bgr: np.ndarray,
        bbox: list[int],
        face_bbox: Optional[list[int]],
    ) -> float:
        zone = self.in_seat_zone(bbox)
        face = 0.0
        if face_bbox is not None:
            face = 1.0 if _bbox_iou(bbox, face_bbox) > 0.05 else 0.0
        inertia = 0.0
        if self.locked_bbox is not None:
            inertia = _bbox_iou(bbox, self.locked_bbox)
        hist = self._hist_similarity(self._bbox_hist(frame_bgr, bbox))
        return (
            self.zone_weight * zone
            + self.face_weight * face
            + self.inertia_weight * inertia
            + self.hist_weight * hist
        )

    def _init_tracker(self, frame_bgr: np.ndarray, bbox: list[int]) -> None:
        tracker = self._create_tracker()
        if tracker is None:
            self._tracker = None
            return
        x1, y1, x2, y2 = bbox
        ok = tracker.init(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
        self._tracker = tracker if ok else None

    def _promote_lock(self, frame_bgr: np.ndarray, bbox: list[int], conf: float, ts_ms: int) -> None:
        self.track_id += 1
        self.locked = True
        self.locked_bbox = _clip_bbox(bbox, self.w, self.h)
        self.lock_conf = float(conf)
        self.pending_hits = 0
        self.pending_bbox = None
        self.lost_frames = 0
        self.last_seen_ts = int(ts_ms)
        self.last_face_ts = int(ts_ms)
        self.hist_ref = self._bbox_hist(frame_bgr, self.locked_bbox)
        self._init_tracker(frame_bgr, self.locked_bbox)

    def unlock(self) -> None:
        self.locked = False
        self.lock_conf = 0.0
        self.locked_bbox = None
        self.pending_bbox = None
        self.pending_hits = 0
        self.lost_frames = 0
        self._tracker = None

    def update_tracking(self, frame_bgr: np.ndarray, ts_ms: int) -> None:
        if not self.locked or self.locked_bbox is None:
            return
        if self._tracker is None:
            self.lost_frames += 1
            if self.lost_frames >= self.lock_lost_frames:
                self.unlock()
            return
        ok, box = self._tracker.update(frame_bgr)
        if not ok:
            self.lost_frames += 1
            if self.lost_frames >= self.lock_lost_frames:
                self.unlock()
            return
        x, y, w, h = box
        bbox = _clip_bbox([int(x), int(y), int(x + w), int(y + h)], self.w, self.h)
        self.locked_bbox = bbox
        self.lock_conf = max(0.4, self.lock_conf * 0.98 + 0.02)
        self.last_seen_ts = int(ts_ms)
        self.lost_frames = 0

        # Keep lock anchored to seat prior.
        if self.in_seat_zone(bbox) < 0.05:
            self.lost_frames += 1
            if self.lost_frames >= self.lock_lost_frames:
                self.unlock()

    def report_face(self, face_bbox: Optional[list[int]], ts_ms: int) -> None:
        if not self.locked:
            return
        if face_bbox is not None and self.locked_bbox is not None:
            if _bbox_iou(face_bbox, self.locked_bbox) > 0.05:
                self.last_face_ts = int(ts_ms)
                self.lost_frames = max(0, self.lost_frames - 1)
                return
        if ts_ms - self.last_face_ts > self.face_missing_ms:
            self.lost_frames += 1
            if self.lost_frames >= self.lock_lost_frames:
                self.unlock()

    def should_reacquire(self, frame_idx: int, full_reacquire_every_n: int) -> bool:
        if not self.locked:
            return True
        if self.lost_frames > 0:
            return True
        if full_reacquire_every_n > 0 and frame_idx % full_reacquire_every_n == 0:
            return True
        return False

    def update_from_detections(
        self,
        frame_bgr: np.ndarray,
        ts_ms: int,
        person_boxes: list[list[int]],
        face_bbox: Optional[list[int]],
    ) -> None:
        if not person_boxes:
            if self.locked:
                self.lost_frames += 1
                if self.lost_frames >= self.lock_lost_frames:
                    self.unlock()
            return

        best_bbox: Optional[list[int]] = None
        best_score = -1.0
        for bbox in person_boxes:
            b = _clip_bbox([int(v) for v in bbox], self.w, self.h)
            score = self._candidate_score(frame_bgr, b, face_bbox)
            if score > best_score:
                best_score = score
                best_bbox = b
        if best_bbox is None:
            return

        hist_sim = self._hist_similarity(self._bbox_hist(frame_bgr, best_bbox))
        if hist_sim < self.lock_hist_min and self.locked:
            # Keep current lock instead of jumping identities.
            return

        if self.locked:
            self.locked_bbox = best_bbox
            self.lock_conf = float(max(0.0, min(1.0, best_score)))
            self.hist_ref = self._bbox_hist(frame_bgr, best_bbox)
            self._init_tracker(frame_bgr, best_bbox)
            self.lost_frames = 0
            self.last_seen_ts = int(ts_ms)
            return

        # Pre-lock staging to avoid locking to transient false positives.
        if self.pending_bbox is not None and _bbox_iou(best_bbox, self.pending_bbox) > 0.2:
            self.pending_hits += 1
        else:
            self.pending_bbox = best_bbox
            self.pending_hits = 1

        if self.pending_hits >= self.lock_min_frames:
            self._promote_lock(frame_bgr, best_bbox, best_score, ts_ms)

    def expanded_roi(
        self, margin: float, min_w: int, min_h: int
    ) -> Optional[list[int]]:
        if not self.locked_bbox:
            return None
        x1, y1, x2, y2 = self.locked_bbox
        bw = max(min_w, int((x2 - x1) * margin))
        bh = max(min_h, int((y2 - y1) * margin))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        nx1 = cx - bw // 2
        ny1 = cy - bh // 2
        nx2 = nx1 + bw
        ny2 = ny1 + bh
        return _clip_bbox([nx1, ny1, nx2, ny2], self.w, self.h)
