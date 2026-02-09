import os
from collections import deque
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class RiskOutput:
    attention: dict
    risk: dict
    alerts: dict


class RiskEngine:
    """Temporal, hysteresis-based driver risk scoring."""

    def __init__(self):
        self.policy = os.environ.get("RISK_POLICY", "AGGRESSIVE").upper()
        self.perclos_window_ms = int(_env_float("PERCLOS_WINDOW_SEC", 20.0) * 1000.0)
        self.perclos_warn = _env_float("PERCLOS_WARN", 0.22)
        self.perclos_alert = _env_float("PERCLOS_ALERT", 0.30)
        self.offroad_warn_ms = _env_int("OFFROAD_WARN_MS", 900)
        self.offroad_alert_ms = _env_int("OFFROAD_ALERT_MS", 1600)
        self.phone_warn_ms = _env_int("PHONE_WARN_MS", 800)
        self.phone_alert_ms = _env_int("PHONE_ALERT_MS", 1400)
        self.alert_cooldown_ms = _env_int("ALERT_COOLDOWN_MS", 3000)
        self.yaw_thresh = _env_float("HEAD_YAW_WARN_DEG", 18.0)
        self.pitch_thresh = _env_float("HEAD_PITCH_WARN_DEG", 20.0)
        self.one_eye_warn_ms = _env_int("ONE_EYE_WARN_MS", 1500)
        self.one_eye_score = _env_float("ONE_EYE_WARN_SCORE", 12.0)
        self.lock_unstable_score = _env_float("LOCK_UNSTABLE_SCORE", 10.0)
        self.not_visible_score = _env_float("NOT_VISIBLE_SCORE", 40.0)
        self.require_seen_driver = os.environ.get("RISK_REQUIRE_SEEN_DRIVER", "1") == "1"

        self.eye_hist: deque[tuple[int, int]] = deque()
        self.blinks: deque[int] = deque()
        self.prev_closed = False
        self.offroad_start: Optional[int] = None
        self.phone_start: Optional[int] = None
        self.closed_start: Optional[int] = None
        self.one_eye_start: Optional[int] = None
        self.not_visible_start: Optional[int] = None
        self.state = "NORMAL"
        self.state_started_ms = 0
        self.last_alert_ms = 0
        self.seen_driver_once = False

    def _duration(self, start_ms: Optional[int], now_ms: int) -> int:
        if start_ms is None:
            return 0
        return max(0, now_ms - start_ms)

    def _perclos(self, now_ms: int) -> float:
        while self.eye_hist and now_ms - self.eye_hist[0][0] > self.perclos_window_ms:
            self.eye_hist.popleft()
        if not self.eye_hist:
            return 0.0
        closed = sum(v for _, v in self.eye_hist)
        return float(closed) / float(len(self.eye_hist))

    def _blink_rate(self, now_ms: int) -> float:
        horizon_ms = 60_000
        while self.blinks and now_ms - self.blinks[0] > horizon_ms:
            self.blinks.popleft()
        return float(len(self.blinks))

    def update(
        self,
        ts_ms: int,
        driver_locked: bool,
        face_bbox: Optional[list[int]],
        eyes: Optional[dict],
        objects: list[dict],
    ) -> RiskOutput:
        reason_codes: list[str] = []
        visible = driver_locked and face_bbox is not None
        if driver_locked:
            self.seen_driver_once = True
        score_allowed = self.seen_driver_once or (not self.require_seen_driver)

        yaw = float(eyes.get("yaw_deg", 0.0)) if (eyes and visible) else 0.0
        pitch = float(eyes.get("pitch_deg", 0.0)) if (eyes and visible) else 0.0

        phone_present = any("phone" in str(o.get("name", "")).lower() for o in objects)
        eyes_closed = False
        one_eye_closed = False
        if eyes:
            left_ear = float(eyes.get("left_ear", 0.0))
            right_ear = float(eyes.get("right_ear", 0.0))
            left_closed = (eyes.get("left_state") == "CLOSED") or (left_ear > 0.55)
            right_closed = (eyes.get("right_state") == "CLOSED") or (right_ear > 0.55)
            eyes_closed = left_closed and right_closed
            one_eye_closed = (left_closed != right_closed)

        if not visible:
            # Do not keep stale closure state when the face is not currently visible.
            eyes_closed = False
            one_eye_closed = False

        self.eye_hist.append((ts_ms, 1 if eyes_closed else 0))
        if eyes_closed and not self.prev_closed:
            self.closed_start = ts_ms
        if (not eyes_closed) and self.prev_closed and self._duration(self.closed_start, ts_ms) > 80:
            self.blinks.append(ts_ms)
        self.prev_closed = eyes_closed
        if not eyes_closed:
            self.closed_start = None

        if one_eye_closed:
            if self.one_eye_start is None:
                self.one_eye_start = ts_ms
        else:
            self.one_eye_start = None
        one_eye_ms = self._duration(self.one_eye_start, ts_ms)

        offroad = abs(yaw) > self.yaw_thresh or abs(pitch) > self.pitch_thresh
        if offroad:
            if self.offroad_start is None:
                self.offroad_start = ts_ms
        else:
            self.offroad_start = None
        offroad_ms = self._duration(self.offroad_start, ts_ms)

        if phone_present:
            if self.phone_start is None:
                self.phone_start = ts_ms
        else:
            self.phone_start = None
        phone_ms = self._duration(self.phone_start, ts_ms)

        if not visible:
            if self.not_visible_start is None:
                self.not_visible_start = ts_ms
        else:
            self.not_visible_start = None
        not_visible_ms = self._duration(self.not_visible_start, ts_ms)

        perclos = self._perclos(ts_ms)
        blink_rate = self._blink_rate(ts_ms)

        score = 0.0
        if perclos >= self.perclos_warn:
            score += 25.0
            reason_codes.append("EYES_CLOSED_SUSTAINED")
        if perclos >= self.perclos_alert:
            score += 20.0
        if one_eye_ms >= self.one_eye_warn_ms:
            score += self.one_eye_score
            reason_codes.append("ONE_EYE_CLOSED_SUSTAINED")
        if offroad_ms >= self.offroad_warn_ms:
            score += 20.0
            reason_codes.append("HEAD_OFFROAD_SUSTAINED")
        if offroad_ms >= self.offroad_alert_ms:
            score += 20.0
        if phone_ms >= self.phone_warn_ms:
            score += 15.0
            reason_codes.append("PHONE_DISTRACTION")
        if phone_ms >= self.phone_alert_ms:
            score += 15.0
        if score_allowed and not_visible_ms >= 800:
            score += self.not_visible_score
            reason_codes.append("DRIVER_NOT_VISIBLE")
        if score_allowed and (not driver_locked):
            score += self.lock_unstable_score
            reason_codes.append("LOCK_UNSTABLE")

        next_state = "NORMAL"
        if score >= 60.0:
            next_state = "ALERT"
        elif score >= 30.0:
            next_state = "WARN"

        if next_state != self.state:
            self.state = next_state
            self.state_started_ms = ts_ms

        warn = self.state in ("WARN", "ALERT")
        alert = self.state == "ALERT"
        if alert:
            if ts_ms - self.last_alert_ms > self.alert_cooldown_ms:
                self.last_alert_ms = ts_ms

        attention = {
            "perclos": round(perclos, 4),
            "blink_rate": round(blink_rate, 3),
            "head_yaw": round(yaw, 2),
            "head_pitch": round(pitch, 2),
            "offroad_ms": int(offroad_ms),
        }
        risk = {
            "score": round(score, 2),
            "state": self.state,
            "reason_codes": reason_codes[:3],
        }
        alerts = {
            "warn": warn,
            "alert": alert,
            "type": self.state.lower() if warn else "none",
            "started_ms": int(self.state_started_ms),
        }
        return RiskOutput(attention=attention, risk=risk, alerts=alerts)
