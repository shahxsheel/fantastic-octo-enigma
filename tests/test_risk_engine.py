import unittest

from src.infer.risk_engine import RiskEngine


class RiskEngineTests(unittest.TestCase):
    def test_alert_on_sustained_phone_and_offroad(self):
        eng = RiskEngine()
        ts = 0
        out = None
        for _ in range(40):
            ts += 100
            out = eng.update(
                ts_ms=ts,
                driver_locked=True,
                face_bbox=[100, 100, 260, 300],
                eyes={
                    "left_state": "OPEN",
                    "right_state": "OPEN",
                    "left_ear": 0.1,
                    "right_ear": 0.1,
                    "yaw_deg": 35.0,
                    "pitch_deg": 0.0,
                },
                objects=[{"name": "cell phone", "conf": 0.8, "bbox": [150, 180, 220, 260]}],
            )
        self.assertIsNotNone(out)
        self.assertTrue(out.alerts["warn"])
        self.assertIn(out.risk["state"], ("WARN", "ALERT"))

    def test_no_startup_warn_before_driver_seen(self):
        eng = RiskEngine()
        ts = 0
        out = None
        for _ in range(20):
            ts += 100
            out = eng.update(
                ts_ms=ts,
                driver_locked=False,
                face_bbox=None,
                eyes=None,
                objects=[],
            )
        self.assertIsNotNone(out)
        self.assertEqual(out.risk["state"], "NORMAL")

    def test_one_eye_sustained_generates_warn_signal(self):
        eng = RiskEngine()
        ts = 0
        # Mark that we have a known driver first.
        eng.update(
            ts_ms=ts,
            driver_locked=True,
            face_bbox=[100, 100, 260, 300],
            eyes={
                "left_state": "OPEN",
                "right_state": "OPEN",
                "left_ear": 0.1,
                "right_ear": 0.1,
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
            },
            objects=[],
        )

        out = None
        for _ in range(30):
            ts += 100
            out = eng.update(
                ts_ms=ts,
                driver_locked=True,
                face_bbox=[100, 100, 260, 300],
                eyes={
                    "left_state": "CLOSED",
                    "right_state": "OPEN",
                    "left_ear": 0.65,
                    "right_ear": 0.1,
                    "yaw_deg": 0.0,
                    "pitch_deg": 0.0,
                },
                objects=[],
            )
        self.assertIsNotNone(out)
        self.assertTrue(out.risk["score"] > 0.0)
        self.assertIn("ONE_EYE_CLOSED_SUSTAINED", out.risk["reason_codes"])


if __name__ == "__main__":
    unittest.main()
