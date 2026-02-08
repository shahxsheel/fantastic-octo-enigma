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


if __name__ == "__main__":
    unittest.main()
