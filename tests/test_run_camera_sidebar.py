import unittest

from src.camera.run_camera import _build_sidebar_rows, build_sidebar_panel


class SidebarTests(unittest.TestCase):
    def test_rows_include_both_eyes_and_alert_footer(self):
        result = {
            "objects": [{"name": "person", "conf": 0.95, "bbox": [1, 1, 10, 10]}],
            "eyes": {
                "left_state": "OPEN",
                "left_pct": 91.0,
                "right_state": "OPEN",
                "right_pct": 93.0,
            },
            "driver": {"locked": True, "lock_conf": 0.88},
            "risk": {"state": "ALERT", "score": 77.0, "reason_codes": ["PHONE_DISTRACTION"]},
            "alerts": {"alert": True, "warn": True},
            "attention": {"head_yaw": 2.0, "perclos": 0.12},
        }
        rows = _build_sidebar_rows(fps=20.0, latency_sec=0.05, result=result)
        texts = [r[0] for r in rows]
        self.assertTrue(any(t.startswith("L OPEN") for t in texts))
        self.assertTrue(any(t.startswith("R OPEN") for t in texts))
        self.assertEqual(texts[-1], "ALERT: ATTEND ROAD")

    def test_panel_shape_matches_frame_height(self):
        x0, panel = build_sidebar_panel((720, 1280, 3), 24.5, 0.03, result=None)
        self.assertEqual(panel.shape[0], 720)
        self.assertEqual(panel.shape[2], 3)
        self.assertGreaterEqual(x0, 0)
        self.assertLess(x0, 1280)


if __name__ == "__main__":
    unittest.main()
