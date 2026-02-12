import unittest
from collections import deque

from src.infer.face_eye_mediapipe import FaceEyeEstimatorMediaPipe


class FaceEyeCacheTests(unittest.TestCase):
    def test_roi_cache_is_bounded_fifo(self):
        est = FaceEyeEstimatorMediaPipe.__new__(FaceEyeEstimatorMediaPipe)
        est._roi_by_ts = {}
        est._roi_ts_queue = deque()
        est._last_roi_info = None

        for ts in range(100):
            est._remember_roi(ts, {"offset_x": ts, "offset_y": ts})

        self.assertLessEqual(len(est._roi_by_ts), 64)
        self.assertNotIn(0, est._roi_by_ts)
        self.assertIn(99, est._roi_by_ts)


if __name__ == "__main__":
    unittest.main()
