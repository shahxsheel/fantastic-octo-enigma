import unittest

import numpy as np

from src.infer.driver_lock import DriverLock


class DriverLockTests(unittest.TestCase):
    def test_lock_promotes_after_stable_candidates(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lock = DriverLock(frame.shape)
        ts = 0
        person = [60, 120, 260, 420]  # left-seat box (LHD prior)
        for _ in range(lock.lock_min_frames + 1):
            ts += 100
            lock.update_from_detections(frame, ts, [person], face_bbox=[90, 170, 210, 300])
        st = lock.state()
        self.assertTrue(st.locked)
        self.assertIsNotNone(st.bbox)
        health = lock.lock_health(ts)
        self.assertTrue(health["stable"])


if __name__ == "__main__":
    unittest.main()
