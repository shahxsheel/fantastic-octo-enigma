import unittest

from src.infer.run_infer import (
    _effective_phone_every_n,
    _is_n_frame,
    _should_run_roi_phone,
)


class RunInferScheduleTests(unittest.TestCase):
    def test_zero_cadence_disables_periodic(self):
        self.assertFalse(_is_n_frame(12, 0))
        self.assertFalse(_is_n_frame(12, -1))

    def test_phone_cadence_tightens_when_unstable_or_risky(self):
        self.assertEqual(_effective_phone_every_n(4, lock_stable=True, risk_state="NORMAL"), 4)
        self.assertEqual(_effective_phone_every_n(4, lock_stable=False, risk_state="NORMAL"), 2)
        self.assertEqual(_effective_phone_every_n(4, lock_stable=True, risk_state="WARN"), 2)
        self.assertEqual(_effective_phone_every_n(0, lock_stable=True, risk_state="ALERT"), 0)

    def test_phone_pass_never_runs_on_full_yolo_frame(self):
        self.assertFalse(
            _should_run_roi_phone(
                has_driver_roi=True,
                run_full_yolo=True,
                frame_count=8,
                phone_every_n=2,
            )
        )
        self.assertTrue(
            _should_run_roi_phone(
                has_driver_roi=True,
                run_full_yolo=False,
                frame_count=8,
                phone_every_n=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
