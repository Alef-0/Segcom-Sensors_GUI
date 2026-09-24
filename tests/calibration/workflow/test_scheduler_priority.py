"""Non-privileged tests for calibration scheduler priority handling."""

import unittest
from unittest.mock import patch

from calibration.scheduler_priority import CalibrationSchedulerPriority


class CalibrationSchedulerPriorityTests(unittest.TestCase):
    def test_priority_is_raised_and_restored(self):
        current = {"nice": 0}

        def getpriority(_which, _who):
            return current["nice"]

        def setpriority(_which, _who, value):
            current["nice"] = value

        priority = CalibrationSchedulerPriority("test calibration work")
        with (
            patch("calibration.scheduler_priority.os.getpriority", side_effect=getpriority),
            patch("calibration.scheduler_priority.os.setpriority", side_effect=setpriority),
        ):
            enabled = priority.enable()
            restored = priority.restore()

        self.assertTrue(enabled["applied"])
        self.assertEqual(enabled["effective_nice"], -10)
        self.assertEqual(restored["effective_nice"], 0)
        self.assertEqual(current["nice"], 0)

        with (
            patch("calibration.scheduler_priority.os.getpriority", side_effect=getpriority),
            patch("calibration.scheduler_priority.os.setpriority", side_effect=setpriority),
        ):
            enabled_again = priority.enable()
        self.assertTrue(enabled_again["applied"])
        self.assertEqual(current["nice"], -10)

    def test_permission_failure_is_nonfatal_and_reported(self):
        priority = CalibrationSchedulerPriority("test calibration work")
        with (
            patch("calibration.scheduler_priority.os.getpriority", return_value=0),
            patch(
                "calibration.scheduler_priority.os.setpriority",
                side_effect=PermissionError("not permitted"),
            ),
        ):
            status = priority.enable()

        self.assertFalse(status["applied"])
        self.assertEqual(status["effective_nice"], 0)
        self.assertIn("not permitted", status["error"])


if __name__ == "__main__":
    unittest.main()
