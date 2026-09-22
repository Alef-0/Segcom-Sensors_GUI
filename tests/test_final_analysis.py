import unittest

import numpy as np

from calibration.final_analysis import Goals, _score


class FinalAnalysisTests(unittest.TestCase):
    def test_error_equal_to_maximum_threshold_fails_strict_goal(self):
        metrics = _score(
            np.asarray([[80.0, 90.0]]),
            np.asarray([70.0]),
            Goals(median_ms=15.0, threshold_ms=10.0),
        )

        self.assertEqual(metrics["maximum_absolute_ms"], 10.0)
        self.assertEqual(metrics["errors_at_or_above_threshold"], 1)
        self.assertFalse(metrics["strict_maximum_goal_passed"])
        self.assertFalse(metrics["conditional_goal_passed"])



if __name__ == "__main__":
    unittest.main()
