import unittest

import numpy as np

from calibration.final_analysis import (
    FEATURE_NAMES,
    Goals,
    _evaluate,
    _score,
)


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

    def test_added_delivery_delay_does_not_change_capture_correction(self):
        flags = {
            "all_grid_cells_decoded": True,
            "more_codes_than_configured_visible": False,
            "visibility_contradiction": False,
            "upstream_position_warnings": False,
            "display_timing_flag": False,
            "history_warmup": False,
            "manual_qr_values": False,
        }
        rows = []
        for frame_number, media, arrival in (
            (1, 1_000_000_000, 1_100_000_000),
            (2, 2_000_000_000, 2_120_000_000),
        ):
            rows.append({
                "recording": "synthetic",
                "frame_number": frame_number,
                "filename": f"images/{frame_number}.jpg",
                "segment": 1,
                "features": [None] * len(FEATURE_NAMES),
                "targets": {
                    "newest_generation": [75.0, 85.0],
                    "newest_visibility": None,
                    "common_visibility": None,
                },
                "flags": flags,
                "display_period_ms": 10.0,
                "media_reference_monotonic_ns": media,
                "application_arrival_monotonic_ns": arrival,
                "observable_arrival_minus_media_ms": (arrival - media) / 1e6,
            })
        model = {
            "recipe": {"kind": "fixed_interval"},
            "correction_ms": 80.0,
        }

        metrics, predictions = _evaluate(model, rows, Goals())

        self.assertTrue(metrics["conditional_goal_passed"])
        self.assertEqual(
            predictions[1]["estimated_capture_monotonic_ns"]
            - predictions[0]["estimated_capture_monotonic_ns"],
            1_000_000_000,
        )
        self.assertEqual(
            predictions[1]["estimated_arrival_delay_ms"]
            - predictions[0]["estimated_arrival_delay_ms"],
            20.0,
        )


if __name__ == "__main__":
    unittest.main()
