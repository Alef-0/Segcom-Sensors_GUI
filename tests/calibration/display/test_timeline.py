"""Focused recording calibration tests."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration.qr import DETECTION_BATCH_SIZE, grid_bounds, timestamp_payload
from calibration.recording_display import (
    CalibrationWindow,
    DEFAULT_INTRINSICS,
    PRESENTATIONS_CSV,
    DisplayTimeline,
    RecordingAnalyzer,
    _maximum_interval_consensus,
)
from tests.calibration.support import (
    FakeReader,
    RecordingFixtureMixin,
    display_rows,
    image_fixture,
)

class TestDisplayTimeline(RecordingFixtureMixin, unittest.TestCase):
    def test_timeline_checks_following_replacement(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder, count=6)
            timeline = DisplayTimeline(folder)
            self.assertEqual(timeline.marker_status(2), ("Clean", []))
            rows[5]["late_submit"] = True
            (folder / "display_timestamps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            timeline = DisplayTimeline(folder)
            status, issues = timeline.marker_status(2)
            self.assertEqual(status, "Timing suspect")
            self.assertIn("replacement_late_submission", issues)

    def test_immediate_interval_boundary_issue_marks_observation_suspect(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder, count=6)
            rows[4]["irregular_interval"] = True
            (folder / "display_timestamps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            marker = rows[3]
            model = RecordingAnalyzer(
                folder,
                DEFAULT_INTRINSICS,
                reader=FakeReader(
                    [timestamp_payload(marker["marker_ns"])],
                    [[60, 60, 80, 80]],
                ),
            )

            check = model.check_frame(model.analyze(0))

            self.assertEqual(check["timing_status"], "Timing suspect")
            self.assertIn("interval_end_irregular_interval", check["issues"])

    def test_timeline_reconstructs_software_presentation_intervals(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder, count=6)
            timeline = DisplayTimeline(folder)
            frames = rows[1:]
            start = frames[1]["flip_return_ns"]
            end = frames[4]["flip_return_ns"]

            self.assertEqual(timeline.active_index_at(start + 1_000_000), 1)
            self.assertEqual(
                [event["display_index"] for event in timeline.events_between(start, end)],
                [2, 3, 4],
            )
            interval = timeline.observation_interval(1, start + 5_000_000)
            self.assertAlmostEqual(interval["offset_interval_upper_ms"], 5.0)
            self.assertAlmostEqual(
                interval["offset_interval_lower_ms"],
                5.0 - 1000.0 / 60.0,
                places=5,
            )
            self.assertEqual(
                interval["current_presentation"]["presentation_event_kind"],
                "legacy_flip_return",
            )

    def test_interval_consensus_finds_maximum_overlap_range(self):
        consensus = _maximum_interval_consensus([
            (95.0, 105.0),
            (98.0, 108.0),
            (100.0, 110.0),
        ])

        self.assertIsNotNone(consensus)
        self.assertEqual(consensus["offset_range_lower_ms"], 100.0)
        self.assertEqual(consensus["offset_range_upper_ms"], 105.0)
        self.assertGreaterEqual(consensus["estimated_offset_ms"], 100.0)
        self.assertLessEqual(consensus["estimated_offset_ms"], 105.0)
        self.assertEqual(consensus["maximum_consistent_frames"], 3)

    def test_thirty_hz_observations_reconstruct_intervening_100_hz_flips(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder, count=20, period=10_000_000)
            model = RecordingAnalyzer(
                folder,
                DEFAULT_INTRINSICS,
                reader=FakeReader([], []),
            )
            exposure_ns = (
                10_025_000_000,
                10_058_333_333,
                10_091_666_666,
                10_125_000_000,
            )
            observed_indices = (2, 5, 9, 12)
            frame_reports = []
            for exposure, observed in zip(exposure_ns, observed_indices):
                reference = exposure + 100_000_000
                start = model.timeline.presentation_times[observed]
                end = model.timeline.presentation_times[observed + 1]
                frame_reports.append({
                    "validation": "accepted_clean",
                    "camera_reference_monotonic_ns": reference,
                    "latest_display_index": observed,
                    "offset_interval_lower_ms": (reference - end) / 1e6,
                    "offset_interval_upper_ms": (reference - start) / 1e6,
                    "arrival_offset_interval_lower_ms": None,
                    "arrival_offset_interval_upper_ms": None,
                    "software_transition_margin_ms": 1.0,
                })

            analysis = model._annotate_interval_analysis(frame_reports)

            estimate = analysis["primary_pts"]["estimated_offset_ms"]
            self.assertGreaterEqual(estimate, 97.333)
            self.assertLessEqual(estimate, 100.667)
            self.assertEqual(
                [frame["presentation_classification"] for frame in frame_reports],
                ["stable_expected"] * 4,
            )
            self.assertEqual(
                [
                    len(frame["display_events_since_previous_camera"])
                    for frame in frame_reports
                ],
                [0, 3, 4, 3],
            )

    def test_previous_qr_at_software_boundary_is_expected_transition(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder, count=12, period=10_000_000)
            model = RecordingAnalyzer(
                folder,
                DEFAULT_INTRINSICS,
                reader=FakeReader([], []),
            )
            support = [{
                "validation": "accepted_clean",
                "camera_reference_monotonic_ns": 10_125_000_000,
                "latest_display_index": 2,
                "offset_interval_lower_ms": 95.0,
                "offset_interval_upper_ms": 105.0,
                "arrival_offset_interval_lower_ms": None,
                "arrival_offset_interval_upper_ms": None,
                "software_transition_margin_ms": 1.0,
            }]
            boundary = model.timeline.presentation_times[5]
            transition = {
                "validation": "accepted_timing_suspect",
                "camera_reference_monotonic_ns": boundary + 100_500_000,
                "latest_display_index": 4,
                "offset_interval_lower_ms": 100.5,
                "offset_interval_upper_ms": 110.5,
                "arrival_offset_interval_lower_ms": None,
                "arrival_offset_interval_upper_ms": None,
                "software_transition_margin_ms": 1.0,
            }

            model._annotate_interval_analysis([*support, transition])

            self.assertEqual(
                transition["presentation_classification"],
                "expected_flip_transition",
            )
            self.assertEqual(transition["expected_display_index"], 5)
