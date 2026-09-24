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

class TestRecordingAnalyzer(RecordingFixtureMixin, unittest.TestCase):
    def test_saved_media_monotonic_time_is_used_directly(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=FakeReader([], []))
            row = {**model.rows[0], "media_monotonic_ns": 12_345_678_901}

            self.assertEqual(model.pts_monotonic_ns(row), 12_345_678_901)

    def test_analysis_keeps_contrast_detected_cell_when_qr_reader_finds_nothing(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            image_path = folder / "images" / "camera_000001.jpg"
            image = image_fixture("bright_qr_cell_0.png")
            cv2.imwrite(str(image_path), image)
            model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=FakeReader([], []))
            model.undistorter.image = lambda frame, _alpha: frame.copy()

            result = model.analyze(0)

            self.assertEqual(len(result["observations"]), 1)
            self.assertIsNone(result["observations"][0]["raw"])
            self.assertEqual(result["observations"][0]["cell"], 0)
            self.assertEqual(len(result["qr_evidence"]["detections"]), 1)
            self.assertEqual(
                result["qr_evidence"]["detections"][0]["detection_method"],
                "screen_contrast",
            )

    def test_empty_frame_skips_qr_reader_creation(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            with patch("calibration.recording_display.create_qreader",
                       side_effect=AssertionError("empty screen should skip QReader")) as factory:
                model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS)
                result = model.analyze(0)

            self.assertEqual(result["observations"], [])
            factory.assert_not_called()

    def test_qreader_box_is_matched_and_grid_cell_mismatch_is_reported(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder)
            marker = rows[3]
            reader = FakeReader([timestamp_payload(marker["marker_ns"])], [[10, 10, 30, 30]])
            model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=reader)
            result = model.analyze(0)
            self.assertEqual(result["latest"]["display_index"], marker["index"])
            self.assertTrue(result["latest"]["mismatch"])
            self.assertEqual(result["latest"]["status"], "Grid cell mismatch")

    def test_partial_grid_values_choose_the_latest_readable_marker(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder, count=14, grid_qrs=12, visible_qrs=8)
            frames = rows[1:]
            model = RecordingAnalyzer(
                folder,
                DEFAULT_INTRINSICS,
                reader=FakeReader([], []),
            )
            result = model.analyze(0)
            qrs = [None] * 12
            for cell in (0, 5, 11):
                qrs[cell] = timestamp_payload(frames[cell]["marker_ns"])
            model.set_manual_values(0, {
                "pts_ns": result["row"]["pts_ns"],
                "ntp_ns": result["row"]["reference_ntp_ns"],
                "qrs": tuple(qrs),
            })
            check = model.check_frame(result)
            self.assertTrue(check["valid"])
            self.assertEqual(check["matched_readable_qrs"], 3)
            self.assertEqual(check["latest_cell"], 11)
            self.assertEqual(check["latest_marker"]["index"], 11)

    def test_twelve_cell_recording_uses_journal_grid_for_automatic_detections(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder, count=14, grid_qrs=12, visible_qrs=12)
            frames = rows[1:]
            selected_cells = (0, 5, 11)
            boxes = []
            for cell in selected_cells:
                left, top, right, bottom = grid_bounds(100, 100, 12)[cell]
                boxes.append([left + 2, top + 2, right - 2, bottom - 2])
            reader = FakeReader(
                [timestamp_payload(frames[cell]["marker_ns"]) for cell in selected_cells],
                boxes,
            )
            model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=reader)

            check = model.check_frame(model.analyze(0))

            self.assertTrue(check["valid"])
            self.assertEqual(check["matched_readable_qrs"], 3)
            self.assertEqual(check["latest_cell"], 11)
            self.assertEqual(check["latest_marker"]["index"], 11)

    def test_automatic_selection_uses_journal_cell_when_camera_grid_is_shifted(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder, count=8, grid_qrs=10, visible_qrs=5)
            frames = rows[1:]
            selected_indices = (1, 2, 3, 4, 5, 6)
            boxes = (
                [25, 10, 35, 30],
                [45, 10, 55, 30],
                [48, 10, 58, 30],
                [65, 10, 75, 30],
                [65, 60, 75, 80],
                [45, 60, 55, 80],
            )
            reader = FakeReader(
                [timestamp_payload(frames[index]["marker_ns"]) for index in selected_indices],
                boxes,
            )
            model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=reader)

            result = model.analyze(0)
            check = model.check_frame(result)

            self.assertTrue(check["valid"])
            self.assertEqual(check["matched_readable_qrs"], len(selected_indices))
            self.assertEqual(check["latest_cell"], frames[6]["cell"])
            self.assertEqual(check["latest_marker"]["index"], 6)
            self.assertTrue(any(item["mismatch"] for item in result["observations"]))
            self.assertEqual(len(check["position_warnings"]), 4)

    def test_legacy_calibration_journal_and_loose_image_remain_readable(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder, legacy=True)
            model = RecordingAnalyzer(
                folder,
                DEFAULT_INTRINSICS,
                reader=FakeReader([], []),
            )

        self.assertEqual(model.rows[0]["filename"], "camera_000001.jpg")
