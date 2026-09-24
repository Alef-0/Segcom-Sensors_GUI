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

class TestRecordingReview(RecordingFixtureMixin, unittest.TestCase):
    def test_saved_review_restores_values_and_boxes_without_decoding_or_writing(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary) / "recording"
            folder.mkdir()
            model, _ = self.four_value_model(folder)
            report = model.summarize(0.25, threading.Event(), lambda *_: None)
            model.save_report(report)
            path = model.output_folder / "calibration_analysis.json"
            before = path.read_bytes()
            with patch("calibration.recording_display.create_qreader", side_effect=AssertionError("must not decode")):
                reopened = RecordingAnalyzer(folder, DEFAULT_INTRINSICS)
                result = reopened.inspect(0)
                check = reopened.check_frame(result)
            self.assertEqual(list(check["values"]["qrs"]), report["frames"][0]["qr_values_ms"])
            self.assertEqual(check["latest_marker"]["index"], report["frames"][0]["latest_display_index"])
            self.assertEqual(len(result["observations"]), len(report["frames"][0]["qr_evidence"]["detections"]))
            self.assertIn("Saved QR results", result["review_notice"])
            self.assertEqual(reopened.cache, {})
            self.assertIsNone(reopened.reader)
            self.assertEqual(path.read_bytes(), before)

    def test_missing_saved_frame_stays_unanalyzed_until_explicit_fresh_decode(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary) / "recording"
            folder.mkdir()
            rows = self.fixture(folder)
            image = image_fixture("bright_qr_cell_0.png")
            cv2.imwrite(str(folder / "images" / "camera_000001.jpg"), image)
            output = folder.with_name("recording_analysis")
            output.mkdir()
            path = output / "calibration_analysis.json"
            path.write_text(json.dumps({"frames": [], "analysis_alpha": 0.25}))
            reader = FakeReader([timestamp_payload(rows[3]["marker_ns"])], [[60, 60, 80, 80]])
            with patch("calibration.recording_display.create_qreader", return_value=reader) as factory:
                model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS)
                model.undistorter.image = lambda frame, _alpha: frame.copy()
                saved = model.inspect(0)
                self.assertIn("No saved QR results", saved["review_notice"])
                self.assertFalse(model.check_frame(saved)["valid"])
                factory.assert_not_called()
                model.begin_fresh_analysis()
                fresh = model.inspect(0)
                self.assertTrue(model.check_frame(fresh)["valid"])
                factory.assert_called_once()
                self.assertNotIn("saved_values", fresh)

    def test_saved_boxes_hidden_for_different_undistortion_but_values_retained(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary) / "recording"
            folder.mkdir()
            model, _ = self.four_value_model(folder)
            report = model.summarize(0.25, threading.Event(), lambda *_: None)
            model.save_report(report)
            reopened = RecordingAnalyzer(folder, DEFAULT_INTRINSICS)
            result = reopened.inspect(0, alpha=0.5)
            self.assertEqual(result["observations"], [])
            self.assertTrue(reopened.check_frame(result)["valid"])
            self.assertIn("do not match", result["review_notice"])

    def test_saved_values_are_matched_by_filename_not_report_order(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary) / "recording"
            folder.mkdir()
            rows = self.fixture(folder)
            journal = folder / "camera_timestamps.jsonl"
            first = json.loads(journal.read_text())
            second = {**first, "frame": "images/camera_000002.jpg"}
            (folder / second["frame"]).write_bytes((folder / first["frame"]).read_bytes())
            journal.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
            saved = []
            for camera, marker in ((first, rows[2]), (second, rows[3])):
                values = [None] * 4
                values[marker["cell"]] = timestamp_payload(marker["marker_ns"])
                saved.append({"filename": camera["frame"], "qr_values_ms": values})
            output = folder.with_name("recording_analysis")
            output.mkdir()
            (output / "calibration_analysis.json").write_text(json.dumps({"frames": saved[::-1]}))
            with patch("calibration.recording_display.create_qreader", side_effect=AssertionError("must not decode")):
                model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS)
                for index in (0, 1):
                    self.assertEqual(list(model.frame_values(model.inspect(index))["qrs"]), saved[index]["qr_values_ms"])

    def test_malformed_saved_analysis_reports_error_without_automatic_decode(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary) / "recording"
            folder.mkdir()
            self.fixture(folder)
            output = folder.with_name("recording_analysis")
            output.mkdir()
            path = output / "calibration_analysis.json"
            path.write_text("{broken")
            with patch("calibration.recording_display.create_qreader", side_effect=AssertionError("must not decode")):
                model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS)
                self.assertIn("Could not load saved analysis", model.saved_analysis_notice)
                self.assertFalse(model.check_frame(model.inspect(0))["valid"])
            self.assertEqual(path.read_text(), "{broken")

    def test_overlay_hides_unreadable_and_draws_selected_qr_in_black_and_white(self):
        class Variable:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class Canvas:
            def __init__(self):
                self.polygons = []
                self.texts = []

            def delete(self, *_):
                pass

            def winfo_width(self):
                return 100

            def winfo_height(self):
                return 100

            def create_image(self, *_args, **_kwargs):
                pass

            def create_polygon(self, _points, **options):
                self.polygons.append(options)

            def create_text(self, *_args, **options):
                self.texts.append(options)
                return len(self.texts)

            def bbox(self, _label):
                return 0, 0, 20, 10

            def create_rectangle(self, *_args, **_kwargs):
                return 1

            def tag_raise(self, *_args):
                pass

        window = CalibrationWindow.__new__(CalibrationWindow)
        window.resize_job = None
        window.canvas = Canvas()
        window.variant = Variable("Undistorted")
        window.draw_all_boxes = Variable(True)
        window.show_all_times = Variable(True)
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        points = np.asarray(((10, 10), (30, 10), (30, 30), (10, 30)))
        window.current = {
            "undistorted": pixels,
            "original": pixels,
            "observations": [
                {
                    "raw": None,
                    "display_index": None,
                    "cell": 0,
                    "undistorted_points": points,
                    "original_points": points,
                },
                {
                    "raw": "invalid",
                    "display_index": None,
                    "cell": 1,
                    "undistorted_points": points,
                    "original_points": points,
                },
                {
                    "raw": "000000012345",
                    "display_index": 5,
                    "cell": 2,
                    "undistorted_points": points,
                    "original_points": points,
                },
            ],
        }
        window.current_check = {
            "valid": True,
            "latest_marker": {"index": 5},
            "values": {},
        }

        with patch("calibration.recording_display.ImageTk.PhotoImage", return_value=object()):
            window.draw()

        self.assertEqual(
            [polygon["outline"] for polygon in window.canvas.polygons],
            ["#000000", "#ffffff"],
        )
        self.assertEqual([text["text"] for text in window.canvas.texts], ["12.345 s"])

    def test_overlay_marks_bright_content_when_timestamp_cannot_be_read(self):
        class Variable:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class Canvas:
            def __init__(self):
                self.polygons = []
                self.texts = []

            def delete(self, *_):
                pass

            def winfo_width(self):
                return 100

            def winfo_height(self):
                return 100

            def create_image(self, *_args, **_kwargs):
                pass

            def create_polygon(self, _points, **options):
                self.polygons.append(options)

            def create_text(self, *_args, **options):
                self.texts.append(options)
                return len(self.texts)

            def bbox(self, _label):
                return 0, 0, 30, 10

            def create_rectangle(self, *_args, **_kwargs):
                return 1

            def tag_raise(self, *_args):
                pass

        window = CalibrationWindow.__new__(CalibrationWindow)
        window.resize_job = None
        window.canvas = Canvas()
        window.variant = Variable("Undistorted")
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        points = np.asarray(((10, 10), (30, 10), (30, 30), (10, 30)))
        window.current = {
            "undistorted": pixels,
            "original": pixels,
            "latest": None,
            "observations": [{
                "raw": None,
                "detection_method": "screen_contrast",
                "undistorted_points": points,
                "original_points": points,
            }],
        }
        window.current_check = {"valid": False}

        with patch("calibration.recording_display.ImageTk.PhotoImage", return_value=object()):
            window.draw()

        self.assertEqual(
            [polygon["outline"] for polygon in window.canvas.polygons],
            ["#000000", "#00e5ff"],
        )
        self.assertEqual(window.canvas.texts[0]["text"], "Bright content · timestamp unreadable")
