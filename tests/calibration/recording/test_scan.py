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

class TestRecordingScan(RecordingFixtureMixin, unittest.TestCase):
    def test_scan_progress_passes_each_decoded_frame(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            model = RecordingAnalyzer(
                folder,
                DEFAULT_INTRINSICS,
                reader=FakeReader([], []),
            )
            progress = []

            model.summarize(
                0.25,
                threading.Event(),
                lambda done, total, frame: progress.append((done, total, frame["index"])),
            )

            self.assertEqual(progress, [(1, 1, 0)])

    def test_scan_accepts_any_amount_of_missing_qrs_when_one_matches(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder)
            frames = rows[1:]
            reader = FakeReader(
                [
                    *[timestamp_payload(frames[index]["marker_ns"]) for index in (4, 1, 2)],
                    None,
                ],
                [
                    [10, 10, 30, 30], [60, 10, 80, 30],
                    [60, 60, 80, 80], [10, 60, 30, 80],
                ],
            )
            model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=reader)
            report = model.summarize(0.25, threading.Event(), lambda *_: None)
            self.assertEqual(report["processed"], 1)
            self.assertIsNone(report["stopped"])
            self.assertEqual(report["counts"]["accepted_frames"], 1)
            self.assertEqual(report["counts"]["unreadable"], 1)
            self.assertEqual(report["frames"][0]["matched_readable_qrs"], 3)
            self.assertEqual(report["frames"][0]["latest_display_index"], 4)
            self.assertEqual(
                report["readability"]["frames_by_matched_readable_qr_count"],
                {"3": 1},
            )

    def test_scan_excludes_sustained_repeat_from_offset_summary(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder, count=12)
            original = json.loads((folder / "camera_timestamps.jsonl").read_text())
            cameras = []
            for i in range(3):
                filename = f"images/camera_{i+1:06d}.jpg"
                (folder / filename).write_bytes((folder / original["frame"]).read_bytes())
                cameras.append({**original, "frame": filename,
                                "pts_ns": original["pts_ns"] + i * 40_000_000,
                                "running_time_ns": 100_000_000 + i * 40_000_000})
            (folder / "camera_timestamps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in cameras))
            model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=FakeReader(
                [timestamp_payload(rows[5]["marker_ns"])], [[10, 10, 30, 30]]))
            report = model.summarize(0.25, threading.Event(), lambda *_: None)
            self.assertEqual(report["counts"]["accepted_frames"], 0)
            self.assertEqual(report["counts"]["frames_suspected_stale_visual_state"], 3)
            self.assertIsNone(report["median_offset_ms"])
            self.assertEqual(len(report["frames"]), 3)
            self.assertTrue(all(row["validation"] == "skipped_suspected_stale_visual_state"
                                for row in report["frames"]))
            saved = model.save_report(report)
            text = (Path(saved["output_directory"]) / "calibration_frames.csv").read_text()
            self.assertIn("temporal_evidence", text)
            self.assertIn("newest_qr_repeated_beyond_presentation_interval", text)

    def test_scan_skips_only_when_no_readable_qr_matches(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            model = RecordingAnalyzer(
                folder,
                DEFAULT_INTRINSICS,
                reader=FakeReader([], []),
            )
            report = model.summarize(0.25, threading.Event(), lambda *_: None)
            self.assertEqual(report["processed"], 1)
            self.assertIsNone(report["stopped"])
            self.assertEqual(report["counts"]["frames_without_readable_qr"], 1)
            self.assertEqual(report["frames"][0]["validation"], "skipped_no_readable_qr")

    def test_analysis_results_are_saved_beside_recording(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary) / "sample_recording"
            folder.mkdir()
            model, _ = self.four_value_model(folder)
            report = model.summarize(0.25, threading.Event(), lambda *_: None)
            saved = model.save_report(report)
            output = Path(temporary) / "sample_recording_analysis"
            self.assertEqual(saved["output_directory"], str(output))
            json_path = output / "calibration_analysis.json"
            csv_path = output / "calibration_frames.csv"
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["detection_batch_size"], DETECTION_BATCH_SIZE)
            self.assertEqual(loaded["counts"]["accepted_frames"], 1)
            self.assertEqual(loaded["frames"][0]["validation"], "accepted_unknown")
            self.assertIn("qr_values_ms", csv_path.read_text(encoding="utf-8"))
            self.assertTrue((output / PRESENTATIONS_CSV).is_file())
            self.assertEqual(loaded["presentation_timeline_file"], PRESENTATIONS_CSV)

    def test_scan_classifies_frame_inside_observed_presentation_interval(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = self.fixture(folder, count=6)
            camera_journal = folder / "camera_timestamps.jsonl"
            camera_row = json.loads(camera_journal.read_text(encoding="utf-8"))
            camera_row["running_time_ns"] = 100_000_000
            camera_journal.write_text(json.dumps(camera_row) + "\n", encoding="utf-8")
            marker = rows[3]
            reader = FakeReader(
                [timestamp_payload(marker["marker_ns"])],
                [[60, 60, 80, 80]],
            )
            model = RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=reader)

            report = model.summarize(
                0.25,
                threading.Event(),
                lambda *_: None,
            )

            primary = report["presentation_interval_analysis"]["primary_pts"]
            self.assertIsNotNone(primary)
            self.assertEqual(primary["maximum_consistent_frames"], 1)
            self.assertEqual(
                report["frames"][0]["presentation_classification"],
                "stable_expected",
            )
            self.assertEqual(report["frames"][0]["expected_display_index"], 2)
