"""Headless QR calibration tests; no display window or camera is opened."""

import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pygame

from PySide6.QtGui import QGuiApplication, QImage, QPainter

import analyze_calibration_recording as recording_launcher
from calibration.display import (
    DisplayJournal as PygameDisplayJournal,
    FramePacer,
    QRClockRenderer as PygameQRClockRenderer,
)
from calibration.display_qt import (
    BACKGROUND as QT_BACKGROUND,
    FOREGROUND as QT_FOREGROUND,
    DisplayJournal as QtDisplayJournal,
    QRClockRenderer,
    SwapTimingMonitor,
    VISIBLE_QRS,
)
from calibration.quantitative_analysis import analyze_output_directory
from calibration.qr import (
    DETECTION_BATCH_SIZE,
    GRID_LAYOUTS,
    QUIET_ZONE_MODULES,
    cell_index_for,
    decode_qrs_batch,
    decode_qrs_with_quadrant_retries,
    grid_bounds,
    grid_cell_names,
    grid_positions,
    grid_shape,
    order_by_quadrant,
    qr_matrix,
    timestamp_payload,
)
from calibration.recording_display import (
    DEFAULT_INTRINSICS,
    PRESENTATIONS_CSV,
    DisplayTimeline,
    RecordingAnalyzer,
    _maximum_interval_consensus,
)


class FakeReader:
    def __init__(self, decoded, boxes):
        self.decoded = tuple(decoded)
        self.boxes = boxes

    def detect_and_decode(self, image, return_detections=False, is_bgr=False):
        if image.shape[:2] != (100, 100):
            return ((), ()) if return_detections else ()
        detections = tuple({"bbox_xyxy": np.asarray(box, dtype=np.float32), "confidence": 0.9}
                           for box in self.boxes)
        return (self.decoded, detections) if return_detections else self.decoded


def display_rows(count=5, period=16_666_667):
    frames = []
    for index in range(count):
        marker = 10_000_000_000 + index * period
        frames.append({
            "kind": "frame", "index": index, "corner": index % 4,
            "marker_ns": marker, "deadline_ns": marker + 1_000_000,
            "submit_ns": marker + 500_000, "flip_return_ns": marker + 1_000_000,
            "frame_period_ns": period, "interval_ns": period if index else None,
            "late_submit": False, "skipped_periods": 0, "irregular_interval": False,
            "resumed_after_pause": False,
        })
    return [{"kind": "session", "format": "segcom-qr-display-v1"}, *frames]


class QRHelpersTests(unittest.TestCase):
    def test_payload_and_reduced_quiet_zone(self):
        self.assertEqual(timestamp_payload(12_345_678_900_000), "000012345678")
        matrix = qr_matrix("000012345678")
        self.assertEqual(QUIET_ZONE_MODULES, 2)
        self.assertFalse(matrix[:QUIET_ZONE_MODULES].any())
        self.assertFalse(matrix[:, :QUIET_ZONE_MODULES].any())

    def test_bounding_boxes_are_ordered_clockwise_by_sector(self):
        detections = [
            {"raw": "3", "bbox": np.array([10, 60, 30, 80]), "center": (20, 70), "confidence": 1},
            {"raw": "1", "bbox": np.array([10, 10, 30, 30]), "center": (20, 20), "confidence": 1},
            {"raw": "4", "bbox": np.array([60, 60, 80, 80]), "center": (70, 70), "confidence": 1},
            {"raw": "2", "bbox": np.array([60, 10, 80, 30]), "center": (70, 20), "confidence": 1},
        ]
        ordered = order_by_quadrant(detections, (100, 100))
        self.assertEqual([item["raw"] for item in ordered], ["1", "2", "4", "3"])
        self.assertEqual([item["quadrant"] for item in ordered], [0, 1, 2, 3])

    def test_missing_quadrant_is_retried_with_qreader(self):
        class RetryReader:
            def __init__(self):
                self.calls = []

            def detect_and_decode(self, image, return_detections=False, is_bgr=False):
                self.calls.append(image.shape[:2])
                if image.shape[:2] == (100, 100):
                    decoded = ("1", "2", "3")
                    boxes = ([10, 10, 30, 30], [60, 10, 80, 30], [60, 60, 80, 80])
                else:
                    decoded = ("4",) if len(self.calls) == 2 else ()
                    boxes = ([10, 10, 30, 30],) if decoded else ()
                detections = tuple({
                    "bbox_xyxy": np.asarray(box, dtype=np.float32),
                    "confidence": 0.9,
                } for box in boxes)
                return (decoded, detections) if return_detections else decoded

        reader = RetryReader()
        detections = decode_qrs_with_quadrant_retries(
            reader, np.zeros((100, 100, 3), np.uint8)
        )
        ordered = order_by_quadrant(detections, (100, 100))
        self.assertEqual([item["raw"] for item in ordered], ["1", "2", "3", "4"])
        self.assertEqual(reader.calls, [(100, 100), (50, 50)])

    def test_supported_grid_shapes_and_snake_order(self):
        self.assertEqual(tuple(GRID_LAYOUTS), (4, 6, 8, 9, 10, 12))
        self.assertEqual(grid_shape(6), (2, 3))
        self.assertEqual(
            grid_positions(6),
            ((0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)),
        )
        self.assertEqual(cell_index_for((90, 75), (120, 100), 6), 3)
        self.assertEqual(grid_cell_names(4)[2], "Bottom-right")

    def test_batch_detection_uses_one_model_prediction_and_keeps_image_order(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                return [f"prediction-{index}" for index in range(len(kwargs["source"]))]

        class BatchReader:
            def __init__(self):
                self.detector = type("Detector", (), {
                    "model": FakeModel(),
                    "_conf_th": 0.3,
                    "_nms_iou": 0.3,
                })()

            @staticmethod
            def decode(image, detection_result):
                return detection_result["raw"]

        reader = BatchReader()
        images = [np.full((20, 20, 3), index, np.uint8) for index in range(3)]

        with (
            patch("qrdet._prepare_input", side_effect=lambda source, is_bgr: source),
            patch(
                "qrdet._yolo_v8_results_to_dict",
                side_effect=lambda results, image: [{
                    "raw": results,
                    "bbox_xyxy": np.asarray((1, 2, 11, 12), dtype=float),
                    "confidence": 0.9,
                }],
            ),
        ):
            batches = decode_qrs_batch(reader, images)

        self.assertEqual(len(reader.detector.model.calls), 1)
        self.assertEqual(
            [[item["raw"] for item in batch] for batch in batches],
            [["prediction-0"], ["prediction-1"], ["prediction-2"]],
        )

    def test_renderer_keeps_exactly_two_quadrants(self):
        self.qt_app = QGuiApplication.instance() or QGuiApplication(["qr-test"])
        renderer = QRClockRenderer(960, 540)
        for index in range(4):
            renderer.render_next(
                10_000_000_000 + index * 20_000_000,
                index,
            )
        self.assertEqual(VISIBLE_QRS, 2)
        self.assertEqual(sum(value is not None for value in renderer.timestamps), 2)
        self.assertEqual(renderer.metadata()["visible_qrs"], 2)

    def test_renderer_respects_selected_visible_qr_count(self):
        self.qt_app = QGuiApplication.instance() or QGuiApplication(["qr-test"])
        renderer = QRClockRenderer(1920, 1080, visible_qrs=6, grid_qrs=12)
        for index in range(14):
            renderer.render_next(10_000_000_000 + index * 20_000_000, index)
        self.assertEqual(sum(value is not None for value in renderer.timestamps), 6)
        self.assertEqual(renderer.metadata()["visible_qrs"], 6)
        self.assertEqual(renderer.metadata()["grid_qrs"], 12)
        self.assertEqual(renderer.metadata()["grid_columns"], 6)

    def test_qt_renderer_retains_unchanged_cells_and_clears_expired_cell(self):
        self.qt_app = QGuiApplication.instance() or QGuiApplication(["qr-test"])
        renderer = QRClockRenderer(960, 540)
        canvas = QImage(960, 540, QImage.Format.Format_RGB32)

        renderer.render_next(10_000_000_000, 0)
        painter = QPainter(canvas)
        renderer.paint(painter)
        painter.end()
        first_qr = canvas.copy(renderer.qr_rects[0])

        renderer.render_next(10_020_000_000, 1)
        painter = QPainter(canvas)
        renderer.paint(painter)
        painter.end()

        self.assertEqual(canvas.copy(renderer.qr_rects[0]), first_qr)
        self.assertEqual(
            canvas.pixelColor(renderer.underlines[0].center()),
            QT_BACKGROUND,
        )
        self.assertEqual(
            canvas.pixelColor(renderer.underlines[1].center()),
            QT_FOREGROUND,
        )

        renderer.render_next(10_040_000_000, 2)
        painter = QPainter(canvas)
        renderer.paint(painter)
        painter.end()
        self.assertEqual(
            canvas.pixelColor(renderer.qr_rects[0].center()),
            QT_BACKGROUND,
        )

    def test_next_swap_prediction_uses_last_swap_and_skips_elapsed_periods(self):
        monitor = SwapTimingMonitor(60.0)
        period = monitor.period_ns
        first_paint = 10_000_000_000
        self.assertEqual(monitor.predict_next_swap(first_paint), first_paint + period)

        monitor.observe(first_paint, first_paint + 1_000_000, first_paint + period)
        self.assertEqual(
            monitor.predict_next_swap(first_paint + period + 2_000_000),
            first_paint + 2 * period,
        )
        self.assertEqual(
            monitor.predict_next_swap(first_paint + 3 * period + 2_000_000),
            first_paint + 4 * period,
        )

    def test_qr_areas_are_shifted_away_from_timestamp_labels(self):
        self.qt_app = QGuiApplication.instance() or QGuiApplication(["qr-test"])
        renderer = QRClockRenderer(1920, 1080)
        for area, underline, qr in zip(
            renderer.areas,
            renderer.underlines,
            renderer.qr_rects,
        ):
            self.assertGreater(qr.top(), underline.bottom())
            self.assertGreaterEqual(qr.left(), area.left())
            self.assertLessEqual(qr.right(), area.right())
            self.assertLessEqual(qr.bottom(), area.bottom())

    def test_pygame_renderer_matches_variable_qr_grid_behavior(self):
        target = pygame.Surface((1920, 1080))
        renderer = PygameQRClockRenderer(
            target,
            visible_qrs=6,
            grid_qrs=12,
        )

        for index in range(14):
            renderer.render_next(10_000_000_000 + index * 20_000_000, index)

        self.assertEqual(sum(value is not None for value in renderer.timestamps), 6)
        self.assertEqual(renderer.display_indices[1], 13)
        self.assertEqual(renderer.metadata()["display_backend"], "pygame-sdl")
        self.assertEqual(renderer.metadata()["grid_qrs"], 12)
        self.assertEqual(renderer.metadata()["grid_columns"], 6)
        self.assertEqual(renderer.metadata()["visible_qrs"], 6)
        for area, underline, qr in zip(
            renderer.areas,
            renderer.underlines,
            renderer.qr_rects,
        ):
            self.assertGreater(qr.top, underline.bottom)
            self.assertTrue(area.contains(qr))

    def test_pygame_surface_qr_matches_module_drawing(self):
        pygame.font.init()
        surface_target = pygame.Surface((960, 540), depth=32)
        module_target = pygame.Surface((960, 540), depth=32)
        surface_renderer = PygameQRClockRenderer(surface_target)
        module_renderer = PygameQRClockRenderer(module_target)
        qr_rects_id = id(surface_renderer.qr_rects)

        surface_renderer.render_next(
            10_000_000_000,
            0,
            qr_draw_mode="surface",
        )
        module_renderer.render_next(
            10_000_000_000,
            0,
            qr_draw_mode="modules",
        )

        self.assertEqual(
            pygame.image.tobytes(surface_target, "RGB"),
            pygame.image.tobytes(module_target, "RGB"),
        )
        self.assertEqual(id(surface_renderer.qr_rects), qr_rects_id)

    def test_display_journals_stream_rows_without_retaining_frame_history(self):
        for name, journal_class in (
            ("pygame", PygameDisplayJournal),
            ("qt", QtDisplayJournal),
        ):
            with self.subTest(display=name), TemporaryDirectory() as temporary:
                path = Path(temporary) / "display_timestamps.jsonl"
                journal = journal_class(path, {"display_backend": name})
                journal.append(0, {
                    "flip_return_ns": 10_000_000_000,
                    "skipped_periods": 1,
                    "irregular_interval": True,
                    "late_submit": False,
                })
                journal.append(1, {
                    "flip_return_ns": 10_010_000_000,
                    "skipped_periods": 0,
                    "irregular_interval": False,
                    "late_submit": True,
                })
                journal.close()

                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertFalse(hasattr(journal, "frames"))
                self.assertEqual(journal.frame_count, 2)
                self.assertEqual(rows[-1], {
                    "kind": "summary",
                    "frames": 2,
                    "missed_period_candidates": 1,
                    "irregular_intervals": 1,
                    "late_submissions": 1,
                })

    def test_pygame_pacer_keeps_predicted_marker_separate_from_render_timing(self):
        anchor = 10_000_000_000
        pacer = FramePacer(anchor, 60.0)
        paint_start = anchor + 2_000_000
        submit = paint_start + 1_000_000
        flip_return = pacer.deadline_ns + 100_000

        timing = pacer.observe(
            pacer.deadline_ns,
            submit,
            flip_return,
            0,
            paint_start_ns=paint_start,
        )

        self.assertEqual(timing["render_ns"], 1_000_000)
        self.assertEqual(timing["marker_to_flip_ns"], 100_000)
        self.assertFalse(timing["late_submit"])
        self.assertEqual(
            pacer.predict_next_flip(flip_return + 2_000_000),
            flip_return + pacer.period_ns,
        )


class RecordingTests(unittest.TestCase):
    def fixture(
        self,
        folder: Path,
        count=5,
        legacy=False,
        grid_qrs=4,
        visible_qrs=2,
        period=16_666_667,
    ):
        rows = display_rows(count, period)
        rows[0].update({"grid_qrs": grid_qrs, "visible_qrs": visible_qrs})
        for index, row in enumerate(rows[1:]):
            row["cell"] = index % grid_qrs
            row["corner"] = row["cell"]
        (folder / "display_timestamps.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        frame = np.zeros((100, 100, 3), np.uint8)
        image_folder = folder if legacy else folder / "images"
        image_folder.mkdir(exist_ok=True)
        image_name = "camera_000001.jpg" if legacy else "images/camera_000001.jpg"
        cv2.imwrite(str(folder / image_name), frame)
        (folder / "camera_timestamps.jsonl").write_text(json.dumps({
            "frame": image_name, "stream_epoch": 1,
            "pts_ns": 100_000_000, "reference_ntp_ns": 1_700_000_000_000_000_000,
        }) + "\n", encoding="utf-8")
        (folder / "camera_timing_session.json").write_text(json.dumps({
            "epochs": [{"stream_epoch": 1, "pipeline_zero_monotonic_ns": 9_950_000_000}]
        }), encoding="utf-8")
        return rows

    def four_value_model(self, folder: Path, count=5):
        rows = self.fixture(folder, count)
        frames = rows[1:]
        selected = (frames[4], frames[1], frames[2], frames[3])
        boxes = ([10, 10, 30, 30], [60, 10, 80, 30],
                 [60, 60, 80, 80], [10, 60, 30, 80])
        reader = FakeReader([timestamp_payload(row["marker_ns"]) for row in selected], boxes)
        return RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=reader), frames

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


class QuantitativeVerdictTests(unittest.TestCase):
    def test_report_separates_readability_and_preserves_generation_alternatives(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            frames = []
            pts_ns = 0
            for index in range(60):
                step_ms = 20.0 if index % 3 == 1 else 40.0
                pts_ns += int(step_ms * 1e6)
                midpoint = 77.0 if step_ms == 20.0 else 87.0
                partial = index % 4 == 0
                frames.append({
                    "frame_number": index + 1,
                    "filename": f"camera_{index + 1:06d}.jpg",
                    "validation": "accepted_clean",
                    "timing_status": "Clean",
                    "pts_ns": pts_ns,
                    "pts_minus_latest_qr_ms": midpoint,
                    "offset_interval_lower_ms": midpoint - 5.0,
                    "offset_interval_upper_ms": midpoint + 5.0,
                    "matched_readable_qrs": 8 if partial else 10,
                    "grid_qrs": 10,
                    "qr_values_ms": [
                        None if partial and cell in (1, 4) else f"qr-{cell}"
                        for cell in range(10)
                    ],
                    "latest_cell": index % 10,
                    "latest_cell_name": f"Cell {index % 10}",
                })
            source = {
                "recording_directory": "/recordings/sample",
                "grid": {"qr_count": 10},
                "frames": frames,
            }
            (output / "calibration_analysis.json").write_text(
                json.dumps(source), encoding="utf-8"
            )
            with (output / "calibration_frames.csv").open(
                "w", encoding="utf-8", newline=""
            ) as destination:
                writer = csv.DictWriter(destination, fieldnames=frames[0].keys())
                writer.writeheader()
                writer.writerows(frames)

            def graph_file(path, *_args, **_kwargs):
                path.write_bytes(b"PNG")

            with (
                patch("calibration.quantitative_analysis._write_timeline_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_residual_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_histogram", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_readability_graph", side_effect=graph_file),
            ):
                report = analyze_output_directory(output)

            readability = report["readability_analysis"]
            self.assertEqual(readability["fully_readable_frames"], 45)
            self.assertEqual(readability["partial_readability_frames"], 15)
            self.assertEqual(len(readability["selected_newest_by_cell"]), 10)
            self.assertEqual(len(readability["per_cell_readability"]), 10)
            self.assertEqual(readability["per_cell_readability"][1]["readable_pct"], 75.0)
            self.assertIn("pts_cadence_state", report["strategies"])
            self.assertIn("pts_history_selected_linear", report["strategies"])
            self.assertEqual(
                report["strategies"]["pts_history6_linear"]["parameters"]["ridge"],
                1e-3,
            )
            with (output / "calibration_strategy_predictions.csv").open(
                encoding="utf-8", newline=""
            ) as source_file:
                prediction = next(csv.DictReader(source_file))
            self.assertEqual(prediction["readability_class"], "partial")
            self.assertNotEqual(
                prediction["newer_generation_1_interval_lower_ms"], ""
            )

    def test_saved_analysis_defaults_to_png_and_optionally_saves_svg_graphs(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            frames = []
            for index in range(45):
                clean = index != 8
                frames.append({
                    "frame_number": index + 1,
                    "filename": f"camera_{index + 1:06d}.jpg",
                    "validation": "accepted_clean" if clean else "accepted_timing_suspect",
                    "timing_status": "Clean" if clean else "Timing suspect",
                    "pts_ns": index * (20_000_000 if index % 2 else 40_000_000),
                    "pts_minus_latest_qr_ms": 96.0 + index % 4,
                })
            source = {
                "recording_directory": "/recordings/sample",
                "processed": len(frames),
                "display": {
                    "late_submissions": 2,
                    "irregular_intervals": 1,
                    "missed_period_candidates": 0,
                },
                "frames": frames,
            }
            (output / "calibration_analysis.json").write_text(
                json.dumps(source), encoding="utf-8"
            )
            with (output / "calibration_frames.csv").open(
                "w", encoding="utf-8", newline=""
            ) as destination:
                writer = csv.DictWriter(destination, fieldnames=frames[0].keys())
                writer.writeheader()
                writer.writerows(frames)

            def graph_file(path, *_args, save_svg=False, **_kwargs):
                path.write_bytes(b"PNG")
                if save_svg:
                    path.with_suffix(".svg").write_text("<svg/>", encoding="utf-8")

            with (
                patch("calibration.quantitative_analysis._write_timeline_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_residual_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_histogram", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_readability_graph", side_effect=graph_file),
            ):
                report = analyze_output_directory(output)

            self.assertEqual(report["data_quality"]["clean_frames"], 44)
            self.assertEqual(report["data_quality"]["timing_suspect_frames"], 1)
            self.assertAlmostEqual(
                report["verdict"]["current_correction_ms"],
                87.348,
                places=3,
            )
            self.assertTrue((output / "calibration_verdict.md").is_file())
            self.assertEqual(report["graph_formats"], ["png"])
            graph_stems = (
                "calibration_offset_timeline",
                "calibration_residual_cdf",
                "calibration_offset_histogram",
                "calibration_fixed_residual_histogram",
                "calibration_pts_residual_histogram",
                "calibration_readability_diagnostics",
            )
            for stem in graph_stems:
                self.assertTrue((output / f"{stem}.png").is_file())
                self.assertFalse((output / f"{stem}.svg").exists())
                self.assertNotIn(f"{stem}.svg", report["output_files"])

            with (
                patch("calibration.quantitative_analysis._write_timeline_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_residual_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_histogram", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_readability_graph", side_effect=graph_file),
            ):
                svg_report = analyze_output_directory(output, save_svg=True)

            self.assertEqual(svg_report["graph_formats"], ["png", "svg"])
            for stem in graph_stems:
                filename = f"{stem}.svg"
                self.assertIn("<svg", (output / filename).read_text())
                self.assertIn(filename, svg_report["output_files"])

    def test_root_launcher_passes_recording_and_saved_qr_analysis_to_distance_model(self):
        recording = Path("/recordings/sample")
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "distance_analysis.json").write_text("{}", encoding="utf-8")
            with (
                patch.object(recording_launcher, "matplotlib_environment", return_value={}),
                patch.object(recording_launcher.subprocess, "run") as run,
            ):
                recording_launcher._run_distance_analysis(recording, output)

            command = run.call_args.args[0]
            self.assertEqual(
                command[:3],
                [recording_launcher.sys.executable, "-m", "calibration.distance_analysis"],
            )
            self.assertIn(str(recording), command)
            self.assertIn(str(output), command)

    def test_root_launcher_builds_cross_recording_report_for_sibling_analyses(self):
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first = parent / "01_calibration_analysis"
            second = parent / "02_calibration_analysis"
            destination = parent / "calibration_cross_recording_analysis"
            destination.mkdir()
            (destination / "calibration_cross_recording.json").write_text(
                json.dumps({"output_directory": str(destination)}),
                encoding="utf-8",
            )
            with (
                patch.object(
                    recording_launcher,
                    "discover_analysis_directories",
                    return_value=[first, second],
                ),
                patch.object(recording_launcher, "matplotlib_environment", return_value={}),
                patch.object(recording_launcher.subprocess, "run") as run,
            ):
                report = recording_launcher._run_cross_recording_analysis(first)

            self.assertEqual(report["output_directory"], str(destination))
            command = run.call_args.args[0]
            self.assertEqual(
                command[:3],
                [recording_launcher.sys.executable, "-m", "calibration.cross_recording_analysis"],
            )
            self.assertIn(str(first), command)
            self.assertIn(str(second), command)
            self.assertIn(str(destination), command)

    def test_root_launcher_runs_distance_analysis_after_the_window(self):
        recording = Path("/recordings/sample")
        output = Path("/recordings/sample_analysis")
        with (
            patch.object(recording_launcher, "run_recording_display", return_value=output) as display,
            patch.object(recording_launcher, "_run_distance_analysis", return_value={
                "output_directory": str(output)
            }) as analyze,
            patch.object(recording_launcher, "_run_quantitative_analysis", return_value={
                "output_directory": str(output)
            }) as quantitative,
            patch.object(recording_launcher, "_run_cross_recording_analysis", return_value=None) as cross,
            patch("sys.argv", ["analyze_calibration_recording.py", str(recording)]),
        ):
            recording_launcher.main()
        display.assert_called_once()
        analyze.assert_called_once_with(recording, output)
        quantitative.assert_called_once_with(output)
        cross.assert_called_once_with(output)


if __name__ == "__main__":
    unittest.main()
