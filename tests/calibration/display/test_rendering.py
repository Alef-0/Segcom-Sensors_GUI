"""Renderer, display journal, and frame pacing tests."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pygame
from PySide6.QtGui import QGuiApplication, QImage, QPainter

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
from calibration.qr import qr_matrix


class TestDisplayRendering(unittest.TestCase):
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

    def test_qt_renderer_keeps_unscaled_pixmaps_and_records_fixed_mask(self):
        self.qt_app = QGuiApplication.instance() or QGuiApplication(["qr-test"])
        renderer = QRClockRenderer(1920, 1080, qr_mask_pattern=3)
        timestamp_ns = 10_000_000_000

        renderer.render_next(timestamp_ns, 0)

        self.assertEqual(
            renderer.images[timestamp_ns].width(),
            renderer.matrices[timestamp_ns].shape[0],
        )
        self.assertEqual(renderer.metadata()["qr_mask_pattern"], 3)
        self.assertEqual(renderer.metadata()["qr_mask_selection"], "fixed")

    def test_qt_prepared_qr_is_reused_by_deadline_render(self):
        self.qt_app = QGuiApplication.instance() or QGuiApplication(["qr-test"])
        renderer = QRClockRenderer(960, 540, qr_mask_pattern=3)
        timestamp_ns = 10_000_000_000

        with patch("calibration.display_qt.qr_matrix", wraps=qr_matrix) as build_matrix:
            renderer.prepare_qr(timestamp_ns)
            renderer.render_next(timestamp_ns, 0)
            self.assertEqual(build_matrix.call_count, 1)
            self.assertEqual(build_matrix.call_args.kwargs["mask_pattern"], 3)

            renderer.render_next(timestamp_ns + 20_000_000, 1)
            self.assertEqual(build_matrix.call_count, 2)

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

    def test_qt_unscaled_pixmap_matches_module_drawing(self):
        self.qt_app = QGuiApplication.instance() or QGuiApplication(["qr-test"])
        pixmap_canvas = QImage(960, 540, QImage.Format.Format_RGB32)
        module_canvas = QImage(960, 540, QImage.Format.Format_RGB32)
        pixmap_renderer = QRClockRenderer(960, 540)
        module_renderer = QRClockRenderer(960, 540)

        pixmap_renderer.render_next(10_000_000_000, 0)
        module_renderer.render_next(10_000_000_000, 0, cache_pixmap=False)
        pixmap_painter = QPainter(pixmap_canvas)
        pixmap_renderer.paint(pixmap_painter)
        pixmap_painter.end()
        module_painter = QPainter(module_canvas)
        module_renderer.paint(module_painter, cached_pixmaps=False)
        module_painter.end()

        self.assertEqual(pixmap_canvas, module_canvas)

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

    def test_pygame_prepared_qr_is_reused_by_deadline_render(self):
        pygame.font.init()
        renderer = PygameQRClockRenderer(
            pygame.Surface((960, 540), depth=32),
            qr_mask_pattern=2,
        )
        timestamp_ns = 10_000_000_000

        with patch("calibration.display.qr_matrix", wraps=qr_matrix) as build_matrix:
            renderer.prepare_qr(timestamp_ns)
            renderer.render_next(timestamp_ns, 0)
            self.assertEqual(build_matrix.call_count, 1)
            self.assertEqual(build_matrix.call_args.kwargs["mask_pattern"], 2)

            renderer.render_next(timestamp_ns + 20_000_000, 1)
            self.assertEqual(build_matrix.call_count, 2)

        self.assertEqual(renderer.metadata()["qr_mask_pattern"], 2)

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

    def test_pygame_pacer_exposes_configurable_spin_wait(self):
        pacer = FramePacer(10_000_000_000, 100.0, spin_wait_us=250)
        self.assertEqual(pacer.spin_wait_ns, 250_000)
        with self.assertRaisesRegex(ValueError, "Spin wait"):
            FramePacer(10_000_000_000, 100.0, spin_wait_us=-1)
