"""Qt6/OpenGL QR clock for camera-delay calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

try:
    from PySide6.QtCore import QRect, QTimer, Qt
    from PySide6.QtGui import (
        QColor,
        QFont,
        QFontMetrics,
        QGuiApplication,
        QImage,
        QPainter,
        QPixmap,
        QSurfaceFormat,
    )
    from PySide6.QtOpenGL import QOpenGLWindow
except ImportError as error:
    raise SystemExit(
        "PySide6 is required for the Qt display. Install it with:\n"
        "  python3 -m pip install PySide6"
    ) from error

try:
    from calibration.qr import (
        GRID_LAYOUTS,
        QUIET_ZONE_MODULES,
        grid_bounds,
        grid_cell_names,
        grid_positions,
        grid_shape,
        qr_matrix,
        timestamp_payload,
    )
except ModuleNotFoundError:
    # Allows testing display_qt.py next to qr.py without installing the package.
    from qr import (
        GRID_LAYOUTS,
        QUIET_ZONE_MODULES,
        grid_bounds,
        grid_cell_names,
        grid_positions,
        grid_shape,
        qr_matrix,
        timestamp_payload,
    )


DISPLAY_JOURNAL_NAME = "display_timestamps.jsonl"
DISPLAY_FORMAT = "segcom-qr-display-qt-v3"
WINDOW_NAME = "QR Calibration Clock — Qt/OpenGL"
BACKGROUND = QColor(50, 50, 50)
FOREGROUND = QColor(255, 255, 255)
BLACK = QColor(0, 0, 0)
DEFAULT_GRID_QRS = 4
VISIBLE_QRS = 2
UNDERLINE_HEIGHT = 4
DEFAULT_REFRESH_HZ = 60.0
TIMESTAMP_MODES = ("paint-start", "predicted-flip")
DEFAULT_TIMESTAMP_MODE = "predicted-flip"


def format_timestamp(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return f"{seconds:,}".replace(",", " ") + f".{nanoseconds // 1_000_000:03d}"


def grid_areas(width: int, height: int, grid_qrs: int) -> tuple[QRect, ...]:
    return tuple(
        QRect(left, top, right - left, bottom - top)
        for left, top, right, bottom in grid_bounds(width, height, grid_qrs)
    )


class QRClockRenderer:
    """Keep recent QR markers and update only changed retained-buffer regions."""

    def __init__(
        self,
        width: int,
        height: int,
        visible_qrs: int = VISIBLE_QRS,
        grid_qrs: int = DEFAULT_GRID_QRS,
    ):
        grid_shape(grid_qrs)
        if not 1 <= visible_qrs <= grid_qrs:
            raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")
        self.grid_qrs = grid_qrs
        self.visible_qrs = visible_qrs
        self.cell_names = grid_cell_names(grid_qrs)
        self.timestamps: list[int | None] = [None] * grid_qrs
        self.display_indices: list[int | None] = [None] * grid_qrs
        self.matrices: dict[int, object] = {}
        self.images: dict[int, QPixmap] = {}
        self._dirty_cells: set[int] = set()
        self._dirty_underlines: set[int] = set()
        self._force_full_redraw = True
        self.next_cell = 0
        self.newest_cell: int | None = None
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        if width < 320 or height < 240:
            raise ValueError("Canvas must be at least 320 by 240 pixels")

        self.size = (width, height)
        rows, columns = grid_shape(self.grid_qrs)
        self.areas = grid_areas(width, height, self.grid_qrs)

        font_size = max(10, min(width // (columns * 14), height // (rows * 10)))
        self.font = QFont()
        self.font.setPixelSize(font_size)
        self.font_metrics = QFontMetrics(self.font)

        self.qr_rects = tuple(
            self._qr_rect(area)
            for area in self.areas
        )
        self.underlines = tuple(
            self._underline(area)
            for area in self.areas
        )
        self.images = {
            timestamp_ns: self._qr_image(self.qr_rects[cell], self.matrices[timestamp_ns])
            for cell, timestamp_ns in enumerate(self.timestamps)
            if timestamp_ns is not None
        }
        self._force_full_redraw = True
        self._dirty_cells.update(range(self.grid_qrs))

    def _qr_rect(self, area: QRect) -> QRect:
        margin = max(4, min(12, area.width() // 24, area.height() // 24))
        text_height = self.font_metrics.height() + 4
        label_gap = max(4, margin)
        content_top = area.y() + margin + text_height + label_gap + UNDERLINE_HEIGHT
        content_bottom = area.bottom() - margin
        available_height = max(1, content_bottom - content_top + 1)
        side = max(25, min(area.width() - 2 * margin, available_height))

        rect = QRect(0, 0, side, side)
        rect.moveCenter(area.center())
        rect.moveTop(content_top + max(0, (available_height - side) // 2))
        return rect

    def _underline(self, area: QRect) -> QRect:
        width = max(24, area.width() // 3)
        label_height = self.font_metrics.height() + 4
        margin = max(4, min(12, area.width() // 24, area.height() // 24))
        y = area.y() + margin + label_height + 2
        return QRect(area.center().x() - width // 2, y, width, UNDERLINE_HEIGHT)

    def render_next(
        self,
        timestamp_ns: int,
        display_index: int,
        *,
        cache_pixmap: bool = True,
    ) -> int:
        cell = self.next_cell
        expired = (cell - self.visible_qrs) % self.grid_qrs

        if self.newest_cell is not None:
            self._dirty_underlines.add(self.newest_cell)

        expired_timestamp = self.timestamps[expired]
        if expired_timestamp is not None:
            self.matrices.pop(expired_timestamp, None)
            self.images.pop(expired_timestamp, None)
            self.timestamps[expired] = None
            self.display_indices[expired] = None
            self._dirty_cells.add(expired)

        self.timestamps[cell] = timestamp_ns
        self.display_indices[cell] = display_index
        matrix = qr_matrix(timestamp_payload(timestamp_ns))
        self.matrices[timestamp_ns] = matrix
        if cache_pixmap:
            self.images[timestamp_ns] = self._qr_image(self.qr_rects[cell], matrix)
        self._dirty_cells.add(cell)

        self.newest_cell = cell
        self.next_cell = (cell + 1) % self.grid_qrs
        return cell

    @staticmethod
    def _qr_image(rect: QRect, matrix) -> QPixmap:
        modules = matrix.shape[0]
        scale = max(1, min(rect.width(), rect.height()) // modules)
        pixels = (255 - matrix * 255).repeat(scale, axis=0).repeat(scale, axis=1)
        image = QImage(
            pixels.data,
            pixels.shape[1],
            pixels.shape[0],
            pixels.strides[0],
            QImage.Format.Format_Grayscale8,
        )
        # Detach from the temporary NumPy storage before returning.
        return QPixmap.fromImage(image.copy())

    @staticmethod
    def _draw_qr(painter: QPainter, rect: QRect, image: QPixmap) -> QRect:
        bounds = QRect(0, 0, image.width(), image.height())
        bounds.moveCenter(rect.center())
        painter.drawPixmap(bounds, image)
        return bounds

    @staticmethod
    def _draw_qr_modules(painter: QPainter, rect: QRect, matrix) -> QRect:
        modules = matrix.shape[0]
        scale = max(1, min(rect.width(), rect.height()) // modules)
        bounds = QRect(0, 0, modules * scale, modules * scale)
        bounds.moveCenter(rect.center())
        painter.fillRect(bounds, FOREGROUND)
        for row, column in zip(*matrix.nonzero()):
            painter.fillRect(
                bounds.x() + int(column) * scale,
                bounds.y() + int(row) * scale,
                scale,
                scale,
                BLACK,
            )
        return bounds

    def _paint_cell(
        self,
        painter: QPainter,
        cell: int,
        *,
        cached_pixmaps: bool,
    ) -> None:
        timestamp_ns = self.timestamps[cell]
        if timestamp_ns is None:
            return

        if cached_pixmaps:
            self._draw_qr(painter, self.qr_rects[cell], self.images[timestamp_ns])
        else:
            self._draw_qr_modules(
                painter,
                self.qr_rects[cell],
                self.matrices[timestamp_ns],
            )

        display_index = self.display_indices[cell]
        label = f"#{display_index}  {format_timestamp(timestamp_ns)}"
        text_height = self.font_metrics.height() + 4
        area = self.areas[cell]
        margin = max(4, min(12, area.width() // 24, area.height() // 24))
        text_y = area.y() + margin
        text_rect = QRect(area.x(), text_y, area.width(), text_height)
        painter.setPen(FOREGROUND)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            label,
        )

    def paint(
        self,
        painter: QPainter,
        *,
        force_full_redraw: bool = False,
        cached_pixmaps: bool = True,
    ) -> None:
        full_redraw = force_full_redraw or self._force_full_redraw
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setFont(self.font)

        if full_redraw:
            painter.fillRect(0, 0, self.size[0], self.size[1], BACKGROUND)
            cells_to_paint = range(self.grid_qrs)
        else:
            for cell in self._dirty_underlines - self._dirty_cells:
                painter.fillRect(self.underlines[cell], BACKGROUND)
            for cell in self._dirty_cells:
                painter.fillRect(self.areas[cell], BACKGROUND)
            cells_to_paint = sorted(self._dirty_cells)

        for cell in cells_to_paint:
            self._paint_cell(painter, cell, cached_pixmaps=cached_pixmaps)

        if self.newest_cell is not None:
            painter.fillRect(self.underlines[self.newest_cell], FOREGROUND)

        self._dirty_cells.clear()
        self._dirty_underlines.clear()
        self._force_full_redraw = False

    def metadata(self) -> dict:
        return {
            "size": list(self.size),
            "code_format": "qr",
            "qr_border_modules": QUIET_ZONE_MODULES,
            "grid_qrs": self.grid_qrs,
            "grid_rows": grid_shape(self.grid_qrs)[0],
            "grid_columns": grid_shape(self.grid_qrs)[1],
            "cell_order": list(self.cell_names),
            "cell_positions": [list(position) for position in grid_positions(self.grid_qrs)],
            "visible_qrs": self.visible_qrs,
            "indicator_style": "underline",
            "indicator_width": UNDERLINE_HEIGHT,
            "corner_order": (
                ["top-left", "top-right", "bottom-right", "bottom-left"]
                if self.grid_qrs == 4 else None
            ),
            "timestamp_semantics": (
                "marker_ns is the timestamp encoded in the QR; session metadata "
                "states whether it is paint-start or predicted-flip time"
            ),
            "frame_index_semantics": (
                "display index is shown as text and recorded in the journal; "
                "the QR payload remains timestamp-only for decoder compatibility"
            ),
        }


class SwapTimingMonitor:
    """Analyze presentation cadence using Qt's frameSwapped signal."""

    def __init__(self, refresh_hz: float):
        if not math.isfinite(refresh_hz) or not 1 <= refresh_hz <= 1000:
            raise ValueError("Refresh rate must be between 1 and 1000 Hz")

        self.refresh_hz = refresh_hz
        self.period_ns = round(1_000_000_000 / refresh_hz)
        self.last_swap_ns: int | None = None

    def reset(self) -> None:
        self.last_swap_ns = None

    def predict_next_swap(self, paint_start_ns: int) -> int:
        """Predict the first nominal refresh boundary after painting begins."""
        if self.last_swap_ns is None:
            return paint_start_ns + self.period_ns

        predicted_ns = self.last_swap_ns + self.period_ns
        if predicted_ns <= paint_start_ns:
            elapsed_periods = (paint_start_ns - predicted_ns) // self.period_ns + 1
            predicted_ns += elapsed_periods * self.period_ns
        return predicted_ns

    def observe(
        self,
        marker_ns: int,
        submit_ns: int,
        swap_return_ns: int,
        *,
        paint_start_ns: int | None = None,
    ) -> dict:
        if paint_start_ns is None:
            paint_start_ns = marker_ns
        interval_ns = (
            None
            if self.last_swap_ns is None
            else swap_return_ns - self.last_swap_ns
        )

        irregular = (
            interval_ns is not None
            and not 0.75 * self.period_ns <= interval_ns <= 1.25 * self.period_ns
        )

        skipped_periods = 0
        if interval_ns is not None and interval_ns > 1.5 * self.period_ns:
            skipped_periods = max(0, round(interval_ns / self.period_ns) - 1)

        self.last_swap_ns = swap_return_ns

        return {
            "marker_ns": marker_ns,
            "paint_start_ns": paint_start_ns,
            "submit_ns": submit_ns,
            "flip_return_ns": swap_return_ns,
            "frame_period_ns": self.period_ns,
            "interval_ns": interval_ns,
            "render_ns": max(0, submit_ns - paint_start_ns),
            "swap_wait_ns": max(0, swap_return_ns - submit_ns),
            "marker_to_flip_ns": swap_return_ns - marker_ns,
            "late_submit": False,
            "skipped_before_render": 0,
            "missed_after_submit": skipped_periods,
            "skipped_periods": skipped_periods,
            "irregular_interval": irregular,
        }


def timing_issues(row: dict) -> list[str]:
    issues = []
    if row.get("skipped_periods"):
        issues.append("missed_period_candidates")
    # Kept for compatibility with journals created by the former Pygame clock.
    if row.get("late_submit"):
        issues.append("late_submission")
    if row.get("irregular_interval"):
        issues.append("irregular_interval")
    if row.get("resumed_after_pause"):
        issues.append("resumed_after_pause")
    return issues


class DisplayJournal:
    def __init__(self, path: str | Path | None, metadata: dict):
        self.file = (
            None
            if path is None
            else Path(path).open("x", encoding="utf-8", buffering=65536)
        )
        self.frames: list[dict] = []
        self.last_flush_ns = time.monotonic_ns()
        self._closed = False

        self._write({"kind": "session", "format": DISPLAY_FORMAT, **metadata})
        if self.file:
            self.file.flush()

    def _write(self, value: dict) -> None:
        if self.file:
            self.file.write(json.dumps(value, separators=(",", ":")) + "\n")

    def append(self, cell: int, timing: dict) -> None:
        row = {
            "kind": "frame",
            "index": len(self.frames),
            "display_frame": len(self.frames),
            "cell": cell,
            # Retained so older readers can still interpret four-cell journals.
            "corner": cell,
            **timing,
        }
        self.frames.append(row)
        self._write(row)

        issues = timing_issues(row)
        if issues:
            self._write({
                "kind": "timing_event",
                "detected_at_monotonic_ns": row["flip_return_ns"],
                "affected_display_indices": [row["index"]],
                "issues": issues,
            })

        now = time.monotonic_ns()
        if self.file and now - self.last_flush_ns >= 1_000_000_000:
            self.file.flush()
            self.last_flush_ns = now

    def pause(self, paused: bool, timestamp_ns: int) -> None:
        self._write({
            "kind": "pause",
            "paused": paused,
            "monotonic_ns": timestamp_ns,
            "last_frame_index": len(self.frames) - 1,
        })
        if self.file:
            self.file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        counts = {
            "missed_period_candidates": sum(
                row.get("skipped_periods", 0) for row in self.frames
            ),
            "irregular_intervals": sum(
                bool(row.get("irregular_interval")) for row in self.frames
            ),
            "late_submissions": sum(
                bool(row.get("late_submit")) for row in self.frames
            ),
        }
        self._write({"kind": "summary", "frames": len(self.frames), **counts})

        if self.file:
            self.file.close()

        print(
            f"[CALIBRATION] Presented {len(self.frames)} QR markers; "
            f"{counts['missed_period_candidates']} missed-period candidate(s), "
            f"{counts['irregular_intervals']} irregular interval(s), "
            f"{counts['late_submissions']} late predicted submission(s).",
            flush=True,
        )


class QRClockWindow(QOpenGLWindow):
    """VSync-driven QR clock using a directly rendered OpenGL window."""

    def __init__(
        self,
        *,
        expected_refresh_hz: float,
        journal_path: str | None,
        screen_metadata: dict,
        visible_qrs: int,
        grid_qrs: int,
        timestamp_mode: str,
    ):
        super().__init__(QOpenGLWindow.UpdateBehavior.PartialUpdateBlit)

        self.expected_refresh_hz = expected_refresh_hz
        self.journal_path = journal_path
        self.screen_metadata = screen_metadata
        self.visible_qrs = visible_qrs
        self.grid_qrs = grid_qrs
        if timestamp_mode not in TIMESTAMP_MODES:
            choices = ", ".join(TIMESTAMP_MODES)
            raise ValueError(f"Timestamp mode must be one of: {choices}")
        self.timestamp_mode = timestamp_mode

        self.renderer: QRClockRenderer | None = None
        self.monitor = SwapTimingMonitor(expected_refresh_hz)
        self.journal: DisplayJournal | None = None

        self.paused = False
        self.resumed_after_pause = False
        self.pending_frame: dict | None = None
        self.next_display_index = 0

        self.setTitle(WINDOW_NAME)
        self.frameSwapped.connect(self._frame_swapped)

    def initializeGL(self) -> None:
        self.renderer = QRClockRenderer(
            self.width(),
            self.height(),
            self.visible_qrs,
            self.grid_qrs,
        )

        context = self.context()
        actual_format = context.format() if context is not None else self.format()

        metadata = {
            **self.renderer.metadata(),
            **self.screen_metadata,
            "qt_version": self._qt_version(),
            "qt_platform": QGuiApplication.platformName(),
            "requested_swap_interval": 1,
            "actual_swap_interval": actual_format.swapInterval(),
            "swap_behavior": str(actual_format.swapBehavior()),
            "renderable_type": str(actual_format.renderableType()),
            "opengl_version": [
                actual_format.majorVersion(),
                actual_format.minorVersion(),
            ],
            "opengl_profile": str(actual_format.profile()),
            "update_behavior": "PartialUpdateBlit",
            "timestamp_mode": self.timestamp_mode,
            "timestamp_semantics": (
                "marker_ns and the QR payload predict the next frameSwapped time"
                if self.timestamp_mode == "predicted-flip"
                else "marker_ns and the QR payload contain paint-start time"
            ),
            "prediction_period_ns": self.monitor.period_ns,
            "presentation_semantics": (
                "flip_return_ns is sampled from Qt frameSwapped, emitted after the "
                "potentially blocking buffer swap; it is still not a measurement "
                "of physical panel scanout"
            ),
        }

        self.journal = DisplayJournal(self.journal_path, metadata)

    @staticmethod
    def _qt_version() -> str:
        from PySide6.QtCore import qVersion
        return qVersion()

    def resizeGL(self, width: int, height: int) -> None:
        if self.renderer is not None and width >= 320 and height >= 240:
            self.renderer.resize(width, height)

    def paintGL(self) -> None:
        if self.renderer is None or self.paused:
            return

        # Do not queue another frame until frameSwapped confirms this one completed.
        if self.pending_frame is not None:
            return

        paint_start_ns = time.monotonic_ns()
        marker_ns = (
            self.monitor.predict_next_swap(paint_start_ns)
            if self.timestamp_mode == "predicted-flip"
            else paint_start_ns
        )
        display_index = self.next_display_index
        cell = self.renderer.render_next(marker_ns, display_index)

        painter = QPainter(self)
        try:
            self.renderer.paint(painter)
        finally:
            painter.end()

        submit_ns = time.monotonic_ns()
        self.pending_frame = {
            "cell": cell,
            "display_index": display_index,
            "marker_ns": marker_ns,
            "paint_start_ns": paint_start_ns,
            "submit_ns": submit_ns,
        }

    def _frame_swapped(self) -> None:
        swap_return_ns = time.monotonic_ns()

        if self.pending_frame is None:
            if not self.paused:
                self.update()
            return

        pending = self.pending_frame
        self.pending_frame = None

        timing = self.monitor.observe(
            pending["marker_ns"],
            pending["submit_ns"],
            swap_return_ns,
            paint_start_ns=pending["paint_start_ns"],
        )
        timing["prediction_error_ns"] = (
            swap_return_ns - pending["marker_ns"]
            if self.timestamp_mode == "predicted-flip"
            else None
        )
        timing["predicted_flip_ns"] = (
            pending["marker_ns"]
            if self.timestamp_mode == "predicted-flip"
            else None
        )
        timing["late_submit"] = (
            self.timestamp_mode == "predicted-flip"
            and pending["submit_ns"] > pending["marker_ns"]
        )
        timing["resumed_after_pause"] = self.resumed_after_pause
        self.resumed_after_pause = False

        if self.journal is not None:
            self.journal.append(pending["cell"], timing)

        self.next_display_index += 1

        if not self.paused:
            # Qt documents frameSwapped -> update() as the preferred way to
            # continuously repaint synchronized to vertical refresh.
            self.update()

    def keyPressEvent(self, event) -> None:
        if event.isAutoRepeat():
            return

        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
            return

        if event.key() == Qt.Key.Key_P:
            self.paused = not self.paused

            if self.journal is not None:
                self.journal.pause(self.paused, time.monotonic_ns())

            self.setTitle(
                WINDOW_NAME + (" — paused" if self.paused else "")
            )

            if not self.paused:
                self.monitor.reset()
                self.resumed_after_pause = True
                self.update()
            return

        super().keyPressEvent(event)

    def exposeEvent(self, event) -> None:
        super().exposeEvent(event)
        if self.isExposed() and not self.paused:
            self.update()

    def close_journal(self) -> None:
        if self.journal is not None:
            self.journal.close()


def configure_surface_format() -> None:
    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface_format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    surface_format.setSwapInterval(1)
    surface_format.setDepthBufferSize(0)
    surface_format.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(surface_format)


def describe_screens(app: QGuiApplication) -> None:
    print("Qt platform:", QGuiApplication.platformName())
    for index, screen in enumerate(app.screens()):
        geometry = screen.geometry()
        print(
            f"[{index}] {screen.name()}: "
            f"{geometry.width()}x{geometry.height()} "
            f"@ {screen.refreshRate():.3f} Hz "
            f"at ({geometry.x()}, {geometry.y()})"
        )


def screen_catalog(app: QGuiApplication) -> list[dict]:
    """Return stable, JSON-safe monitor descriptions for the launcher GUI."""
    result = []
    primary = app.primaryScreen()
    for index, screen in enumerate(app.screens()):
        geometry = screen.geometry()
        result.append({
            "index": index,
            "name": screen.name() or f"Monitor {index + 1}",
            "width": geometry.width(),
            "height": geometry.height(),
            "x": geometry.x(),
            "y": geometry.y(),
            "refresh_hz": float(screen.refreshRate()),
            "primary": screen is primary,
        })
    return result


def run_calibration_display(
    stop_event=None,
    *,
    width: int = 1920,
    height: int = 1080,
    windowed: bool = False,
    refresh_hz: float | None = None,
    journal_path: str | None = None,
    screen_index: int = 0,
    visible_qrs: int = VISIBLE_QRS,
    grid_qrs: int = DEFAULT_GRID_QRS,
    timestamp_mode: str = DEFAULT_TIMESTAMP_MODE,
    list_screens: bool = False,
) -> int:
    if width < 320 or height < 240:
        raise ValueError("Canvas must be at least 320 by 240 pixels")
    grid_shape(grid_qrs)
    if not 1 <= visible_qrs <= grid_qrs:
        raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")

    configure_surface_format()

    # Do not let command-line arguments belonging to the parent GUI leak into Qt.
    app = QGuiApplication(["segcom-qr-display"])

    if list_screens:
        describe_screens(app)
        return 0

    screens = app.screens()
    if not 0 <= screen_index < len(screens):
        describe_screens(app)
        raise ValueError(
            f"Invalid --screen {screen_index}; available screens: 0..{len(screens) - 1}"
        )

    screen = screens[screen_index]
    reported_refresh_hz = float(screen.refreshRate())
    expected_refresh_hz = refresh_hz
    if expected_refresh_hz is None:
        expected_refresh_hz = (
            reported_refresh_hz
            if math.isfinite(reported_refresh_hz) and reported_refresh_hz >= 1
            else DEFAULT_REFRESH_HZ
        )

    geometry = screen.geometry()
    screen_metadata = {
        "screen_index": screen_index,
        "screen_name": screen.name(),
        "screen_geometry": [
            geometry.x(),
            geometry.y(),
            geometry.width(),
            geometry.height(),
        ],
        "qt_reported_refresh_hz": reported_refresh_hz,
        "expected_refresh_hz": expected_refresh_hz,
    }

    window = QRClockWindow(
        expected_refresh_hz=expected_refresh_hz,
        journal_path=journal_path,
        screen_metadata=screen_metadata,
        visible_qrs=visible_qrs,
        grid_qrs=grid_qrs,
        timestamp_mode=timestamp_mode,
    )
    window.setScreen(screen)

    if windowed:
        window.resize(width, height)
        x = geometry.x() + max(0, (geometry.width() - width) // 2)
        y = geometry.y() + max(0, (geometry.height() - height) // 2)
        window.setPosition(x, y)
        window.show()
    else:
        # Fullscreen uses the monitor's currently configured native mode.
        # Set 120/144/240 Hz in the desktop/display settings before launch.
        window.setGeometry(geometry)
        window.showFullScreen()

    def request_close(*_args) -> None:
        window.close()

    def poll_stop_event() -> None:
        if stop_event is not None and stop_event.is_set():
            request_close()

    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(poll_stop_event)
    timer.start()

    previous_handlers = {}
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signal_number] = signal.signal(
                signal_number, request_close
            )
        except (OSError, ValueError):
            pass

    try:
        return app.exec()
    finally:
        timer.stop()
        window.close_journal()
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display rotating QR camera-calibration timestamps with Qt6/OpenGL"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument(
        "--refresh-hz",
        type=float,
        help=(
            "Expected refresh rate used for timing diagnostics. "
            "Default: refresh rate reported by Qt for the selected monitor."
        ),
    )
    parser.add_argument("--journal", help="New display timing journal path")
    parser.add_argument(
        "--grid-qrs",
        type=int,
        choices=tuple(GRID_LAYOUTS),
        default=DEFAULT_GRID_QRS,
        help="Total QR cells in the display grid (default: 4)",
    )
    parser.add_argument(
        "--visible-qrs",
        type=int,
        choices=range(1, max(GRID_LAYOUTS) + 1),
        default=VISIBLE_QRS,
        help="Number of recent QR codes left visible (default: 2)",
    )
    parser.add_argument(
        "--screen",
        type=int,
        default=0,
        help="Qt screen index to use (default: 0)",
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=TIMESTAMP_MODES,
        default=DEFAULT_TIMESTAMP_MODE,
        help=(
            "QR timestamp source: predict the next swap by default, or retain "
            "the former paint-start timestamp"
        ),
    )
    parser.add_argument(
        "--list-screens",
        action="store_true",
        help="Print detected monitors and refresh rates, then exit",
    )
    parser.add_argument(
        "--list-screens-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()

    if arguments.list_screens_json:
        configure_surface_format()
        app = QGuiApplication(["segcom-qr-screen-list"])
        print(json.dumps(screen_catalog(app), separators=(",", ":")))
        raise SystemExit(0)

    raise SystemExit(
        run_calibration_display(
            width=arguments.width,
            height=arguments.height,
            windowed=arguments.windowed,
            refresh_hz=arguments.refresh_hz,
            journal_path=arguments.journal,
            screen_index=arguments.screen,
            visible_qrs=arguments.visible_qrs,
            grid_qrs=arguments.grid_qrs,
            timestamp_mode=arguments.timestamp_mode,
            list_screens=arguments.list_screens,
        )
    )


if __name__ == "__main__":
    main()
