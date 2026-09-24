"""VSync-driven Qt QR clock used during camera timing calibration."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time

try:
    from PySide6.QtCore import QRect, QTimer, Qt
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPixmap, QSurfaceFormat
    from PySide6.QtOpenGL import QOpenGLWindow
except ImportError as error:
    raise SystemExit("PySide6 is required for the Qt QR display (install PySide6).") from error

from calibration.display_common import (
    BACKGROUND_RGB, DEFAULT_GRID_QRS as COMMON_DEFAULT_GRID_QRS,
    DEFAULT_REFRESH_HZ, DEFAULT_TIMESTAMP_MODE, DISPLAY_JOURNAL_NAME,
    FOREGROUND_RGB, RECORDING_WAIT_RGB,
    STATUS_STRIP_HEIGHT, TIMESTAMP_MODES, UNDERLINE_HEIGHT, VISIBLE_QRS,
    WINDOW_NAME, DisplayJournal, SwapTimingMonitor, format_timestamp,
    grid_areas as _layout_areas,
)
from calibration.qr import (
    GRID_LAYOUTS, QUIET_ZONE_MODULES, QR_MASK_PATTERNS, grid_cell_names,
    grid_positions, grid_shape, qr_matrix, timestamp_payload,
)

BACKGROUND = QColor(*BACKGROUND_RGB)
FOREGROUND = QColor(*FOREGROUND_RGB)
BLACK = QColor(0, 0, 0)
RECORDING_WAIT_RED = QColor(*RECORDING_WAIT_RGB)
DEFAULT_GRID_QRS = COMMON_DEFAULT_GRID_QRS
WINDOW_NAME = "QR Calibration Clock — Qt/OpenGL"
STATUS_BAR_HEIGHT = STATUS_STRIP_HEIGHT
RECORDING_WAIT_BAR_MAX_HEIGHT_PX = 12


def grid_areas(width: int, height: int, grid_qrs: int) -> tuple[QRect, ...]:
    return tuple(QRect(left, top, right - left, bottom - top)
                 for left, top, right, bottom in _layout_areas(width, height, grid_qrs))


class QRClockRenderer:
    """Own QR state and draw a complete deterministic frame on every paint."""

    def __init__(self, width: int, height: int, visible_qrs: int = VISIBLE_QRS,
                 grid_qrs: int = DEFAULT_GRID_QRS, qr_mask_pattern: int | None = None):
        if width < 320 or height < 240:
            raise ValueError("Canvas must be at least 320 by 240 pixels")
        grid_shape(grid_qrs)
        if not 1 <= visible_qrs <= grid_qrs:
            raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")
        if qr_mask_pattern is not None and qr_mask_pattern not in QR_MASK_PATTERNS:
            raise ValueError("QR mask pattern must be from 0 to 7")
        self.grid_qrs, self.visible_qrs = grid_qrs, visible_qrs
        self.qr_mask_pattern = qr_mask_pattern
        self.cell_names = grid_cell_names(grid_qrs)
        self.timestamps: list[int | None] = [None] * grid_qrs
        self.display_indices: list[int | None] = [None] * grid_qrs
        self.matrices: dict[int, object] = {}
        self.images: dict[int, QPixmap] = {}
        self._prepared_payload: str | None = None
        self._prepared_matrix = None
        self._prepared_image: QPixmap | None = None
        self.next_cell = 0
        self.newest_cell: int | None = None
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        if width < 320 or height < 240:
            raise ValueError("Canvas must be at least 320 by 240 pixels")
        self.size = (width, height)
        rows, columns = grid_shape(self.grid_qrs)
        self.areas = grid_areas(width, height, self.grid_qrs)
        self.recording_wait_bar = QRect(2, 3, width - 4, RECORDING_WAIT_BAR_MAX_HEIGHT_PX)
        font_size = max(10, min(width // (columns * 14), height // (rows * 10)))
        self.font = QFont()
        self.font.setPixelSize(font_size)
        self.font_metrics = QFontMetrics(self.font)
        self.qr_rects = tuple(self._qr_rect(area) for area in self.areas)
        self.underlines = tuple(self._underline(area) for area in self.areas)

    def _qr_rect(self, bounds: QRect) -> QRect:
        margin = max(4, min(12, bounds.width() // 24, bounds.height() // 24))
        top = bounds.top() + margin + self.font_metrics.height() + 8 + UNDERLINE_HEIGHT
        available = max(1, bounds.bottom() - margin - top)
        side = max(25, min(bounds.width() - 2 * margin, available))
        rect = QRect(0, 0, side, side)
        rect.moveCenter(bounds.center())
        rect.moveTop(top + max(0, (available - side) // 2))
        return rect

    def _underline(self, bounds: QRect) -> QRect:
        margin = max(4, min(12, bounds.width() // 24, bounds.height() // 24))
        width = max(24, bounds.width() // 3)
        y = bounds.top() + margin + self.font_metrics.height() + 6
        return QRect(bounds.center().x() - width // 2, y, width, UNDERLINE_HEIGHT)

    @staticmethod
    def _pixmap(matrix) -> QPixmap:
        pixels = 255 - matrix * 255
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                       QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(image.copy())

    def prepare_qr(self, timestamp_ns: int) -> None:
        payload = timestamp_payload(timestamp_ns)
        if payload == self._prepared_payload:
            return
        matrix = qr_matrix(payload, mask_pattern=self.qr_mask_pattern)
        self._prepared_payload = payload
        self._prepared_matrix = matrix
        self._prepared_image = self._pixmap(matrix)

    def _prepared(self, timestamp_ns: int):
        if timestamp_payload(timestamp_ns) != self._prepared_payload or self._prepared_matrix is None:
            self.prepare_qr(timestamp_ns)
        return self._prepared_matrix, self._prepared_image

    def render_next(self, timestamp_ns: int, display_index: int, *, cache_pixmap: bool = True) -> int:
        cell = self.next_cell
        expired = (cell - self.visible_qrs) % self.grid_qrs
        old = self.timestamps[expired]
        if old is not None:
            self.timestamps[expired] = self.display_indices[expired] = None
            self.matrices.pop(old, None)
            self.images.pop(old, None)
        matrix, image = self._prepared(timestamp_ns)
        self.timestamps[cell], self.display_indices[cell] = timestamp_ns, display_index
        self.matrices[timestamp_ns] = matrix
        if cache_pixmap and image is not None:
            self.images[timestamp_ns] = image
        self.newest_cell = cell
        self.next_cell = (cell + 1) % self.grid_qrs
        return cell

    @staticmethod
    def _draw_modules(painter: QPainter, rect: QRect, matrix) -> None:
        modules = matrix.shape[0]
        scale = max(1, min(rect.width(), rect.height()) // modules)
        side = modules * scale
        bounds = QRect(0, 0, side, side)
        bounds.moveCenter(rect.center())
        painter.fillRect(bounds, FOREGROUND)
        for row, column in zip(*matrix.nonzero()):
            painter.fillRect(bounds.x() + int(column) * scale, bounds.y() + int(row) * scale,
                             scale, scale, BLACK)

    def paint(self, painter: QPainter, *, force_full_redraw: bool = True,
              cached_pixmaps: bool = True) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.setFont(self.font)
        painter.fillRect(0, 0, *self.size, BACKGROUND)
        for cell, area in enumerate(self.areas):
            timestamp = self.timestamps[cell]
            if timestamp is None:
                continue
            qr_rect = self.qr_rects[cell]
            if cached_pixmaps and timestamp in self.images:
                pixmap = self.images[timestamp]
                modules = pixmap.width()
                scale = max(1, min(qr_rect.width(), qr_rect.height()) // modules)
                bounds = QRect(0, 0, modules * scale, modules * scale)
                bounds.moveCenter(qr_rect.center())
                painter.drawPixmap(bounds, pixmap)
            else:
                self._draw_modules(painter, qr_rect, self.matrices[timestamp])
            margin = max(4, min(12, area.width() // 24, area.height() // 24))
            label = f"#{self.display_indices[cell]}  {format_timestamp(timestamp)}"
            text = QRect(area.x(), area.y() + margin, area.width(), self.font_metrics.height() + 4)
            painter.setPen(FOREGROUND)
            painter.drawText(text, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, label)
        if self.newest_cell is not None:
            painter.fillRect(self.underlines[self.newest_cell], FOREGROUND)

    def metadata(self) -> dict:
        rows, columns = grid_shape(self.grid_qrs)
        return {
            "size": list(self.size), "code_format": "qr", "qr_border_modules": QUIET_ZONE_MODULES,
            "display_backend": "qt-opengl", "grid_qrs": self.grid_qrs,
            "grid_rows": rows, "grid_columns": columns, "cell_order": list(self.cell_names),
            "cell_positions": [list(value) for value in grid_positions(self.grid_qrs)],
            "visible_qrs": self.visible_qrs, "qr_mask_pattern": self.qr_mask_pattern,
            "qr_mask_selection": "automatic" if self.qr_mask_pattern is None else "fixed",
            "indicator_style": "underline", "indicator_width": UNDERLINE_HEIGHT,
            "status_strip_height_px": STATUS_STRIP_HEIGHT,
            "timestamp_semantics": "marker_ns is paint-start or predicted software frame swap time",
            "frame_index_semantics": "display index is a label; QR payload remains timestamp-only",
        }


class QRClockWindow(QOpenGLWindow):
    """Render one QR frame per completed buffer swap; the status strip is drawn last."""

    def __init__(self, *, expected_refresh_hz: float, journal_path: str | None,
                 screen_metadata: dict, visible_qrs: int, grid_qrs: int,
                 timestamp_mode: str, qr_mask_pattern: int | None,
                 recording_wait_event=None, first_qr_event=None):
        super().__init__(QOpenGLWindow.UpdateBehavior.NoPartialUpdate)
        if timestamp_mode not in TIMESTAMP_MODES:
            raise ValueError(f"Timestamp mode must be one of: {', '.join(TIMESTAMP_MODES)}")
        self.expected_refresh_hz = expected_refresh_hz
        self.journal_path, self.screen_metadata = journal_path, screen_metadata
        self.visible_qrs, self.grid_qrs = visible_qrs, grid_qrs
        self.timestamp_mode, self.qr_mask_pattern = timestamp_mode, qr_mask_pattern
        self.recording_wait_event, self.first_qr_event = recording_wait_event, first_qr_event
        self.renderer: QRClockRenderer | None = None
        self.monitor = SwapTimingMonitor(expected_refresh_hz)
        self.journal: DisplayJournal | None = None
        self.paused = self.resumed_after_pause = False
        self.pending_frame: dict | None = None
        self.next_display_index = 0
        self.setTitle(WINDOW_NAME)
        self.frameSwapped.connect(self._frame_swapped)

    def initializeGL(self) -> None:
        self.renderer = QRClockRenderer(self.width(), self.height(), self.visible_qrs,
                                        self.grid_qrs, self.qr_mask_pattern)
        context = self.context()
        fmt = context.format() if context else self.format()
        metadata = {
            **self.renderer.metadata(), **self.screen_metadata,
            "qt_version": __import__("PySide6.QtCore", fromlist=["qVersion"]).qVersion(),
            "qt_platform": QGuiApplication.platformName(), "requested_swap_interval": 1,
            "actual_swap_interval": fmt.swapInterval(), "opengl_version": [fmt.majorVersion(), fmt.minorVersion()],
            "update_behavior": "NoPartialUpdate", "timestamp_mode": self.timestamp_mode,
            "prediction_period_ns": self.monitor.period_ns,
            "presentation_semantics": "software frameSwapped return; physical scanout is not measured",
        }
        self.journal = DisplayJournal(self.journal_path, metadata)

    def resizeGL(self, width: int, height: int) -> None:
        if self.renderer and width >= 320 and height >= 240:
            self.renderer.resize(width, height)

    def paintGL(self) -> None:
        if self.renderer is None or self.paused or self.pending_frame is not None:
            return
        start = time.monotonic_ns()
        marker = self.monitor.predict_next_swap(start) if self.timestamp_mode == "predicted-flip" else start
        index = self.next_display_index
        cell = self.renderer.render_next(marker, index)
        recording_wait_active = bool(self.recording_wait_event and self.recording_wait_event.is_set())
        painter = QPainter(self)
        try:
            self.renderer.paint(painter)
            color = RECORDING_WAIT_RED if recording_wait_active else BACKGROUND
            painter.fillRect(self.renderer.recording_wait_bar, color)
        finally:
            painter.end()
        self.pending_frame = {"cell": cell, "display_index": index, "marker_ns": marker,
                              "paint_start_ns": start, "submit_ns": time.monotonic_ns(),
                              "recording_wait_active": recording_wait_active}

    def _frame_swapped(self) -> None:
        returned = time.monotonic_ns()
        pending, self.pending_frame = self.pending_frame, None
        if pending is None:
            if not self.paused:
                self.update()
            return
        timing = self.monitor.observe(pending["marker_ns"], pending["submit_ns"], returned,
                                      paint_start_ns=pending["paint_start_ns"])
        timing.update({
            "prediction_error_ns": returned - pending["marker_ns"] if self.timestamp_mode == "predicted-flip" else None,
            "predicted_flip_ns": pending["marker_ns"] if self.timestamp_mode == "predicted-flip" else None,
            "presentation_event_kind": "qt_frame_swapped_return",
            "presentation_return_ns": returned, "physical_presentation_measured": False,
            "late_submit": self.timestamp_mode == "predicted-flip" and pending["submit_ns"] > pending["marker_ns"],
            "resumed_after_pause": self.resumed_after_pause,
            "recording_wait_active": pending["recording_wait_active"],
        })
        self.resumed_after_pause = False
        if self.journal:
            self.journal.append(pending["cell"], timing)
        if pending["display_index"] == 0 and self.first_qr_event is not None:
            self.first_qr_event.set()
        self.next_display_index += 1
        if not self.paused:
            next_marker = self.monitor.predict_next_swap(time.monotonic_ns())
            self.update()
            if self.renderer and self.timestamp_mode == "predicted-flip":
                self.renderer.prepare_qr(next_marker)

    def keyPressEvent(self, event) -> None:
        if event.isAutoRepeat():
            return
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        elif event.key() == Qt.Key.Key_P:
            self.paused = not self.paused
            if self.journal:
                self.journal.pause(self.paused, time.monotonic_ns())
            self.setTitle(WINDOW_NAME + (" — paused" if self.paused else ""))
            if not self.paused:
                self.monitor.reset()
                self.resumed_after_pause = True
                self.update()
        else:
            super().keyPressEvent(event)

    def exposeEvent(self, event) -> None:
        super().exposeEvent(event)
        if self.isExposed() and not self.paused:
            self.update()

    def close_journal(self) -> None:
        if self.journal:
            self.journal.close()
            self.journal = None


def configure_surface_format() -> None:
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    fmt.setSwapInterval(1)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)


def screen_catalog(app: QGuiApplication) -> list[dict]:
    primary = app.primaryScreen()
    result = []
    for index, screen in enumerate(app.screens()):
        rect = screen.geometry()
        result.append({"index": index, "name": screen.name() or f"Monitor {index + 1}",
                       "width": rect.width(), "height": rect.height(), "x": rect.x(), "y": rect.y(),
                       "refresh_hz": float(screen.refreshRate()), "primary": screen is primary})
    return result


def describe_screens(app: QGuiApplication) -> None:
    print(json.dumps(screen_catalog(app)))


def run_calibration_display(stop_event=None, *, width: int = 1920, height: int = 1080,
                            windowed: bool = False, refresh_hz: float | None = None,
                            journal_path: str | None = None, screen_index: int = 0,
                            visible_qrs: int = VISIBLE_QRS, grid_qrs: int = DEFAULT_GRID_QRS,
                            timestamp_mode: str = DEFAULT_TIMESTAMP_MODE,
                            qr_mask_pattern: int | None = None, recording_wait_event=None,
                            first_qr_event=None, list_screens: bool = False) -> int:
    if width < 320 or height < 240:
        raise ValueError("Canvas must be at least 320 by 240 pixels")
    grid_shape(grid_qrs)
    if not 1 <= visible_qrs <= grid_qrs:
        raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")
    if qr_mask_pattern is not None and qr_mask_pattern not in QR_MASK_PATTERNS:
        raise ValueError("QR mask pattern must be from 0 to 7")
    if timestamp_mode not in TIMESTAMP_MODES:
        raise ValueError(f"Timestamp mode must be one of: {', '.join(TIMESTAMP_MODES)}")
    configure_surface_format()
    app = QGuiApplication(["segcom-qr-display"])
    screens = app.screens()
    if list_screens:
        print(json.dumps(screen_catalog(app)))
        return 0
    if not 0 <= screen_index < len(screens):
        raise ValueError(f"Invalid screen index {screen_index}; available screens: 0..{len(screens) - 1}")
    screen = screens[screen_index]
    rect = screen.geometry()
    reported = float(screen.refreshRate())
    expected = refresh_hz or (reported if math.isfinite(reported) and reported >= 1 else DEFAULT_REFRESH_HZ)
    window = QRClockWindow(
        expected_refresh_hz=expected, journal_path=journal_path,
        screen_metadata={"screen_index": screen_index, "screen_name": screen.name(),
                         "screen_geometry": [rect.x(), rect.y(), rect.width(), rect.height()],
                         "qt_reported_refresh_hz": reported, "expected_refresh_hz": expected},
        visible_qrs=visible_qrs, grid_qrs=grid_qrs, timestamp_mode=timestamp_mode,
        qr_mask_pattern=qr_mask_pattern, recording_wait_event=recording_wait_event,
        first_qr_event=first_qr_event)
    window.setScreen(screen)
    if windowed:
        window.resize(width, height)
        window.setPosition(rect.x() + max(0, (rect.width() - width) // 2),
                           rect.y() + max(0, (rect.height() - height) // 2))
        window.show()
    else:
        window.setGeometry(rect)
        window.showFullScreen()
    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(lambda: window.close() if stop_event is not None and stop_event.is_set() else None)
    timer.start()
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[signum] = signal.signal(signum, lambda *_: window.close())
        except (OSError, ValueError):
            pass
    try:
        return app.exec()
    finally:
        timer.stop()
        window.close_journal()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display rotating QR camera-calibration timestamps")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--refresh-hz", type=float)
    parser.add_argument("--journal")
    parser.add_argument("--grid-qrs", type=int, choices=tuple(GRID_LAYOUTS), default=DEFAULT_GRID_QRS)
    parser.add_argument("--visible-qrs", type=int, default=VISIBLE_QRS)
    parser.add_argument("--screen", type=int, default=0)
    parser.add_argument("--timestamp-mode", choices=TIMESTAMP_MODES, default=DEFAULT_TIMESTAMP_MODE)
    parser.add_argument("--qr-mask-pattern", type=int, choices=QR_MASK_PATTERNS)
    parser.add_argument("--list-screens-json", action="store_true")
    args = parser.parse_args()
    run_calibration_display(width=args.width, height=args.height, windowed=args.windowed,
                            refresh_hz=args.refresh_hz, journal_path=args.journal,
                            screen_index=args.screen, visible_qrs=args.visible_qrs,
                            grid_qrs=args.grid_qrs, timestamp_mode=args.timestamp_mode,
                            qr_mask_pattern=args.qr_mask_pattern, list_screens=args.list_screens_json)


if __name__ == "__main__":
    main()
