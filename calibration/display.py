"""SDL QR clock for camera timing calibration."""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from calibration.display_common import (
    BACKGROUND_RGB, DEFAULT_GRID_QRS, DEFAULT_QR_DRAW_MODE, DEFAULT_REFRESH_HZ,
    DEFAULT_SPIN_WAIT_US, DEFAULT_TIMESTAMP_MODE, DISPLAY_JOURNAL_NAME,
    FOREGROUND_RGB, QR_DRAW_MODES, RECORDING_WAIT_RGB, STATUS_STRIP_HEIGHT,
    TIMESTAMP_MODES, UNDERLINE_HEIGHT, VISIBLE_QRS, WINDOW_NAME, DisplayJournal,
    FramePacer, format_timestamp, grid_areas as _layout_areas,
)
from calibration.qr import (
    GRID_LAYOUTS, QUIET_ZONE_MODULES, QR_MASK_PATTERNS, grid_cell_names,
    grid_positions, grid_shape, qr_matrix, timestamp_payload,
)

REFRESH_HZ = DEFAULT_REFRESH_HZ


def parse_spin_wait_us(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("spin wait must be a whole number") from error
    if not 0 <= result <= 5_000:
        raise argparse.ArgumentTypeError("spin wait must be from 0 to 5000 microseconds")
    return result


def grid_areas(width: int, height: int, grid_qrs: int) -> tuple[pygame.Rect, ...]:
    return tuple(pygame.Rect(left, top, right - left, bottom - top)
                 for left, top, right, bottom in _layout_areas(width, height, grid_qrs))


def quadrant_areas(width: int, height: int) -> tuple[pygame.Rect, ...]:
    return grid_areas(width, height, DEFAULT_GRID_QRS)


class QRClockRenderer:
    """Retain visible timestamps and redraw only the cell being replaced."""

    def __init__(self, target: pygame.Surface, visible_qrs: int = VISIBLE_QRS,
                 grid_qrs: int = DEFAULT_GRID_QRS, qr_mask_pattern: int | None = None):
        width, height = target.get_size()
        if width < 320 or height < 240:
            raise ValueError("Canvas must be at least 320 by 240 pixels")
        grid_shape(grid_qrs)
        if not 1 <= visible_qrs <= grid_qrs:
            raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")
        if qr_mask_pattern is not None and qr_mask_pattern not in QR_MASK_PATTERNS:
            raise ValueError("QR mask pattern must be from 0 to 7")
        if not pygame.font.get_init():
            pygame.font.init()
        self.target, self.size = target, (width, height)
        self.grid_qrs, self.visible_qrs = grid_qrs, visible_qrs
        self.qr_mask_pattern = qr_mask_pattern
        self.cell_names = grid_cell_names(grid_qrs)
        rows, columns = grid_shape(grid_qrs)
        self.areas = grid_areas(width, height, grid_qrs)
        self.font = pygame.font.Font(None, max(10, min(width // (columns * 14), height // (rows * 10))))
        self.recording_wait_bar = pygame.Rect(2, 3, width - 4, 12)
        self.qr_rects = [self._qr_rect(area) for area in self.areas]
        self.underlines = tuple(self._underline(area) for area in self.areas)
        self.timestamps: list[int | None] = [None] * grid_qrs
        self.display_indices: list[int | None] = [None] * grid_qrs
        self._surfaces: dict[tuple[int, int], tuple[pygame.Surface, pygame.Surface]] = {}
        self._prepared_payload: str | None = None
        self._prepared_matrix = None
        self.next_cell = 0
        self.newest_cell: int | None = None
        target.fill(BACKGROUND_RGB)

    def _qr_rect(self, area: pygame.Rect) -> pygame.Rect:
        margin = max(4, min(12, area.width // 24, area.height // 24))
        label_gap = self.font.get_linesize() + 8 + UNDERLINE_HEIGHT
        top = area.top + margin + label_gap
        available = max(1, area.bottom - margin - top)
        side = max(25, min(area.width - 2 * margin, available))
        rect = pygame.Rect(0, 0, side, side)
        rect.centerx = area.centerx
        rect.top = top + max(0, (available - side) // 2)
        return rect

    def _underline(self, area: pygame.Rect) -> pygame.Rect:
        margin = max(4, min(12, area.width // 24, area.height // 24))
        width = max(24, area.width // 3)
        return pygame.Rect(area.centerx - width // 2,
                           area.top + margin + self.font.get_linesize() + 6,
                           width, UNDERLINE_HEIGHT)

    def prepare_qr(self, timestamp_ns: int) -> None:
        payload = timestamp_payload(timestamp_ns)
        if payload != self._prepared_payload:
            self._prepared_payload = payload
            self._prepared_matrix = qr_matrix(payload, mask_pattern=self.qr_mask_pattern)

    def _matrix(self, timestamp_ns: int):
        self.prepare_qr(timestamp_ns)
        return self._prepared_matrix

    def _draw_modules(self, rect: pygame.Rect, matrix) -> pygame.Rect:
        scale = max(1, min(rect.width, rect.height) // matrix.shape[0])
        bounds = pygame.Rect(0, 0, matrix.shape[0] * scale, matrix.shape[1] * scale)
        bounds.center = rect.center
        self.target.fill(FOREGROUND_RGB, bounds)
        for row, column in zip(*matrix.nonzero()):
            self.target.fill((0, 0, 0), (bounds.x + int(column) * scale,
                                         bounds.y + int(row) * scale, scale, scale))
        return bounds

    def _draw_surface(self, rect: pygame.Rect, matrix) -> pygame.Rect:
        modules = matrix.shape[0]
        scale = max(1, min(rect.width, rect.height) // modules)
        surfaces = self._surfaces.get((modules, scale))
        if surfaces is None:
            surfaces = (pygame.Surface((modules, modules), depth=24),
                        pygame.Surface((modules * scale, modules * scale), depth=24))
            self._surfaces[(modules, scale)] = surfaces
        source, scaled = surfaces
        pixels = pygame.surfarray.pixels3d(source)
        pixels[...] = (255 - matrix.T * 255)[:, :, None]
        del pixels
        pygame.transform.scale(source, scaled.get_size(), scaled)
        bounds = scaled.get_rect(center=rect.center)
        self.target.blit(scaled, bounds)
        return bounds

    def render_next(self, timestamp_ns: int, display_index: int,
                    *, qr_draw_mode: str = DEFAULT_QR_DRAW_MODE) -> int:
        if qr_draw_mode not in QR_DRAW_MODES:
            raise ValueError(f"QR draw mode must be one of: {', '.join(QR_DRAW_MODES)}")
        cell = self.next_cell
        expired = (cell - self.visible_qrs) % self.grid_qrs
        if self.newest_cell is not None:
            self.target.fill(BACKGROUND_RGB, self.underlines[self.newest_cell])
        if self.timestamps[expired] is not None:
            self.target.fill(BACKGROUND_RGB, self.areas[expired])
            self.timestamps[expired] = self.display_indices[expired] = None
        self.target.fill(BACKGROUND_RGB, self.areas[cell])
        matrix = self._matrix(timestamp_ns)
        bounds = (self._draw_surface(self.qr_rects[cell], matrix) if qr_draw_mode == "surface"
                  else self._draw_modules(self.qr_rects[cell], matrix))
        label = self.font.render(f"#{display_index}  {format_timestamp(timestamp_ns)}",
                                 True, FOREGROUND_RGB, BACKGROUND_RGB)
        text_rect = label.get_rect(centerx=self.areas[cell].centerx,
                                   top=self.areas[cell].top + max(4, min(12, self.areas[cell].width // 24)))
        self.target.blit(label, text_rect)
        underline = self.underlines[cell].copy()
        underline.top = text_rect.bottom + 2
        self.target.fill(FOREGROUND_RGB, underline)
        self.timestamps[cell], self.display_indices[cell] = timestamp_ns, display_index
        self.qr_rects[cell] = bounds
        self.newest_cell = cell
        self.next_cell = (cell + 1) % self.grid_qrs
        return cell

    def metadata(self) -> dict:
        rows, columns = grid_shape(self.grid_qrs)
        return {
            "size": list(self.size), "code_format": "qr", "qr_border_modules": QUIET_ZONE_MODULES,
            "display_backend": "pygame-sdl", "grid_qrs": self.grid_qrs,
            "grid_rows": rows, "grid_columns": columns, "cell_order": list(self.cell_names),
            "cell_positions": [list(value) for value in grid_positions(self.grid_qrs)],
            "visible_qrs": self.visible_qrs, "qr_mask_pattern": self.qr_mask_pattern,
            "qr_mask_selection": "automatic" if self.qr_mask_pattern is None else "fixed",
            "indicator_style": "underline", "indicator_width": UNDERLINE_HEIGHT,
            "status_strip_height_px": STATUS_STRIP_HEIGHT,
            "timestamp_semantics": "marker_ns is paint-start or predicted software flip time",
            "frame_index_semantics": "display index is a label; QR payload remains timestamp-only",
            "layouts": [{"area": list(area), "qr": list(qr), "underline": list(line)}
                        for area, qr, line in zip(self.areas, self.qr_rects, self.underlines)],
        }


def run_calibration_display(stop_event=None, *, width: int = 1920, height: int = 1080,
                            windowed: bool = False, refresh_hz: float = REFRESH_HZ,
                            journal_path: str | None = None, screen_index: int = 0,
                            visible_qrs: int = VISIBLE_QRS, grid_qrs: int = DEFAULT_GRID_QRS,
                            timestamp_mode: str = DEFAULT_TIMESTAMP_MODE,
                            qr_draw_mode: str = DEFAULT_QR_DRAW_MODE,
                            qr_mask_pattern: int | None = None,
                            spin_wait_us: int = DEFAULT_SPIN_WAIT_US,
                            recording_wait_event=None, first_qr_event=None) -> None:
    if width < 320 or height < 240:
        raise ValueError("Canvas must be at least 320 by 240 pixels")
    grid_shape(grid_qrs)
    if not 1 <= visible_qrs <= grid_qrs:
        raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")
    if timestamp_mode not in TIMESTAMP_MODES or qr_draw_mode not in QR_DRAW_MODES:
        raise ValueError("Unsupported timestamp or QR drawing mode")
    if qr_mask_pattern is not None and qr_mask_pattern not in QR_MASK_PATTERNS:
        raise ValueError("QR mask pattern must be from 0 to 7")
    if not 0 <= spin_wait_us <= 5_000:
        raise ValueError("Spin wait must be from 0 to 5000 microseconds")

    pygame.display.init()
    pygame.font.init()
    journal = None
    try:
        displays = pygame.display.get_num_displays()
        if not 0 <= screen_index < displays:
            raise ValueError(f"Invalid screen index {screen_index}; available screens: 0..{displays - 1}")
        flags = pygame.SCALED | (0 if windowed else pygame.FULLSCREEN)
        requested = (width, height) if windowed else pygame.display.get_desktop_sizes()[screen_index]
        try:
            screen = pygame.display.set_mode(requested, flags, display=screen_index, vsync=1)
        except pygame.error as error:
            raise RuntimeError("Pygame could not create the QR calibration display") from error
        pygame.display.set_caption(WINDOW_NAME + " — Pygame/SDL")
        renderer = QRClockRenderer(screen, visible_qrs, grid_qrs, qr_mask_pattern)
        pygame.display.flip()
        pacer = FramePacer(time.monotonic_ns(), refresh_hz, spin_wait_us)
        journal = DisplayJournal(journal_path, {
            **renderer.metadata(), "screen_index": screen_index, "requested_refresh_hz": refresh_hz,
            "pygame_version": pygame.version.ver, "sdl_version": list(pygame.get_sdl_version()),
            "display_driver": pygame.display.get_driver(), "vsync_requested": True,
            "timestamp_mode": timestamp_mode, "qr_draw_mode": qr_draw_mode,
            "prediction_period_ns": pacer.period_ns,
            "presentation_semantics": "presentation_return_ns is pygame flip return, not physical scanout",
        })
        paused = resumed = exiting = False

        def poll_controls() -> bool:
            nonlocal paused, resumed, exiting
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                    exiting = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    paused = not paused
                    journal.pause(paused, time.monotonic_ns())
                    if not paused:
                        pacer.last_flip_ns = None
                        resumed = True
            if stop_event is not None and stop_event.is_set():
                exiting = True
            return exiting

        while not poll_controls():
            if paused:
                pygame.time.wait(30)
                continue
            ready, skipped = pacer.wait(poll_controls)
            if not ready:
                continue
            start = time.monotonic_ns()
            marker = pacer.predict_next_flip(start) if timestamp_mode == "predicted-flip" else start
            cell = renderer.render_next(marker, journal.frame_count, qr_draw_mode=qr_draw_mode)
            recording_wait_active = bool(recording_wait_event and recording_wait_event.is_set())
            color = RECORDING_WAIT_RGB if recording_wait_active else BACKGROUND_RGB
            pygame.draw.rect(screen, color, renderer.recording_wait_bar)
            submit = time.monotonic_ns()
            pygame.display.flip()
            returned = time.monotonic_ns()
            timing = pacer.observe(marker, submit, returned, skipped, paint_start_ns=start)
            timing.update({
                "prediction_error_ns": returned - marker if timestamp_mode == "predicted-flip" else None,
                "predicted_flip_ns": marker if timestamp_mode == "predicted-flip" else None,
                "presentation_event_kind": "pygame_display_flip_return",
                "presentation_return_ns": returned, "physical_presentation_measured": False,
                "resumed_after_pause": resumed,
                "recording_wait_active": recording_wait_active,
            })
            resumed = False
            journal.append(cell, timing)
            if journal.frame_count == 1 and first_qr_event is not None:
                first_qr_event.set()
            if timestamp_mode == "predicted-flip":
                renderer.prepare_qr(pacer.predict_next_flip(time.monotonic_ns()))
    finally:
        if journal is not None:
            journal.close()
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Display rotating QR camera-calibration timestamps")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--refresh-hz", type=float, default=REFRESH_HZ)
    parser.add_argument("--journal")
    parser.add_argument("--screen", type=int, default=0)
    parser.add_argument("--grid-qrs", type=int, choices=tuple(GRID_LAYOUTS), default=DEFAULT_GRID_QRS)
    parser.add_argument("--visible-qrs", type=int, default=VISIBLE_QRS)
    parser.add_argument("--timestamp-mode", choices=TIMESTAMP_MODES, default=DEFAULT_TIMESTAMP_MODE)
    parser.add_argument("--qr-draw-mode", choices=QR_DRAW_MODES, default=DEFAULT_QR_DRAW_MODE)
    parser.add_argument("--qr-mask-pattern", type=int, choices=QR_MASK_PATTERNS)
    parser.add_argument("--spin-wait-us", type=parse_spin_wait_us, default=DEFAULT_SPIN_WAIT_US)
    args = parser.parse_args()
    run_calibration_display(width=args.width, height=args.height, windowed=args.windowed,
                            refresh_hz=args.refresh_hz, journal_path=args.journal,
                            screen_index=args.screen, visible_qrs=args.visible_qrs,
                            grid_qrs=args.grid_qrs, timestamp_mode=args.timestamp_mode,
                            qr_draw_mode=args.qr_draw_mode, qr_mask_pattern=args.qr_mask_pattern,
                            spin_wait_us=args.spin_wait_us)


if __name__ == "__main__":
    main()
