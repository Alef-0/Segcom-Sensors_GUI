"""Single-canvas QR clock for camera-delay calibration."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

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


DISPLAY_JOURNAL_NAME = "display_timestamps.jsonl"
DISPLAY_FORMAT = "segcom-qr-display-pygame-v2"
WINDOW_NAME = "QR Calibration Clock — Pygame/SDL"
BACKGROUND = (50, 50, 50)
FOREGROUND = (255, 255, 255)
BLACK = (0, 0, 0)
DEFAULT_GRID_QRS = 4
VISIBLE_QRS = 2
REFRESH_HZ = 60.0
UNDERLINE_HEIGHT = 4


def format_timestamp(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return f"{seconds:,}".replace(",", " ") + f".{nanoseconds // 1_000_000:03d}"


def grid_areas(width: int, height: int, grid_qrs: int) -> tuple[pygame.Rect, ...]:
    return tuple(
        pygame.Rect(left, top, right - left, bottom - top)
        for left, top, right, bottom in grid_bounds(width, height, grid_qrs)
    )


def quadrant_areas(width: int, height: int) -> tuple[pygame.Rect, ...]:
    """Compatibility wrapper for the original four-cell renderer."""
    return grid_areas(width, height, DEFAULT_GRID_QRS)


class QRClockRenderer:
    """Draw the requested newest QR timestamps on one persistent surface."""

    def __init__(
        self,
        target: pygame.Surface,
        visible_qrs: int = VISIBLE_QRS,
        grid_qrs: int = DEFAULT_GRID_QRS,
    ):
        width, height = target.get_size()
        if width < 320 or height < 240:
            raise ValueError("Canvas must be at least 320 by 240 pixels")
        grid_shape(grid_qrs)
        if not 1 <= visible_qrs <= grid_qrs:
            raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")
        if not pygame.font.get_init():
            pygame.font.init()
        self.target = target
        self.size = target.get_size()
        self.grid_qrs = grid_qrs
        self.visible_qrs = visible_qrs
        self.cell_names = grid_cell_names(grid_qrs)
        self.areas = grid_areas(width, height, grid_qrs)
        rows, columns = grid_shape(grid_qrs)
        font_size = max(10, min(width // (columns * 14), height // (rows * 10)))
        self.font = pygame.font.Font(None, font_size)
        self.qr_rects = tuple(self._qr_rect(area) for area in self.areas)
        self.underlines = tuple(self._underline(area) for area in self.areas)
        self.timestamps: list[int | None] = [None] * grid_qrs
        self.display_indices: list[int | None] = [None] * grid_qrs
        self.next_cell = 0
        self.newest_cell: int | None = None
        target.fill(BACKGROUND)

    def _qr_rect(self, area: pygame.Rect) -> pygame.Rect:
        margin = max(4, min(12, area.width // 24, area.height // 24))
        text_height = self.font.get_linesize() + 4
        label_gap = max(4, margin)
        content_top = area.top + margin + text_height + label_gap + UNDERLINE_HEIGHT
        content_bottom = area.bottom - margin
        available_height = max(1, content_bottom - content_top)
        side = max(25, min(area.width - 2 * margin, available_height))
        rect = pygame.Rect(0, 0, side, side)
        rect.centerx = area.centerx
        rect.top = content_top + max(0, (available_height - side) // 2)
        return rect

    def _underline(self, area: pygame.Rect) -> pygame.Rect:
        width = max(24, area.width // 3)
        margin = max(4, min(12, area.width // 24, area.height // 24))
        y = area.top + margin + self.font.get_linesize() + 6
        return pygame.Rect(area.centerx - width // 2, y, width, UNDERLINE_HEIGHT)

    def _draw_qr(self, rect: pygame.Rect, payload: str) -> pygame.Rect:
        matrix = qr_matrix(payload)
        modules = matrix.shape[0]
        scale = max(1, min(rect.width, rect.height) // modules)
        side = modules * scale
        bounds = pygame.Rect(0, 0, side, side)
        bounds.center = rect.center
        self.target.fill(FOREGROUND, bounds)
        dark = pygame.Rect(0, 0, scale, scale)
        for row, column in zip(*matrix.nonzero()):
            dark.topleft = bounds.x + int(column) * scale, bounds.y + int(row) * scale
            self.target.fill(BLACK, dark)
        return bounds

    def render_next(self, timestamp_ns: int, display_index: int) -> int:
        cell = self.next_cell
        if self.newest_cell is not None:
            self.target.fill(BACKGROUND, self.underlines[self.newest_cell])
        expired = (cell - self.visible_qrs) % self.grid_qrs
        if self.timestamps[expired] is not None:
            self.target.fill(BACKGROUND, self.areas[expired])
            self.timestamps[expired] = None
            self.display_indices[expired] = None
        self.target.fill(BACKGROUND, self.areas[cell])
        bounds = self._draw_qr(self.qr_rects[cell], timestamp_payload(timestamp_ns))
        text = self.font.render(
            f"#{display_index}  {format_timestamp(timestamp_ns)}",
            True,
            FOREGROUND,
            BACKGROUND,
        )
        text_rect = text.get_rect(centerx=self.areas[cell].centerx)
        margin = max(
            4,
            min(12, self.areas[cell].width // 24, self.areas[cell].height // 24),
        )
        text_rect.y = self.areas[cell].top + margin
        self.target.blit(text, text_rect)
        underline = self.underlines[cell]
        underline.top = text_rect.bottom + 2
        self.target.fill(FOREGROUND, underline)
        self.qr_rects = tuple(
            bounds if index == cell else value
            for index, value in enumerate(self.qr_rects)
        )
        self.timestamps[cell] = timestamp_ns
        self.display_indices[cell] = display_index
        self.newest_cell = cell
        self.next_cell = (cell + 1) % self.grid_qrs
        return cell

    def metadata(self) -> dict:
        return {
            "size": list(self.size),
            "code_format": "qr",
            "qr_border_modules": QUIET_ZONE_MODULES,
            "display_backend": "pygame-sdl",
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
            "timestamp_semantics": "monotonic time sampled before drawing, not physical scanout",
            "frame_index_semantics": (
                "display index is shown as text and recorded in the journal; "
                "the QR payload remains timestamp-only for decoder compatibility"
            ),
            "layouts": [
                {"area": list(area), "qr": list(qr), "underline": list(underline)}
                for area, qr, underline in zip(self.areas, self.qr_rects, self.underlines)
            ],
        }


class FramePacer:
    """Maintain an absolute refresh grid and report missed/late presentations."""

    def __init__(self, anchor_ns: int, refresh_hz: float):
        if not math.isfinite(refresh_hz) or not 1 <= refresh_hz <= 1000:
            raise ValueError("Refresh rate must be between 1 and 1000 Hz")
        self.nominal_period_ns = round(1_000_000_000 / refresh_hz)
        self.period_ns = self.nominal_period_ns
        self.deadline_ns = anchor_ns + self.period_ns
        self.render_budget_ns = min(1_500_000, self.period_ns // 3)
        self.last_flip_ns: int | None = None
        self.render_times = deque(maxlen=120)

    def _skip_expired(self, now_ns: int) -> int:
        if now_ns < self.deadline_ns:
            return 0
        skipped = (now_ns - self.deadline_ns) // self.period_ns + 1
        self.deadline_ns += skipped * self.period_ns
        return skipped

    def wait(self, should_stop) -> tuple[bool, int]:
        skipped = self._skip_expired(time.monotonic_ns())
        while True:
            if should_stop():
                return False, skipped
            remaining = self.deadline_ns - self.render_budget_ns - time.monotonic_ns()
            if remaining <= 0:
                extra = self._skip_expired(time.monotonic_ns())
                skipped += extra
                if not extra:
                    return True, skipped
                continue
            if remaining > 1_000_000:
                pygame.time.wait(min(10, max(1, (remaining - 1_000_000) // 1_000_000)))
            else:
                target = self.deadline_ns - self.render_budget_ns
                while time.monotonic_ns() < target:
                    pass

    def observe(self, marker_ns: int, submit_ns: int, flip_return_ns: int, skipped: int) -> dict:
        interval = None if self.last_flip_ns is None else flip_return_ns - self.last_flip_ns
        irregular = interval is not None and not 0.75 * self.period_ns <= interval <= 1.25 * self.period_ns
        missed_after_submit = max(
            0, (flip_return_ns - self.deadline_ns + self.period_ns // 4) // self.period_ns
        )
        result = {
            "marker_ns": marker_ns,
            "deadline_ns": self.deadline_ns,
            "submit_ns": submit_ns,
            "flip_return_ns": flip_return_ns,
            "frame_period_ns": self.period_ns,
            "interval_ns": interval,
            "late_submit": submit_ns > self.deadline_ns,
            "skipped_before_render": skipped,
            "missed_after_submit": missed_after_submit,
            "skipped_periods": skipped + missed_after_submit,
            "irregular_interval": irregular,
        }
        self.render_times.append(max(0, submit_ns - marker_ns))
        ordered = sorted(self.render_times)
        p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
        self.render_budget_ns = min(self.period_ns // 2, max(500_000, p95 + 750_000))
        self.deadline_ns += (missed_after_submit + 1) * self.period_ns
        self.last_flip_ns = flip_return_ns
        return result


def timing_issues(row: dict) -> list[str]:
    issues = []
    if row.get("skipped_periods"):
        issues.append("missed_period_candidates")
    if row.get("late_submit"):
        issues.append("late_submission")
    if row.get("irregular_interval"):
        issues.append("irregular_interval")
    if row.get("resumed_after_pause"):
        issues.append("resumed_after_pause")
    return issues


class DisplayJournal:
    def __init__(self, path: str | Path | None, metadata: dict):
        self.file = None if path is None else Path(path).open("x", encoding="utf-8", buffering=65536)
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
            "missed_period_candidates": sum(row.get("skipped_periods", 0) for row in self.frames),
            "irregular_intervals": sum(bool(row.get("irregular_interval")) for row in self.frames),
            "late_submissions": sum(bool(row.get("late_submit")) for row in self.frames),
        }
        self._write({"kind": "summary", "frames": len(self.frames), **counts})
        if self.file:
            self.file.close()
        print(
            f"[CALIBRATION] Presented {len(self.frames)} QR markers; "
            f"{counts['missed_period_candidates']} missed-period candidate(s), "
            f"{counts['irregular_intervals']} irregular interval(s), "
            f"{counts['late_submissions']} late submission(s).",
            flush=True,
        )


def run_calibration_display(
    stop_event=None,
    *,
    width: int = 1920,
    height: int = 1080,
    windowed: bool = False,
    refresh_hz: float = REFRESH_HZ,
    journal_path: str | None = None,
    screen_index: int = 0,
    visible_qrs: int = VISIBLE_QRS,
    grid_qrs: int = DEFAULT_GRID_QRS,
) -> None:
    """Sample, draw, and flip on one thread without queued display frames."""
    if width < 320 or height < 240:
        raise ValueError("Canvas must be at least 320 by 240 pixels")
    grid_shape(grid_qrs)
    if not 1 <= visible_qrs <= grid_qrs:
        raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")

    pygame.display.init()
    pygame.font.init()
    journal = None
    try:
        display_count = pygame.display.get_num_displays()
        if not 0 <= screen_index < display_count:
            raise ValueError(
                f"Invalid --screen {screen_index}; available displays: 0..{display_count - 1}"
            )
        flags = pygame.SCALED | (0 if windowed else pygame.FULLSCREEN)
        try:
            requested_size = (
                (width, height)
                if windowed
                else pygame.display.get_desktop_sizes()[screen_index]
            )
            screen = pygame.display.set_mode(
                requested_size,
                flags,
                display=screen_index,
                vsync=1,
            )
        except pygame.error as error:
            raise RuntimeError("Pygame could not create the QR calibration display") from error
        pygame.display.set_caption(WINDOW_NAME)
        renderer = QRClockRenderer(
            screen,
            visible_qrs=visible_qrs,
            grid_qrs=grid_qrs,
        )
        pygame.display.flip()
        pacer = FramePacer(time.monotonic_ns(), refresh_hz)
        journal = DisplayJournal(journal_path, {
            **renderer.metadata(),
            "screen_index": screen_index,
            "requested_refresh_hz": refresh_hz,
            "pygame_version": pygame.version.ver,
            "sdl_version": list(pygame.get_sdl_version()),
            "display_driver": pygame.display.get_driver(),
            "vsync_requested": True,
        })
        paused = False
        exit_requested = False
        resumed_after_pause = False

        def poll_controls() -> bool:
            nonlocal paused, exit_requested, resumed_after_pause, pacer
            if stop_event is not None and stop_event.is_set():
                exit_requested = True
                return True
            toggled = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)
                ):
                    exit_requested = True
                    return True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p and not getattr(event, "repeat", False):
                    paused = not paused
                    toggled = True
                    journal.pause(paused, time.monotonic_ns())
                    if not paused:
                        pacer = FramePacer(time.monotonic_ns(), refresh_hz)
                        resumed_after_pause = True
                    pygame.display.set_caption(WINDOW_NAME + (" — paused" if paused else ""))
            return exit_requested or paused or toggled

        clock = pygame.time.Clock()
        while True:
            if paused:
                poll_controls()
                if exit_requested:
                    break
                clock.tick(30)
                continue
            ready, skipped = pacer.wait(poll_controls)
            if not ready:
                if exit_requested:
                    break
                continue
            marker_ns = time.monotonic_ns()
            cell = renderer.render_next(marker_ns, len(journal.frames))
            submit_ns = time.monotonic_ns()
            pygame.display.flip()
            flip_return_ns = time.monotonic_ns()
            timing = pacer.observe(marker_ns, submit_ns, flip_return_ns, skipped)
            timing["resumed_after_pause"] = resumed_after_pause
            resumed_after_pause = False
            journal.append(cell, timing)
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
        help="SDL display index to use (default: 0)",
    )
    arguments = parser.parse_args()
    run_calibration_display(
        width=arguments.width,
        height=arguments.height,
        windowed=arguments.windowed,
        refresh_hz=arguments.refresh_hz,
        journal_path=arguments.journal,
        screen_index=arguments.screen,
        visible_qrs=arguments.visible_qrs,
        grid_qrs=arguments.grid_qrs,
    )


if __name__ == "__main__":
    main()
