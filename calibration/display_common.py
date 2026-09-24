"""Timing, layout, and journal helpers shared by both QR display backends."""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import time

from calibration.qr import grid_positions, grid_shape

DISPLAY_JOURNAL_NAME = "display_timestamps.jsonl"
DISPLAY_FORMAT = "segcom-qr-display-v4"
WINDOW_NAME = "QR Calibration Clock"
BACKGROUND_RGB = (50, 50, 50)
FOREGROUND_RGB = (255, 255, 255)
RECORDING_WAIT_RGB = (220, 30, 30)
STATUS_STRIP_HEIGHT = 18
DEFAULT_GRID_QRS = 4
VISIBLE_QRS = 2
UNDERLINE_HEIGHT = 4
DEFAULT_REFRESH_HZ = 60.0
TIMESTAMP_MODES = ("paint-start", "predicted-flip")
DEFAULT_TIMESTAMP_MODE = "predicted-flip"
QR_DRAW_MODES = ("surface", "modules")
DEFAULT_QR_DRAW_MODE = "surface"
DEFAULT_SPIN_WAIT_US = 1_000


def format_timestamp(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    return f"{seconds:,}".replace(",", " ") + f".{nanoseconds // 1_000_000:03d}"


def grid_areas(width: int, height: int, grid_qrs: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return cell rectangles below a reserved status strip."""
    rows, columns = grid_shape(grid_qrs)
    content_height = max(1, height - STATUS_STRIP_HEIGHT)
    return tuple(
        (width * column // columns,
         STATUS_STRIP_HEIGHT + content_height * row // rows,
         width * (column + 1) // columns,
         STATUS_STRIP_HEIGHT + content_height * (row + 1) // rows)
        for row, column in grid_positions(grid_qrs)
    )


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


class SwapTimingMonitor:
    """Estimate nominal refresh boundaries and record software swap returns."""

    def __init__(self, refresh_hz: float):
        if not math.isfinite(refresh_hz) or not 1 <= refresh_hz <= 1000:
            raise ValueError("Refresh rate must be between 1 and 1000 Hz")
        self.refresh_hz = refresh_hz
        self.period_ns = round(1_000_000_000 / refresh_hz)
        self.last_swap_ns: int | None = None

    def reset(self) -> None:
        self.last_swap_ns = None

    def predict_next_swap(self, paint_start_ns: int) -> int:
        predicted = paint_start_ns + self.period_ns if self.last_swap_ns is None else self.last_swap_ns + self.period_ns
        if predicted <= paint_start_ns:
            predicted += ((paint_start_ns - predicted) // self.period_ns + 1) * self.period_ns
        return predicted

    def observe(self, marker_ns: int, submit_ns: int, swap_return_ns: int,
                *, paint_start_ns: int | None = None) -> dict:
        paint_start_ns = marker_ns if paint_start_ns is None else paint_start_ns
        interval = None if self.last_swap_ns is None else swap_return_ns - self.last_swap_ns
        skipped = max(0, round(interval / self.period_ns) - 1) if interval and interval > 1.5 * self.period_ns else 0
        self.last_swap_ns = swap_return_ns
        return {
            "marker_ns": marker_ns,
            "paint_start_ns": paint_start_ns,
            "submit_ns": submit_ns,
            "flip_return_ns": swap_return_ns,
            "frame_period_ns": self.period_ns,
            "interval_ns": interval,
            "render_ns": max(0, submit_ns - paint_start_ns),
            "swap_wait_ns": max(0, swap_return_ns - submit_ns),
            "marker_to_flip_ns": swap_return_ns - marker_ns,
            "late_submit": False,
            "skipped_before_render": 0,
            "missed_after_submit": skipped,
            "skipped_periods": skipped,
            "irregular_interval": interval is not None and not .75 * self.period_ns <= interval <= 1.25 * self.period_ns,
        }


class FramePacer:
    """Pace SDL frames against an absolute refresh grid and retain timing metrics."""

    def __init__(self, anchor_ns: int, refresh_hz: float, spin_wait_us: int = DEFAULT_SPIN_WAIT_US):
        if not math.isfinite(refresh_hz) or not 1 <= refresh_hz <= 1000:
            raise ValueError("Refresh rate must be between 1 and 1000 Hz")
        if not 0 <= spin_wait_us <= 5_000:
            raise ValueError("Spin wait must be from 0 to 5000 microseconds")
        self.nominal_period_ns = round(1_000_000_000 / refresh_hz)
        self.period_ns = self.nominal_period_ns
        self.deadline_ns = anchor_ns + self.period_ns
        self.render_budget_ns = min(1_500_000, self.period_ns // 3)
        self.spin_wait_ns = spin_wait_us * 1_000
        self.last_flip_ns: int | None = None
        self.render_times = deque(maxlen=120)

    def _skip_expired(self, now_ns: int) -> int:
        if now_ns < self.deadline_ns:
            return 0
        skipped = (now_ns - self.deadline_ns) // self.period_ns + 1
        self.deadline_ns += skipped * self.period_ns
        return skipped

    def predict_next_flip(self, paint_start_ns: int) -> int:
        predicted = self.deadline_ns if self.last_flip_ns is None else self.last_flip_ns + self.period_ns
        if predicted <= paint_start_ns:
            predicted += ((paint_start_ns - predicted) // self.period_ns + 1) * self.period_ns
        return predicted

    def wait(self, should_stop) -> tuple[bool, int]:
        import pygame

        skipped = self._skip_expired(time.monotonic_ns())
        while True:
            if should_stop():
                return False, skipped
            remaining = self.deadline_ns - self.render_budget_ns - time.monotonic_ns()
            if remaining <= 0:
                additional = self._skip_expired(time.monotonic_ns())
                skipped += additional
                if not additional:
                    return True, skipped
                continue
            if remaining > self.spin_wait_ns:
                pygame.time.wait(min(10, max(1, (remaining - self.spin_wait_ns) // 1_000_000)))
            else:
                deadline = self.deadline_ns - self.render_budget_ns
                while time.monotonic_ns() < deadline:
                    if should_stop():
                        return False, skipped

    def observe(self, marker_ns: int, submit_ns: int, flip_return_ns: int, skipped: int,
                *, paint_start_ns: int | None = None) -> dict:
        paint_start_ns = marker_ns if paint_start_ns is None else paint_start_ns
        interval = None if self.last_flip_ns is None else flip_return_ns - self.last_flip_ns
        missed = max(0, (flip_return_ns - self.deadline_ns + self.period_ns // 4) // self.period_ns)
        render_ns = max(0, submit_ns - paint_start_ns)
        row = {
            "marker_ns": marker_ns, "paint_start_ns": paint_start_ns,
            "deadline_ns": self.deadline_ns, "submit_ns": submit_ns,
            "flip_return_ns": flip_return_ns, "frame_period_ns": self.period_ns,
            "interval_ns": interval, "late_submit": submit_ns > self.deadline_ns,
            "skipped_before_render": skipped, "missed_after_submit": missed,
            "skipped_periods": skipped + missed,
            "irregular_interval": interval is not None and not .75 * self.period_ns <= interval <= 1.25 * self.period_ns,
            "render_ns": render_ns, "swap_wait_ns": max(0, flip_return_ns - submit_ns),
            "marker_to_flip_ns": flip_return_ns - marker_ns,
        }
        self.render_times.append(render_ns)
        ordered = sorted(self.render_times)
        p95 = ordered[math.ceil(.95 * len(ordered)) - 1]
        self.render_budget_ns = min(self.period_ns // 2, max(500_000, p95 + 750_000))
        self.deadline_ns += (missed + 1) * self.period_ns
        self.last_flip_ns = flip_return_ns
        return row


class DisplayJournal:
    """Append timing rows as frames arrive; never retain the full display history."""

    def __init__(self, path: str | Path | None, metadata: dict):
        self.file = None if path is None else Path(path).open("x", encoding="utf-8", buffering=65536)
        self.frame_count = 0
        self.counts = {"missed_period_candidates": 0, "irregular_intervals": 0, "late_submissions": 0}
        self.last_flush_ns = time.monotonic_ns()
        self._closed = False
        self._write({"kind": "session", "format": DISPLAY_FORMAT, **metadata})
        if self.file:
            self.file.flush()

    def _write(self, row: dict) -> None:
        if self.file:
            self.file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def append(self, cell: int, timing: dict) -> None:
        row = {"kind": "frame", "index": self.frame_count, "display_frame": self.frame_count,
               "cell": cell, "corner": cell, **timing}
        self.frame_count += 1
        self.counts["missed_period_candidates"] += row.get("skipped_periods", 0)
        self.counts["irregular_intervals"] += bool(row.get("irregular_interval"))
        self.counts["late_submissions"] += bool(row.get("late_submit"))
        self._write(row)
        issues = timing_issues(row)
        if issues:
            self._write({"kind": "timing_event", "detected_at_monotonic_ns": row.get("flip_return_ns"),
                         "affected_display_indices": [row["index"]], "issues": issues})
        now = time.monotonic_ns()
        if self.file and now - self.last_flush_ns >= 1_000_000_000:
            self.file.flush()
            self.last_flush_ns = now

    def pause(self, paused: bool, timestamp_ns: int) -> None:
        self._write({"kind": "pause", "paused": paused, "monotonic_ns": timestamp_ns,
                     "last_frame_index": self.frame_count - 1})
        if self.file:
            self.file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._write({"kind": "summary", "frames": self.frame_count, **self.counts})
        if self.file:
            self.file.close()
