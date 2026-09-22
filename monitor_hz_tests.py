"""Compare Qt/OpenGL and Pygame/SDL presentation timing on one monitor.

This program measures software-visible presentation timing. It cannot measure
HDMI transport, monitor processing, panel scanout, or photon output latency.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
GRID_CHOICES = (4, 6, 8, 9, 10, 12)


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _spin_wait_us(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a whole number") from error
    if not 0 <= number <= 5_000:
        raise argparse.ArgumentTypeError("must be from 0 to 5000 microseconds")
    return number


def _screen_catalog() -> list[dict]:
    """Query Qt screens in a separate process so other GUI backends stay isolated."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "calibration.display_qt",
            "--list-screens-json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Qt could not enumerate monitors: {detail}")

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Qt returned no monitor information")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Qt returned invalid monitor information: " + lines[-1]
        ) from error


def list_screens() -> int:
    screens = _screen_catalog()
    print("Qt/OpenGL screens:")
    for screen in screens:
        primary = " (primary)" if screen.get("primary") else ""
        print(
            f"  [{screen['index']}] {screen['name']}: "
            f"{screen['width']}x{screen['height']} "
            f"@ {screen['refresh_hz']:.3f} Hz "
            f"at ({screen['x']}, {screen['y']}){primary}"
        )

    print("\nPygame/SDL displays:")
    try:
        import os

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        pygame.display.init()
        try:
            sizes = pygame.display.get_desktop_sizes()
            for index, (width, height) in enumerate(sizes):
                print(f"  [{index}] {width}x{height}")
        finally:
            pygame.display.quit()
    except Exception as error:
        print(f"  unavailable: {error}")

    print(
        "\nQt and SDL enumerate displays independently. Confirm that the chosen "
        "index identifies the same physical monitor in both lists."
    )
    return 0


def _percentile_ns(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index] / 1_000_000


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} ms"


def print_summary(
    *,
    backend: str,
    workload: str,
    expected_hz: float,
    samples: list[dict],
    extra: dict | None = None,
) -> None:
    print(f"\n=== {backend} / {workload} ===")
    if extra:
        for key, value in extra.items():
            print(f"{key}: {value}")

    if len(samples) < 2:
        print("Not enough completed presentations to calculate a result.")
        return

    period_ns = round(1_000_000_000 / expected_hz)
    intervals = [
        sample["interval_ns"]
        for sample in samples
        if sample.get("interval_ns") is not None
    ]
    schedule = [
        sample["schedule_ns"]
        for sample in samples
        if sample.get("schedule_ns") is not None
    ]
    paint = [sample["paint_ns"] for sample in samples]
    swap = [sample["swap_ns"] for sample in samples]
    prediction_errors = [
        sample["prediction_error_ns"]
        for sample in samples
        if sample.get("prediction_error_ns") is not None
    ]
    automatic_gc = any(sample["automatic_gc"] for sample in samples)

    elapsed_ns = samples[-1]["flip_ns"] - samples[0]["flip_ns"]
    effective_hz = (len(samples) - 1) * 1_000_000_000 / elapsed_ns
    irregular = sum(
        not 0.75 * period_ns <= interval <= 1.25 * period_ns
        for interval in intervals
    )
    missed = sum(
        max(0, round(interval / period_ns) - 1)
        if interval > 1.5 * period_ns
        else 0
        for interval in intervals
    )

    print(f"Expected refresh: {expected_hz:.3f} Hz ({period_ns / 1e6:.3f} ms)")
    print(f"Completed presentations: {len(samples)}")
    print(f"Effective presentation rate: {effective_hz:.3f} Hz")
    print(
        "Presentation interval median / P95 / maximum: "
        f"{_format_ms(_percentile_ns(intervals, 0.50))} / "
        f"{_format_ms(_percentile_ns(intervals, 0.95))} / "
        f"{_format_ms(_percentile_ns(intervals, 1.00))}"
    )
    print(
        "Wait/scheduling median / P95: "
        f"{_format_ms(_percentile_ns(schedule, 0.50))} / "
        f"{_format_ms(_percentile_ns(schedule, 0.95))}"
    )
    print(
        "Paint phase median / P95: "
        f"{_format_ms(_percentile_ns(paint, 0.50))} / "
        f"{_format_ms(_percentile_ns(paint, 0.95))}"
    )
    print(
        "Flip-return phase median / P95: "
        f"{_format_ms(_percentile_ns(swap, 0.50))} / "
        f"{_format_ms(_percentile_ns(swap, 0.95))}"
    )
    print(f"Automatic cyclic GC: {'enabled' if automatic_gc else 'disabled'}")
    print("Manual garbage collection: never called")
    if prediction_errors:
        median_error = _percentile_ns(prediction_errors, 0.50)
        absolute_p95 = _percentile_ns(
            [abs(value) for value in prediction_errors],
            0.95,
        )
        print(
            "Predicted flip error median / absolute P95: "
            f"{_format_ms(median_error)} / {_format_ms(absolute_p95)}"
        )
    print(f"Irregular intervals: {irregular} / {len(intervals)}")
    print(f"Missed-period candidates: {missed}")

    slow = sorted(
        (sample for sample in samples if sample.get("interval_ns") is not None),
        key=lambda sample: sample["interval_ns"],
        reverse=True,
    )[:5]
    print("Slowest presentation intervals:")
    for sample in slow:
        prediction_error = sample.get("prediction_error_ns")
        prediction_detail = (
            "prediction error n/a"
            if prediction_error is None
            else f"prediction error {prediction_error / 1e6:.3f} ms"
        )
        print(
            "  display "
            f"{sample['display_index']}: "
            f"interval {sample['interval_ns'] / 1e6:.3f} ms, "
            f"paint {sample['paint_ns'] / 1e6:.3f} ms, "
            f"flip-return {sample['swap_ns'] / 1e6:.3f} ms, "
            f"{prediction_detail}"
        )


def print_interpretation_guide() -> None:
    print("\nHow to interpret the four tests:")
    print("  - Solid good, QR poor in both: QR generation/drawing is the bottleneck.")
    print("  - Pygame good, both Qt tests poor: Qt/OpenGL scheduling or its driver path.")
    print("  - Qt solid good, Qt QR poor: the OpenGL-backed QPainter QR path.")
    print("  - Both solid tests poor: investigate compositor, VSync, and display mode.")


def run_qt_test(args: argparse.Namespace, screen: dict, expected_hz: float) -> int:
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QColor, QGuiApplication, QPainter
        from PySide6.QtOpenGL import QOpenGLWindow
    except ImportError as error:
        raise RuntimeError("PySide6 is required for the Qt/OpenGL test") from error

    from calibration.display_qt import (
        QRClockRenderer,
        SwapTimingMonitor,
        configure_surface_format,
    )

    gc_disabled_for_test = args.disable_gc
    configure_surface_format()
    app = QGuiApplication(["monitor-hz-tests"])
    qt_screens = app.screens()
    if not 0 <= args.screen < len(qt_screens):
        raise ValueError(
            f"Qt screen {args.screen} is unavailable; use --list-screens"
        )

    class TestWindow(QOpenGLWindow):
        def __init__(self) -> None:
            update_behavior = (
                QOpenGLWindow.UpdateBehavior.PartialUpdateBlit
                if args.qt_update_mode == "partial"
                else QOpenGLWindow.UpdateBehavior.NoPartialUpdate
            )
            super().__init__(update_behavior)
            self.samples: list[dict] = []
            self.renderer = None
            self.monitor = SwapTimingMonitor(expected_hz)
            self.pending: dict | None = None
            self.last_flip_ns: int | None = None
            self.started_ns: int | None = None
            self.frame_index = 0
            self.setTitle(f"Monitor Hz test - Qt {args.workload}")
            self.frameSwapped.connect(self.frame_swapped)

        def initializeGL(self) -> None:
            if args.workload == "qr":
                self.renderer = QRClockRenderer(
                    self.width(),
                    self.height(),
                    args.visible_qrs,
                    args.grid_qrs,
                    args.qr_mask_pattern,
                )

        def resizeGL(self, width: int, height: int) -> None:
            if self.renderer is not None and width >= 320 and height >= 240:
                self.renderer.resize(width, height)

        def paintGL(self) -> None:
            if self.pending is not None:
                return

            paint_start_ns = time.monotonic_ns()
            marker_ns = (
                self.monitor.predict_next_swap(paint_start_ns)
                if args.timestamp_mode == "predicted-flip"
                else paint_start_ns
            )
            painter = QPainter(self)
            try:
                if args.workload == "qr":
                    self.renderer.render_next(
                        marker_ns,
                        self.frame_index,
                        cache_pixmap=args.qt_draw_mode == "pixmap",
                    )
                    self.renderer.paint(
                        painter,
                        force_full_redraw=args.qt_update_mode == "full",
                        cached_pixmaps=args.qt_draw_mode == "pixmap",
                    )
                else:
                    color = (
                        QColor(48, 48, 48)
                        if self.frame_index % 2
                        else QColor(52, 52, 52)
                    )
                    painter.fillRect(0, 0, self.width(), self.height(), color)
            finally:
                painter.end()

            self.pending = {
                "marker_ns": marker_ns,
                "paint_start_ns": paint_start_ns,
                "submit_ns": time.monotonic_ns(),
                "previous_flip_ns": self.last_flip_ns,
            }

        def frame_swapped(self) -> None:
            flip_ns = time.monotonic_ns()
            if self.pending is None:
                self.update()
                return

            pending = self.pending
            self.pending = None
            timing = self.monitor.observe(
                pending["marker_ns"],
                pending["submit_ns"],
                flip_ns,
                paint_start_ns=pending["paint_start_ns"],
            )
            if self.started_ns is None:
                self.started_ns = flip_ns

            if flip_ns - self.started_ns >= args.warmup * 1_000_000_000:
                previous = pending["previous_flip_ns"]
                self.samples.append(
                    {
                        "interval_ns": (
                            None if previous is None else flip_ns - previous
                        ),
                        "schedule_ns": (
                            None
                            if previous is None
                            else pending["paint_start_ns"] - previous
                        ),
                        "paint_ns": timing["render_ns"],
                        "swap_ns": flip_ns - pending["submit_ns"],
                        "prediction_error_ns": (
                            flip_ns - pending["marker_ns"]
                            if args.timestamp_mode == "predicted-flip"
                            else None
                        ),
                        "flip_ns": flip_ns,
                        "display_index": self.frame_index,
                        "predicted_flip_ns": pending["marker_ns"],
                        "flip_return_ns": flip_ns,
                        "presentation_event_kind": "qt_frame_swapped_return",
                        "physical_presentation_measured": False,
                        "automatic_gc": not gc_disabled_for_test,
                    }
                )

            self.last_flip_ns = flip_ns
            self.frame_index += 1
            self.update()
            if self.renderer is not None and args.timestamp_mode == "predicted-flip":
                self.renderer.prepare_qr(
                    self.monitor.predict_next_swap(time.monotonic_ns())
                )

    window = TestWindow()
    window.setScreen(qt_screens[args.screen])
    geometry = qt_screens[args.screen].geometry()
    if args.windowed:
        window.resize(args.width, args.height)
        window.setPosition(
            geometry.x() + max(0, (geometry.width() - args.width) // 2),
            geometry.y() + max(0, (geometry.height() - args.height) // 2),
        )
        window.show()
    else:
        window.setGeometry(geometry)
        window.showFullScreen()

    QTimer.singleShot(round((args.warmup + args.duration) * 1000), window.close)
    automatic_gc_was_enabled = gc.isenabled()
    if gc_disabled_for_test:
        gc.disable()
    try:
        app.exec()
    finally:
        if gc_disabled_for_test and automatic_gc_was_enabled:
            gc.enable()

    context = window.context()
    actual_format = context.format() if context is not None else window.format()
    print_summary(
        backend=f"Qt/OpenGL ({args.qt_update_mode}, {args.qt_draw_mode})",
        workload=args.workload,
        expected_hz=expected_hz,
        samples=window.samples,
        extra={
            "Screen": (
                f"[{args.screen}] {screen['name']} "
                f"{screen['width']}x{screen['height']}"
            ),
            "Qt-reported refresh": f"{screen['refresh_hz']:.3f} Hz",
            "Actual swap interval": actual_format.swapInterval(),
            "Swap behavior": str(actual_format.swapBehavior()),
            "Timestamp mode": args.timestamp_mode,
            "QR drawing": args.qt_draw_mode,
            "QR mask": (
                "automatic"
                if args.qr_mask_pattern is None
                else f"fixed {args.qr_mask_pattern}"
            ),
            "Retained QR matrices / pixmaps": (
                f"{len(window.renderer.matrices)} / {len(window.renderer.images)}"
                if window.renderer is not None else "0 / 0"
            ),
        },
    )
    return 0 if len(window.samples) >= 2 else 1


def run_pygame_test(
    args: argparse.Namespace,
    screen: dict,
    expected_hz: float,
) -> int:
    import os

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("Pygame is required for the Pygame/SDL test") from error

    from calibration.display import FramePacer, QRClockRenderer

    automatic_gc_was_enabled = gc.isenabled()
    gc_disabled_for_test = args.disable_gc
    if gc_disabled_for_test:
        gc.disable()
    pygame.display.init()
    pygame.font.init()
    samples: list[dict] = []
    try:
        sizes = pygame.display.get_desktop_sizes()
        if not 0 <= args.screen < len(sizes):
            raise ValueError(
                f"SDL display {args.screen} is unavailable; use --list-screens"
            )

        flags = pygame.SCALED | (0 if args.windowed else pygame.FULLSCREEN)
        requested_size = (
            (args.width, args.height) if args.windowed else sizes[args.screen]
        )
        surface = pygame.display.set_mode(
            requested_size,
            flags,
            display=args.screen,
            vsync=1,
        )
        pygame.display.set_caption(f"Monitor Hz test - Pygame {args.workload}")
        renderer = (
            QRClockRenderer(
                surface,
                visible_qrs=args.visible_qrs,
                grid_qrs=args.grid_qrs,
                qr_mask_pattern=args.qr_mask_pattern,
            )
            if args.workload == "qr"
            else None
        )

        pygame.display.flip()
        start_ns = time.monotonic_ns()
        collect_after_ns = start_ns + round(args.warmup * 1_000_000_000)
        stop_ns = collect_after_ns + round(args.duration * 1_000_000_000)
        pacer = FramePacer(
            start_ns,
            expected_hz,
            spin_wait_us=args.pygame_spin_wait_us,
        )
        last_flip_ns: int | None = None
        frame_index = 0
        stopped = False

        if renderer is not None and args.timestamp_mode == "predicted-flip":
            renderer.prepare_qr(pacer.predict_next_flip(time.monotonic_ns()))

        def should_stop() -> bool:
            nonlocal stopped
            if time.monotonic_ns() >= stop_ns:
                stopped = True
                return True
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN
                    and event.key in (pygame.K_q, pygame.K_ESCAPE)
                ):
                    stopped = True
                    return True
            return False

        while not stopped:
            ready, _skipped = pacer.wait(should_stop)
            if not ready:
                continue

            paint_start_ns = time.monotonic_ns()
            marker_ns = (
                pacer.predict_next_flip(paint_start_ns)
                if args.timestamp_mode == "predicted-flip"
                else paint_start_ns
            )
            if renderer is not None:
                renderer.render_next(
                    marker_ns,
                    frame_index,
                    qr_draw_mode=args.pygame_draw_mode,
                )
            else:
                color = (48, 48, 48) if frame_index % 2 else (52, 52, 52)
                surface.fill(color)
            submit_ns = time.monotonic_ns()
            pygame.display.flip()
            flip_ns = time.monotonic_ns()
            timing = pacer.observe(
                marker_ns,
                submit_ns,
                flip_ns,
                _skipped,
                paint_start_ns=paint_start_ns,
            )

            if flip_ns >= collect_after_ns:
                samples.append(
                    {
                        "interval_ns": (
                            None if last_flip_ns is None else flip_ns - last_flip_ns
                        ),
                        "schedule_ns": (
                            None
                            if last_flip_ns is None
                            else paint_start_ns - last_flip_ns
                        ),
                        "paint_ns": timing["render_ns"],
                        "swap_ns": flip_ns - submit_ns,
                        "prediction_error_ns": (
                            flip_ns - marker_ns
                            if args.timestamp_mode == "predicted-flip"
                            else None
                        ),
                        "flip_ns": flip_ns,
                        "display_index": frame_index,
                        "predicted_flip_ns": marker_ns,
                        "flip_return_ns": flip_ns,
                        "presentation_event_kind": "pygame_display_flip_return",
                        "physical_presentation_measured": False,
                        "automatic_gc": not gc_disabled_for_test,
                    }
                )
            last_flip_ns = flip_ns
            frame_index += 1
            if renderer is not None and args.timestamp_mode == "predicted-flip":
                renderer.prepare_qr(pacer.predict_next_flip(time.monotonic_ns()))
    finally:
        pygame.quit()
        if gc_disabled_for_test and automatic_gc_was_enabled:
            gc.enable()

    sdl_size = sizes[args.screen]
    print_summary(
        backend=f"Pygame/SDL ({args.pygame_draw_mode})",
        workload=args.workload,
        expected_hz=expected_hz,
        samples=samples,
        extra={
            "SDL display": f"[{args.screen}] {sdl_size[0]}x{sdl_size[1]}",
            "Expected physical monitor": (
                f"{screen['name']} {screen['width']}x{screen['height']}"
            ),
            "VSync requested": "yes",
            "Timestamp mode": args.timestamp_mode,
            "QR drawing": args.pygame_draw_mode,
            "QR mask": (
                "automatic"
                if args.qr_mask_pattern is None
                else f"fixed {args.qr_mask_pattern}"
            ),
            "Pygame spin wait": f"{args.pygame_spin_wait_us} us",
            "Reusable QR surface pairs": (
                len(renderer._qr_surfaces) if renderer is not None else 0
            ),
        },
    )
    return 0 if len(samples) >= 2 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare monitor presentation cadence using Qt/OpenGL and "
            "Pygame/SDL with solid-color and QR workloads."
        )
    )
    parser.add_argument(
        "--list-screens",
        action="store_true",
        help="list Qt and SDL displays, then exit",
    )
    parser.add_argument(
        "--screen",
        type=int,
        help="display index to test; required unless --list-screens is used",
    )
    parser.add_argument(
        "--backend",
        choices=("all", "qt", "pygame"),
        default="all",
        help="presentation backend to test (default: all)",
    )
    parser.add_argument(
        "--workload",
        choices=("all", "solid", "qr"),
        default="all",
        help="drawing workload to test (default: all)",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        default=10.0,
        help="measured seconds per test (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=_positive_float,
        default=1.0,
        help="unmeasured warm-up seconds per test (default: 1)",
    )
    parser.add_argument(
        "--refresh-hz",
        type=_positive_float,
        help=(
            "expected rate for diagnostics and Pygame pacing; this does not "
            "change the monitor mode (default: selected Qt screen rate)"
        ),
    )
    parser.add_argument(
        "--grid-qrs",
        type=int,
        choices=GRID_CHOICES,
        default=10,
        help="QR grid size (default: 10, matching the 100 Hz recordings)",
    )
    parser.add_argument(
        "--visible-qrs",
        type=int,
        default=5,
        help="recent QR codes kept visible (default: 5)",
    )
    parser.add_argument(
        "--qt-update-mode",
        choices=("partial", "full"),
        default="partial",
        help=(
            "Qt retained partial updates or a forced full repaint "
            "(default: partial)"
        ),
    )
    parser.add_argument(
        "--qt-draw-mode",
        choices=("pixmap", "modules"),
        default="pixmap",
        help="Qt cached-pixmap or original module-by-module QR drawing",
    )
    parser.add_argument(
        "--pygame-draw-mode",
        choices=("surface", "modules"),
        default="surface",
        help="Pygame single-surface or original module-by-module QR drawing",
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=("predicted-flip", "paint-start"),
        default="predicted-flip",
        help=(
            "QR timestamp source: predicted presentation or paint start "
            "(default: predicted-flip)"
        ),
    )
    parser.add_argument(
        "--qr-mask-pattern",
        type=int,
        choices=range(8),
        help=(
            "use a fixed QR mask (0-7) to compare generation speed; default: "
            "automatic lowest-penalty selection"
        ),
    )
    parser.add_argument(
        "--pygame-spin-wait-us",
        type=_spin_wait_us,
        default=1_000,
        help=(
            "Pygame final busy-wait duration in microseconds; lower values save "
            "CPU but can add scheduling jitter (default: 1000)"
        ),
    )
    parser.add_argument(
        "--disable-gc",
        action="store_true",
        help=(
            "disable automatic cyclic GC during presentation without running "
            "manual collections"
        ),
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="use a centered window instead of fullscreen",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser


def _forwarded_arguments(
    args: argparse.Namespace,
    backend: str,
    workload: str,
) -> list[str]:
    result = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--screen",
        str(args.screen),
        "--backend",
        backend,
        "--workload",
        workload,
        "--duration",
        str(args.duration),
        "--warmup",
        str(args.warmup),
        "--grid-qrs",
        str(args.grid_qrs),
        "--visible-qrs",
        str(args.visible_qrs),
        "--qt-update-mode",
        args.qt_update_mode,
        "--qt-draw-mode",
        args.qt_draw_mode,
        "--pygame-draw-mode",
        args.pygame_draw_mode,
        "--timestamp-mode",
        args.timestamp_mode,
        "--pygame-spin-wait-us",
        str(args.pygame_spin_wait_us),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    if args.refresh_hz is not None:
        result.extend(("--refresh-hz", str(args.refresh_hz)))
    if args.windowed:
        result.append("--windowed")
    if args.disable_gc:
        result.append("--disable-gc")
    if args.qr_mask_pattern is not None:
        result.extend(("--qr-mask-pattern", str(args.qr_mask_pattern)))
    return result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_screens:
        return list_screens()
    if args.screen is None:
        parser.error("--screen INDEX is required for a test; use --list-screens")
    if args.screen < 0:
        parser.error("--screen must be zero or greater")
    if not 1 <= args.visible_qrs <= args.grid_qrs:
        parser.error("--visible-qrs must be between 1 and --grid-qrs")
    if args.width < 320 or args.height < 240:
        parser.error("--width and --height must be at least 320x240")

    backends = ("qt", "pygame") if args.backend == "all" else (args.backend,)
    workloads = ("solid", "qr") if args.workload == "all" else (args.workload,)
    if len(backends) * len(workloads) > 1:
        screens = _screen_catalog()
        if not 0 <= args.screen < len(screens):
            raise ValueError(
                f"Screen {args.screen} is unavailable; use --list-screens"
            )
        if args.refresh_hz is None:
            args.refresh_hz = float(screens[args.screen]["refresh_hz"])
        print(
            "Running each backend/workload in a separate process. Press Q or "
            "Escape to stop the active test early."
        )
        status = 0
        for backend in backends:
            for workload in workloads:
                result = subprocess.run(
                    _forwarded_arguments(args, backend, workload),
                    cwd=ROOT,
                    check=False,
                )
                status = status or result.returncode
        print_interpretation_guide()
        print(
            "\nThese results stop at the application/window-system presentation "
            "boundary. HDMI transmission, monitor processing, scanout, and panel "
            "response require an optical or hardware measurement."
        )
        return status

    screens = _screen_catalog()
    if not 0 <= args.screen < len(screens):
        raise ValueError(
            f"Screen {args.screen} is unavailable; use --list-screens"
        )
    screen = screens[args.screen]
    expected_hz = args.refresh_hz or float(screen["refresh_hz"])

    print(
        f"Testing screen [{args.screen}] {screen['name']} at an expected "
        f"{expected_hz:.3f} Hz. Press Q or Escape to stop early."
    )
    if args.backend == "qt":
        return run_qt_test(args, screen, expected_hz)
    return run_pygame_test(args, screen, expected_hz)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
