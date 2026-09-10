from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from multiprocessing import get_context
from queue import Empty

import FreeSimpleGUI as sg

import application_core as base
from calibration.qr import GRID_LAYOUTS
from sensors.camera.camera_gstreamer import gstreamer_main
from sensors.camera.camera_pipeline import available_decoder_backends
from sensors.radar.connection_main import create_connection_communication
from sensors.gps.gps_connection import main as gps_main
from menu_configurations import Configurations
from processing.playback.playback import playback_main
from processing.playback.snapshot_playback import snapshot_playback_main


DISPLAY_JOURNAL_NAME = "display_timestamps.jsonl"
CALIBRATION_DECODER_BACKENDS = {
    "Usual NVIDIA": "rtx",
    "ARM / Jetson": "orin",
    "CPU": "cpu",
}


@dataclass
class RuntimeState:
    pending_playback_folder: str | None = None
    pending_snapshot_playback: dict | None = None
    recording_stop_pending: bool = False
    calibration_recording_deadline: float | None = None
    calibration_recording_root: str | None = None
    calibration_prepared_folder: str | None = None
    calibration_recording_folder: str | None = None
    pending_calibration_camera: dict | None = None
    calibration_clock_process: object | None = None
    calibration_clock_stop_event: object | None = None
    calibration_clock_error_queue: object | None = None
    visualization_process: object | None = None
    process_context: object | None = None


def _resolution(values):
    try:
        width = int(str(values.get("playback_width", "1280")).strip())
        height = int(str(values.get("playback_height", "720")).strip())
    except ValueError as error:
        raise ValueError("Playback width and height must be integers") from error
    if width <= 0 or height <= 0:
        raise ValueError("Playback width and height must be positive")
    return width, height


def _point_cutoff(values):
    try:
        cutoff = float(str(values.get("point_cutoff", "15")).strip())
    except ValueError as error:
        raise ValueError("Point cutoff must be a number in meters") from error
    if cutoff <= 0:
        raise ValueError("Point cutoff must be greater than zero")
    return cutoff


def _graph_resolution(values):
    try:
        width = int(str(values.get("graph_width", "800")).strip())
        height = int(str(values.get("graph_height", "600")).strip())
    except ValueError as error:
        raise ValueError("Graph width and height must be integers") from error
    if width <= 100 or height <= 100:
        raise ValueError("Graph width and height must be greater than 100 pixels")
    return width, height


def _graph_range(values):
    try:
        x_range = float(str(values.get("graph_x_range", "15")).strip())
        y_range = float(str(values.get("graph_y_range", "15")).strip())
    except ValueError as error:
        raise ValueError("Graph X and Y ranges must be numbers in meters") from error
    if (
        not math.isfinite(x_range)
        or not math.isfinite(y_range)
        or x_range <= 0
        or y_range <= 0
    ):
        raise ValueError("Graph X and Y ranges must be greater than zero")
    return x_range, y_range


def _camera_latency_settings(values):
    try:
        pipeline_latency_ms = int(str(values.get("camera_pipeline_latency", "145")).strip())
        adjustment_ms = float(str(values.get("camera_latency_adjustment", "109")).strip())
    except ValueError as error:
        raise ValueError("Both camera latency values must be numeric") from error
    if pipeline_latency_ms < 0:
        raise ValueError("rtspsrc latency cannot be negative")
    return pipeline_latency_ms, adjustment_ms


def _camera_recording_rate(values):
    try:
        frames_per_30 = int(str(values.get("camera_recording_rate", "30")).strip())
    except ValueError as error:
        raise ValueError("Recorded camera frames must be a whole number") from error
    if not 1 <= frames_per_30 <= 30:
        raise ValueError("Recorded camera frames must be between 1 and 30")
    return frames_per_30


def _calibration_decoder_backend(values):
    label = str(values.get("calibration_decoder", "Usual NVIDIA")).strip()
    try:
        return CALIBRATION_DECODER_BACKENDS[label]
    except KeyError as error:
        choices = ", ".join(CALIBRATION_DECODER_BACKENDS)
        raise ValueError(f"Select a camera pipeline: {choices}") from error


def _calibration_screen_index(values):
    selection = str(values.get("calibration_screen", "0")).strip()
    try:
        index = int(selection.split(":", 1)[0])
    except ValueError as error:
        raise ValueError("Select a valid QR display monitor") from error
    if index < 0:
        raise ValueError("Select a valid QR display monitor")
    return index


def _calibration_grid_qrs(values):
    try:
        grid_qrs = int(str(values.get("calibration_grid_qrs", "4")).strip())
    except ValueError as error:
        raise ValueError("QR grid amount must be a whole number") from error
    if grid_qrs not in GRID_LAYOUTS:
        choices = ", ".join(str(value) for value in GRID_LAYOUTS)
        raise ValueError(f"QR grid amount must be one of: {choices}")
    return grid_qrs


def _calibration_visible_qrs(values, grid_qrs=4):
    try:
        visible_qrs = int(str(values.get("calibration_visible_qrs", "2")).strip())
    except ValueError as error:
        raise ValueError(
            f"Visible QR codes must be a whole number from 1 to {grid_qrs}"
        ) from error
    if not 1 <= visible_qrs <= grid_qrs:
        raise ValueError(f"Visible QR codes must be from 1 to {grid_qrs}")
    return visible_qrs


def _validate_calibration_decoder(values):
    decoder_backend = _calibration_decoder_backend(values)
    available_decoder_backends(decoder_backend, strict=True)
    return decoder_backend


def _set_transposition(active, config, send_radar, send_cam, message=None):
    active = bool(active)
    if active:
        config.window["choose_2"].update(value=True)
        send_radar.send(("choose", 2))
        send_cam.send(("choose", 2))
    payload = {"active": active}
    send_radar.send(("transposition", payload))
    send_cam.send(("transposition", payload))
    config.change_transposition(active, message)


def _qt_screen_choices():
    """Read Qt's monitor order in a disposable process, before creating Tk."""
    command = [
        sys.executable,
        "-m",
        "calibration.display_qt",
        "--list-screens-json",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        catalog = json.loads(completed.stdout.strip().splitlines()[-1])
        if completed.returncode == 0 and catalog:
            return [
                (
                    f"{item['index']}: {item['name']} @ "
                    f"{item['refresh_hz']:.3f} Hz"
                )
                for item in catalog
            ]
    except (IndexError, json.JSONDecodeError, KeyError, OSError, subprocess.TimeoutExpired):
        pass
    # Still allow an external screen to be selected if discovery is unavailable.
    return [
        "0: Primary @ unknown Hz",
        "1: Monitor 2 @ unknown Hz",
        "2: Monitor 3 @ unknown Hz",
        "3: Monitor 4 @ unknown Hz",
    ]


def _run_calibration_clock_process(stop_event, error_queue, display_options):
    try:
        from calibration.display_qt import run_calibration_display

        exit_code = run_calibration_display(stop_event, **display_options)
        if exit_code:
            raise RuntimeError(f"Qt display exited with status {exit_code}")
    except KeyboardInterrupt:
        return
    except BaseException as error:
        message = str(error).strip() or type(error).__name__
        try:
            error_queue.put(message, timeout=0.5)
        except Exception:
            pass
        raise


def _request_calibration_camera(
    values,
    config,
    runtime,
    send_radar,
    send_cam,
    send_playback,
    send_snapshot_playback,
):
    if config.calibration_camera:
        runtime.pending_calibration_camera = None
        runtime.calibration_recording_deadline = None
        runtime.calibration_recording_root = None
        runtime.calibration_prepared_folder = None
        if config.calibration_recording or runtime.calibration_recording_folder:
            send_cam.send(("record_stop", None))
        send_cam.send(("calibration_camera", {"active": False}))
        return

    try:
        pipeline_latency_ms, adjustment_ms = _camera_latency_settings(values)
        recording_frames_per_30 = _camera_recording_rate(values)
        decoder_backend = _calibration_decoder_backend(values)
    except ValueError as error:
        config.show_calibration_error(str(error))
        return

    config.set_calibration_camera_pending()
    runtime.pending_calibration_camera = {
        "pipeline_latency_ms": pipeline_latency_ms,
        "latency_adjustment_ms": adjustment_ms,
        "recording_frames_per_30": recording_frames_per_30,
        "decoder_backend": decoder_backend,
    }
    if config.recording or config.recording_pending:
        base._request_recording_stop(config, runtime, send_cam)
    if config.connected_radar:
        send_radar.send(("conn_radar", None))
    if config.playback_pending:
        runtime.pending_playback_folder = None
        config.change_playback({"active": False, "completed": False})
    if config.playback:
        send_playback.send(("playback_stop", None))
    if config.snapshot_playback_pending:
        runtime.pending_snapshot_playback = None
        config.change_snapshot_playback({"active": False, "completed": False})
    if config.snapshot_playback:
        send_snapshot_playback.send(("snapshot_playback_stop", None))
    _maybe_open_calibration_camera(config, runtime, send_cam)


def _maybe_open_calibration_camera(config, runtime, send_cam):
    payload = runtime.pending_calibration_camera
    if (
        payload is None
        or config.connected_radar
        or config.recording
        or config.recording_pending
        or config.playback
        or config.playback_pending
        or config.snapshot_playback
        or config.snapshot_playback_pending
    ):
        return
    runtime.pending_calibration_camera = None
    recording_frames_per_30 = payload.pop("recording_frames_per_30")
    decoder_backend = payload.pop("decoder_backend")
    send_cam.send(("camera_decoder_backend", {"backend": decoder_backend}))
    send_cam.send(("camera_latency_settings", payload))
    send_cam.send((
        "camera_recording_rate",
        {"frames_per_30": recording_frames_per_30},
    ))
    send_cam.send(("calibration_camera", {"active": True}))


def _start_calibration_clock(values, config, runtime):
    process = runtime.calibration_clock_process
    if process is not None and process.is_alive():
        return

    try:
        screen_index = _calibration_screen_index(values)
        grid_qrs = _calibration_grid_qrs(values)
        visible_qrs = _calibration_visible_qrs(values, grid_qrs)
        _validate_calibration_decoder(values)
    except (RuntimeError, ValueError) as error:
        config.show_calibration_error(str(error))
        return

    recording_root = None
    if config.calibration_camera and config.connected_cam and not config.calibration_recording:
        recording_root = Path(values.get("record_folder", "")).expanduser()
        if not recording_root.is_dir():
            config.show_calibration_error(
                "Select an existing recording destination before starting QR calibration"
            )
            return

    # Prepare the destination before starting the display so its first marker
    # has evidence too. JPEG capture still starts after the three-second delay.
    journal_path = None
    if recording_root is not None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            folder = recording_root.resolve() / f"calibration_camera_4_{timestamp}"
            folder.mkdir(parents=False, exist_ok=False)
        except OSError as error:
            config.show_calibration_error(f"Could not create calibration recording: {error}")
            return
        runtime.calibration_prepared_folder = str(folder)
        journal_path = str(folder / DISPLAY_JOURNAL_NAME)

    stop_event = runtime.process_context.Event()
    error_queue = runtime.process_context.Queue(1)
    process = runtime.process_context.Process(
        target=_run_calibration_clock_process,
        args=(stop_event, error_queue, {
            "journal_path": journal_path,
            "screen_index": screen_index,
            "visible_qrs": visible_qrs,
            "grid_qrs": grid_qrs,
        }),
        name="calibration-clock",
    )
    try:
        process.start()
    except (OSError, RuntimeError) as error:
        error_queue.close()
        runtime.calibration_prepared_folder = None
        config.show_calibration_error(f"Could not start QR display: {error}")
        return
    runtime.calibration_clock_process = process
    runtime.calibration_clock_stop_event = stop_event
    runtime.calibration_clock_error_queue = error_queue
    config.change_calibration_clock(True)

    if not (config.calibration_camera and config.connected_cam):
        runtime.calibration_recording_deadline = None
        runtime.calibration_recording_root = None
        config.window["calibration_status"].update(
            "QR ACTIVE — CAMERA 4 IS NOT OPEN, SO RECORDING WAS NOT SCHEDULED"
        )
        return
    if config.calibration_recording:
        runtime.calibration_recording_deadline = None
        runtime.calibration_recording_root = None
        config.window["calibration_status"].update(
            "QR ACTIVE — CAMERA 4 IS ALREADY RECORDING"
        )
        return
    assert recording_root is not None
    runtime.calibration_recording_root = str(recording_root.resolve())
    runtime.calibration_recording_deadline = time.monotonic() + 3.0
    config.window["calibration_status"].update("QR ACTIVE — RECORDING IN 3 SECONDS")


def _service_calibration(config, runtime, send_cam):
    process = runtime.calibration_clock_process
    if process is not None and not process.is_alive():
        process.join(timeout=0.1)
        error_message = None
        if runtime.calibration_clock_error_queue is not None:
            try:
                error_message = runtime.calibration_clock_error_queue.get_nowait()
            except Empty:
                pass
            runtime.calibration_clock_error_queue.close()
        runtime.calibration_clock_process = None
        runtime.calibration_clock_stop_event = None
        runtime.calibration_clock_error_queue = None
        runtime.calibration_recording_deadline = None
        runtime.calibration_recording_root = None
        runtime.calibration_prepared_folder = None
        if config.calibration_recording or runtime.calibration_recording_folder:
            send_cam.send(("record_stop", None))
        config.change_calibration_clock(False)
        if process.exitcode:
            config.show_calibration_error(
                error_message
                or "The QR display stopped with an error; check the terminal output."
            )

    deadline = runtime.calibration_recording_deadline
    if deadline is None or time.monotonic() < deadline:
        return
    runtime.calibration_recording_deadline = None
    if not (config.calibration_camera and config.connected_cam):
        runtime.calibration_recording_root = None
        runtime.calibration_prepared_folder = None
        config.window["calibration_status"].update(
            "BARCODE ACTIVE — CAMERA 4 CLOSED BEFORE RECORDING"
        )
        return

    runtime.calibration_recording_root = None
    prepared = runtime.calibration_prepared_folder
    runtime.calibration_prepared_folder = None
    if not prepared or not Path(prepared).is_dir():
        config.show_calibration_error("The prepared calibration recording folder is missing")
        return
    runtime.calibration_recording_folder = prepared
    send_cam.send((
        "record_start",
        {"folders": {4: prepared}, "calibration": True,
         "display_journal": DISPLAY_JOURNAL_NAME},
    ))


def _maybe_start_snapshot_playback(config, runtime, send_snapshot_playback):
    if (
        runtime.pending_snapshot_playback
        and not config.recording
        and not config.recording_pending
        and not config.connected_radar
        and not config.connected_cam
        and not config.snapshot_playback
    ):
        payload = runtime.pending_snapshot_playback
        runtime.pending_snapshot_playback = None
        send_snapshot_playback.send(("snapshot_playback_start", payload))


def _disconnect_live_for_snapshot_playback(
    config, runtime, send_radar, send_cam, send_snapshot_playback
):
    if config.connected_cam:
        send_cam.send(("conn_cam", None))
    if config.connected_radar:
        send_radar.send(("conn_radar", None))
    _maybe_start_snapshot_playback(config, runtime, send_snapshot_playback)


def _request_snapshot_playback(
    values, config, runtime, send_radar, send_cam, send_snapshot_playback
):
    if config.snapshot_playback:
        send_snapshot_playback.send(("snapshot_playback_stop", None))
        return

    folder = Path(values.get("snapshot_playback_folder", "")).expanduser()
    if not folder.is_dir():
        config.show_snapshot_playback_error(
            "Select an existing snapshot playback folder"
        )
        return
    try:
        width, height = _resolution(values)
    except ValueError as error:
        config.show_snapshot_playback_error(str(error))
        return

    runtime.pending_snapshot_playback = {
        "folder": str(folder.resolve()),
        "snapshot_folder": str(Path(values.get("snapshot_folder", "")).expanduser().resolve()),
        "width": width,
        "height": height,
        "synced_only": bool(values.get("snapshot_playback_synced_only", True)),
    }
    config.set_snapshot_playback_pending()
    if config.recording or config.recording_pending:
        base._request_recording_stop(config, runtime, send_cam)
    else:
        _disconnect_live_for_snapshot_playback(
            config, runtime, send_radar, send_cam, send_snapshot_playback
        )


def _start_calibration_visualization(values, config, runtime):
    process = runtime.visualization_process
    if process is not None and process.poll() is None:
        return
    folder_text = str(values.get("visualization_folder", "")).strip()
    folder = Path(folder_text).expanduser()
    if not folder_text or not folder.is_dir() or not any(
        (folder / name).is_file() for name in ("camera_timestamps.jsonl", "camera_timestamps.json")
    ):
        sg.popup_error("Select a calibration folder containing camera_timestamps.jsonl or camera_timestamps.json",
                       title="Visualization")
        return
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "analyze_calibration_recording.py"),
        str(folder.resolve()),
    ]
    try:
        runtime.visualization_process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parent))
    except OSError as error:
        sg.popup_error(f"Could not start calibration visualization: {error}", title="Visualization")
        return
    config.window["visualization_open"].update(disabled=True)
    config.window["visualization_status"].update("Viewer open — close its window to select another recording here.")


def _service_visualization(config, runtime):
    process = runtime.visualization_process
    if process is not None and process.poll() is not None:
        runtime.visualization_process = None
        config.window["visualization_open"].update(disabled=False)
        config.window["visualization_status"].update(
            "Viewer closed." if process.returncode == 0 else f"Viewer exited with error {process.returncode}; see terminal details.")


def _stop_visualization(runtime):
    process = runtime.visualization_process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def _handle_gui_event(
    event, values, config, runtime,
    send_radar, send_cam, send_gps, send_playback, send_snapshot_playback,
    shutdown_event,
):
    if event == "calibration_grid_qrs":
        config.change_calibration_qr_grid(
            values.get("calibration_grid_qrs", 4),
            values.get("calibration_visible_qrs", 2),
        )
        return
    if event == "transposition_toggle":
        _set_transposition(
            not config.transposition,
            config,
            send_radar,
            send_cam,
        )
        return
    if (
        isinstance(event, str)
        and event.startswith("choose_")
        and event != "choose_2"
        and config.transposition
    ):
        _set_transposition(False, config, send_radar, send_cam)
    if event == "visualization_open":
        _start_calibration_visualization(values, config, runtime)
        return
    if event == "snapshot_playback_toggle":
        if not config.snapshot_playback and config.transposition:
            _set_transposition(False, config, send_radar, send_cam)
        _request_snapshot_playback(
            values, config, runtime, send_radar, send_cam, send_snapshot_playback
        )
        return
    if event == "snapshot_playback_pause":
        send_snapshot_playback.send(("snapshot_playback_pause", None))
        return
    if event == "snapshot_playback_previous":
        send_snapshot_playback.send(("snapshot_playback_previous", None))
        return
    if event == "snapshot_playback_next":
        send_snapshot_playback.send(("snapshot_playback_next", None))
        return
    if event == "snapshot_playback_snapshot":
        folder = Path(values.get("snapshot_folder", "")).expanduser()
        if not folder.is_dir():
            config.show_snapshot_playback_snapshot_error(
                "Select an existing snapshot destination folder"
            )
            return
        send_snapshot_playback.send((
            "snapshot_playback_snapshot",
            {"folder": str(folder.resolve())},
        ))
        return
    if event == "playback_resolution_apply":
        try:
            width, height = _resolution(values)
        except ValueError as error:
            sg.popup_error(str(error), title="Playback resolution error")
            return
        payload = {"width": width, "height": height}
        send_cam.send(("playback_resolution", payload))
        send_playback.send(("playback_resolution", payload))
        send_snapshot_playback.send(("playback_resolution", payload))
        config.change_playback_resolution(width, height)
        return
    if event == "point_cutoff_apply":
        try:
            cutoff = _point_cutoff(values)
        except ValueError as error:
            sg.popup_error(str(error), title="Point cutoff error")
            return
        payload = {"distance": cutoff}
        send_radar.send(("point_cutoff", payload))
        send_playback.send(("point_cutoff", payload))
        send_snapshot_playback.send(("point_cutoff", payload))
        config.change_point_cutoff(cutoff)
        return
    if event == "graph_settings_apply":
        try:
            width, height = _graph_resolution(values)
            x_range, y_range = _graph_range(values)
        except ValueError as error:
            sg.popup_error(str(error), title="Graph settings error")
            return
        resolution_payload = {"width": width, "height": height}
        range_payload = {"x_range": x_range, "y_range": y_range}
        send_radar.send(("graph_resolution", resolution_payload))
        send_radar.send(("graph_range", range_payload))
        send_playback.send(("graph_resolution", resolution_payload))
        send_playback.send(("graph_range", range_payload))
        send_snapshot_playback.send(("graph_resolution", resolution_payload))
        send_snapshot_playback.send(("graph_range", range_payload))
        config.change_graph_resolution(width, height)
        config.change_graph_range(x_range, y_range)
        return
    if event == "calibration_latency_apply":
        try:
            pipeline_latency_ms, adjustment_ms = _camera_latency_settings(values)
        except ValueError as error:
            config.show_calibration_error(str(error))
            return
        payload = {
            "pipeline_latency_ms": pipeline_latency_ms,
            "latency_adjustment_ms": adjustment_ms,
        }
        send_cam.send(("camera_latency_settings", payload))
        send_radar.send(("camera_latency_adjustment", payload))
        send_snapshot_playback.send(("camera_latency_adjustment", payload))
        return
    if event == "calibration_camera_toggle":
        if not config.calibration_camera and config.transposition:
            _set_transposition(False, config, send_radar, send_cam)
        _request_calibration_camera(
            values,
            config,
            runtime,
            send_radar,
            send_cam,
            send_playback,
            send_snapshot_playback,
        )
        return
    if event == "recording_rate_apply":
        try:
            frames_per_30 = _camera_recording_rate(values)
        except ValueError as error:
            config.show_calibration_error(str(error))
            return
        send_cam.send((
            "camera_recording_rate",
            {"frames_per_30": frames_per_30},
        ))
        return
    if event == "calibration_clock_start":
        _start_calibration_clock(values, config, runtime)
        return

    if event == "playback_toggle" and not config.playback and config.transposition:
        _set_transposition(False, config, send_radar, send_cam)

    base._handle_gui_event(
        event, values, config, runtime,
        send_radar, send_cam, send_gps, send_playback, shutdown_event,
    )
    if isinstance(event, str) and event.startswith("filter"):
        send_snapshot_playback.send((event, values))


def _apply_status_message(
    message, payload, config, runtime,
    send_radar, send_cam, send_playback, send_snapshot_playback,
):
    if message == "snapshot_playback_state":
        config.change_snapshot_playback(payload)
        _maybe_open_calibration_camera(config, runtime, send_cam)
        return
    if message == "snapshot_playback_progress":
        config.change_snapshot_playback_progress(payload)
        return
    if message == "snapshot_playback_error":
        runtime.pending_snapshot_playback = None
        config.show_snapshot_playback_error(payload)
        return
    if message == "snapshot_playback_snapshot_saved":
        config.show_snapshot_playback_snapshot_saved(payload)
        return
    if message == "snapshot_playback_snapshot_error":
        config.show_snapshot_playback_snapshot_error(payload)
        return
    if message == "playback_error" and isinstance(payload, dict):
        payload = payload.get("message", "Playback failed")
    if message in ("graph_resolution_error", "graph_range_error"):
        sg.popup_error(payload, title="Graph display error")
        return
    if message == "calibration_camera_state":
        config.change_calibration_camera(payload.get("active"))
        if not payload.get("active"):
            runtime.calibration_recording_deadline = None
            runtime.calibration_recording_root = None
            runtime.calibration_prepared_folder = None
        return
    if message == "calibration_recording_state":
        state = dict(payload)
        state["folder"] = runtime.calibration_recording_folder or ""
        config.change_calibration_recording(state)
        if not state.get("active"):
            runtime.calibration_recording_folder = None
        return
    if message == "camera_latency_state":
        config.change_calibration_latencies(
            payload["pipeline_latency_ms"],
            payload["latency_adjustment_ms"],
        )
        return
    if message == "camera_latency_error":
        config.show_calibration_error(payload)
        return
    if message == "camera_pipeline_error":
        config.show_calibration_error(payload)
        return
    if message == "transposition_state":
        config.change_transposition(
            payload.get("active"),
            payload.get("message"),
        )
        return
    if message == "transposition_error":
        send_radar.send(("transposition", {"active": False}))
        config.change_transposition(False, f"ERROR · {payload}")
        return
    if message == "camera_recording_rate_state":
        config.change_recording_rate(payload["frames_per_30"])
        return
    if message == "camera_recording_rate_error":
        config.show_calibration_error(payload)
        return
    if message == "camera_recording_drop":
        config.change_camera_recording_drop(payload)
        return
    if message == "camera_ntp_time":
        config.change_camera_ntp(payload)
        return
    if message == "camera_snapshot" and payload.get("calibration"):
        return

    base._apply_status_message(
        message, payload, config, runtime,
        send_radar, send_cam, send_playback,
    )

    if message == "change_cam" and not payload:
        config.change_camera_ntp({"available": False})

    if message in ("change_radar", "change_cam"):
        _maybe_start_snapshot_playback(config, runtime, send_snapshot_playback)
    elif message == "recording_state" and not payload.get("active"):
        if runtime.pending_snapshot_playback:
            _disconnect_live_for_snapshot_playback(
                config, runtime, send_radar, send_cam, send_snapshot_playback
            )
    if message in ("change_radar", "recording_state", "playback_state"):
        _maybe_open_calibration_camera(config, runtime, send_cam)


def _drain_status_queue(
    all_queue, config, runtime,
    send_radar, send_cam, send_playback, send_snapshot_playback,
):
    while True:
        try:
            message, payload = all_queue.get_nowait()
        except Empty:
            return
        _apply_status_message(
            message, payload, config, runtime,
            send_radar, send_cam, send_playback, send_snapshot_playback,
        )


def _run_event_loop(
    config, all_queue, runtime,
    send_radar, send_cam, send_gps, send_playback, send_snapshot_playback,
    shutdown_event,
):
    while not shutdown_event.is_set():
        event, values = config.read()
        _handle_gui_event(
            event, values, config, runtime,
            send_radar, send_cam, send_gps, send_playback, send_snapshot_playback,
            shutdown_event,
        )
        _drain_status_queue(
            all_queue, config, runtime,
            send_radar, send_cam, send_playback, send_snapshot_playback,
        )
        _service_calibration(config, runtime, send_cam)
        _service_visualization(config, runtime)


def _stop_calibration_clock(runtime):
    process = runtime.calibration_clock_process
    if process is None:
        return
    if runtime.calibration_clock_stop_event is not None:
        runtime.calibration_clock_stop_event.set()
    process.join(timeout=2.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    if runtime.calibration_clock_error_queue is not None:
        runtime.calibration_clock_error_queue.close()
    runtime.calibration_clock_process = None
    runtime.calibration_clock_stop_event = None
    runtime.calibration_clock_error_queue = None


def main():
    process_context = get_context("spawn")
    shutdown_event = process_context.Event()

    def signal_handler(_sig, _frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    sg.set_options(font=("Helvetica", 12))
    all_queue = process_context.Queue(128)
    transposition_channel = process_context.Queue(1)

    receive_radar, send_radar = process_context.Pipe()
    receive_cam, send_cam = process_context.Pipe()
    receive_gps, send_gps = process_context.Pipe()
    receive_playback, send_playback = process_context.Pipe()
    receive_snapshot_playback, send_snapshot_playback = process_context.Pipe()

    config = Configurations(calibration_screens=_qt_screen_choices())
    _, values = config.read()
    runtime = RuntimeState()
    runtime.process_context = process_context

    processes = [
        process_context.Process(
            target=create_connection_communication,
            args=(
                values,
                receive_radar,
                all_queue,
                shutdown_event,
                transposition_channel,
            ),
        ),
        process_context.Process(
            target=gstreamer_main,
            args=(
                receive_cam,
                all_queue,
                shutdown_event,
                transposition_channel,
            ),
        ),
        process_context.Process(
            target=gps_main,
            args=(receive_gps, all_queue, shutdown_event),
        ),
        process_context.Process(
            target=playback_main,
            args=(receive_playback, all_queue, shutdown_event, values),
        ),
        process_context.Process(
            target=snapshot_playback_main,
            args=(receive_snapshot_playback, all_queue, shutdown_event, values),
        ),
    ]
    for process in processes:
        process.start()

    try:
        _run_event_loop(
            config, all_queue, runtime,
            send_radar, send_cam, send_gps, send_playback, send_snapshot_playback,
            shutdown_event,
        )
    finally:
        _stop_calibration_clock(runtime)
        _stop_visualization(runtime)
        try:
            base._shutdown(
                processes,
                (send_radar, send_cam, send_gps, send_playback, send_snapshot_playback),
                config,
                shutdown_event,
            )
        finally:
            transposition_channel.close()


if __name__ == "__main__":
    main()
