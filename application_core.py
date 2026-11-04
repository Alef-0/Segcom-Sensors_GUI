from dataclasses import dataclass
from pathlib import Path
import re
import signal
from multiprocessing import get_context
from queue import Empty
import uuid

import FreeSimpleGUI as sg

from sensors.camera.camera_gstreamer import gstreamer_main
from sensors.radar.connection_main import create_connection_communication
from sensors.gps.gps_connection import main as gps_main
from processing.visualization.filter_schema import RCS_KEY
from menu_configurations import Configurations
from processing.playback.playback import playback_main
from processing.recording.point_cloud_recorder import (
    LEGACY_RECORDING_METADATA_NAME,
    LEGACY_TIMESTAMPS_METADATA_NAME,
    RECORDING_METADATA_NAME,
    TIMESTAMPS_METADATA_NAME,
)


@dataclass
class RuntimeState:
    pending_playback_folder: str | None = None
    recording_stop_pending: bool = False


def check_popup():
    layout = [
        [sg.Text("Digite [Alohomora] para confirmar salvar permanentemente nos radares!", justification="center")],
        [sg.Input("", key="passwd", expand_x=True, justification="center")],
        [sg.Push(), sg.Ok(), sg.Cancel(), sg.Push()],
    ]
    window = sg.Window("PASSWORD", layout)
    try:
        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                return False
            if event == "Ok":
                return values["passwd"] == "Alohomora"
    finally:
        window.close()


def _join_processes(processes, timeout=3.0, terminate_timeout=1.0):
    for process in processes:
        process.join(timeout)

    remaining = [process for process in processes if process.is_alive()]
    for process in remaining:
        process.terminate()
    for process in remaining:
        process.join(terminate_timeout)

    # Some workers install a SIGTERM handler so Process.terminate() can request
    # cleanup. If native GUI or media cleanup is stuck, that request may never
    # finish; do not let the parent wait forever during application shutdown.
    remaining = [process for process in remaining if process.is_alive()]
    for process in remaining:
        process.kill()
    for process in remaining:
        process.join(terminate_timeout)


def _start_recording(values, config, send_radar):
    folder = Path(values.get("record_folder", "")).expanduser()
    channels = [
        channel
        for channel in range(1, 4)
        if values.get(f"record_radar_{channel}")
    ]
    if not folder.is_dir():
        sg.popup_error("Select an existing destination folder", title="Recording error")
        return
    if not channels:
        sg.popup_error("Select at least one group to record", title="Recording error")
        return
    missing_devices = [
        name
        for name, connected in (
            ("radar", config.connected_radar),
            ("camera", config.connected_cam),
        )
        if not connected
    ]
    if missing_devices:
        missing = " and ".join(missing_devices)
        result = sg.popup_ok_cancel(
            f"The {missing} {'are' if len(missing_devices) > 1 else 'is'} not open. "
            "Continue anyway? Only data from open devices will be saved.",
            title="Recording without all devices",
        )
        if result != "OK":
            return
    config.set_recording_pending(True)
    send_radar.send(("record_start", {"folder": str(folder.resolve()), "channels": channels}))


def _start_snapshot(values, config, send_cam):
    folder = Path(values.get("snapshot_folder", "")).expanduser()
    if not folder.is_dir():
        config.show_snapshot_error("Select an existing snapshot destination folder")
        return
    if not config.connected_radar or not config.connected_cam:
        config.show_snapshot_error("Connect both the radar and camera before taking a snapshot")
        return

    channel = next(
        (
            channel
            for channel in range(1, 4)
            if values.get(f"snapshot_group_{channel}")
        ),
        None,
    )
    if channel is None:
        config.show_snapshot_error("Select a radar and camera group")
        return

    request_id = uuid.uuid4().hex
    config.set_snapshot_pending(request_id, channel)
    send_cam.send((
        "snapshot_capture",
        {
            "request_id": request_id,
            "folder": str(folder.resolve()),
            "channel": channel,
        },
    ))


def _request_recording_stop(config, runtime, send_cam):
    if runtime.recording_stop_pending:
        return
    runtime.recording_stop_pending = True
    config.set_recording_pending(False)
    send_cam.send(("record_stop", None))


def _is_recording_folder(folder: Path) -> bool:
    return (
        folder.is_dir()
        and any(folder.rglob("*.pcd"))
        and (
            (folder / RECORDING_METADATA_NAME).is_file()
            or (folder / TIMESTAMPS_METADATA_NAME).is_file()
            or (folder / LEGACY_RECORDING_METADATA_NAME).is_file()
            or (folder / LEGACY_TIMESTAMPS_METADATA_NAME).is_file()
        )
    )


def _maybe_start_playback(config, runtime, send_playback):
    if (
        runtime.pending_playback_folder
        and not config.recording
        and not config.recording_pending
        and not config.connected_radar
        and not config.connected_cam
        and not config.playback
    ):
        folder = runtime.pending_playback_folder
        runtime.pending_playback_folder = None
        send_playback.send(("playback_start", {"folder": folder}))


def _disconnect_live_for_playback(config, runtime, send_radar, send_cam, send_playback):
    if config.connected_cam:
        send_cam.send(("conn_cam", None))
    if config.connected_radar:
        send_radar.send(("conn_radar", None))
    _maybe_start_playback(config, runtime, send_playback)


def _request_playback(
    values,
    config,
    runtime,
    send_radar,
    send_cam,
    send_playback,
):
    if config.playback:
        send_playback.send(("playback_stop", None))
        return

    folder = Path(values.get("playback_folder", "")).expanduser()
    if not _is_recording_folder(folder):
        sg.popup_error(
            "Select a recording folder containing PCD files and recording metadata",
            title="Playback error",
        )
        return

    runtime.pending_playback_folder = str(folder.resolve())
    config.set_playback_pending()
    if config.recording or config.recording_pending:
        _request_recording_stop(config, runtime, send_cam)
    else:
        _disconnect_live_for_playback(
            config,
            runtime,
            send_radar,
            send_cam,
            send_playback,
        )


def _handle_gui_event(
    event,
    values,
    config,
    runtime,
    send_radar,
    send_cam,
    send_gps,
    send_playback,
    shutdown_event,
):
    if event == sg.WINDOW_CLOSED:
        shutdown_event.set()
        return

    match event:
        case "Send":
            if config.connected_radar:
                send_radar.send((event, values))
            config.window["save_nvm"].update(button_color=("black", "white"))
        case "save_nvm":
            if config.connected_radar and check_popup():
                config.window["save_nvm"].update(button_color=("white", "green"))
                send_radar.send((event, values))
        case "record_toggle":
            if config.recording:
                _request_recording_stop(config, runtime, send_cam)
            else:
                _start_recording(values, config, send_radar)
        case "snapshot_capture":
            _start_snapshot(values, config, send_cam)
        case "playback_toggle":
            _request_playback(
                values,
                config,
                runtime,
                send_radar,
                send_cam,
                send_playback,
            )
        case "playback_stop":
            if config.playback:
                send_playback.send(("playback_stop", None))
        case "playback_restart":
            if config.playback:
                send_playback.send(("playback_restart", None))
        case "playback_previous_5s":
            if config.playback:
                send_playback.send(("playback_seek", {"seconds": -5.0}))
        case "playback_next_5s":
            if config.playback:
                send_playback.send(("playback_seek", {"seconds": 5.0}))
        case key if isinstance(key, str) and key.startswith("filter"):
            if event == RCS_KEY:
                config.window["RCS_FILTER_VALUE"].update(f"{values[RCS_KEY]:.1f}")
            send_radar.send((event, values))
            send_playback.send((event, values))
        case key if isinstance(key, str) and re.match(r"^choose_", key):
            choice = int(event.rsplit("_", 1)[1])
            send_radar.send(("choose", choice))
            send_cam.send(("choose", choice))
        case key if isinstance(key, str) and re.match(r"^conn_", key):
            if config.playback or config.playback_pending:
                return
            target = {
                "conn_radar": send_radar,
                "conn_cam": send_cam,
                "conn_gps": send_gps,
            }.get(event)
            if target:
                target.send((event, None))
        case "gps_maps":
            send_gps.send((event, None))
        case "DISTANCE":
            config.window["SLIDER_VAL"].update(int(values["DISTANCE"]))


def _apply_status_message(
    message,
    payload,
    config,
    runtime,
    send_radar,
    send_cam,
    send_playback,
):
    match message:
        case "message_201":
            config.change_radar(payload)
        case "received_messages":
            config.change_received_messages(payload)
        case "change_radar":
            config.change_connection_radar(payload)
            _maybe_start_playback(config, runtime, send_playback)
        case "change_cam":
            config.change_connection_cam(payload)
            _maybe_start_playback(config, runtime, send_playback)
        case "gps_text":
            config.window[message].update(payload)
        case "conn_gps":
            config.change_connection_gps(payload)
        case "recording_state":
            config.change_recording(payload)
            if payload.get("active"):
                send_cam.send(("record_start", {"folders": payload.get("folders", {})}))
            else:
                send_cam.send(("record_stop", None))
            if not payload.get("active") and runtime.pending_playback_folder:
                _disconnect_live_for_playback(
                    config,
                    runtime,
                    send_radar,
                    send_cam,
                    send_playback,
                )
        case "recording_progress":
            config.change_recording_progress(payload)
        case "recording_error":
            runtime.recording_stop_pending = False
            send_cam.send(("record_stop", None))
            config.show_recording_error(payload)
        case "camera_snapshot":
            send_radar.send(("record_camera", payload))
        case "manual_snapshot_frame":
            send_radar.send(("snapshot_capture", payload))
        case "manual_snapshot_error":
            config.show_snapshot_error(payload.get("message", "Camera snapshot failed"))
        case "snapshot_saved":
            config.change_snapshot_saved(payload)
        case "snapshot_error":
            config.show_snapshot_error(payload.get("message", "Snapshot failed"))
        case "camera_recording_state":
            if not payload.get("active") and runtime.recording_stop_pending:
                runtime.recording_stop_pending = False
                send_radar.send(("record_stop", None))
        case "camera_recording_error":
            config.show_camera_recording_error(payload)
        case "playback_state":
            config.change_playback(payload)
        case "playback_progress":
            config.change_playback_progress(payload)
        case "playback_error":
            runtime.pending_playback_folder = None
            config.show_playback_error(payload)


def _drain_status_queue(
    all_queue,
    config,
    runtime,
    send_radar,
    send_cam,
    send_playback,
):
    while True:
        try:
            message, payload = all_queue.get_nowait()
        except Empty:
            return
        _apply_status_message(
            message,
            payload,
            config,
            runtime,
            send_radar,
            send_cam,
            send_playback,
        )


def _run_event_loop(
    config,
    all_queue,
    runtime,
    send_radar,
    send_cam,
    send_gps,
    send_playback,
    shutdown_event,
):
    while not shutdown_event.is_set():
        event, values = config.read()
        _handle_gui_event(
            event,
            values,
            config,
            runtime,
            send_radar,
            send_cam,
            send_gps,
            send_playback,
            shutdown_event,
        )
        _drain_status_queue(
            all_queue,
            config,
            runtime,
            send_radar,
            send_cam,
            send_playback,
        )


def _shutdown(processes, pipes, config, shutdown_event):
    shutdown_event.set()
    for pipe in pipes:
        try:
            pipe.send(("STOP", None))
            pipe.close()
        except (BrokenPipeError, EOFError, OSError):
            pass
    _join_processes(processes)
    config.window.close()


def main():
    process_context = get_context("spawn")
    shutdown_event = process_context.Event()

    def signal_handler(_sig, _frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sg.set_options(font=("Helvetica", 12))
    all_queue = process_context.Queue(128)

    receive_radar, send_radar = process_context.Pipe()
    receive_cam, send_cam = process_context.Pipe()
    receive_gps, send_gps = process_context.Pipe()
    receive_playback, send_playback = process_context.Pipe()

    config = Configurations()
    _, values = config.read()
    runtime = RuntimeState()

    processes = [
        process_context.Process(
            target=create_connection_communication,
            args=(values, receive_radar, all_queue, shutdown_event),
        ),
        process_context.Process(
            target=gstreamer_main,
            args=(receive_cam, all_queue, shutdown_event),
        ),
        process_context.Process(
            target=gps_main,
            args=(receive_gps, all_queue, shutdown_event),
        ),
        process_context.Process(
            target=playback_main,
            args=(receive_playback, all_queue, shutdown_event, values),
        ),
    ]
    for process in processes:
        process.start()

    try:
        _run_event_loop(
            config,
            all_queue,
            runtime,
            send_radar,
            send_cam,
            send_gps,
            send_playback,
            shutdown_event,
        )
    finally:
        _shutdown(
            processes,
            (send_radar, send_cam, send_gps, send_playback),
            config,
            shutdown_event,
        )


if __name__ == "__main__":
    main()
