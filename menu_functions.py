from dataclasses import dataclass
from pathlib import Path
from queue import Empty
import re
import uuid
import FreeSimpleGUI as sg

from processing.recording.point_cloud_recorder import RECORDING_METADATA_NAME, TIMESTAMPS_METADATA_NAME
from processing.visualization.filter_schema import RCS_KEY


@dataclass
class WorkerPipes:
    """Bundles IPC pipes to avoid threading multiple separate channels."""
    radar: object
    cam: object
    gps: object
    playback: object
    snapshot: object

    def all_pipes(self):
        return (self.radar, self.cam, self.gps, self.playback, self.snapshot)


@dataclass
class RuntimeState:
    """Tracks asynchronous operation state across GUI ticks."""
    pending_playback_folder: str | None = None
    pending_snapshot_playback: dict | None = None
    recording_stop_pending: bool = False
    process_context: object | None = None


def check_popup():
    """Prompt password dialog before persisting settings to radar NVM."""
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


def join_processes(processes, timeout=3.0, terminate_timeout=1.0):
    """Wait for child processes to terminate, escalating to SIGKILL if stuck."""
    for proc in processes:
        proc.join(timeout)

    remaining = [p for p in processes if p.is_alive()]  # check still running
    for proc in remaining:
        proc.terminate()
    for proc in remaining:
        proc.join(terminate_timeout)

    remaining = [p for p in processes if p.is_alive()]  # escalate to kill if stuck
    for proc in remaining:
        proc.kill()
    for proc in remaining:
        proc.join(terminate_timeout)


def shutdown_workers(processes, pipes, controls, shutdown_event):
    """Stop all background processes and close IPC channels cleanly."""
    shutdown_event.set()
    pipe_list = pipes.all_pipes() if isinstance(pipes, WorkerPipes) else pipes
    for pipe in pipe_list:
        try:
            pipe.send(("STOP", None))
            pipe.close()
        except (BrokenPipeError, EOFError, OSError):
            pass
    join_processes(processes)
    controls.close()


def is_recording_folder(folder: Path) -> bool:
    """Check if directory contains point cloud files and metadata."""
    return (
        folder.is_dir()
        and any(folder.rglob("*.pcd"))
        and ((folder / RECORDING_METADATA_NAME).is_file() or (folder / TIMESTAMPS_METADATA_NAME).is_file())
    )


def start_recording(values, controls, pipes):
    """Validate destination and trigger multi-channel radar/camera recording."""
    folder = Path(values.get("record_folder", "")).expanduser()
    channels = [ch for ch in range(1, 4) if values.get(f"record_radar_{ch}")]
    radar_pipe = pipes.radar if isinstance(pipes, WorkerPipes) else pipes
    if not folder.is_dir():
        sg.popup_error("Select an existing destination folder", title="Recording error")
        return
    if not channels:
        sg.popup_error("Select at least one group to record", title="Recording error")
        return
    missing = [name for name, conn in (("radar", controls.connected_radar), ("camera", controls.connected_cam)) if not conn]
    if missing:
        msg = f"The {' and '.join(missing)} {'are' if len(missing) > 1 else 'is'} not open. Continue anyway?"
        if sg.popup_ok_cancel(msg, title="Recording without all devices") != "OK":
            return
    controls.set_recording_pending(True)
    radar_pipe.send(("record_start", {"folder": str(folder.resolve()), "channels": channels}))


def request_recording_stop(controls, runtime, pipes):
    """Request camera worker to halt recording."""
    if runtime.recording_stop_pending:
        return
    runtime.recording_stop_pending = True
    controls.set_recording_pending(False)
    cam_pipe = pipes.cam if isinstance(pipes, WorkerPipes) else pipes
    cam_pipe.send(("record_stop", None))


def start_snapshot(values, controls, pipes):
    """Capture single synchronized frame from active camera and radar."""
    folder = Path(values.get("snapshot_folder", "")).expanduser()
    if not folder.is_dir():
        controls.show_snapshot_error("Select an existing snapshot destination folder")
        return
    if not controls.connected_radar or not controls.connected_cam:
        controls.show_snapshot_error("Connect both the radar and camera before taking a snapshot")
        return
    channel = next((ch for ch in range(1, 4) if values.get(f"snapshot_group_{ch}")), None)
    if channel is None:
        controls.show_snapshot_error("Select a radar and camera group")
        return
    req_id = uuid.uuid4().hex  # unique request identifier
    controls.set_snapshot_pending(req_id, channel)
    cam_pipe = pipes.cam if isinstance(pipes, WorkerPipes) else pipes
    cam_pipe.send(("snapshot_capture", {"request_id": req_id, "folder": str(folder.resolve()), "channel": channel}))


def maybe_start_playback(controls, runtime, pipes):
    """Start playback worker if live devices are idle and disconnected."""
    if (
        runtime.pending_playback_folder
        and not controls.recording and not controls.recording_pending
        and not controls.connected_radar and not controls.connected_cam
        and not controls.playback
    ):
        folder = runtime.pending_playback_folder
        runtime.pending_playback_folder = None
        pb_pipe = pipes.playback if isinstance(pipes, WorkerPipes) else pipes
        pb_pipe.send(("playback_start", {"folder": folder}))


def disconnect_live_for_playback(controls, runtime, pipes):
    """Disconnect active radar and camera before starting playback."""
    if controls.connected_cam:
        pipes.cam.send(("conn_cam", None))
    if controls.connected_radar:
        pipes.radar.send(("conn_radar", None))
    maybe_start_playback(controls, runtime, pipes)


def request_playback(values, controls, runtime, pipes):
    """Initiate or stop standard recording playback."""
    if controls.playback:
        pipes.playback.send(("playback_stop", None))
        return
    folder = Path(values.get("playback_folder", "")).expanduser()
    if not is_recording_folder(folder):
        sg.popup_error("Select a recording folder containing PCD files and recording metadata", title="Playback error")
        return
    runtime.pending_playback_folder = str(folder.resolve())
    controls.set_playback_pending()
    if controls.recording or controls.recording_pending:
        request_recording_stop(controls, runtime, pipes)
    else:
        disconnect_live_for_playback(controls, runtime, pipes)


def maybe_start_snapshot_playback(controls, runtime, pipes):
    """Start snapshot playback worker if live devices are disconnected."""
    if (
        runtime.pending_snapshot_playback
        and not controls.recording and not controls.recording_pending
        and not controls.connected_radar and not controls.connected_cam
        and not controls.snapshot_playback
    ):
        payload = runtime.pending_snapshot_playback
        runtime.pending_snapshot_playback = None
        pipes.snapshot.send(("snapshot_playback_start", payload))


def disconnect_live_for_snapshot_playback(controls, runtime, pipes):
    """Disconnect live sensors before starting snapshot playback."""
    if controls.connected_cam:
        pipes.cam.send(("conn_cam", None))
    if controls.connected_radar:
        pipes.radar.send(("conn_radar", None))
    maybe_start_snapshot_playback(controls, runtime, pipes)


def request_snapshot_playback(values, controls, runtime, pipes):
    """Initiate or stop snapshot sequence playback."""
    if controls.snapshot_playback:
        pipes.snapshot.send(("snapshot_playback_stop", None))
        return
    folder = Path(values.get("snapshot_playback_folder", "")).expanduser()
    if not folder.is_dir():
        controls.show_snapshot_playback_error("Select an existing snapshot playback folder")
        return
    try:
        w, h = controls.validate_playback_resolution(values)
    except ValueError as err:
        controls.show_snapshot_playback_error(str(err))
        return
    runtime.pending_snapshot_playback = {
        "folder": str(folder.resolve()),
        "snapshot_folder": str(Path(values.get("snapshot_folder", "")).expanduser().resolve()),
        "width": w, "height": h,
        "synced_only": bool(values.get("snapshot_playback_synced_only", True)),
    }
    controls.set_snapshot_playback_pending()
    if controls.recording or controls.recording_pending:
        request_recording_stop(controls, runtime, pipes)
    else:
        disconnect_live_for_snapshot_playback(controls, runtime, pipes)


def set_transposition(active, controls, pipes, message=None):
    """Toggle camera-radar coordinate transposition mode."""
    active = bool(active)
    if active:
        controls.window["choose_2"].update(value=True)
        pipes.radar.send(("choose", 2))
        pipes.cam.send(("choose", 2))
    payload = {"active": active}
    pipes.radar.send(("transposition", payload))
    pipes.cam.send(("transposition", payload))
    controls.update_transposition(active, message)


def handle_gui_event(event, values, controls, runtime, pipes, shutdown_event):
    """Dispatch GUI events to appropriate hardware pipes and state handlers."""
    if   event in (sg.WINDOW_CLOSED, None):             shutdown_event.set()
    elif event in ("playback_width", "playback_height", "graph_width", "graph_height"):
        controls.sanitize_numeric_input(event, values, allow_float=False)
    elif event in ("point_cutoff", "graph_x_range", "graph_y_range"):
        controls.sanitize_numeric_input(event, values, allow_float=True)
    elif event == "transposition_toggle":               set_transposition(not controls.transposition, controls, pipes)
    elif isinstance(event, str) and event.startswith("choose_") and event != "choose_2" and controls.transposition:
        set_transposition(False, controls, pipes)
    elif isinstance(event, str) and re.match(r"^choose_", event):
        choice = int(event.rsplit("_", 1)[1])
        pipes.radar.send(("choose", choice))
        pipes.cam.send(("choose", choice))
    elif event == "snapshot_playback_toggle":
        if not controls.snapshot_playback and controls.transposition:
            set_transposition(False, controls, pipes)
        request_snapshot_playback(values, controls, runtime, pipes)
    elif event in ("snapshot_playback_pause", "snapshot_playback_previous", "snapshot_playback_next"):
        pipes.snapshot.send((event, None))
    elif event == "snapshot_playback_snapshot":
        dest = Path(values.get("snapshot_folder", "")).expanduser()
        if not dest.is_dir():
            controls.show_error("Select an existing snapshot destination folder", "Snapshot error")
        else:
            pipes.snapshot.send(("snapshot_playback_snapshot", {"folder": str(dest.resolve())}))
    elif event == "playback_resolution_apply":
        try:
            w, h = controls.validate_playback_resolution(values)
            payload = {"width": w, "height": h}
            pipes.cam.send(("playback_resolution", payload))
            pipes.playback.send(("playback_resolution", payload))
            pipes.snapshot.send(("playback_resolution", payload))
            controls.update_playback_resolution(w, h)
        except ValueError as err:
            controls.show_error(str(err), "Playback resolution error")
    elif event == "point_cutoff_apply":
        try:
            cutoff = controls.validate_point_cutoff(values)
            payload = {"distance": cutoff}
            pipes.radar.send(("point_cutoff", payload))
            pipes.playback.send(("point_cutoff", payload))
            pipes.snapshot.send(("point_cutoff", payload))
            controls.update_point_cutoff(cutoff)
        except ValueError as err:
            controls.show_error(str(err), "Point cutoff error")
    elif event == "graph_settings_apply":
        try:
            w, h, x, y = controls.validate_graph_settings(values)
            res_payload, rng_payload = {"width": w, "height": h}, {"x_range": x, "y_range": y}
            for p in (pipes.radar, pipes.playback, pipes.snapshot):
                p.send(("graph_resolution", res_payload))
                p.send(("graph_range", rng_payload))
            controls.update_graph_resolution(w, h)
            controls.update_graph_range(x, y)
        except ValueError as err:
            controls.show_error(str(err), "Graph settings error")
    elif event == "recording_rate_apply":
        try:
            fps = controls.validate_recording_rate(values)
            pipes.cam.send(("camera_recording_rate", {"frames_per_30": fps}))
        except ValueError as err:
            controls.show_error(str(err), "Recording rate error")
    elif event == "playback_toggle":
        if not controls.playback and controls.transposition:
            set_transposition(False, controls, pipes)
        request_playback(values, controls, runtime, pipes)
    elif event == "playback_stop" and controls.playback:        pipes.playback.send(("playback_stop", None))
    elif event == "playback_restart" and controls.playback:     pipes.playback.send(("playback_restart", None))
    elif event == "playback_previous_5s" and controls.playback: pipes.playback.send(("playback_seek", {"seconds": -5.0}))
    elif event == "playback_next_5s" and controls.playback:     pipes.playback.send(("playback_seek", {"seconds": 5.0}))
    elif event == "Send":
        if controls.connected_radar:                            pipes.radar.send((event, values))
        controls.window["save_nvm"].update(button_color=("black", "white"))
    elif event == "save_nvm":
        if controls.connected_radar and check_popup():
            controls.window["save_nvm"].update(button_color=("white", "green"))
            pipes.radar.send((event, values))
    elif event == "record_toggle":
        if controls.recording:                                  request_recording_stop(controls, runtime, pipes)
        else:                                                   start_recording(values, controls, pipes)
    elif event == "snapshot_capture":                           start_snapshot(values, controls, pipes)
    elif isinstance(event, str) and event.startswith("filter"):
        if event == RCS_KEY:                                    controls.window["RCS_FILTER_VALUE"].update(f"{values[RCS_KEY]:.1f}")
        pipes.radar.send((event, values))
        pipes.playback.send((event, values))
        pipes.snapshot.send((event, values))
    elif isinstance(event, str) and event.startswith("conn_"):
        if not (controls.playback or controls.playback_pending):
            target = {"conn_radar": pipes.radar, "conn_cam": pipes.cam, "conn_gps": pipes.gps}.get(event)
            if target:                                          target.send((event, None))
    elif event == "gps_maps":                                   pipes.gps.send((event, None))
    elif event == "DISTANCE":                                   controls.window["SLIDER_VAL"].update(int(values["DISTANCE"]))


def apply_status_message(message, payload, controls, runtime, pipes):
    """Process status notification from workers and update GUI state."""
    if   message == "snapshot_playback_state":          controls.update_snapshot_playback(payload)
    elif message == "snapshot_playback_progress":       controls.update_snapshot_playback_progress(payload)
    elif message == "snapshot_playback_error":
        runtime.pending_snapshot_playback = None
        controls.show_snapshot_playback_error(payload)
    elif message == "snapshot_playback_snapshot_saved": controls.show_snapshot_playback_snapshot_saved(payload)
    elif message == "snapshot_playback_snapshot_error": controls.show_snapshot_playback_snapshot_error(payload)
    elif message == "playback_error":
        runtime.pending_playback_folder = None
        err = payload.get("message", "Playback failed") if isinstance(payload, dict) else payload
        controls.show_playback_error(err)
    elif message in ("graph_resolution_error", "graph_range_error", "camera_pipeline_error"):
        controls.show_error(payload, "Display error")
    elif message == "transposition_state":              controls.update_transposition(payload.get("active"), payload.get("message"))
    elif message == "transposition_error":
        pipes.radar.send(("transposition", {"active": False}))
        controls.update_transposition(False, f"ERROR · {payload}")
    elif message == "camera_recording_rate_state":      controls.update_recording_rate(payload["frames_per_30"])
    elif message == "camera_recording_rate_error":      controls.show_error(payload, "Recording rate error")
    elif message == "camera_recording_drop":            pass
    elif message == "camera_ntp_time":                  controls.update_camera_ntp(payload)
    elif message == "message_201":                      controls.update_radar_telemetry(payload)
    elif message == "received_messages":                controls.update_received_messages(payload)
    elif message == "change_radar":
        controls.update_radar_connection(payload)
        maybe_start_playback(controls, runtime, pipes)
        maybe_start_snapshot_playback(controls, runtime, pipes)
    elif message == "change_cam":
        controls.update_cam_connection(payload)
        if not payload:
            controls.update_camera_ntp({"available": False})
        maybe_start_playback(controls, runtime, pipes)
        maybe_start_snapshot_playback(controls, runtime, pipes)
    elif message == "gps_text":                         controls.window[message].update(payload)
    elif message == "conn_gps":                         controls.update_gps_connection(payload)
    elif message == "recording_state":
        controls.update_recording_state(payload)
        if payload.get("active"):
            pipes.cam.send(("record_start", {"folders": payload.get("folders", {})}))
        else:
            pipes.cam.send(("record_stop", None))
            if runtime.pending_playback_folder:
                disconnect_live_for_playback(controls, runtime, pipes)
            if getattr(runtime, "pending_snapshot_playback", None):
                disconnect_live_for_snapshot_playback(controls, runtime, pipes)
    elif message == "recording_progress":               controls.update_recording_progress(payload)
    elif message == "recording_error":
        runtime.recording_stop_pending = False
        pipes.cam.send(("record_stop", None))
        controls.show_recording_error(payload)
    elif message == "camera_snapshot":                  pipes.radar.send(("record_camera", payload))
    elif message == "manual_snapshot_frame":            pipes.radar.send(("snapshot_capture", payload))
    elif message == "manual_snapshot_error":            controls.show_snapshot_error(payload.get("message", "Camera snapshot failed"))
    elif message == "snapshot_saved":                   controls.update_snapshot_saved(payload)
    elif message == "snapshot_error":                   controls.show_snapshot_error(payload.get("message", "Snapshot failed"))
    elif message == "camera_recording_state":
        if not payload.get("active") and runtime.recording_stop_pending:
            runtime.recording_stop_pending = False
            pipes.radar.send(("record_stop", None))
    elif message == "camera_recording_error":           controls.show_camera_recording_error(payload)
    elif message == "playback_state":                   controls.update_playback_state(payload)
    elif message == "playback_progress":                controls.update_playback_progress(payload)


def drain_status_queue(all_queue, controls, runtime, pipes):
    """Empty worker message queue and update GUI states."""
    while True:
        try:
            msg, payload = all_queue.get_nowait()
        except Empty:
            return
        apply_status_message(msg, payload, controls, runtime, pipes)


def run_event_loop(controls, all_queue, runtime, pipes, shutdown_event):
    """Continuously poll window events and process asynchronous worker queues."""
    while not shutdown_event.is_set():
        event, values = controls.read()
        handle_gui_event(event, values, controls, runtime, pipes, shutdown_event)
        drain_status_queue(all_queue, controls, runtime, pipes)
