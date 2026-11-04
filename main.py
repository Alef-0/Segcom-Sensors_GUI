import argparse
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
from uuid import uuid4

import FreeSimpleGUI as sg

import application_core as base
from calibration_workbench import CalibrationWorkbench
from sensors.camera.camera_gstreamer import gstreamer_main
from sensors.radar.connection_main import create_connection_communication
from sensors.gps.gps_connection import main as gps_main
from menu_configurations import Configurations
from processing.playback.playback import playback_main
from processing.playback.snapshot_playback import snapshot_playback_main


@dataclass
class RuntimeState:
    pending_playback_folder: str | None = None
    pending_snapshot_playback: dict | None = None
    recording_stop_pending: bool = False
    process_context: object | None = None
    workbench: object | None = None
    active_channel: int = 2


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


def _camera_recording_rate(values):
    try:
        frames_per_30 = int(str(values.get("camera_recording_rate", "30")).strip())
    except ValueError as error:
        raise ValueError("Recorded camera frames must be a whole number") from error
    if not 1 <= frames_per_30 <= 30:
        raise ValueError("Recorded camera frames must be between 1 and 30")
    return frames_per_30


def _codet_gate(values):
    try:
        gate = float(str(values.get("global_codet_gate", "0.50")).strip())
    except ValueError as error:
        raise ValueError("Co-detection gate must be a number in meters") from error
    if not math.isfinite(gate) or gate <= 0:
        raise ValueError("Co-detection gate must be greater than zero")
    return gate


def _radar_pose_values(values, channel):
    pod = "A" if int(channel) == 1 else "C"
    try:
        ox = float(str(values.get(f"radar_pose_x_{channel}", "0")).strip())
        oy = float(str(values.get(f"radar_pose_y_{channel}", "0")).strip())
        yaw = float(str(values.get(f"radar_pose_yaw_{channel}", "0")).strip())
    except ValueError as error:
        raise ValueError(f"Radar {pod} X, Y, and Yaw must be numbers") from error
    if not (math.isfinite(ox) and math.isfinite(oy) and math.isfinite(yaw)):
        raise ValueError(f"Radar {pod} X, Y, and Yaw must be finite numbers")
    return {"offset_x_m": ox, "offset_y_m": oy, "yaw_deg": yaw}


def _global_map_payload(values, active, *, clear_accumulated=False):
    return {
        "active": bool(active),
        "color_by_pod": not bool(values.get("global_color_dyn", False)),
        "show_cones": bool(values.get("global_show_cones", True)),
        "show_codet": bool(values.get("global_show_codet", True)),
        "codet_gate_m": _codet_gate(values) if active else 0.50,
        "accumulate_pairs": bool(values.get("global_accumulate_pairs", False)),
        "clear_accumulated": bool(clear_accumulated),
    }


def _transposition_source(values):
    return "global" if values.get("transposition_src_global") else "local"


def _set_transposition(
    active,
    config,
    send_radar,
    send_cam,
    *,
    source="local",
    group=2,
    message=None,
):
    active = bool(active)
    payload = {"active": active, "source": str(source), "group": int(group)}
    send_radar.send(("transposition", payload))
    send_cam.send(("transposition", payload))
    config.change_transposition(active, message)


def _set_global_map(active, values, config, send_radar):
    active = bool(active)
    try:
        payload = _global_map_payload(values, active)
    except ValueError as error:
        sg.popup_error(str(error), title="Global Map error")
        return
    send_radar.send(("global_map_settings", payload))
    config.change_global_map(active)


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


def _handle_gui_event(
    event, values, config, runtime,
    send_radar, send_cam, send_gps, send_playback, send_snapshot_playback,
    shutdown_event,
):
    if event == "transposition_toggle":
        _set_transposition(
            not config.transposition,
            config,
            send_radar,
            send_cam,
            source=_transposition_source(values),
            group=runtime.active_channel,
        )
        return
    if event in ("transposition_src_local", "transposition_src_global"):
        if config.transposition:
            _set_transposition(
                True,
                config,
                send_radar,
                send_cam,
                source=_transposition_source(values),
                group=runtime.active_channel,
            )
        return
    if isinstance(event, str) and event.startswith("choose_"):
        try:
            runtime.active_channel = int(event.rsplit("_", 1)[1])
        except ValueError:
            pass
    if event == "global_map_toggle":
        _set_global_map(not config.global_map_active, values, config, send_radar)
        return
    if event in (
        "global_color_pod",
        "global_color_dyn",
        "global_show_cones",
        "global_show_codet",
        "global_accumulate_pairs",
    ):
        try:
            payload = _global_map_payload(values, config.global_map_active)
        except ValueError as error:
            sg.popup_error(str(error), title="Global Map error")
            return
        send_radar.send(("global_map_settings", payload))
        return
    if event == "global_clear_pairs":
        send_radar.send(("global_map_settings", {"clear_accumulated": True}))
        config.window["global_overlap_status"].update("Co-Det: A↔B 0 | C↔B 0 (cleared)")
        return
    if event in ("radar_poses_apply", "radar_poses_save"):
        try:
            pose_1 = _radar_pose_values(values, 1)
            pose_3 = _radar_pose_values(values, 3)
        except ValueError as error:
            sg.popup_error(str(error), title="Radar pose error")
            return
        save_flag = event == "radar_poses_save"
        config.extrinsics_registry.set_radar_pose(
            1, pose_1["offset_x_m"], pose_1["offset_y_m"], pose_1["yaw_deg"]
        )
        config.extrinsics_registry.set_radar_pose(
            3, pose_3["offset_x_m"], pose_3["offset_y_m"], pose_3["yaw_deg"]
        )
        if save_flag:
            config.extrinsics_registry.save()
            send_cam.send(("extrinsics_reload", None))
            config.refresh_calibration_badges()
        send_radar.send((
            "radar_extrinsics_update",
            {"1": pose_1, "3": pose_3, "save": False},
        ))
        return
    if event in ("radar_autofit_1", "radar_autofit_3"):
        ch = 1 if event.endswith("_1") else 3
        send_radar.send(("radar_auto_fit", {"channel": ch}))
        return
    if event == "calibration_reload_json":
        config.refresh_calibration_badges()
        pose_a = config.extrinsics_registry.get_radar_pose(1)
        pose_c = config.extrinsics_registry.get_radar_pose(3)
        config.update_radar_pose_inputs(1, pose_a.offset_x_m, pose_a.offset_y_m, pose_a.yaw_deg)
        config.update_radar_pose_inputs(3, pose_c.offset_x_m, pose_c.offset_y_m, pose_c.yaw_deg)
        send_radar.send(("radar_extrinsics_reload", None))
        send_cam.send(("extrinsics_reload", None))
        return
    if event == "calibration_open_workbench":
        if runtime.workbench is not None and not runtime.workbench.closed:
            runtime.workbench.close()
        target_ch = 1 if values.get("cal_pod_1") else 3
        config.extrinsics_registry.reload()

        def _request_live_pair(ch):
            send_cam.send((
                "calibration_capture",
                {"request_id": uuid4().hex, "target_channel": int(ch)},
            ))

        def _on_saved(_ch, _res):
            config.refresh_calibration_badges()
            send_radar.send(("radar_extrinsics_reload", None))
            send_cam.send(("extrinsics_reload", None))

        runtime.workbench = CalibrationWorkbench(
            config.extrinsics_registry,
            target_channel=target_ch,
            request_live_pair_cb=_request_live_pair,
            on_saved_cb=_on_saved,
        )
        return
    if event == "snapshot_playback_toggle":
        if not config.snapshot_playback:
            if config.transposition:
                _set_transposition(False, config, send_radar, send_cam)
            if config.global_map_active:
                _set_global_map(False, values, config, send_radar)
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
    if event == "recording_rate_apply":
        try:
            frames_per_30 = _camera_recording_rate(values)
        except ValueError as error:
            sg.popup_error(str(error), title="Recording rate error")
            return
        send_cam.send((
            "camera_recording_rate",
            {"frames_per_30": frames_per_30},
        ))
        return

    if event == "playback_toggle" and not config.playback:
        if config.transposition:
            _set_transposition(False, config, send_radar, send_cam)
        if config.global_map_active:
            _set_global_map(False, values, config, send_radar)

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
    if message == "camera_pipeline_error":
        sg.popup_error(payload, title="Camera pipeline error")
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
    if message == "global_map_telemetry":
        config.change_global_telemetry(payload)
        return
    if message == "radar_auto_fit_result":
        ch = int(payload["channel"])
        ox = float(payload["offset_x_m"])
        oy = float(payload["offset_y_m"])
        yaw = float(payload["yaw_deg"])
        config.extrinsics_registry.set_radar_pose(ch, ox, oy, yaw)
        config.update_radar_pose_inputs(ch, ox, oy, yaw)
        pod = "A" if ch == 1 else "C"
        config.window["global_overlap_status"].update(
            f"Auto-fit {pod}: {payload.get('pairs', 0)} pairs, "
            f"RMSE {payload.get('rmse_cm', 0.0):.1f} cm (SAVE POSES to persist)"
        )
        return
    if message == "radar_auto_fit_error":
        sg.popup_error(str(payload), title="Radar Auto-Fit error")
        return
    if message == "radar_extrinsics_error":
        sg.popup_error(str(payload), title="Radar Pose error")
        return
    if message == "radar_extrinsics_saved":
        return
    if message == "calibration_camera_pair":
        send_radar.send(("calibration_match_radar", payload))
        return
    if message == "calibration_pair_ready":
        if runtime.workbench is not None and not runtime.workbench.closed:
            runtime.workbench.receive_live_pair(payload)
        return
    if message == "calibration_capture_error":
        if runtime.workbench is not None and not runtime.workbench.closed:
            runtime.workbench.report_error(str(payload))
        else:
            sg.popup_error(str(payload), title="Calibration capture error")
        return
    if message == "camera_recording_rate_state":
        config.change_recording_rate(payload["frames_per_30"])
        return
    if message == "camera_recording_rate_error":
        sg.popup_error(payload, title="Camera recording rate error")
        return
    if message == "camera_ntp_time":
        config.change_camera_ntp(payload)
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
        if runtime.workbench is not None:
            if not runtime.workbench.closed:
                runtime.workbench.poll()
            if runtime.workbench.closed:
                runtime.workbench = None


def parse_arguments(argv=None):
    """Parse command-line arguments for the GUI application."""
    parser = argparse.ArgumentParser(description="Segcom Sensors GUI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug and timing warning messages",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
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

    config = Configurations()
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
            kwargs={"debug": args.debug},
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
        if runtime.workbench is not None and not runtime.workbench.closed:
            runtime.workbench.close()
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
