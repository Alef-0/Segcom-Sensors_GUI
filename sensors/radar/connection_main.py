from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import queue
import signal
import time

import cv2 as cv

from sensors.radar.connection_communication import Can_Connection
from sensors.radar.cluster_messages import (
    Clusters_messages,
    read_701_cluster_list as r701,
    read_702_quality_info as r702,
)
from sensors.radar.connection_packages import (
    create_200_radar_configuration as c200,
    read_201_radar_state_extended as r201,
)
from sensors.radar.object_messages import (
    Objects_messages,
    read_60a_object_status as r60a,
    read_60b_object_general as r60b,
    read_60c_object_quality as r60c,
    read_60d_object_extended as r60d,
    read_60e_object_warning as r60e,
)
from processing.visualization.graph_draw import Graph_radar
from processing.visualization.graph_filter import Filter_graph
from processing.visualization.multi_radar_fusion import (
    ChannelFrameCache,
    MultiSensorRegistry,
)
from processing.visualization.transposition import (
    clear_latest,
    global_transposition_payload,
    put_latest,
    transposition_payload,
)
from processing import ManualSnapshotWriter, RadarRecordingSession
from processing.recording.point_cloud_recorder import CAMERA_DELAY_SECONDS

RADAR_CHANNELS = (1, 2, 3)
STATUS_FRAME_TYPES = {0x600: "cluster", 0x60A: "object"}
FRAME_HISTORY_SECONDS = 3.0
MAX_SNAPSHOT_SYNC_ERROR_SECONDS = 0.5


@dataclass(frozen=True)
class RadarFrame:
    recorded_at: datetime
    frame_type: str
    points: tuple


def _put_status(pool, message, payload, *, critical=False):
    try:
        if critical:
            pool.put((message, payload), timeout=0.5)
        else:
            pool.put_nowait((message, payload))
    except queue.Full:
        pass


def _configuration_label(options, index):
    try:
        return options[index]
    except (IndexError, TypeError):
        return f"Unknown ({index})"


def treat_201_message(channel, payload, pool):
    (
        distance,
        radar_power,
        output_type,
        rcs_threshold,
        send_quality,
        send_ext_info,
        ctrl_relay,
        _,
    ) = r201(payload)
    values = {
        f"DISTANCE_{channel}": distance * 2,
        f"RPW_{channel}": _configuration_label(
            ("STANDARD", "-3db TX", "-6db TX", "-9db TX"), radar_power
        ),
        f"OUT_{channel}": _configuration_label(
            ("None", "Objects", "Clusters"), output_type
        ),
        f"RCS_{channel}": _configuration_label(
            ("Standard", "High Sensitivity"), rcs_threshold
        ),
        f"QUALITY_{channel}": ["Inactive", "Active"][send_quality],
        f"EXT_{channel}": ["Inactive", "Active"][send_ext_info],
        f"RELAY_{channel}": ["Inactive", "Active"][ctrl_relay],
    }
    _put_status(pool, "message_201", values)


def send_configuration_message(values, connection, save_nvm):
    data = c200(
        values["CHECK_DISTANCE"], int(values["DISTANCE"] / 2),
        values["CHECK_RPW"], ["STANDARD", "-3dB Tx gain", "-6dB Tx gain", "-9dB Tx gain"].index(values["RPW"]),
        values["CHECK_OUT"], ["NONE", "OBJECT", "CLUSTERS"].index(values["OUT"]),
        values["CHECK_RCS"], ["STANDARD", "HIGH SENSITIVITY"].index(values["RCS"]),
        True, int(bool(values["CHECK_QUALITY"])),
        save_nvm,
        ok_ext=True, ext_info=int(bool(values["CHECK_EXTENDED"])),
        ok_relay=True, ctrl_relay=int(bool(values["CHECK_RELAY"])),
    )
    for channel in RADAR_CHANNELS:
        if values.get(f"send_{channel}") or values.get("send_all"):
            message = connection.packet_struct.pack(8, 0, 0x200, 0, data.to_bytes(8, "big"), channel)
            connection.send_message(message)


def _stop_recording(recording, pool, recording_ready):
    recording_ready.clear()
    if not recording.active:
        return
    try:
        counts = recording.stop()
        _put_status(
            pool,
            "recording_state",
            {"active": False, "counts": counts},
            critical=True,
        )
    except Exception as error:
        _put_status(pool, "recording_error", str(error), critical=True)
        _put_status(
            pool,
            "recording_state",
            {"active": False, "counts": {}},
            critical=True,
        )


def create_connection_communication(
    initial_values,
    pipe,
    pool,
    shutdown_event,
    transposition_channel=None,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
    radar_choice = next(
        int(key.rsplit("_", 1)[1])
        for key, selected in initial_values.items()
        if isinstance(key, str) and key.startswith("choose_") and selected
    )

    try:
        camera_delay_seconds = float(
            initial_values.get("camera_latency_adjustment", 65)
        ) / 1000.0
    except (TypeError, ValueError):
        camera_delay_seconds = CAMERA_DELAY_SECONDS

    connection = Can_Connection()
    cluster_messages = {channel: Clusters_messages() for channel in RADAR_CHANNELS}
    object_messages = {channel: Objects_messages() for channel in RADAR_CHANNELS}
    frame_types: dict[int, str] = {}
    frame_timestamps: dict[int, datetime] = {}
    frame_history = {channel: deque() for channel in RADAR_CHANNELS}
    recording_ready: dict[int, bool] = {}
    received_message_ids: set[int] = set()
    transposition_active = False
    transposition_source = "local"
    transposition_group = int(radar_choice)
    global_map_active = bool(initial_values.get("global_map_active", False))
    global_color_by_pod = bool(initial_values.get("global_color_pod", True))
    global_show_cones = bool(initial_values.get("global_show_cones", True))
    global_show_codet = bool(initial_values.get("global_show_codet", True))
    global_codet_gate_m = 0.50
    global_accumulate_pairs = False
    last_global_render_mono = 0.0
    last_global_telemetry_mono = 0.0
    registry = MultiSensorRegistry()
    global_cache = ChannelFrameCache()
    graph = Graph_radar(
        initial_values.get("point_cutoff", 15.0),
        initial_values.get("graph_width", 800),
        initial_values.get("graph_height", 600),
        initial_values.get("graph_x_range", 15.0),
        initial_values.get("graph_y_range", 15.0),
    )
    filters = Filter_graph(initial_values)

    def _emit_global_telemetry(snapshot, now_mono, *, force=False):
        nonlocal last_global_telemetry_mono
        if not force and (now_mono - last_global_telemetry_mono) < 0.25:
            return
        last_global_telemetry_mono = now_mono
        ab = snapshot.co_detections.get("AB", ())
        cb = snapshot.co_detections.get("CB", ())
        rmse_ab = math.sqrt(sum(m["distance_m"] ** 2 for m in ab) / len(ab)) if ab else None
        rmse_cb = math.sqrt(sum(m["distance_m"] ** 2 for m in cb) / len(cb)) if cb else None
        _put_status(pool, "global_map_telemetry", {
            "channels": snapshot.channel_status,
            "ab_count": len(ab),
            "cb_count": len(cb),
            "ab_rmse_cm": None if rmse_ab is None else rmse_ab * 100.0,
            "cb_rmse_cm": None if rmse_cb is None else rmse_cb * 100.0,
        })

    def _refresh_global_view(*, force=False):
        nonlocal last_global_render_mono
        now_mono = time.monotonic()
        if not force and (now_mono - last_global_render_mono) < 0.045:
            return None
        last_global_render_mono = now_mono
        snap = global_cache.build_snapshot(
            registry,
            distance_cutoff=graph.distance_cutoff,
            color_by_pod=global_color_by_pod,
            compute_overlap=global_show_codet or global_accumulate_pairs,
            dist_gate_m=global_codet_gate_m,
            accumulate_pairs=global_accumulate_pairs,
            now_monotonic=now_mono,
        )
        if global_map_active and connection.connected:
            graph.show_global_snapshot(
                snap,
                show_cones=global_show_cones,
                show_co_detections=global_show_codet,
            )
            _emit_global_telemetry(snap, now_mono, force=force)
        return snap

    def report_progress(channel, count):
        _put_status(pool, "recording_progress", {"channel": channel, "count": count})

    def report_received_message(can_id):
        if can_id in received_message_ids:
            return
        received_message_ids.add(can_id)
        _put_status(pool, "received_messages", tuple(sorted(received_message_ids)))

    recording = RadarRecordingSession(report_progress, camera_delay_seconds)

    def frame_messages(channel, frame_type):
        return cluster_messages[channel] if frame_type == "cluster" else object_messages[channel]

    def remember_frame(channel, frame):
        history = frame_history[channel]
        history.append(frame)
        cutoff = frame.recorded_at - timedelta(seconds=FRAME_HISTORY_SECONDS)
        while history and history[0].recorded_at < cutoff:
            history.popleft()

    def begin_frame(channel, frame_type, recorded_at, status=None):
        previous_type = frame_types.get(channel)
        if previous_type is not None:
            previous_messages = frame_messages(channel, previous_type)
            points = previous_messages.snapshot()
            if previous_type == "cluster":
                x, y, colors = filters.filter_point_sequence(points)
            else:
                x, y, colors = filters.filter_object_sequence(points)
            filtered_points = filters.last_points
            global_cache.update_channel(
                channel,
                filtered_points,
                colors,
                frame_type=previous_type,
                recorded_at=frame_timestamps[channel],
            )

            snap = None
            if global_map_active:
                snap = _refresh_global_view()
            elif channel == radar_choice:
                graph.show_points(x, y, colors, filtered_points)

            if transposition_active:
                if transposition_source == "global":
                    if snap is None:
                        snap = global_cache.build_snapshot(
                            registry,
                            distance_cutoff=graph.distance_cutoff,
                            color_by_pod=global_color_by_pod,
                            compute_overlap=False,
                        )
                    put_latest(
                        transposition_channel,
                        global_transposition_payload(
                            snap,
                            recorded_at=frame_timestamps[channel],
                            frame_type=previous_type,
                        ),
                    )
                elif channel == transposition_group:
                    put_latest(
                        transposition_channel,
                        transposition_payload(
                            filtered_points,
                            colors,
                            frame_type=previous_type,
                            recorded_at=frame_timestamps[channel],
                            distance_cutoff=graph.distance_cutoff,
                            group=channel,
                        ),
                    )

            frame = RadarFrame(
                recorded_at=frame_timestamps[channel],
                frame_type=previous_type,
                points=points,
            )
            remember_frame(channel, frame)
            if recording_ready.get(channel, False):
                recording.submit(
                    channel,
                    points,
                    frame.recorded_at,
                    frame_type=previous_type,
                )
            previous_messages.clear()

        current_messages = frame_messages(channel, frame_type)
        current_messages.clear()
        if frame_type == "object":
            current_messages.fill_60a(status)
        frame_types[channel] = frame_type
        frame_timestamps[channel] = recorded_at
        if channel in recording.channels:
            recording_ready[channel] = True

    def _serialize_radar_points_for_calibration(channel, target_time):
        """Return filtered unified-B and local radar targets closest to target_time for calibration."""
        history = frame_history.get(channel)
        if not connection.connected or not history:
            return (), None
        frame = min(history, key=lambda item: abs((item.recorded_at - target_time).total_seconds()))
        sync_err = abs((frame.recorded_at - target_time).total_seconds())
        if sync_err > MAX_SNAPSHOT_SYNC_ERROR_SECONDS:
            return (), sync_err
        if frame.frame_type == "cluster":
            _, _, cols = filters.filter_point_sequence(frame.points)
        else:
            _, _, cols = filters.filter_object_sequence(frame.points)
        f_pts = filters.last_points
        pose = registry.get_radar_pose(channel)
        serialized = []
        for pt, col in zip(f_pts, cols):
            lat, lon = pt.dist_latitude, pt.dist_long
            if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            if math.hypot(lat, lon) > graph.distance_cutoff:
                continue
            uni = pose.transform_xy([[lat, lon]])[0]
            serialized.append({
                "channel": int(channel),
                "local_lat": float(lat),
                "local_long": float(lon),
                "unified_lat_b": float(uni[0]),
                "unified_long_b": float(uni[1]),
                "rcs": getattr(pt, "rcs", None),
                "color": tuple(int(c) for c in col),
            })
        return tuple(serialized), sync_err

    def match_calibration_radar(values):
        target_ch = int(values.get("target_channel", 3))
        cam_b_time = datetime.fromisoformat(values["captured_at_b"]) - timedelta(seconds=camera_delay_seconds)
        cam_s_time = datetime.fromisoformat(values["captured_at_s"]) - timedelta(seconds=camera_delay_seconds)
        pts_b, err_b = _serialize_radar_points_for_calibration(2, cam_b_time)
        pts_s, err_s = _serialize_radar_points_for_calibration(target_ch, cam_s_time)
        result = dict(values)
        result["radar_points_b"] = pts_b
        result["radar_points_s"] = pts_s
        result["radar_sync_ms_b"] = None if err_b is None else round(err_b * 1000.0, 1)
        result["radar_sync_ms_s"] = None if err_s is None else round(err_s * 1000.0, 1)
        _put_status(pool, "calibration_pair_ready", result, critical=True)

    def handle_radar_auto_fit(values):
        ch = int(values.get("channel", 3))
        try:
            # Feed current live co-detections first so stationary scenes work immediately
            _refresh_global_view(force=True)
            snap = global_cache.build_snapshot(
                registry,
                distance_cutoff=graph.distance_cutoff,
                color_by_pod=True,
                compute_overlap=True,
                dist_gate_m=max(global_codet_gate_m, 1.5),
                accumulate_pairs=True,
            )
            pose, rmse_m, count = global_cache.solve_accumulated_se2(ch)
            registry.set_radar_pose(ch, pose.offset_x_m, pose.offset_y_m, pose.yaw_deg)
            _put_status(pool, "radar_auto_fit_result", {
                "channel": ch,
                "offset_x_m": pose.offset_x_m,
                "offset_y_m": pose.offset_y_m,
                "yaw_deg": pose.yaw_deg,
                "rmse_cm": rmse_m * 100.0,
                "pairs": count,
            }, critical=True)
            _refresh_global_view(force=True)
        except Exception as error:
            _put_status(pool, "radar_auto_fit_error", str(error), critical=True)

    def save_manual_snapshot(values):
        request_id = values.get("request_id")
        try:
            if not connection.connected:
                raise RuntimeError("Connect the radar before taking a snapshot")

            channel = int(values.get("channel", 0))
            if channel not in RADAR_CHANNELS:
                raise ValueError(f"Unsupported radar group: {channel}")

            camera_recorded_at = datetime.fromisoformat(values["captured_at"])
            target_time = camera_recorded_at - timedelta(seconds=camera_delay_seconds)
            history = frame_history[channel]
            if not history:
                raise RuntimeError(
                    "No complete radar frame is available for the selected group"
                )

            frame = min(
                history,
                key=lambda item: abs(
                    (item.recorded_at - target_time).total_seconds()
                ),
            )
            sync_error = abs((frame.recorded_at - target_time).total_seconds())
            if sync_error > MAX_SNAPSHOT_SYNC_ERROR_SECONDS:
                raise RuntimeError(
                    "No radar frame was close enough to the delayed camera frame "
                    f"({sync_error * 1000:.1f} ms difference)"
                )

            result = ManualSnapshotWriter(
                values["folder"],
                camera_delay_seconds=camera_delay_seconds,
            ).save(
                frame.points,
                frame.recorded_at,
                frame.frame_type,
                values["image_bytes"],
                camera_recorded_at,
                pts=values.get("pts"),
                ntp=values.get("ntp"),
                arrival_time=values.get("arrival_time", values.get("captured_at")),
            )
            result["request_id"] = request_id
            _put_status(pool, "snapshot_saved", result, critical=True)
        except Exception as error:
            _put_status(
                pool,
                "snapshot_error",
                {"request_id": request_id, "message": str(error)},
                critical=True,
            )

    try:
        while not shutdown_event.is_set():
            try:
                while pipe.poll():
                    event, values = pipe.recv()
                    if event == "STOP":
                        shutdown_event.set()
                    elif event == "conn_radar":
                        connection.change_connection()
                        received_message_ids.clear()
                        global_cache.clear()
                        _put_status(pool, "received_messages", ())
                        _put_status(pool, "change_radar", connection.connected, critical=True)
                        cv.destroyAllWindows()
                        if not connection.connected:
                            _stop_recording(recording, pool, recording_ready)
                    elif event == "Send" and connection.connected:
                        send_configuration_message(values, connection, False)
                    elif event == "save_nvm" and connection.connected:
                        send_configuration_message(values, connection, True)
                    elif event == "choose":
                        radar_choice = int(values)
                        transposition_group = int(values)
                    elif event == "transposition":
                        transposition_active = bool(values.get("active"))
                        transposition_source = str(values.get("source", transposition_source))
                        if "group" in values and values["group"] is not None:
                            transposition_group = int(values["group"])
                        if not transposition_active:
                            clear_latest(transposition_channel)
                    elif event == "global_map_settings":
                        global_map_active = bool(values.get("active", global_map_active))
                        global_color_by_pod = bool(values.get("color_by_pod", global_color_by_pod))
                        global_show_cones = bool(values.get("show_cones", global_show_cones))
                        global_show_codet = bool(values.get("show_codet", global_show_codet))
                        global_codet_gate_m = float(values.get("codet_gate_m", global_codet_gate_m))
                        if "accumulate_pairs" in values:
                            global_accumulate_pairs = bool(values["accumulate_pairs"])
                        if values.get("clear_accumulated"):
                            global_cache.clear_accumulated_pairs()
                        if global_map_active and connection.connected:
                            _refresh_global_view(force=True)
                    elif event == "radar_extrinsics_update":
                        try:
                            for ch in (1, 3):
                                item = values.get(str(ch)) or values.get(ch)
                                if isinstance(item, dict):
                                    registry.set_radar_pose(
                                        ch,
                                        float(item.get("offset_x_m", 0.0)),
                                        float(item.get("offset_y_m", 0.0)),
                                        float(item.get("yaw_deg", 0.0)),
                                        float(item.get("offset_z_m", 0.0)),
                                    )
                            if values.get("save"):
                                registry.save()
                            _put_status(pool, "radar_extrinsics_saved", {"saved": bool(values.get("save"))})
                            if global_map_active and connection.connected:
                                _refresh_global_view(force=True)
                        except Exception as error:
                            _put_status(pool, "radar_extrinsics_error", str(error), critical=True)
                    elif event == "radar_extrinsics_reload":
                        registry.reload()
                        if global_map_active and connection.connected:
                            _refresh_global_view(force=True)
                    elif event == "radar_auto_fit":
                        handle_radar_auto_fit(values or {})
                    elif event == "calibration_match_radar":
                        match_calibration_radar(values)
                    elif event == "record_start":
                        try:
                            folders = recording.start(values["folder"], values["channels"])
                            recording_ready.clear()
                            recording_ready.update({channel: False for channel in recording.channels})
                            _put_status(
                                pool,
                                "recording_state",
                                {"active": True, "folders": folders, "counts": {}},
                                critical=True,
                            )
                        except Exception as error:
                            _put_status(pool, "recording_error", str(error), critical=True)
                    elif event == "record_camera":
                        captured_at = values.get("captured_at", "")
                        pts = values.get("pts")
                        ntp = values.get("ntp")
                        arrival_time = values.get("arrival_time", captured_at)
                        for channel, filename in values.get("files", {}).items():
                            recording.add_camera_snapshot(
                                int(channel),
                                filename,
                                captured_at,
                                pts=pts,
                                ntp=ntp,
                                arrival_time=arrival_time,
                            )
                    elif event == "record_stop":
                        _stop_recording(recording, pool, recording_ready)
                    elif event == "snapshot_capture":
                        save_manual_snapshot(values)
                    elif event == "point_cutoff":
                        graph.set_distance_cutoff(values.get("distance", 15.0))
                    elif event == "graph_resolution":
                        graph.set_resolution(
                            values.get("width", 800), values.get("height", 600)
                        )
                    elif event == "graph_range":
                        graph.set_range(
                            values.get("x_range", 15.0),
                            values.get("y_range", 15.0),
                        )
                    elif event == "camera_latency_adjustment":
                        try:
                            camera_delay_seconds = float(
                                values.get("latency_adjustment_ms")
                            ) / 1000.0
                            recording.set_camera_delay_seconds(camera_delay_seconds)
                        except (AttributeError, TypeError, ValueError):
                            _put_status(
                                pool,
                                "camera_latency_error",
                                "Program-wide pipeline latency adjustment must be numeric",
                            )
                    elif isinstance(event, str) and event.startswith("filter"):
                        filters.update_values(event, values)
            except (EOFError, OSError):
                shutdown_event.set()

            recording_error = recording.poll_error()
            if recording_error is not None:
                _stop_recording(recording, pool, recording_ready)

            if not connection.connected:
                shutdown_event.wait(0.01)
                continue

            connection.read_chunk()
            if global_map_active and (time.monotonic() - last_global_render_mono) >= 0.20:
                _refresh_global_view(force=True)
            while connection.can_create_can():
                message = connection.create_package()
                channel = message.canChannel
                if channel not in cluster_messages:
                    continue

                report_received_message(message.canId)
                if message.canId == 0x201:
                    treat_201_message(channel, message.canData, pool)

                frame_type = STATUS_FRAME_TYPES.get(message.canId)
                if frame_type is not None:
                    status = r60a(message.canData) if frame_type == "object" else None
                    begin_frame(
                        channel,
                        frame_type,
                        datetime.now().astimezone(),
                        status,
                    )
                elif frame_types.get(channel) == "cluster":
                    if message.canId == 0x701:
                        cluster_messages[channel].fill_701(r701(message.canData))
                    elif message.canId == 0x702:
                        cluster_messages[channel].fill_702(r702(message.canData))
                elif frame_types.get(channel) == "object":
                    if message.canId == 0x60B:
                        object_messages[channel].fill_60b(r60b(message.canData))
                    elif message.canId == 0x60C:
                        object_messages[channel].fill_60c(r60c(message.canData))
                    elif message.canId == 0x60D:
                        object_messages[channel].fill_60d(r60d(message.canData))
                    elif message.canId == 0x60E:
                        object_messages[channel].fill_60e(r60e(message.canData))
    finally:
        clear_latest(transposition_channel)
        _stop_recording(recording, pool, recording_ready)
        if connection.sock:
            connection.sock.close()
        cv.destroyAllWindows()
