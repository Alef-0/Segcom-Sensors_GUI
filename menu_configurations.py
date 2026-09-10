from datetime import datetime, timezone
from pathlib import Path

import FreeSimpleGUI as sg

from interface_core import Configurations as BaseConfigurations


class Configurations(BaseConfigurations):
    def __init__(self, calibration_screens=None):
        self.snapshot_playback = False
        self.snapshot_playback_pending = False
        self.snapshot_playback_paused = False
        self.playback_width = 1280
        self.playback_height = 720
        self.point_cutoff = 15.0
        self.graph_width = 800
        self.graph_height = 600
        self.graph_x_range = 15
        self.graph_y_range = 15
        self.calibration_camera = False
        self.calibration_camera_pending = False
        self.calibration_clock = False
        self.calibration_recording = False
        self.transposition = False
        self.calibration_screens = list(
            calibration_screens or ["0: Primary @ unknown Hz"]
        )
        super().__init__()

    def centralize_combos(self):
        super().centralize_combos()
        for key in (
            "calibration_decoder",
            "calibration_screen",
            "calibration_grid_qrs",
            "calibration_visible_qrs",
        ):
            self.window[key].Widget.configure(justify="center")

    def create_radar_division(self):
        columns = []
        names = ("LEFT", "MIDDLE", "RIGHT")
        letters = ("A", "B", "C")
        for channel, (name, letter) in enumerate(zip(names, letters), start=1):
            column = sg.Column([
                [sg.Text(f"{name} {letter}", justification="center", expand_x=True)],
                [sg.Text("Distance", expand_x=True), sg.Text("XXX", key=f"DISTANCE_{channel}", justification="right")],
                [sg.Text("Radar", expand_x=True), sg.Text("XXX", key=f"RPW_{channel}", justification="right")],
                [sg.Text("Output", expand_x=True), sg.Text("XXX", key=f"OUT_{channel}", justification="right")],
                [sg.Text("RCS", expand_x=True), sg.Text("XXX", key=f"RCS_{channel}", justification="right")],
                [sg.Text("Quality", expand_x=True), sg.Text("XXX", key=f"QUALITY_{channel}", justification="right")],
                [sg.Text("Extended Info", expand_x=True), sg.Text("XXX", key=f"EXT_{channel}", justification="right")],
                [sg.Text("Control Relay", expand_x=True), sg.Text("XXX", key=f"RELAY_{channel}", justification="right")],
                [sg.Radio(f"Visualizar Grupo {letter}", "visu_radar", key=f"choose_{channel}", default=channel == 2, enable_events=True)],
            ])
            columns.extend([sg.Push(), column, sg.Push()])
            if channel != 3:
                columns.append(sg.VSep())

        self.FRAME = [sg.Frame(
            "Real Time Configurations",
            [
                columns,
                [
                    sg.Push(),
                    sg.Text(
                        "MESSAGES: --",
                        key="received_messages",
                        expand_x=True,
                        justification="center",
                    ),
                    sg.VSep(),
                    sg.Text(
                        "CAMERA NTP: --",
                        key="camera_ntp_time",
                        expand_x=True,
                        justification="center",
                    ),
                    sg.Push(),
                ],
            ],
            expand_x=True,
            title_location=sg.TITLE_LOCATION_TOP,
        )]

    def create_options(self):
        choices = sg.Frame(
            "Radar",
            [[
                sg.Radio("1", "choose", key="send_1"),
                sg.Radio("2", "choose", key="send_2"),
                sg.Radio("3", "choose", key="send_3"),
                sg.Radio("all", "choose", key="send_all", default=True),
            ]],
            title_location=sg.TITLE_LOCATION_TOP,
        )
        column1 = sg.Column([
            [
                sg.Checkbox("Radar Power", key="CHECK_RPW", default=True),
                sg.Push(),
                sg.Combo(self.POWER, self.POWER[3], key="RPW", readonly=True),
            ],
            [
                sg.Checkbox("RCS Threshold", key="CHECK_RCS", default=True),
                sg.Push(),
                sg.Combo(self.RCS, self.RCS[1], key="RCS", readonly=True),
            ],
            [sg.Button("Send", expand_x=True), choices],
            [
                sg.Button(
                    "SAVE in Non Volatile Memory",
                    key="save_nvm",
                    expand_x=True,
                    button_color=("black", "white"),
                )
            ],
        ], expand_x=True)
        column2 = sg.Column([
            [
                sg.Checkbox("Output Type", key="CHECK_OUT", default=True),
                sg.Push(),
                sg.Combo(self.OUTPUT, self.OUTPUT[2], key="OUT", readonly=True, size=(15, 1)),
            ],
            [
                sg.Push(),
                sg.Checkbox("Quality", key="CHECK_QUALITY", default=True),
                sg.Checkbox("Extended Info", key="CHECK_EXTENDED", default=True),
                sg.Checkbox("Control Relay", key="CHECK_RELAY", default=True),
                sg.Push(),
            ],
            [
                sg.Button(
                    "OPEN RADAR",
                    key="conn_radar",
                    expand_x=True,
                    button_color=("white", "green"),
                ),
                sg.VSep(),
                sg.Button(
                    "OPEN GPS",
                    key="conn_gps",
                    button_color=("white", "green"),
                ),
                sg.Button("MAPS", key="gps_maps"),
            ],
            [
                sg.Button(
                    "OPEN CAM",
                    key="conn_cam",
                    expand_x=True,
                    button_color=("white", "green"),
                ),
                sg.VSep(),
                sg.Text(
                    "0° 0' 0\" N, 0° 0' 0\" E",
                    key="gps_text",
                    expand_x=True,
                    justification="center",
                ),
            ],
        ], expand_x=True)
        self.options = [
            [
                sg.Checkbox("Max Distance", key="CHECK_DISTANCE", default=True),
                sg.Text("196", key="SLIDER_VAL"),
                sg.Slider(
                    (196, 260),
                    196,
                    orientation="h",
                    resolution=1,
                    key="DISTANCE",
                    disable_number_display=True,
                    enable_events=True,
                    expand_x=True,
                ),
            ],
            [column1, sg.VSep(), column2],
        ]

    @staticmethod
    def _create_snapshot_layout():
        layout = BaseConfigurations._create_snapshot_layout()[:-1]
        layout.extend([
            [sg.HorizontalSeparator()],
            [
                sg.Text("Playback folder"),
                sg.Input(default_text=str(Path.cwd()), key="snapshot_playback_folder", expand_x=True),
                sg.FolderBrowse("SELECT", key="snapshot_playback_browse", target="snapshot_playback_folder"),
                sg.Checkbox(
                    "Image + PCD",
                    key="snapshot_playback_synced_only",
                    default=True,
                ),
            ],
            [
                sg.Push(),
                sg.Button(
                    "START PLAYBACK",
                    key="snapshot_playback_toggle",
                    button_color=("white", "green"),
                ),
                sg.VSep(),
                sg.Button("PREVIOUS", key="snapshot_playback_previous", disabled=True),
                sg.Button("PAUSE", key="snapshot_playback_pause", disabled=True),
                sg.Button("NEXT", key="snapshot_playback_next", disabled=True),
                sg.Button("SNAPSHOT CURRENT", key="snapshot_playback_snapshot", disabled=True),
                sg.Push(),
            ],
            [
                sg.Text(
                    "",
                    key="snapshot_playback_status",
                    expand_x=True,
                    justification="center",
                    pad=(0, 0),
                )
            ],
        ])
        return layout

    @staticmethod
    def _create_general_configurations_layout():
        return [
            [
                sg.Column([[
                    sg.Text("Camera Resolution"),
                    sg.Input("1280", key="playback_width", size=(8, 1), justification="right"),
                    sg.Text("×"),
                    sg.Input("720", key="playback_height", size=(8, 1), justification="right"),
                    sg.Button("APPLY", key="playback_resolution_apply"),
                ]], expand_x=True),
                sg.VSep(),
                sg.Column([[
                    sg.Text("Recorded frames (out of 30)"),
                    sg.Combo(
                        tuple(range(1, 31)),
                        30,
                        key="camera_recording_rate",
                        size=(5, 1),
                        readonly=True,
                    ),
                    sg.Button("APPLY", key="recording_rate_apply"),
                ]]),
            ],
            [
                sg.Push(),
                sg.Text("1280 × 720", key="playback_resolution_status"),
                sg.VSep(),
                sg.Text("30 / 30 frames (30 FPS)", key="recording_rate_status"),
                sg.Push(),
            ],
            [sg.HorizontalSeparator()],
            [
                sg.Column([[
                    sg.Text("Point cutoff (m)"),
                    sg.Input("15", key="point_cutoff", size=(6, 1), justification="right"),
                    sg.Button("APPLY", key="point_cutoff_apply"),
                ]]),
                sg.VSep(),
                sg.Column([[
                    sg.Text("Graph Resolution"),
                    sg.Input("800", key="graph_width", size=(6, 1), justification="right"),
                    sg.Text("×"),
                    sg.Input("600", key="graph_height", size=(6, 1), justification="right"),
                ], [
                    sg.Text("Graph Range (m)"),
                    sg.Text("X ±"),
                    sg.Input("15", key="graph_x_range", size=(5, 1), justification="right"),
                    sg.Text("Y 0–"),
                    sg.Input("15", key="graph_y_range", size=(5, 1), justification="right"),
                    sg.Button("APPLY", key="graph_settings_apply"),
                ]]),
            ],
            [
                sg.Push(),
                sg.Text("Cutoff 15.0 m", key="point_cutoff_status"),
                sg.VSep(),
                sg.Text("800 × 600", key="graph_resolution_status"),
                sg.VSep(),
                sg.Text("X ±15 m | Y 0–15 m", key="graph_range_status"),
                sg.Push(),
            ],
        ]

    @staticmethod
    def _create_calibration_layout(screen_choices=None):
        screen_choices = list(screen_choices or ["0: Primary @ unknown Hz"])
        return [
            [
                sg.Push(),
                sg.Text("Camera latency"),
                sg.Input("145", key="camera_pipeline_latency", size=(9, 1), justification="right"),
                sg.Text("Pipeline Adjustment (ms)"),
                sg.Input("109", key="camera_latency_adjustment", size=(9, 1), justification="right"),
                sg.Button("APPLY LATENCIES", key="calibration_latency_apply"),
                sg.Push(),
            ],
            [
                sg.Push(),
                sg.Text("Current latencies: 145 ms / 109 ms", key="calibration_latency_status"),
                sg.Text("Camera pipeline"),
                sg.Combo(
                    ("Usual NVIDIA", "ARM / Jetson", "CPU"),
                    "Usual NVIDIA",
                    key="calibration_decoder",
                    readonly=True,
                    size=(15, 1),
                ),
                sg.Push(),
            ],
            [
                sg.Push(),
                sg.Text("Display monitor"),
                sg.Combo(
                    screen_choices,
                    screen_choices[0],
                    key="calibration_screen",
                    readonly=True,
                    size=(24, 1),
                ),
                sg.Text("QR grid"),
                sg.Combo(
                    (4, 6, 8, 9, 10, 12),
                    4,
                    key="calibration_grid_qrs",
                    readonly=True,
                    enable_events=True,
                    size=(5, 1),
                ),
                sg.Text("QR codes kept"),
                sg.Combo(
                    tuple(range(1, 5)),
                    2,
                    key="calibration_visible_qrs",
                    readonly=True,
                    size=(5, 1),
                ),
                sg.Push(),
            ],
            [sg.HorizontalSeparator()],
            [
                sg.Push(),
                sg.Text("Qt/OpenGL QR calibration clock"),
                sg.Button(
                    "OPEN CALIBRATION CAMERA 4",
                    key="calibration_camera_toggle",
                    button_color=("white", "green"),
                ),
                sg.Button(
                    "START QR CALIBRATION",
                    key="calibration_clock_start",
                    button_color=("white", "green"),
                ),
                sg.Text("", key="calibration_status"),
                sg.Push(),
            ],
            [
                sg.Text(
                    "The fullscreen QR view records after 3 seconds and stops recording when closed.",
                    expand_x=True,
                    justification="center",
                )
            ],
        ]

    @staticmethod
    def _create_visualization_layout():
        root = Path(__file__).resolve().parent
        return [
            [sg.Text("Calibration recording"),
             sg.Input(str(root / "recordings"), key="visualization_folder", expand_x=True),
             sg.FolderBrowse("SELECT", target="visualization_folder")],
            [sg.Push(), sg.Button("OPEN CALIBRATION VISUALIZATION", key="visualization_open"), sg.Push()],
            [sg.Text("Browse recorded frames, compare decoded timestamps, and inspect the marked source of each offset.",
                     expand_x=True, justification="center")],
            [sg.Text("Select a calibration folder to open the separate viewer.", key="visualization_status",
                     expand_x=True, justification="center")],
        ]

    def create_radar_control(self):
        tabs = [[
            sg.Tab("Configurations", self.options),
            sg.Tab("Record", self._create_record_layout()),
            sg.Tab("Snapshots", self._create_snapshot_layout()),
            sg.Tab("Display", self._create_general_configurations_layout()),
            sg.Tab("Calibration", self._create_calibration_layout(self.calibration_screens)),
            sg.Tab("Visualization", self._create_visualization_layout()),
        ]]
        self.radar_control = sg.Frame(
            "General Control",
            [[sg.TabGroup(tabs, expand_x=True, pad=(0, 0))]],
            expand_x=True,
            title_location=sg.TITLE_LOCATION_TOP,
            pad=(0, 0),
        )

    @staticmethod
    def _create_transposition_layout():
        return [
            [
                sg.Push(),
                sg.Text("Camera B + Radar B"),
                sg.Button(
                    "ENABLE TRANSPOSITION",
                    key="transposition_toggle",
                    button_color=("white", "green"),
                ),
                sg.Push(),
            ],
            [
                sg.Text(
                    "OFF · camera points use the current radar filters and distance cutoff",
                    key="transposition_status",
                    expand_x=True,
                    justification="center",
                )
            ],
        ]

    def create_filters(self):
        tabs = [[
            sg.Tab("Basic", self._create_filter_layout()),
            sg.Tab("Cluster Options", self._create_cluster_filter_layout()),
            sg.Tab("Transposition", self._create_transposition_layout()),
        ]]
        self.filters = sg.Frame(
            "Radar controls",
            [[sg.TabGroup(tabs, expand_x=True)]],
            expand_x=True,
            title_location=sg.TITLE_LOCATION_TOP,
        )

    def _refresh_mode_controls(self):
        live_blocked = (
            self.playback
            or self.playback_pending
            or self.snapshot_playback
            or self.snapshot_playback_pending
            or self.calibration_camera
            or self.calibration_camera_pending
        )
        recording_inputs_disabled = self.recording or self.recording_pending or live_blocked
        for key in ("record_folder", "record_browse", "record_radar_1", "record_radar_2", "record_radar_3"):
            self.window[key].update(disabled=recording_inputs_disabled)

        record_disabled = (
            self.recording_pending
            or self.snapshot_pending
            or live_blocked
        )
        self.window["record_toggle"].update(disabled=record_disabled)

        snapshot_inputs_disabled = (
            self.snapshot_pending
            or self.recording
            or self.recording_pending
            or live_blocked
        )
        for key in (
            "snapshot_folder", "snapshot_browse",
            "snapshot_group_1", "snapshot_group_2", "snapshot_group_3",
        ):
            self.window[key].update(disabled=snapshot_inputs_disabled)
        self.window["snapshot_capture"].update(
            disabled=(
                snapshot_inputs_disabled
                or not self.connected_radar
                or not self.connected_cam
            )
        )

        playback_inputs_disabled = (
            self.recording
            or self.recording_pending
            or self.snapshot_pending
            or self.playback
            or self.playback_pending
            or self.snapshot_playback
            or self.snapshot_playback_pending
        )
        for key in ("playback_folder", "playback_browse"):
            self.window[key].update(disabled=playback_inputs_disabled)
        self.window["playback_toggle"].update(
            disabled=(
                self.recording
                or self.recording_pending
                or self.snapshot_pending
                or self.playback
                or self.playback_pending
                or self.snapshot_playback
                or self.snapshot_playback_pending
            )
        )
        transport_disabled = not self.playback
        for key in (
            "playback_stop",
            "playback_restart",
            "playback_previous_5s",
            "playback_next_5s",
        ):
            self.window[key].update(disabled=transport_disabled)

        snapshot_playback_inputs_disabled = (
            self.recording
            or self.recording_pending
            or self.snapshot_pending
            or self.playback
            or self.playback_pending
            or self.snapshot_playback
            or self.snapshot_playback_pending
        )
        for key in (
            "snapshot_playback_folder",
            "snapshot_playback_browse",
            "snapshot_playback_synced_only",
        ):
            self.window[key].update(disabled=snapshot_playback_inputs_disabled)
        self.window["snapshot_playback_toggle"].update(
            disabled=(
                self.recording
                or self.recording_pending
                or self.snapshot_pending
                or self.playback
                or self.playback_pending
                or self.snapshot_playback_pending
            )
        )
        transport_disabled = not self.snapshot_playback
        for key in (
            "snapshot_playback_previous",
            "snapshot_playback_pause",
            "snapshot_playback_next",
            "snapshot_playback_snapshot",
        ):
            self.window[key].update(disabled=transport_disabled)

        for key in ("conn_radar", "conn_cam"):
            self.window[key].update(disabled=live_blocked or self.snapshot_pending)
        for channel in range(1, 4):
            self.window[f"choose_{channel}"].update(
                disabled=self.calibration_camera or self.calibration_camera_pending
            )
        self.window["calibration_camera_toggle"].update(
            disabled=(
                self.calibration_camera_pending
                or (self.recording_pending and not self.calibration_camera)
                or self.snapshot_pending
            )
        )
        self.window["calibration_clock_start"].update(
            disabled=self.calibration_clock
        )
        settings_disabled = self.calibration_camera or self.calibration_camera_pending
        self.window["calibration_decoder"].update(disabled=settings_disabled)
        self.window["calibration_screen"].update(disabled=self.calibration_clock)
        self.window["calibration_grid_qrs"].update(disabled=self.calibration_clock)
        self.window["calibration_visible_qrs"].update(disabled=self.calibration_clock)
        self.window["transposition_toggle"].update(
            disabled=(
                self.playback
                or self.playback_pending
                or self.snapshot_playback
                or self.snapshot_playback_pending
                or self.calibration_camera
                or self.calibration_camera_pending
            )
        )

    def change_received_messages(self, message_ids):
        messages = ", ".join(f"0x{message_id:03X}" for message_id in message_ids) or "--"
        self.window["received_messages"].update(f"MESSAGES: {messages}")

    def change_camera_ntp(self, payload):
        if not payload or not payload.get("available"):
            self.window["camera_ntp_time"].update("CAMERA NTP: --")
            return
        seconds, nanoseconds = divmod(int(payload["ntp_unix_ns"]), 1_000_000_000)
        try:
            ntp_time = datetime.fromtimestamp(seconds, timezone.utc)
        except (OverflowError, OSError, ValueError):
            self.window["camera_ntp_time"].update("CAMERA NTP: INVALID")
            return
        text = (
            f"CAMERA {payload.get('channel', '?')} NTP: "
            f"{ntp_time:%Y-%m-%d %H:%M:%S}.{nanoseconds // 1_000_000:03d} UTC"
        )
        if payload.get("offset_ms") is not None:
            text += f" | OFFSET {float(payload['offset_ms']):+.3f} ms"
        self.window["camera_ntp_time"].update(text)

    def change_snapshot_saved(self, payload):
        if (
            self.snapshot_request_id is not None
            and payload.get("request_id") != self.snapshot_request_id
        ):
            return
        self.snapshot_pending = False
        self.snapshot_request_id = None
        self.window["snapshot_capture"].update(
            "CAPTURE SNAPSHOT",
            button_color=("white", "green"),
        )
        self.window["snapshot_status"].update(
            f"SAVED: {payload['point_cloud']} + {payload['camera_frame']}"
        )
        self._refresh_mode_controls()

    def set_snapshot_playback_pending(self):
        self.snapshot_playback_pending = True
        self.window["snapshot_playback_toggle"].update("PREPARING...", disabled=True)
        self.window["snapshot_playback_status"].update("STOPPING LIVE MONITORING")
        self._refresh_mode_controls()

    def change_snapshot_playback(self, payload):
        self.snapshot_playback = bool(payload.get("active"))
        self.snapshot_playback_pending = False
        self.snapshot_playback_paused = bool(payload.get("paused", False))
        self.window["snapshot_playback_toggle"].update(
            "STOP PLAYBACK" if self.snapshot_playback else "START PLAYBACK",
            button_color=("white", "red" if self.snapshot_playback else "green"),
        )
        self.window["snapshot_playback_pause"].update(
            "PLAY" if self.snapshot_playback_paused else "PAUSE"
        )
        if self.snapshot_playback:
            current = payload.get("current", 1)
            total = payload.get("total", 0)
            state = "PAUSED" if self.snapshot_playback_paused else "PLAYING"
            self.window["snapshot_playback_status"].update(f"{state} {current} / {total}")
        elif payload.get("completed"):
            self.window["snapshot_playback_status"].update("COMPLETED")
        else:
            self.window["snapshot_playback_status"].update("")
        self._refresh_mode_controls()

    def change_snapshot_playback_progress(self, payload):
        state = "PAUSED" if self.snapshot_playback_paused else "PLAYING"
        self.window["snapshot_playback_status"].update(
            f"{state} {payload['current']} / {payload['total']} — {payload['file']} + {payload['image']}"
        )

    def change_snapshot_playback_pause(self, payload):
        self.snapshot_playback_paused = bool(payload.get("paused"))
        self.window["snapshot_playback_pause"].update(
            "PLAY" if self.snapshot_playback_paused else "PAUSE"
        )
        self._refresh_mode_controls()

    def show_snapshot_playback_error(self, message):
        self.change_snapshot_playback({"active": False, "completed": False})
        sg.popup_error(message, title="Snapshot playback error")

    def show_snapshot_playback_snapshot_saved(self, payload):
        self.window["snapshot_playback_status"].update(
            f"SAVED: {payload['point_cloud']} + {payload['camera_frame']}"
        )

    def show_snapshot_playback_snapshot_error(self, message):
        sg.popup_error(message, title="Playback snapshot error")

    def change_playback_resolution(self, width, height):
        self.playback_width = width
        self.playback_height = height
        self.window["playback_resolution_status"].update(f"{width} × {height}")

    def change_point_cutoff(self, cutoff):
        self.point_cutoff = cutoff
        self.window["point_cutoff_status"].update(f"Cutoff {cutoff:.1f} m")

    def change_graph_resolution(self, width, height):
        self.graph_width = width
        self.graph_height = height
        self.window["graph_resolution_status"].update(f"{width} × {height}")

    def change_graph_range(self, x_range, y_range):
        self.graph_x_range = x_range
        self.graph_y_range = y_range
        self.window["graph_range_status"].update(
            f"X ±{x_range:g} m | Y 0–{y_range:g} m"
        )

    def set_calibration_camera_pending(self):
        self.calibration_camera_pending = True
        self.window["calibration_camera_toggle"].update("OPENING CAMERA 4...", disabled=True)
        self.window["calibration_status"].update("CLOSING RADARS AND OPENING CAMERA 4")
        self._refresh_mode_controls()

    def change_calibration_camera(self, active):
        self.calibration_camera = bool(active)
        self.calibration_camera_pending = False
        self.window["calibration_camera_toggle"].update(
            "CLOSE CALIBRATION CAMERA 4" if active else "OPEN CALIBRATION CAMERA 4",
            button_color=("white", "red" if active else "green"),
        )
        if active:
            self.window["calibration_status"].update("CAMERA 4 OPEN")
        elif not self.calibration_recording:
            self.window["calibration_status"].update("")
        self._refresh_mode_controls()

    def change_calibration_clock(self, active):
        self.calibration_clock = bool(active)
        self.window["calibration_clock_start"].update(
            "QR ACTIVE" if active else "START QR CALIBRATION",
            disabled=active,
            button_color=("white", "red" if active else "green"),
        )
        self._refresh_mode_controls()

    def change_calibration_qr_grid(self, grid_qrs, visible_qrs):
        grid_qrs = int(grid_qrs)
        choices = tuple(range(1, grid_qrs + 1))
        try:
            selected = int(visible_qrs)
        except (TypeError, ValueError):
            selected = 1
        selected = min(grid_qrs, max(1, selected))
        self.window["calibration_visible_qrs"].update(
            values=choices,
            value=selected,
        )

    def change_calibration_recording(self, payload):
        self.calibration_recording = bool(payload.get("active"))
        if self.calibration_recording:
            self.window["calibration_status"].update(
                f"RECORDING CAMERA 4 — {payload.get('folder', '')}"
            )
        else:
            count = payload.get("count", 0)
            dropped = payload.get("dropped", 0)
            self.window["calibration_status"].update(
                (
                    f"SAVED {count} CAMERA FRAMES — DROPPED {dropped}"
                    if count or dropped
                    else "CAMERA 4 OPEN"
                )
            )

    def change_camera_recording_drop(self, payload):
        dropped = payload.get("dropped", payload.get("missing", 1))
        if self.calibration_recording:
            self.window["calibration_status"].update(
                f"RECORDING CAMERA 4 — DROPPED {dropped} FRAME(S)"
            )

    def change_calibration_latencies(self, pipeline_latency_ms, adjustment_ms):
        self.window["calibration_latency_status"].update(
            f"Current latencies: {pipeline_latency_ms} ms / {adjustment_ms:g} ms"
        )

    def change_transposition(self, active, message=None):
        self.transposition = bool(active)
        self.window["transposition_toggle"].update(
            "DISABLE TRANSPOSITION" if self.transposition else "ENABLE TRANSPOSITION",
            button_color=("white", "red" if self.transposition else "green"),
        )
        if message is None:
            message = (
                "ON · GROUP B · waiting for camera and radar frames"
                if self.transposition
                else "OFF · camera points use the current radar filters and distance cutoff"
            )
        self.window["transposition_status"].update(message)
        self._refresh_mode_controls()

    def change_recording_rate(self, frames_per_30):
        self.window["recording_rate_status"].update(
            f"{frames_per_30} / 30 frames ({frames_per_30} FPS)"
        )

    def show_calibration_error(self, message):
        sg.popup_error(message, title="Calibration error")
