from pathlib import Path
import FreeSimpleGUI as sg
from sensors.filter import (
    AMBIGUITY_STATE_OPTIONS, DYNAMIC_PROPERTY_OPTIONS, INVALID_STATE_OPTIONS, PDH_KEY, RCS_KEY,
)

POWER_OPTIONS = ["STANDARD", "-3dB Tx gain", "-6dB Tx gain", "-9dB Tx gain"]
OUTPUT_OPTIONS = ["NONE", "OBJECT", "CLUSTERS"]
RCS_OPTIONS = ["STANDARD", "HIGH SENSITIVITY"]


def option_control(option, include_color=False):
    """Build checkbox with optional disabled color button."""
    ctrls = [sg.Checkbox(option.label, key=option.key, default=option.default, disabled=not option.enabled, enable_events=True)]
    if include_color:
        ctrls.append(sg.Button("", button_color=option.color, disabled=True))
    return ctrls


def build_radar_columns():
    """Construct the 3 radar channel telemetry columns."""
    columns, names, letters = [], ("LEFT", "MIDDLE", "RIGHT"), ("A", "B", "C")
    for channel, (name, letter) in enumerate(zip(names, letters), start=1):
        col = sg.Column([
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
        columns.extend([sg.Push(), col, sg.Push()])
        if channel != 3:
            columns.append(sg.VSep())
    return columns


def build_telemetry_frame():
    """Construct top frame displaying real-time radar, messages, and camera NTP status."""
    status_row = [
        sg.Push(),
        sg.Text("MESSAGES: --", key="received_messages", expand_x=True, justification="center"),
        sg.VSep(),
        sg.Text("CAMERA NTP: --", key="camera_ntp_time", expand_x=True, justification="center"),
        sg.Push(),
    ]
    return sg.Frame("Real Time Configurations", [build_radar_columns(), status_row], expand_x=True, title_location=sg.TITLE_LOCATION_TOP)


def build_radar_options():
    """Construct radar hardware configuration controls tab."""
    choices = sg.Frame("Radar", [[sg.Radio(str(i), "choose", key=f"send_{i}") for i in (1, 2, 3)] + [sg.Radio("all", "choose", key="send_all", default=True)]], title_location=sg.TITLE_LOCATION_TOP)
    col1 = sg.Column([
        [sg.Checkbox("Radar Power", key="CHECK_RPW", default=True), sg.Push(), sg.Combo(POWER_OPTIONS, POWER_OPTIONS[3], key="RPW", readonly=True)],
        [sg.Checkbox("RCS Threshold", key="CHECK_RCS", default=True), sg.Push(), sg.Combo(RCS_OPTIONS, RCS_OPTIONS[1], key="RCS", readonly=True)],
        [sg.Button("Send", expand_x=True), choices],
        [sg.Button("SAVE in Non Volatile Memory", key="save_nvm", expand_x=True, button_color=("black", "white"))],
    ], expand_x=True)
    col2 = sg.Column([
        [sg.Checkbox("Output Type", key="CHECK_OUT", default=True), sg.Push(), sg.Combo(OUTPUT_OPTIONS, OUTPUT_OPTIONS[2], key="OUT", readonly=True, size=(15, 1))],
        [sg.Push(), sg.Checkbox("Quality", key="CHECK_QUALITY", default=True), sg.Checkbox("Extended Info", key="CHECK_EXTENDED", default=True), sg.Checkbox("Control Relay", key="CHECK_RELAY", default=True), sg.Push()],
        [sg.Button("OPEN RADAR", key="conn_radar", expand_x=True, button_color=("white", "green")), sg.VSep(), sg.Button("OPEN GPS", key="conn_gps", button_color=("white", "green")), sg.Button("MAPS", key="gps_maps")],
        [sg.Button("OPEN CAM", key="conn_cam", expand_x=True, button_color=("white", "green")), sg.VSep(), sg.Text("0° 0' 0\" N, 0° 0' 0\" E", key="gps_text", expand_x=True, justification="center")],
    ], expand_x=True)
    distance_row = [
        sg.Checkbox("Max Distance", key="CHECK_DISTANCE", default=True), sg.Text("196", key="SLIDER_VAL"),
        sg.Slider((196, 260), 196, orientation="h", resolution=1, key="DISTANCE", disable_number_display=True, enable_events=True, expand_x=True),
    ]
    return [distance_row, [col1, sg.VSep(), col2]]


def build_record_tab():
    """Construct recording and playback controls tab."""
    rec_root = Path.cwd() / "recordings"
    rec_root.mkdir(exist_ok=True)
    radar_choices = [sg.Text("Group:")] + [sg.Checkbox(letter, key=f"record_radar_{ch}", default=False) for ch, letter in ((1, "A"), (2, "B"), (3, "C"))]
    return [
        [sg.Text("Destination folder"), sg.Input("./recordings", key="record_folder", expand_x=True, enable_events=True), sg.FolderBrowse("SELECT", key="record_browse", target="record_folder", initial_folder="./recordings")],
        [*radar_choices, sg.Push(), sg.Text("IDLE", key="record_status", justification="center"), sg.Text("RADAR CLOSED | CAMERA CLOSED", key="record_devices", justification="center"), sg.Push(),
         sg.Button("START RECORDING", key="record_toggle", button_color=("white", "green"), disabled=True), sg.Button("SNAPSHOT", key="snapshot_capture", button_color=("white", "green"), disabled=True)],
        [sg.HorizontalSeparator()],
        [sg.Text("Playback folder"), sg.Input("./recordings", key="playback_folder", expand_x=True, enable_events=True), sg.FolderBrowse("SELECT", key="playback_browse", target="playback_folder", initial_folder="./recordings"), sg.Checkbox("Image + PCD", key="playback_synced_only", default=True)],
        [sg.Push(), sg.Button("START PLAYBACK", key="playback_toggle", button_color=("white", "green")), sg.VSep(),
         sg.Button("PREVIOUS", key="playback_previous", disabled=True), sg.Button("PAUSE", key="playback_pause", disabled=True), sg.Button("NEXT", key="playback_next", disabled=True), sg.Button("SNAPSHOT CURRENT", key="playback_snapshot", disabled=True), sg.Push()],
        [sg.Text("", key="playback_status", expand_x=True, justification="center", pad=(0, 0))],
    ]


build_snapshot_tab = build_record_tab


def build_video_tab():
    """Construct camera resolution, FPS, jitter, and offset calibration tab."""
    col_res = sg.Column([[
        sg.Text("Camera Resolution"),
        sg.Input("1280", key="playback_width", size=(8, 1), justification="center", enable_events=True),
        sg.Text("×"),
        sg.Input("720", key="playback_height", size=(8, 1), justification="center", enable_events=True),
        sg.Button("APPLY", key="playback_resolution_apply"),
    ]], element_justification="center")
    col_fps = sg.Column([[
        sg.Text("Recorded frames (out of 30)"),
        sg.Combo(tuple(range(1, 31)), 30, key="camera_recording_rate", size=(5, 1), readonly=True),
        sg.Button("APPLY", key="recording_rate_apply"),
    ]], element_justification="center")
    col_latency = sg.Column([[
        sg.Text("Jitter (ms)"),
        sg.Input("145", key="camera_pipeline_latency", size=(6, 1), justification="center", enable_events=True),
        sg.Text("Offset (ms)"),
        sg.Input("87.3", key="camera_latency_adjustment", size=(6, 1), justification="center", enable_events=True),
        sg.Button("APPLY", key="camera_latency_apply"),
    ]], element_justification="center")
    status_row1 = [
        sg.Push(),
        sg.Text("1280 × 720", key="playback_resolution_status"),
        sg.VSep(),
        sg.Text("30 / 30 frames (30 FPS)", key="recording_rate_status"),
        sg.Push(),
    ]
    status_row2 = [
        sg.Push(),
        sg.Text("Jitter: 145 ms | Offset: 87.3 ms", key="camera_latency_status"),
        sg.Push(),
    ]
    return [
        [sg.Push(), col_res, sg.VSep(), col_fps, sg.Push()],
        status_row1,
        [sg.HorizontalSeparator()],
        [sg.Push(), col_latency, sg.Push()],
        status_row2,
    ]


def build_graph_tab():
    """Construct radar point cutoff, graph range/resolution, and transposition tab."""
    col_cutoff = sg.Column([[
        sg.Text("Point cutoff (m)"),
        sg.Input("15", key="point_cutoff", size=(6, 1), justification="center", enable_events=True),
        sg.Button("APPLY", key="point_cutoff_apply"),
    ]], element_justification="center")
    col_graph = sg.Column([
        [sg.Text("Graph Resolution"), sg.Input("800", key="graph_width", size=(6, 1), justification="center", enable_events=True), sg.Text("×"), sg.Input("600", key="graph_height", size=(6, 1), justification="center", enable_events=True)],
        [sg.Text("Graph Range (m)"), sg.Text("X ±"), sg.Input("15", key="graph_x_range", size=(5, 1), justification="center", enable_events=True), sg.Text("Y 0–"), sg.Input("15", key="graph_y_range", size=(5, 1), justification="center", enable_events=True), sg.Button("APPLY", key="graph_settings_apply")],
    ], element_justification="center")
    status_row = [
        sg.Push(),
        sg.Text("Cutoff 15.0 m", key="point_cutoff_status"),
        sg.VSep(),
        sg.Text("800 × 600", key="graph_resolution_status"),
        sg.VSep(),
        sg.Text("X ±15 m | Y 0–15 m", key="graph_range_status"),
        sg.Push(),
    ]
    transposition_row = [
        sg.Push(),
        sg.Text("Camera B + Radar B"),
        sg.Button("ENABLE TRANSPOSITION", key="transposition_toggle", button_color=("white", "green")),
        sg.Push(),
    ]
    transposition_status = [
        sg.Text("OFF · camera points use current radar filters and distance cutoff", key="transposition_status", expand_x=True, justification="center"),
    ]
    return [
        [sg.Push(), col_cutoff, sg.VSep(), col_graph, sg.Push()],
        status_row,
        [sg.HorizontalSeparator()],
        transposition_row,
        transposition_status,
    ]


def build_display_tab():
    """Compatibility alias for graph tab layout."""
    return build_graph_tab()


def build_filter_tab():
    """Construct radar point cloud filter tab."""
    dynamic_rows = []
    for start in (0, 4):
        row = [sg.Push()] + [ctrl for opt in DYNAMIC_PROPERTY_OPTIONS[start:start + 4] for ctrl in option_control(opt, include_color=True)] + [sg.Push()]
        dynamic_rows.append(row)
    pdh = sg.Column([
        [sg.Text("PDH0 - False Alarm Probability (zero is invalid)", expand_x=True, justification="center")],
        [sg.Slider((1, 7), 3, orientation="h", tick_interval=1, disable_number_display=True, expand_x=True, enable_events=True, key=PDH_KEY)],
    ])
    rcs = sg.Column([
        [sg.Text("Minimum RCS (dBm²)", expand_x=True, justification="center")],
        [sg.Slider((-64.0, 63.5), -20.0, orientation="h", resolution=0.5, expand_x=True, enable_events=True, key=RCS_KEY, disable_number_display=True)],
        [sg.Push(), sg.Text("-20.0", key="RCS_FILTER_VALUE"), sg.Push()],
    ], expand_x=True)
    return [[sg.Column(dynamic_rows, justification="center")], [sg.HorizontalSeparator()], [pdh, sg.VSep(), rcs]]


def build_cluster_filter_tab():
    """Construct radar cluster filter options tab."""
    ambiguity = sg.Column([
        [sg.Text("Ambiguity State", justification="center", expand_x=True)],
        [sg.Push()] + [c for opt in AMBIGUITY_STATE_OPTIONS[:2] for c in option_control(opt)] + [sg.Push()],
        [sg.Push()] + [c for opt in AMBIGUITY_STATE_OPTIONS[2:] for c in option_control(opt)] + [sg.Push()],
    ], justification="center", expand_x=True, vertical_alignment="top")
    inv_rows = []
    for start in range(0, len(INVALID_STATE_OPTIONS), 6):
        inv_rows.append([sg.Push()] + [c for opt in INVALID_STATE_OPTIONS[start:start + 6] for c in option_control(opt)] + [sg.Push()])
    invalid = sg.Column([[sg.Text("Cluster Invalid State", expand_x=True, justification="center")], *inv_rows], expand_x=True, vertical_alignment="top")
    return [[ambiguity, sg.VSep(), invalid]]


def build_transposition_tab():
    """Construct camera-radar transposition toggle tab."""
    return [
        [sg.Push(), sg.Text("Camera B + Radar B"), sg.Button("ENABLE TRANSPOSITION", key="transposition_toggle", button_color=("white", "green")), sg.Push()],
        [sg.Text("OFF · camera points use current radar filters and distance cutoff", key="transposition_status", expand_x=True, justification="center")],
    ]


def build_main_layout():
    """Assemble full application window layout."""
    control_tabs = [
        sg.Tab("Configurations", build_radar_options()),
        sg.Tab("Record", build_record_tab()),
        sg.Tab("Graph", build_graph_tab()),
        sg.Tab("Video", build_video_tab()),
    ]

    filter_tabs = [
        sg.Tab("Basic", build_filter_tab()),
        sg.Tab("Cluster Options", build_cluster_filter_tab()),
    ]
    radar_control = sg.Frame("General Control", [[sg.TabGroup([control_tabs], expand_x=True, pad=(0, 0))]], expand_x=True, title_location=sg.TITLE_LOCATION_TOP, pad=(0, 0))
    filters = sg.Frame("Radar controls", [[sg.TabGroup([filter_tabs], expand_x=True)]], expand_x=True, title_location=sg.TITLE_LOCATION_TOP)
    return [[build_telemetry_frame()], [radar_control], [filters]]
