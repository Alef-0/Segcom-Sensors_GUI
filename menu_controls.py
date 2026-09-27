from datetime import datetime, timezone
import math
import re
import FreeSimpleGUI as sg

from menu_layout import build_main_layout


class MenuControls:
    """Controls window lifecycle, widget state transitions, and input validation."""
    RADAR_LETTERS = {1: "A", 2: "B", 3: "C"}

    def __init__(self, font=("Helvetica", 12), theme="SystemDefaultForReal"):
        sg.set_options(font=font)
        sg.theme(theme)
        self.connected_radar = False
        self.connected_cam = False
        self.connected_gps = False
        self.recording = False
        self.recording_pending = False
        self.recording_counts = {}
        self.snapshot_pending = False
        self.snapshot_request_id = None
        self.playback = False
        self.playback_pending = False
        self.playback_paused = False
        self.playback_width = 1280
        self.playback_height = 720
        self.point_cutoff = 15.0
        self.graph_width = 800
        self.graph_height = 600
        self.graph_x_range = 15.0
        self.graph_y_range = 15.0
        self.transposition = False
        self.camera_pipeline_latency = 145
        self.camera_latency_adjustment = 87.3

        self.window = sg.Window("Configurations Menu", build_main_layout(), finalize=True)
        for element in self.window.element_list():
            if isinstance(element, sg.Combo):
                popdown = element.Widget.tk.eval(f"ttk::combobox::PopdownWindow {element.Widget}")
                element.Widget.tk.call(f"{popdown}.f.l", "configure", "-font", ("Helvetica", 11), "-justify", "center")
        for key in ("RPW", "OUT", "RCS"):
            self.window[key].Widget.configure(justify="center")
        self.refresh_mode_controls()

    def read(self, timeout=10):
        """Read events from the active window with a default timeout."""
        return self.window.read(timeout=timeout)

    def close(self):
        """Close window instance."""
        self.window.close()

    def sanitize_numeric_input(self, key, values, allow_float=False):
        """Filter non-numeric characters from active input field."""
        raw = str(values.get(key, ""))
        pattern = r"[^0-9.]" if allow_float else r"[^0-9]"
        cleaned = re.sub(pattern, "", raw)
        if allow_float and cleaned.count(".") > 1:
            head, tail = cleaned.split(".", 1)
            cleaned = head + "." + tail.replace(".", "")
        if cleaned != raw:
            self.window[key].update(cleaned)
        return cleaned

    def validate_playback_resolution(self, values):
        """Validate playback/camera width and height."""
        width = int(str(values.get("playback_width", "1280")).strip())
        height = int(str(values.get("playback_height", "720")).strip())
        if width <= 0 or height <= 0:
            raise ValueError("Playback width and height must be positive")
        return width, height

    def validate_point_cutoff(self, values):
        """Validate point cloud cutoff distance."""
        cutoff = float(str(values.get("point_cutoff", "15")).strip())
        if cutoff <= 0:
            raise ValueError("Point cutoff must be greater than zero")
        return cutoff

    def validate_graph_settings(self, values):
        """Validate graph resolution and range parameters."""
        w = int(str(values.get("graph_width", "800")).strip())
        h = int(str(values.get("graph_height", "600")).strip())
        x = float(str(values.get("graph_x_range", "15")).strip())
        y = float(str(values.get("graph_y_range", "15")).strip())
        if w <= 100 or h <= 100:
            raise ValueError("Graph width and height must be greater than 100 pixels")
        if not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0:
            raise ValueError("Graph X and Y ranges must be greater than zero")
        return w, h, x, y

    def validate_recording_rate(self, values):
        """Validate recorded camera FPS."""
        fps = int(str(values.get("camera_recording_rate", "30")).strip())
        if not 1 <= fps <= 30:
            raise ValueError("Recorded camera frames must be between 1 and 30")
        return fps

    def validate_camera_latency(self, values):
        """Validate camera pipeline latency (jitter) and latency adjustment (offset)."""
        jitter = int(str(values.get("camera_pipeline_latency", "145")).strip())
        offset = float(str(values.get("camera_latency_adjustment", "87.3")).strip())
        if jitter < 0:
            raise ValueError("Camera latency (jitter) must be a non-negative integer")
        return jitter, offset

    @property
    def snapshot_playback(self):
        return self.playback

    @snapshot_playback.setter
    def snapshot_playback(self, val):
        self.playback = val

    @property
    def snapshot_playback_pending(self):
        return self.playback_pending

    @snapshot_playback_pending.setter
    def snapshot_playback_pending(self, val):
        self.playback_pending = val

    @property
    def snapshot_playback_paused(self):
        return self.playback_paused

    @snapshot_playback_paused.setter
    def snapshot_playback_paused(self, val):
        self.playback_paused = val

    def refresh_mode_controls(self):
        """Enable or disable widgets depending on current operating state."""
        live_blocked = (self.playback or self.playback_pending)
        rec_inputs_disabled = self.recording or self.recording_pending or live_blocked
        for key in ("record_folder", "record_browse", "record_radar_1", "record_radar_2", "record_radar_3"):
            if key in self.window.key_dict:
                self.window[key].update(disabled=rec_inputs_disabled)

        if "record_toggle" in self.window.key_dict:
            self.window["record_toggle"].update(disabled=(self.recording_pending or self.snapshot_pending or live_blocked))

        if "snapshot_capture" in self.window.key_dict:
            self.window["snapshot_capture"].update(disabled=(
                self.snapshot_pending or self.recording or self.recording_pending or live_blocked
                or not self.connected_radar or not self.connected_cam
            ))

        pb_inputs_disabled = self.recording or self.recording_pending or self.snapshot_pending or self.playback_pending
        for key in ("playback_folder", "playback_browse", "playback_synced_only"):
            if key in self.window.key_dict:
                self.window[key].update(disabled=(pb_inputs_disabled or self.playback))

        if "playback_toggle" in self.window.key_dict:
            self.window["playback_toggle"].update(disabled=pb_inputs_disabled)
        for key in ("playback_previous", "playback_pause", "playback_next", "playback_snapshot"):
            if key in self.window.key_dict:
                self.window[key].update(disabled=not self.playback)

        for key in ("conn_radar", "conn_cam"):
            if key in self.window.key_dict:
                self.window[key].update(disabled=live_blocked or self.snapshot_pending)
        for ch in range(1, 4):
            if f"choose_{ch}" in self.window.key_dict:
                self.window[f"choose_{ch}"].update(disabled=False)
        if "transposition_toggle" in self.window.key_dict:
            self.window["transposition_toggle"].update(disabled=False)

    def update_record_device_status(self):
        """Update connection indicator texts."""
        status = f"RADAR {'OPEN' if self.connected_radar else 'CLOSED'} | CAMERA {'OPEN' if self.connected_cam else 'CLOSED'}"
        if "record_devices" in self.window.key_dict:
            self.window["record_devices"].update(status)
        if "playback_devices" in self.window.key_dict:
            self.window["playback_devices"].update(status)

    def update_radar_connection(self, connection):
        """Update radar connection state and indicators."""
        self.connected_radar = bool(connection)
        self.window["conn_radar"].update("CLOSE RADAR" if connection else "OPEN RADAR", button_color=("white", "red" if connection else "green"))
        self.update_record_device_status()
        self.refresh_mode_controls()
        if not connection:
            self.update_received_messages(())
            for ch in range(1, 4):
                self.update_radar_telemetry({f"{k}_{ch}": "XXX" for k in ("DISTANCE", "RPW", "OUT", "RCS", "QUALITY", "EXT", "RELAY")})

    def update_cam_connection(self, connection):
        """Update camera connection state and indicators."""
        self.connected_cam = bool(connection)
        self.window["conn_cam"].update("CLOSE CAM" if connection else "OPEN CAM", button_color=("white", "red" if connection else "green"))
        self.update_record_device_status()
        self.refresh_mode_controls()

    def update_gps_connection(self, connection):
        """Update GPS connection button."""
        self.connected_gps = bool(connection)
        self.window["conn_gps"].update("CLOSE GPS" if connection else "OPEN GPS", button_color=("white", "red" if connection else "green"))

    def update_received_messages(self, message_ids):
        """Format and display received CAN message IDs."""
        msgs = ", ".join(f"0x{m:03X}" for m in message_ids) or "--"
        self.window["received_messages"].update(f"MESSAGES: {msgs}")

    def update_camera_ntp(self, payload):
        """Update camera NTP timestamp and offset display."""
        if not payload or not payload.get("available"):
            self.window["camera_ntp_time"].update("CAMERA NTP: --")
            return
        secs, ns = divmod(int(payload["ntp_unix_ns"]), 1_000_000_000)
        try:
            ntp_time = datetime.fromtimestamp(secs, timezone.utc)
        except (OverflowError, OSError, ValueError):
            self.window["camera_ntp_time"].update("CAMERA NTP: INVALID")
            return
        txt = f"CAMERA {payload.get('channel', '?')} NTP: {ntp_time:%Y-%m-%d %H:%M:%S}.{ns // 1_000_000:03d} UTC"
        if payload.get("offset_ms") is not None:
            txt += f" | OFFSET {float(payload['offset_ms']):+.3f} ms"
        self.window["camera_ntp_time"].update(txt)

    def set_recording_pending(self, starting):
        """Set recording transitional state."""
        self.recording_pending = True
        self.window["record_toggle"].update("STARTING..." if starting else "STOPPING...", disabled=True)
        self.refresh_mode_controls()

    def update_recording_state(self, payload):
        """Update recording status button and labels."""
        self.recording = bool(payload.get("active"))
        self.recording_pending = False
        self.recording_counts = dict(payload.get("counts", {}))
        self.window["record_toggle"].update("STOP RECORDING" if self.recording else "START RECORDING", button_color=("white", "red" if self.recording else "green"))
        if self.recording:
            letters = ", ".join(self.RADAR_LETTERS[int(ch)] for ch in sorted(payload.get("folders", {})))
            self.window["record_status"].update(f"RECORDING: {letters}")
        else:
            self._update_recording_count_text("SAVED" if self.recording_counts else "IDLE")
        self.refresh_mode_controls()

    def update_recording_progress(self, payload):
        """Update recorded point cloud frame counts."""
        self.recording_counts[payload["channel"]] = payload["count"]
        self._update_recording_count_text("RECORDING")

    def _update_recording_count_text(self, prefix):
        if not self.recording_counts:
            self.window["record_status"].update(prefix)
            return
        counts = " | ".join(f"{self.RADAR_LETTERS[ch]}: {cnt}" for ch, cnt in sorted(self.recording_counts.items()))
        self.window["record_status"].update(f"{prefix} — {counts}")

    def set_snapshot_pending(self, request_id, channel):
        """Mark snapshot capture as pending."""
        self.snapshot_pending = True
        self.snapshot_request_id = request_id
        msg = f"CAPTURING GROUP {self.RADAR_LETTERS[channel]}..."
        if "snapshot_status" in self.window.key_dict:
            self.window["snapshot_status"].update(msg)
        if "record_status" in self.window.key_dict:
            self.window["record_status"].update(msg)
        if "snapshot_capture" in self.window.key_dict:
            self.window["snapshot_capture"].update("CAPTURING...", disabled=True)
        self.refresh_mode_controls()

    def update_snapshot_saved(self, payload):
        """Acknowledge completed snapshot."""
        if self.snapshot_request_id is not None and payload.get("request_id") != self.snapshot_request_id:
            return
        self.snapshot_pending = False
        self.snapshot_request_id = None
        if "snapshot_capture" in self.window.key_dict:
            self.window["snapshot_capture"].update("SNAPSHOT", button_color=("white", "green"))
        msg = f"SAVED: {payload.get('point_cloud', '')} + {payload.get('camera_frame', '')}"
        if "snapshot_status" in self.window.key_dict:
            self.window["snapshot_status"].update(msg)
        if "record_status" in self.window.key_dict:
            self.window["record_status"].update(msg)
        self.refresh_mode_controls()

    def show_snapshot_error(self, message):
        """Show snapshot capture error dialog."""
        self.snapshot_pending = False
        self.snapshot_request_id = None
        if "snapshot_capture" in self.window.key_dict:
            self.window["snapshot_capture"].update("SNAPSHOT", button_color=("white", "green"))
        if "snapshot_status" in self.window.key_dict:
            self.window["snapshot_status"].update("FAILED")
        if "record_status" in self.window.key_dict:
            self.window["record_status"].update("SNAPSHOT FAILED")
        self.refresh_mode_controls()
        sg.popup_error(message, title="Snapshot error")

    def set_playback_pending(self):
        """Mark playback as preparing."""
        self.playback_pending = True
        if "playback_toggle" in self.window.key_dict:
            self.window["playback_toggle"].update("PREPARING...", disabled=True)
        if "playback_status" in self.window.key_dict:
            self.window["playback_status"].update("STOPPING LIVE MONITORING")
        self.refresh_mode_controls()

    def update_playback_state(self, payload):
        """Update playback transport controls and indicators."""
        self.playback = bool(payload.get("active"))
        self.playback_pending = False
        self.playback_paused = bool(payload.get("paused", False))
        if "playback_toggle" in self.window.key_dict:
            self.window["playback_toggle"].update(
                "STOP PLAYBACK" if self.playback else "START PLAYBACK",
                button_color=("white", "red" if self.playback else "green"),
            )
        if "playback_pause" in self.window.key_dict:
            self.window["playback_pause"].update("PLAY" if self.playback_paused else "PAUSE")
        if "playback_status" in self.window.key_dict:
            if self.playback:
                state = "PAUSED" if self.playback_paused else "PLAYING"
                self.window["playback_status"].update(f"{state} {payload.get('current', 1)} / {payload.get('total', 0)}")
            elif payload.get("completed"):
                self.window["playback_status"].update("COMPLETED")
            else:
                self.window["playback_status"].update("IDLE")
        self.refresh_mode_controls()

    def update_playback_progress(self, payload):
        """Update active playback frame progress."""
        state = "PAUSED" if self.playback_paused else "PLAYING"
        file_desc = payload.get("file") or f"{payload.get('point_cloud', '')} + {payload.get('image', '')}"
        if "playback_status" in self.window.key_dict:
            self.window["playback_status"].update(f"{state} {payload['current']} / {payload['total']} — {file_desc}")

    def update_playback_pause(self, payload):
        """Update pause/play toggle button."""
        self.playback_paused = bool(payload.get("paused"))
        if "playback_pause" in self.window.key_dict:
            self.window["playback_pause"].update("PLAY" if self.playback_paused else "PAUSE")
        self.refresh_mode_controls()

    def show_playback_snapshot_saved(self, payload):
        """Show confirmation of saved playback snapshot."""
        if "playback_status" in self.window.key_dict:
            self.window["playback_status"].update(f"SAVED: {payload.get('point_cloud', '')} + {payload.get('camera_frame', '')}")

    def show_playback_snapshot_error(self, message):
        """Show error message when playback snapshot fails."""
        self.show_error(message, "Playback snapshot error")

    def show_playback_error(self, message):
        self.update_playback_state({"active": False})
        self.show_error(message, "Playback error")

    def update_transposition(self, active, message=None):
        """Toggle transposition state and update button/label."""
        self.transposition = bool(active)
        self.window["transposition_toggle"].update("DISABLE TRANSPOSITION" if self.transposition else "ENABLE TRANSPOSITION", button_color=("white", "red" if self.transposition else "green"))
        if message is None:
            message = "ON · GROUP B · waiting for frames" if self.transposition else "OFF · camera points use current filters and cutoff"
        self.window["transposition_status"].update(message)
        self.refresh_mode_controls()

    def update_playback_resolution(self, width, height):
        self.playback_width, self.playback_height = width, height
        self.window["playback_resolution_status"].update(f"{width} × {height}")

    def update_point_cutoff(self, cutoff):
        self.point_cutoff = cutoff
        self.window["point_cutoff_status"].update(f"Cutoff {cutoff:.1f} m")

    def update_graph_resolution(self, width, height):
        self.graph_width, self.graph_height = width, height
        self.window["graph_resolution_status"].update(f"{width} × {height}")

    def update_graph_range(self, x_range, y_range):
        self.graph_x_range, self.graph_y_range = x_range, y_range
        self.window["graph_range_status"].update(f"X ±{x_range:g} m | Y 0–{y_range:g} m")

    def update_recording_rate(self, fps):
        self.window["recording_rate_status"].update(f"{fps} / 30 frames ({fps} FPS)")

    def update_camera_latency(self, pipeline_latency_ms, adjustment_ms):
        """Update camera latency status label."""
        self.camera_pipeline_latency = pipeline_latency_ms
        self.camera_latency_adjustment = adjustment_ms
        self.window["camera_latency_status"].update(
            f"Jitter: {pipeline_latency_ms} ms | Offset: {adjustment_ms:g} ms"
        )

    def update_radar_telemetry(self, values_dict):
        for k, v in values_dict.items():
            if k in self.window.key_dict:
                self.window[k].update(v)

    def show_error(self, message, title="Error"):
        sg.popup_error(message, title=title)

    def show_recording_error(self, message):
        self.update_recording_state({"active": False})
        self.show_error(message, "Recording error")

    def show_playback_error(self, message):
        self.update_playback_state({"active": False})
        self.show_error(message, "Playback error")

    def show_camera_recording_error(self, message):
        self.show_error(message, "Camera snapshot error")

    set_snapshot_playback_pending = set_playback_pending
    update_snapshot_playback = update_playback_state
    update_snapshot_playback_progress = update_playback_progress
    update_snapshot_playback_pause = update_playback_pause
    show_snapshot_playback_error = show_playback_error
    show_snapshot_playback_snapshot_saved = show_playback_snapshot_saved
    show_snapshot_playback_snapshot_error = show_playback_snapshot_error
