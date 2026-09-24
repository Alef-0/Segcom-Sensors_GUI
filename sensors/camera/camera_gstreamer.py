import queue
import signal
import socket
import threading
import time

import cv2 as cv
import gi
import numpy as np

from calibration import CALIBRATION_PIPELINE_RESTART_DELAY_SECONDS
from calibration.scheduler_priority import CalibrationSchedulerPriority
from processing import CameraSnapshotRecorder
from processing.visualization.transposition import (
    RADAR_GROUP_B,
    RadarCameraOverlay,
    clear_latest,
    get_latest,
)
from sensors.camera.camera_pipeline import (
    available_decoder_backends,
    build_camera_pipeline,
)
from sensors.camera.camera_reference_clock import ReferenceClockObserver
from sensors.camera.camera_timebase import FrameTimestampPolicy
from sensors.camera.timing_defaults import (
    DEFAULT_CAMERA_CALIBRATION_VERSION,
    DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS,
)

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

Gst.init(None)

CAMERA_PIPELINE_LATENCY_MS = 145
CAMERA_FRAME_RATE = 30
DEFAULT_DISPLAY_WIDTH = 1280
DEFAULT_DISPLAY_HEIGHT = 720
MAX_PIPELINE_ATTEMPTS = 3
FIRST_FRAME_TIMEOUT_SECONDS = 5.0
PIPELINE_RETRY_DELAY_SECONDS = 0.5
TIMESTAMP_WARNING_INTERVAL_SECONDS = 5.0
MANUAL_SNAPSHOT_TIMESTAMP_TIMEOUT_SECONDS = 5.0
NTP_UI_UPDATE_INTERVAL_SECONDS = 0.1
_RESULT_FAILURE = "failure"
_RESULT_RESTART = "restart"
_RESULT_CLOSED = "closed"


class GStreamerPipeline:
    def __init__(self, conn, pool, shutdown_event, transposition_channel=None):
        self.pipeline = None
        self.main_loop = None
        self.frames = queue.Queue(maxsize=1)
        self.channel = 2
        self.normal_channel = 2
        self.communicate = conn
        self.pool = pool
        self.shutdown_event = shutdown_event
        self.connected = False
        self.source_ids = []
        self.first_frame_received = False
        self.attempt_started = 0.0
        self.exit_reason = _RESULT_FAILURE
        self.channel_changed = False
        self.display_width = DEFAULT_DISPLAY_WIDTH
        self.display_height = DEFAULT_DISPLAY_HEIGHT
        self.pipeline_latency_ms = CAMERA_PIPELINE_LATENCY_MS
        self.latency_adjustment_ms = DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS
        self.recording_frames_per_30 = CAMERA_FRAME_RATE
        self.calibration_mode = False
        self.calibration_recording = False
        self.calibration_pipeline_restart_delay_seconds: float | None = None
        self.calibration_restart_waiting_for_preview = False
        self.calibration_scheduler_priority = CalibrationSchedulerPriority(
            "calibration camera and recorder"
        )
        self.snapshot_recorder = CameraSnapshotRecorder(
            self._report_snapshot,
            self._report_recording_drop,
        )
        self.snapshot_recorder.prepare()
        self.timestamp_policy = FrameTimestampPolicy(
            capture_correction_ms=self.latency_adjustment_ms,
            capture_calibration_version=DEFAULT_CAMERA_CALIBRATION_VERSION,
        )
        self.reference_clock = ReferenceClockObserver()
        self.stream_epoch = 0
        self.decoder_preference = "auto"
        self.decoder_backends = available_decoder_backends()
        self.decoder_backend_index = 0
        self.last_pipeline_error: str | None = None
        self._pipeline_error_reported = False
        self._last_timestamp_warning = 0.0
        self._last_published_camera_ntp_ns: int | None = None
        self._last_ntp_ui_update = 0.0
        self._has_frame_ntp = False
        self._last_writer_drop_warning = 0.0
        self._last_pts_gap_warning = 0.0
        self._pending_pts_gap_candidates = 0
        self._manual_snapshot_lock = threading.Lock()
        self._pending_manual_snapshot: dict | None = None
        self.transposition_channel = transposition_channel
        self.transposition_active = False
        self.transposition_payload = None
        self.capture_size: tuple[int, int] | None = None
        self.transposition_overlay = None
        self.transposition_error = None
        try:
            self.transposition_overlay = RadarCameraOverlay.from_json()
        except Exception as error:
            self.transposition_error = str(error).strip() or type(error).__name__

    @staticmethod
    def create_url(channel):
        return f"rtsp://admin:l1v3user5@192.168.1.108:554/cam/realmonitor?channel={channel}&subtype=0"

    def _put_status(self, message, payload, *, timeout=0.2):
        try:
            self.pool.put((message, payload), timeout=timeout)
        except queue.Full:
            pass

    def _report_snapshot(self, payload):
        self._put_status("camera_snapshot", payload)

    def _report_recording_drop(self, payload):
        now = time.monotonic()
        if now - self._last_writer_drop_warning < 1.0:
            return
        print(
            "[WARNING][CAMERA] Dropped a selected camera frame because the "
            f"image writer queue is full; {payload['dropped']} writer-queue "
            "drop(s) in this recording"
        )
        self._put_status("camera_recording_drop", payload)
        self._last_writer_drop_warning = now

    def _set_transposition(self, value):
        active = bool(value.get("active"))
        if active and self.transposition_overlay is None:
            self.transposition_active = False
            self.transposition_payload = None
            clear_latest(self.transposition_channel)
            message = self.transposition_error or "Could not load camera_matrixes.json"
            self._put_status("transposition_error", message)
            return
        self.transposition_active = active
        if not active:
            self.transposition_payload = None
            clear_latest(self.transposition_channel)
        self._put_status(
            "transposition_state",
            {
                "active": active,
                "message": (
                    "ON · GROUP B · camera matrix loaded"
                    if active
                    else "OFF · camera points use the current radar filters and distance cutoff"
                ),
            },
        )

    def _report_unusual_pts_gap(self, timing):
        if not timing.get("large_pts_gap_candidate"):
            return
        self._pending_pts_gap_candidates += 1
        now = time.monotonic()
        if now - self._last_pts_gap_warning < 1.0:
            return
        payload = {
            "reason": "unusual PTS gap",
            "candidates": self._pending_pts_gap_candidates,
            "pts_delta_ns": timing.get("pts_delta_ns"),
            "channel": self.channel,
        }
        print(
            f"[WARNING][CAMERA] Camera channel {self.channel} had "
            f"{self._pending_pts_gap_candidates} unusual PTS gap candidate(s); "
            "this is not counted as confirmed frame loss"
        )
        self._put_status("camera_timing_warning", payload)
        self._pending_pts_gap_candidates = 0
        self._last_pts_gap_warning = now

    def _start_snapshot_recording(self, value):
        """Enable the waiting writer on the running stream, preserving decoder history."""
        if self.snapshot_recorder.active:
            self._put_status("camera_recording_error", "Camera recording is already active")
            return
        if not self.connected:
            self._put_status("camera_recording_error", "Connect the camera before starting camera recording")
            self._put_status(
                "calibration_recording_state" if value.get("calibration") else "camera_recording_state",
                {"active": False},
            )
            return
        calibration = bool(value.get("calibration"))
        try:
            if calibration:
                # Also covers callers that start a calibration recording
                # without first toggling the calibration-camera control.
                self.calibration_scheduler_priority.enable()
            self._last_writer_drop_warning = 0.0
            self._last_pts_gap_warning = 0.0
            self._pending_pts_gap_candidates = 0
            self.snapshot_recorder.start(
                value.get("folders", {}),
                calibration=calibration,
                latency_adjustment_ms=self.latency_adjustment_ms,
                timing_session={
                    "camera_channel": self.channel,
                    "decoder_backend": self.current_decoder_backend.name,
                    "pipeline_latency_ms": self.pipeline_latency_ms,
                    "stream_epoch_at_start": self.stream_epoch,
                    "pipeline_restarted_for_recording": False,
                    "display_journal": value.get("display_journal"),
                    "scheduler_priority": (
                        self.calibration_scheduler_priority.status()
                        if calibration
                        else None
                    ),
                    "timing_contract": {
                        "media_reference": (
                            "pipeline clock anchor plus segment-mapped buffer running time"
                        ),
                        "application_arrival_boundary": (
                            "capture appsink new-sample callback entry"
                        ),
                        "correction_sign": (
                            "positive correction is subtracted from media reference"
                        ),
                        "capture_reference": (
                            "provisional corrected camera timestamp; physical exposure "
                            "start/midpoint is not established"
                        ),
                        "arrival_delay": (
                            "application arrival monotonic time minus estimated capture time"
                        ),
                        "units": "integer nanoseconds unless a field name says otherwise",
                    },
                },
            )
            self.calibration_recording = calibration
            message = (
                "calibration_recording_state"
                if calibration
                else "camera_recording_state"
            )
            self._put_status(message, {"active": True})
        except Exception as error:
            self._put_status("camera_recording_error", str(error))
            message = (
                "calibration_recording_state"
                if calibration
                else "camera_recording_state"
            )
            self._put_status(message, {"active": False})

    def _stop_snapshot_recording(self):
        was_calibration = self.calibration_recording
        try:
            self.snapshot_recorder.record_timing_events(self.reference_clock.poll())
            self.snapshot_recorder.update_transport_stats(
                self.reference_clock.transport_stats(),
                stream_epoch=self.stream_epoch,
            )
            count = self.snapshot_recorder.stop()
            self.calibration_recording = False
            confirmed_not_saved = (
                self.snapshot_recorder.frames_dropped
                + self.snapshot_recorder.frames_rejected_invalid_timing
            )
            self._put_status(
                "calibration_recording_state"
                if was_calibration
                else "camera_recording_state",
                {
                    "active": False,
                    "count": count,
                    "dropped": confirmed_not_saved,
                    "writer_dropped": self.snapshot_recorder.frames_dropped,
                    "pipeline_dropped": (
                        self.snapshot_recorder.frames_rejected_invalid_timing
                    ),
                    "pts_gap_candidates": (
                        self.snapshot_recorder.unusual_pts_gap_candidates
                    ),
                },
            )
        except Exception as error:
            self.calibration_recording = False
            self._put_status("camera_recording_error", str(error))
            self._put_status(
                "calibration_recording_state"
                if was_calibration
                else "camera_recording_state",
                {"active": False},
            )

    def _connect_camera(self):
        try:
            with socket.create_connection(("192.168.1.108", 554), timeout=2):
                self.connected = self.reset_decoder_selection()
        except OSError:
            self.connected = False
        self._put_status("change_cam", self.connected)
        return self.connected

    def _set_calibration_camera(self, active):
        active = bool(active)
        if active:
            # New GStreamer and image-writer threads inherit this preference.
            self.calibration_scheduler_priority.enable()
            self.calibration_mode = True
            self.channel = 4
            self.channel_changed = True
            if not self.connected:
                self._connect_camera()
            if not self.connected:
                self.calibration_mode = False
                self.channel = self.normal_channel
                self.calibration_scheduler_priority.restore()
            if self.connected:
                self.exit_reason = _RESULT_RESTART
            self._put_status(
                "calibration_camera_state",
                {"active": self.connected, "channel": 4},
            )
            return self.connected

        self.calibration_mode = False
        self.channel = self.normal_channel
        self.channel_changed = True
        self.connected = False
        self.exit_reason = _RESULT_CLOSED
        self._fail_manual_snapshot("Calibration camera closed before taking the snapshot")
        self._put_status("change_cam", False)
        self._put_status("calibration_camera_state", {"active": False, "channel": 4})
        self.calibration_scheduler_priority.restore()
        return True

    def _set_latency_settings(self, value):
        try:
            pipeline_latency_ms = int(value.get("pipeline_latency_ms"))
            adjustment_ms = float(value.get("latency_adjustment_ms"))
        except (AttributeError, TypeError, ValueError):
            self._put_status(
                "camera_latency_error",
                "Both camera latency values must be numeric",
            )
            return False
        if pipeline_latency_ms < 0:
            self._put_status(
                "camera_latency_error",
                "RTSP source latency cannot be negative",
            )
            return False
        restart = pipeline_latency_ms != self.pipeline_latency_ms and self.connected
        self.pipeline_latency_ms = pipeline_latency_ms
        self.latency_adjustment_ms = adjustment_ms
        self.timestamp_policy.set_capture_correction_ms(adjustment_ms)
        self.snapshot_recorder.set_latency_adjustment_ms(adjustment_ms)
        self._put_status(
            "camera_latency_state",
            {
                "pipeline_latency_ms": pipeline_latency_ms,
                "latency_adjustment_ms": adjustment_ms,
            },
        )
        if restart:
            self.exit_reason = _RESULT_RESTART
        return restart

    def _set_recording_rate(self, value):
        try:
            numeric_value = float(value.get("frames_per_30"))
            if not numeric_value.is_integer():
                raise ValueError
            frames_per_30 = int(numeric_value)
            self.snapshot_recorder.set_recorded_frames_per_30(frames_per_30)
        except (AttributeError, TypeError, ValueError):
            self._put_status(
                "camera_recording_rate_error",
                "Recorded camera frames must be a whole number between 1 and 30",
            )
            return
        self.recording_frames_per_30 = frames_per_30
        self._put_status(
            "camera_recording_rate_state",
            {"frames_per_30": frames_per_30},
        )

    def _set_decoder_backend(self, value):
        preference = str(value.get("backend", "auto")).strip().lower()
        if preference not in ("auto", "rtx", "orin", "cpu"):
            self._put_status(
                "camera_pipeline_error",
                f"Unsupported camera pipeline selection: {preference!r}",
            )
            return False
        if preference == self.decoder_preference:
            return False

        self.decoder_preference = preference
        self._pipeline_error_reported = False
        self.last_pipeline_error = None
        if not self.connected:
            return False
        if not self.reset_decoder_selection():
            self.connected = False
            self._put_status("change_cam", False)
            return True
        self.exit_reason = _RESULT_RESTART
        return True

    def _set_display_resolution(self, value):
        try:
            width = int(value.get("width", self.display_width))
            height = int(value.get("height", self.display_height))
        except (AttributeError, TypeError, ValueError):
            self._put_status(
                "playback_resolution_error",
                "Playback width and height must be positive integers",
            )
            return False
        if width <= 0 or height <= 0:
            self._put_status(
                "playback_resolution_error",
                "Playback width and height must be positive integers",
            )
            return False
        if (width, height) == (self.display_width, self.display_height):
            return False
        self.display_width = width
        self.display_height = height
        if self.connected:
            self.exit_reason = _RESULT_RESTART
            return True
        return False

    def _fail_manual_snapshot(self, message):
        with self._manual_snapshot_lock:
            request = self._pending_manual_snapshot
            self._pending_manual_snapshot = None
        if request is None:
            return
        self._put_status(
            "manual_snapshot_error",
            {
                "request_id": request.get("request_id"),
                "message": message,
            },
        )

    def _queue_manual_snapshot(self, value):
        if not self.connected:
            self._put_status(
                "manual_snapshot_error",
                {
                    "request_id": value.get("request_id"),
                    "message": "Connect the camera before taking a snapshot",
                },
            )
            return False

        channel = int(value.get("channel", 0))
        if channel not in (1, 2, 3):
            self._put_status(
                "manual_snapshot_error",
                {
                    "request_id": value.get("request_id"),
                    "message": f"Unsupported camera group: {channel}",
                },
            )
            return False

        request = dict(value)
        request["restore_channel"] = self.channel
        request["queued_at"] = time.monotonic()
        with self._manual_snapshot_lock:
            if self._pending_manual_snapshot is not None:
                self._put_status(
                    "manual_snapshot_error",
                    {
                        "request_id": value.get("request_id"),
                        "message": "A camera snapshot is already pending",
                    },
                )
                return False
            self._pending_manual_snapshot = request

        if channel != self.channel:
            self.channel = channel
            self.channel_changed = True
            self.exit_reason = _RESULT_RESTART
            return True
        return False

    def _restore_channel_after_snapshot(self, channel):
        if channel != self.channel:
            self.channel = channel
            self.channel_changed = True
            self.exit_reason = _RESULT_RESTART
            if self.main_loop:
                self.main_loop.quit()
        return GLib.SOURCE_REMOVE

    def _emit_manual_snapshot(self, frame, captured_at):
        with self._manual_snapshot_lock:
            request = self._pending_manual_snapshot
            if request is None or int(request["channel"]) != self.channel:
                return
            self._pending_manual_snapshot = None

        success, encoded = cv.imencode(".jpg", frame)
        if not success:
            self._put_status(
                "manual_snapshot_error",
                {
                    "request_id": request.get("request_id"),
                    "message": "Could not encode the camera snapshot",
                },
            )
            return

        self._put_status(
            "manual_snapshot_frame",
            {
                "request_id": request.get("request_id"),
                "folder": request.get("folder"),
                "channel": int(request["channel"]),
                "captured_at": captured_at.isoformat(timespec="microseconds"),
                "image_bytes": encoded.tobytes(),
            },
            timeout=1.0,
        )

        restore_channel = int(request.get("restore_channel", self.channel))
        if restore_channel != self.channel:
            GLib.idle_add(self._restore_channel_after_snapshot, restore_channel)

    @staticmethod
    def _sample_to_frame(sample):
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            caps = sample.get_caps().get_structure(0)
            width = caps.get_value("width")
            height = caps.get_value("height")
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3).copy()
            return frame
        finally:
            buffer.unmap(map_info)

    def _reset_camera_ntp_observation(self):
        self._last_published_camera_ntp_ns = None
        self._last_ntp_ui_update = 0.0
        self._has_frame_ntp = False
        self._put_status(
            "camera_ntp_time",
            {"available": False, "channel": self.channel},
        )

    def _publish_camera_ntp(self, camera_ntp_ns, offset_ms=None):
        now = time.monotonic()
        if (
            self._last_published_camera_ntp_ns is not None
            and now - self._last_ntp_ui_update < NTP_UI_UPDATE_INTERVAL_SECONDS
        ):
            return
        self._last_published_camera_ntp_ns = int(camera_ntp_ns)
        self._last_ntp_ui_update = now
        self._put_status(
            "camera_ntp_time",
            {
                "available": True,
                "channel": self.channel,
                "ntp_unix_ns": int(camera_ntp_ns),
                "offset_ms": offset_ms,
            },
        )

    def _observe_camera_ntp(self, camera_ntp_ns, offset_seconds):
        if camera_ntp_ns is None:
            return
        camera_ntp_ns = int(camera_ntp_ns)
        self._has_frame_ntp = True
        self._publish_camera_ntp(
            camera_ntp_ns,
            None if offset_seconds is None else offset_seconds * 1_000.0,
        )

    def check_camera_ntp(self):
        self.snapshot_recorder.record_timing_events(self.reference_clock.poll())
        self.snapshot_recorder.update_transport_stats(
            self.reference_clock.transport_stats(),
            stream_epoch=self.stream_epoch,
        )
        if not self._has_frame_ntp:
            camera_ntp_ns = self.reference_clock.latest_sender_report_ntp_ns
            if (
                camera_ntp_ns is not None
                and camera_ntp_ns != self._last_published_camera_ntp_ns
            ):
                self._publish_camera_ntp(camera_ntp_ns)
        return GLib.SOURCE_CONTINUE

    def on_new_display_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        frame = self._sample_to_frame(sample)
        if frame is None:
            return Gst.FlowReturn.ERROR
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(frame)
        return Gst.FlowReturn.OK

    def _capture_queue_levels(self):
        pipeline = getattr(self, "pipeline", None)
        queue_element = (
            pipeline.get_by_name("capture_queue")
            if pipeline is not None and hasattr(pipeline, "get_by_name")
            else None
        )
        if queue_element is None:
            return {}
        output = {}
        for property_name, field_name in (
            ("current-level-buffers", "capture_queue_level_buffers"),
            ("current-level-bytes", "capture_queue_level_bytes"),
            ("current-level-time", "capture_queue_level_time_ns"),
        ):
            try:
                output[field_name] = int(queue_element.get_property(property_name))
            except (AttributeError, TypeError, ValueError):
                pass
        return output

    def _retain_capture_failure(self, reason, timing):
        note_invalid_frame = getattr(
            self.snapshot_recorder, "note_invalid_timing_frame", None
        )
        if note_invalid_frame is not None:
            note_invalid_frame(reason=reason, timing=timing)

    def on_new_capture_sample(self, sink):
        application_arrival_monotonic_ns = time.monotonic_ns()
        application_arrival_unix_ns = time.time_ns()
        sample = sink.emit("pull-sample")
        sample_pulled_monotonic_ns = time.monotonic_ns()
        sample_pulled_unix_ns = time.time_ns()
        if sample is None:
            self._retain_capture_failure(
                "capture appsink returned no sample",
                {
                    "application_arrival_monotonic_ns": application_arrival_monotonic_ns,
                    "application_arrival_unix_ns": application_arrival_unix_ns,
                    "sample_pulled_monotonic_ns": sample_pulled_monotonic_ns,
                    "sample_pulled_unix_ns": sample_pulled_unix_ns,
                },
            )
            return Gst.FlowReturn.ERROR
        frame = self._sample_to_frame(sample)
        frame_converted_monotonic_ns = time.monotonic_ns()
        frame_converted_unix_ns = time.time_ns()
        if frame is None:
            self._retain_capture_failure(
                "capture sample could not be converted to an image",
                {
                    "application_arrival_monotonic_ns": application_arrival_monotonic_ns,
                    "application_arrival_unix_ns": application_arrival_unix_ns,
                    "sample_pulled_monotonic_ns": sample_pulled_monotonic_ns,
                    "sample_pulled_unix_ns": sample_pulled_unix_ns,
                    "frame_converted_monotonic_ns": frame_converted_monotonic_ns,
                    "frame_converted_unix_ns": frame_converted_unix_ns,
                },
            )
            return Gst.FlowReturn.ERROR
        shape = getattr(frame, "shape", ())
        if len(shape) >= 2:
            self.capture_size = (shape[1], shape[0])
        first_frame = not self.first_frame_received
        self.first_frame_received = True
        if first_frame:
            print(
                f"[DEBUG][CAMERA] Camera channel {self.channel} is using the "
                f"{self.current_decoder_backend.name} decoder"
            )

        timestamp = self.timestamp_policy.timestamp_for_sample(
            sample,
            application_arrival_monotonic_ns=application_arrival_monotonic_ns,
            application_arrival_unix_ns=application_arrival_unix_ns,
            sample_pulled_monotonic_ns=sample_pulled_monotonic_ns,
            sample_pulled_unix_ns=sample_pulled_unix_ns,
            frame_converted_monotonic_ns=frame_converted_monotonic_ns,
            frame_converted_unix_ns=frame_converted_unix_ns,
            capture_queue_levels=self._capture_queue_levels(),
        )
        if not timestamp.valid:
            self._reject_synchronized_frame(
                timestamp.reason or "unknown timestamp error",
                timing=timestamp.timing,
            )
            return Gst.FlowReturn.OK

        self._observe_camera_ntp(
            timestamp.camera_ntp_ns,
            timestamp.reference_clock_offset_seconds,
        )
        timing = dict(timestamp.timing or {})
        timing.update({
            "timestamp_source": timestamp.source,
            "reference_clock_offset_seconds": (
                timestamp.reference_clock_offset_seconds
            ),
        })
        self._report_unusual_pts_gap(timing)
        self.snapshot_recorder.submit(
            frame,
            captured_at=timestamp.captured_at,
            timing=timing,
        )
        self._emit_manual_snapshot(frame, timestamp.captured_at)
        return Gst.FlowReturn.OK

    def _reject_synchronized_frame(self, reason, *, timing=None):
        note_invalid_frame = getattr(
            self.snapshot_recorder,
            "note_invalid_timing_frame",
            None,
        )
        if note_invalid_frame is not None:
            note_invalid_frame(reason=reason, timing=timing)
        now = time.monotonic()
        if now - self._last_timestamp_warning >= TIMESTAMP_WARNING_INTERVAL_SECONDS:
            print(f"[DEBUG][CAMERA] Skipping unsynchronized camera frame: {reason}")
            self._put_status("camera_timestamp_warning", reason)
            self._last_timestamp_warning = now

        with self._manual_snapshot_lock:
            request = self._pending_manual_snapshot
            queued_at = request.get("queued_at", now) if request is not None else now
        if request is not None and now - queued_at >= MANUAL_SNAPSHOT_TIMESTAMP_TIMEOUT_SECONDS:
            self._fail_manual_snapshot(
                "Camera frames did not contain a valid synchronization timestamp"
            )

    def on_message(self, _bus, message):
        if message.type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if structure is not None and structure.get_name() == "drop-msg":
                self.snapshot_recorder.record_timing_events(({
                    "event": "jitterbuffer_drop",
                    "stream_epoch": self.stream_epoch,
                    "received_monotonic_ns": time.monotonic_ns(),
                    "details": structure.to_string(),
                },))
            return
        if message.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            self.exit_reason = _RESULT_FAILURE
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                self.last_pipeline_error = f"{error}; {debug}"
                print(f"[DEBUG][CAMERA] GStreamer error: {error}; {debug}")
            if self.main_loop:
                self.main_loop.quit()

    def process_commands(self):
        restart = False
        try:
            while self.communicate.poll():
                event, value = self.communicate.recv()
                if event == "STOP":
                    self._stop_snapshot_recording()
                    self._fail_manual_snapshot("Camera process stopped before taking the snapshot")
                    self.shutdown_event.set()
                    self.connected = False
                    self.exit_reason = _RESULT_CLOSED
                    restart = True
                elif event == "choose":
                    self.normal_channel = value
                    if self.calibration_mode or value == self.channel:
                        continue
                    self.channel = value
                    self.channel_changed = True
                    self.capture_size = None
                    self.transposition_payload = None
                    self._clear_frames()
                    if self.connected:
                        self.exit_reason = _RESULT_RESTART
                        restart = True
                elif event == "conn_cam":
                    if self.calibration_mode:
                        continue
                    if self.connected:
                        self.connected = False
                        self.exit_reason = _RESULT_CLOSED
                        self._fail_manual_snapshot("Camera disconnected before taking the snapshot")
                        self._put_status("change_cam", False)
                        restart = True
                    else:
                        self._connect_camera()
                elif event == "calibration_camera":
                    restart = self._set_calibration_camera(value.get("active")) or restart
                elif event == "calibration_pipeline_restart":
                    if not self.connected or not self.calibration_mode:
                        self._put_status(
                            "calibration_pipeline_restart_error",
                            {"message": "Camera 4 must be open before restarting for calibration"},
                        )
                    elif self.snapshot_recorder.active:
                        self._put_status(
                            "calibration_pipeline_restart_error",
                            {"message": "Stop the current camera recording before restarting for calibration"},
                        )
                    else:
                        self.calibration_pipeline_restart_delay_seconds = float(
                            value.get(
                                "delay_seconds",
                                CALIBRATION_PIPELINE_RESTART_DELAY_SECONDS,
                            )
                        )
                        self.exit_reason = _RESULT_RESTART
                        restart = True
                elif event == "camera_decoder_backend":
                    restart = self._set_decoder_backend(value) or restart
                elif event == "camera_latency_settings":
                    restart = self._set_latency_settings(value) or restart
                elif event == "camera_recording_rate":
                    self._set_recording_rate(value)
                elif event == "transposition":
                    self._set_transposition(value)
                elif event == "record_start":
                    self._start_snapshot_recording(value)
                elif event == "record_stop":
                    self._stop_snapshot_recording()
                elif event == "snapshot_capture":
                    restart = self._queue_manual_snapshot(value) or restart
                elif event == "playback_resolution":
                    restart = self._set_display_resolution(value) or restart
        except (EOFError, OSError):
            self.shutdown_event.set()
            self.connected = False
            self.exit_reason = _RESULT_CLOSED
            restart = True

        snapshot_error = self.snapshot_recorder.poll_error()
        if snapshot_error is not None and self.snapshot_recorder.active:
            self._stop_snapshot_recording()

        if (restart or self.shutdown_event.is_set()) and self.main_loop:
            self.main_loop.quit()
        return GLib.SOURCE_CONTINUE

    def display_latest_frame(self):
        try:
            frame = self.frames.get_nowait()
        except queue.Empty:
            return GLib.SOURCE_CONTINUE
        if (
            self.transposition_active
            and not self.calibration_mode
            and self.channel == RADAR_GROUP_B
            and self.transposition_overlay is not None
            and self.capture_size is not None
        ):
            self.transposition_payload = get_latest(
                self.transposition_channel,
                self.transposition_payload,
            )
            try:
                frame = self.transposition_overlay.draw(
                    frame,
                    self.transposition_payload,
                    source_size=self.capture_size,
                )
            except (KeyError, TypeError, ValueError, cv.error) as error:
                self.transposition_active = False
                self.transposition_payload = None
                clear_latest(self.transposition_channel)
                self._put_status("transposition_error", str(error))
        cv.imshow("CALIBRATION CAMERA 4" if self.calibration_mode else "CAMERA", frame)
        cv.waitKey(1)
        if self.calibration_restart_waiting_for_preview:
            self.calibration_restart_waiting_for_preview = False
            self._put_status(
                "calibration_pipeline_ready",
                {"channel": self.channel, "stream_epoch": self.stream_epoch},
            )
        return GLib.SOURCE_CONTINUE

    def check_first_frame(self):
        if self.first_frame_received:
            return GLib.SOURCE_CONTINUE
        if time.monotonic() - self.attempt_started < FIRST_FRAME_TIMEOUT_SECONDS:
            return GLib.SOURCE_CONTINUE
        print(
            f"[DEBUG][CAMERA] Camera channel {self.channel} did not produce a frame within "
            f"{FIRST_FRAME_TIMEOUT_SECONDS:.1f} seconds"
        )
        self.last_pipeline_error = (
            f"The {self.current_decoder_backend.name} camera pipeline did not "
            f"produce a frame within {FIRST_FRAME_TIMEOUT_SECONDS:.1f} seconds"
        )
        self.exit_reason = _RESULT_FAILURE
        if self.main_loop:
            self.main_loop.quit()
        return GLib.SOURCE_CONTINUE

    def _clear_frames(self):
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    @property
    def current_decoder_backend(self):
        return self.decoder_backends[self.decoder_backend_index]

    def reset_decoder_selection(self):
        try:
            self.decoder_backends = available_decoder_backends(
                self.decoder_preference,
                strict=self.decoder_preference != "auto",
            )
        except RuntimeError as error:
            self._report_pipeline_error(str(error))
            print(f"[DEBUG][CAMERA] {error}")
            return False
        self.decoder_backend_index = 0
        self._pipeline_error_reported = False
        self.last_pipeline_error = None
        return True

    def _report_pipeline_error(self, message):
        if self._pipeline_error_reported:
            return
        self._pipeline_error_reported = True
        self._put_status("camera_pipeline_error", message)

    def advance_decoder_backend(self):
        next_index = self.decoder_backend_index + 1
        if next_index >= len(self.decoder_backends):
            return False
        previous = self.current_decoder_backend.name
        self.decoder_backend_index = next_index
        print(
            f"[DEBUG][CAMERA] {previous} decoder failed; trying "
            f"{self.current_decoder_backend.name}"
        )
        return True

    def _remove_sources(self):
        for source_id in self.source_ids:
            if source_id:
                GLib.source_remove(source_id)
        self.source_ids.clear()

    @staticmethod
    def _destroy_window():
        for window_name in ("CAMERA", "CALIBRATION CAMERA 4"):
            try:
                cv.destroyWindow(window_name)
                cv.waitKey(1)
            except cv.error:
                pass

    def run(self):
        pipeline_str = build_camera_pipeline(
            self.current_decoder_backend,
            display_width=self.display_width,
            display_height=self.display_height,
            latency_ms=self.pipeline_latency_ms,
        )
        bus = None
        self._clear_frames()
        self._reset_camera_ntp_observation()
        self.first_frame_received = False
        self.attempt_started = time.monotonic()
        self.exit_reason = _RESULT_FAILURE

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            source = self.pipeline.get_by_name("source")
            display_sink = self.pipeline.get_by_name("display_sink")
            capture_sink = self.pipeline.get_by_name("capture_sink")
            source.set_property("location", self.create_url(self.channel))
            self.stream_epoch += 1
            self.timestamp_policy.reset(
                self.pipeline,
                stream_epoch=self.stream_epoch,
            )
            self.reference_clock.reset(self.stream_epoch)
            self.reference_clock.attach_rtsp_source(source)
            display_sink.connect("new-sample", self.on_new_display_sample)
            capture_sink.connect("new-sample", self.on_new_capture_sample)
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            self.main_loop = GLib.MainLoop()

            self.source_ids = [
                GLib.timeout_add(10, self.process_commands),
                GLib.timeout_add(10, self.display_latest_frame),
                GLib.timeout_add(100, self.check_first_frame),
                GLib.timeout_add(500, self.check_camera_ntp),
            ]

            if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                self.last_pipeline_error = (
                    f"GStreamer could not start camera channel {self.channel} "
                    f"with the {self.current_decoder_backend.name} pipeline"
                )
                print(f"[DEBUG][CAMERA] {self.last_pipeline_error}")
                if self.decoder_preference != "auto":
                    self._report_pipeline_error(self.last_pipeline_error)
                return self.exit_reason, self.first_frame_received

            self.main_loop.run()
            return self.exit_reason, self.first_frame_received
        except Exception as error:
            self.last_pipeline_error = (
                f"Could not build the {self.current_decoder_backend.name} "
                f"camera pipeline: {error}"
            )
            print(f"[DEBUG][CAMERA] {self.last_pipeline_error}")
            if self.decoder_preference != "auto":
                self._report_pipeline_error(self.last_pipeline_error)
            return _RESULT_FAILURE, self.first_frame_received
        finally:
            self.snapshot_recorder.record_timing_events(self.reference_clock.poll())
            self.snapshot_recorder.update_transport_stats(
                self.reference_clock.transport_stats(),
                stream_epoch=self.stream_epoch,
            )
            self._remove_sources()
            if bus is not None:
                bus.remove_signal_watch()
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
            self._destroy_window()
            self.pipeline = None
            self.main_loop = None


def gstreamer_main(connection, pool, shutdown_event, transposition_channel=None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
    pipeline = GStreamerPipeline(
        connection,
        pool,
        shutdown_event,
        transposition_channel,
    )
    failed_attempts = 0

    try:
        while not shutdown_event.is_set():
            pipeline.process_commands()
            if (
                pipeline.calibration_pipeline_restart_delay_seconds is not None
                and pipeline.main_loop is None
            ):
                delay = pipeline.calibration_pipeline_restart_delay_seconds
                pipeline.calibration_pipeline_restart_delay_seconds = None
                if not pipeline.connected:
                    pipeline._put_status(
                        "calibration_pipeline_restart_error",
                        {"message": "Camera 4 closed before its video preview resumed"},
                    )
                    continue
                shutdown_event.wait(delay)
                if shutdown_event.is_set():
                    continue
                pipeline.calibration_restart_waiting_for_preview = True
            if pipeline.channel_changed:
                failed_attempts = 0
                pipeline.channel_changed = False
            if not pipeline.connected:
                failed_attempts = 0
                shutdown_event.wait(0.05)
                continue

            result, received_frame = pipeline.run()
            if result == _RESULT_RESTART:
                delay = pipeline.calibration_pipeline_restart_delay_seconds
                if delay is not None:
                    pipeline.calibration_pipeline_restart_delay_seconds = None
                    shutdown_event.wait(delay)
                    if shutdown_event.is_set():
                        continue
                    pipeline.calibration_restart_waiting_for_preview = True
                failed_attempts = 0
                continue
            if result == _RESULT_CLOSED:
                if pipeline.calibration_pipeline_restart_delay_seconds is not None:
                    pipeline.calibration_pipeline_restart_delay_seconds = None
                    pipeline._put_status(
                        "calibration_pipeline_restart_error",
                        {"message": "Camera 4 closed before its video preview resumed"},
                    )
                if pipeline.calibration_restart_waiting_for_preview:
                    pipeline.calibration_restart_waiting_for_preview = False
                    pipeline._put_status(
                        "calibration_pipeline_restart_error",
                        {"message": "Camera 4 closed before its video preview resumed"},
                    )
                failed_attempts = 0
                continue
            if shutdown_event.is_set() or not pipeline.connected:
                continue

            if not received_frame and pipeline.advance_decoder_backend():
                failed_attempts = 0
                continue

            if received_frame:
                failed_attempts = 0
            failed_attempts += 1
            if failed_attempts >= MAX_PIPELINE_ATTEMPTS:
                print(
                    f"[DEBUG][CAMERA] Camera channel {pipeline.channel} failed after "
                    f"{MAX_PIPELINE_ATTEMPTS} attempts"
                )
                pipeline._report_pipeline_error(
                    pipeline.last_pipeline_error
                    or (
                        f"Camera channel {pipeline.channel} failed after "
                        f"{MAX_PIPELINE_ATTEMPTS} attempts with the "
                        f"{pipeline.current_decoder_backend.name} pipeline"
                    )
                )
                if pipeline.calibration_restart_waiting_for_preview:
                    pipeline.calibration_restart_waiting_for_preview = False
                    pipeline._put_status(
                        "calibration_pipeline_restart_error",
                        {"message": "Camera video did not return after the calibration restart"},
                    )
                pipeline.connected = False
                pipeline._fail_manual_snapshot("Camera pipeline failed before taking the snapshot")
                if pipeline.snapshot_recorder.active:
                    pipeline._stop_snapshot_recording()
                pipeline._put_status("change_cam", False)
                if pipeline.calibration_mode:
                    pipeline.calibration_mode = False
                    pipeline.channel = pipeline.normal_channel
                    pipeline._put_status(
                        "calibration_camera_state",
                        {"active": False, "channel": 4},
                    )
                failed_attempts = 0
                continue

            shutdown_event.wait(PIPELINE_RETRY_DELAY_SECONDS)
    finally:
        pipeline._stop_snapshot_recording()
        pipeline.snapshot_recorder.close()
        pipeline._fail_manual_snapshot("Camera process stopped before taking the snapshot")
        pipeline._remove_sources()
        if pipeline.pipeline:
            pipeline.pipeline.set_state(Gst.State.NULL)
        cv.destroyAllWindows()
