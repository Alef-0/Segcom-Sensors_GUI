"""GStreamer RTSP pipeline for camera streaming, capture, and transposition."""

from datetime import datetime, timezone
import queue
import signal
import socket
import time

import cv2 as cv
import gi
import numpy as np

from sensors.recording import CameraRecorder
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
NTP_UI_UPDATE_INTERVAL_SECONDS = 0.1
DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS = 87.348
_RESULT_FAILURE = "failure"
_RESULT_RESTART = "restart"
_RESULT_CLOSED = "closed"


class GStreamerPipeline:
    """Manages GStreamer RTSP stream, display frames, and snapshot capture."""

    def __init__(self, conn, pool, shutdown_event, transposition_channel=None, include_display_sink=True):
        self.include_display_sink = include_display_sink
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
        self.snapshot_recorder = CameraRecorder(self._report_snapshot, self._report_recording_drop)
        self.snapshot_recorder.prepare()
        self.decoder_preference = "auto"
        self.decoder_backends = available_decoder_backends()
        self.decoder_backend_index = 0
        self.last_pipeline_error: str | None = None
        self._pipeline_error_reported = False
        self._last_pts: int | None = None
        self._last_timestamp_warning = 0.0
        self._last_ntp_ui_update = 0.0
        self._last_writer_drop_warning = 0.0
        self.snapshot_request: dict | None = None
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
            f"image writer queue is full; {payload['dropped']} writer-queue drop(s)"
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
        self._put_status("transposition_state", {
            "active": active,
            "message": ("ON · GROUP B · camera matrix loaded" if active
                        else "OFF · camera points use the current radar filters and distance cutoff"),
        })

    def _start_snapshot_recording(self, value):
        if self.snapshot_recorder.active:
            self._put_status("camera_recording_error", "Camera recording is already active")
            return
        if not self.connected:
            self._put_status("camera_recording_error", "Connect the camera before starting camera recording")
            self._put_status("camera_recording_state", {"active": False})
            return
        try:
            self._last_writer_drop_warning = 0.0
            self.snapshot_recorder.start(
                value.get("folders", {}),
                latency_adjustment_ms=self.latency_adjustment_ms,
            )
            self._put_status("camera_recording_state", {"active": True})
        except Exception as error:
            self._put_status("camera_recording_error", str(error))
            self._put_status("camera_recording_state", {"active": False})

    def _stop_snapshot_recording(self):
        try:
            count = self.snapshot_recorder.stop()
            dropped = (
                self.snapshot_recorder.frames_dropped
                + self.snapshot_recorder.frames_rejected_invalid_timing
            )
            self._put_status("camera_recording_state", {
                "active": False,
                "count": count,
                "dropped": dropped,
            })
        except Exception as error:
            self._put_status("camera_recording_error", str(error))
            self._put_status("camera_recording_state", {"active": False})

    def _connect_camera(self):
        try:
            with socket.create_connection(("192.168.1.108", 554), timeout=2):
                self.connected = self.reset_decoder_selection()
        except OSError:
            self.connected = False
        self._put_status("change_cam", self.connected)
        return self.connected

    @staticmethod
    def _sample_to_frame(sample):
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            caps = sample.get_caps().get_structure(0)
            width, height = caps.get_value("width"), caps.get_value("height")
            return np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3).copy()
        finally:
            buffer.unmap(map_info)

    def _reset_camera_ntp_observation(self):
        self._last_ntp_ui_update = 0.0
        self._put_status("camera_ntp_time", {"available": False, "channel": self.channel})

    def _update_ntp_display(self, ntp_ns, meta):
        now = time.monotonic()
        if now - self._last_ntp_ui_update < NTP_UI_UPDATE_INTERVAL_SECONDS:
            return
        self._last_ntp_ui_update = now
        try:
            ref_str = meta.reference.to_string() if meta.reference else ""
            if "ntp" in ref_str:
                unix_ns = ntp_ns - 2_208_988_800 * 1_000_000_000
            elif "unix" in ref_str:
                unix_ns = ntp_ns
            else:
                unix_ns = None
            if unix_ns is not None and unix_ns > 0:
                self._put_status("camera_ntp_time", {
                    "available": True,
                    "channel": self.channel,
                    "ntp_unix_ns": unix_ns,
                })
        except Exception:
            pass

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

    def on_new_capture_sample(self, sink):
        monotonic_ns = time.monotonic_ns()
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        if buffer is None:
            self.snapshot_recorder.note_invalid_timing_frame(
                reason="sample has no buffer",
                timing={"application_arrival_monotonic_ns": monotonic_ns},
            )
            return Gst.FlowReturn.ERROR

        pts = buffer.pts if buffer.pts < Gst.CLOCK_TIME_NONE else -1
        if pts < 0 or (self._last_pts is not None and pts <= self._last_pts):
            self.snapshot_recorder.note_invalid_timing_frame(
                reason="invalid PTS",
                timing={"application_arrival_monotonic_ns": monotonic_ns},
            )
            now = time.monotonic()
            if now - self._last_timestamp_warning >= TIMESTAMP_WARNING_INTERVAL_SECONDS:
                print(f"[DEBUG][CAMERA] Skipping invalid PTS frame: {pts}")
                self._put_status("camera_timestamp_warning", "invalid PTS")
                self._last_timestamp_warning = now
            return Gst.FlowReturn.OK

        self._last_pts = pts
        frame = self._sample_to_frame(sample)
        if frame is None:
            return Gst.FlowReturn.ERROR

        shape = getattr(frame, "shape", ())
        if len(shape) >= 2:
            self.capture_size = (shape[1], shape[0])

        if not self.first_frame_received:
            self.first_frame_received = True
            print(f"[DEBUG][CAMERA] Camera channel {self.channel} using {self.current_decoder_backend.name} decoder")

        ntp = -1
        meta = buffer.get_reference_timestamp_meta(None)
        if meta is not None and meta.timestamp < Gst.CLOCK_TIME_NONE:
            ntp = int(meta.timestamp)
            self._update_ntp_display(ntp, meta)

        captured_at = datetime.now(timezone.utc)
        timing = {
            "pts_ns": pts,
            "camera_ntp_ns": ntp,
            "application_arrival_monotonic_ns": monotonic_ns,
        }

        # 1. Recording: always save while active
        if self.snapshot_recorder.active:
            self.snapshot_recorder.submit(frame, captured_at=captured_at, timing=timing)

        # 2. Snapshot: togglable variation to save this one specific frame at that moment
        snapshot_req = getattr(self, "snapshot_request", None)
        if snapshot_req is not None:
            self.snapshot_request = None
            success, encoded = cv.imencode(".jpg", frame)
            if success:
                self._put_status("manual_snapshot_frame", {
                    "request_id": snapshot_req.get("request_id"),
                    "folder": snapshot_req.get("folder"),
                    "channel": self.channel,
                    "captured_at": captured_at.isoformat(timespec="microseconds"),
                    "image_bytes": encoded.tobytes(),
                }, timeout=1.0)

        return Gst.FlowReturn.OK

    def on_message(self, _bus, message):
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
                    self.shutdown_event.set()
                    self.connected = False
                    self.exit_reason = _RESULT_CLOSED
                    restart = True
                elif event == "choose":
                    if value != self.channel:
                        self.channel = value
                        self.channel_changed = True
                        self.capture_size = None
                        self.transposition_payload = None
                        self._clear_frames()
                        if self.connected:
                            self.exit_reason = _RESULT_RESTART
                            restart = True
                elif event == "conn_cam":
                    if self.connected:
                        self.connected = False
                        self.exit_reason = _RESULT_CLOSED
                        self._put_status("change_cam", False)
                        restart = True
                    else:
                        self._connect_camera()
                elif event == "camera_decoder_backend":
                    self.decoder_preference = value.get("backend", "auto")
                elif event == "camera_latency_settings":
                    self.pipeline_latency_ms = int(value.get("pipeline_latency_ms", self.pipeline_latency_ms))
                    self.latency_adjustment_ms = float(value.get("latency_adjustment_ms", self.latency_adjustment_ms))
                    self.snapshot_recorder.set_latency_adjustment_ms(self.latency_adjustment_ms)
                elif event == "camera_recording_rate":
                    self.recording_frames_per_30 = int(value.get("frames_per_30", self.recording_frames_per_30))
                    self.snapshot_recorder.set_recorded_frames_per_30(self.recording_frames_per_30)
                elif event == "playback_resolution":
                    self.display_width = int(value.get("width", self.display_width))
                    self.display_height = int(value.get("height", self.display_height))
                elif event == "transposition":
                    self._set_transposition(value)
                elif event == "record_start":
                    self._start_snapshot_recording(value)
                elif event == "record_stop":
                    self._stop_snapshot_recording()
                elif event == "snapshot_capture":
                    self.snapshot_request = value
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
        if not self.include_display_sink:
            return GLib.SOURCE_CONTINUE
        try:
            frame = self.frames.get_nowait()
        except queue.Empty:
            return GLib.SOURCE_CONTINUE
        if (
            self.transposition_active
            and self.channel == RADAR_GROUP_B
            and self.transposition_overlay is not None
            and self.capture_size is not None
        ):
            self.transposition_payload = get_latest(self.transposition_channel, self.transposition_payload)
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
        cv.imshow("CAMERA", frame)
        cv.waitKey(1)
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

    def _remove_sources(self):
        for source_id in self.source_ids:
            if source_id:
                GLib.source_remove(source_id)
        self.source_ids.clear()

    def _destroy_window(self):
        if not self.include_display_sink:
            return
        try:
            cv.destroyWindow("CAMERA")
            cv.waitKey(1)
        except cv.error:
            pass

    def run(self):
        pipeline_str = build_camera_pipeline(
            self.current_decoder_backend,
            display_width=self.display_width,
            display_height=self.display_height,
            latency_ms=self.pipeline_latency_ms,
            include_display_sink=self.include_display_sink,
        )
        bus = None
        self._clear_frames()
        self._reset_camera_ntp_observation()
        self.first_frame_received = False
        self._last_pts = None
        self.attempt_started = time.monotonic()
        self.exit_reason = _RESULT_FAILURE

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            source = self.pipeline.get_by_name("source")
            display_sink = self.pipeline.get_by_name("display_sink")
            capture_sink = self.pipeline.get_by_name("capture_sink")
            source.set_property("location", self.create_url(self.channel))
            if display_sink is not None:
                display_sink.connect("new-sample", self.on_new_display_sample)
            if capture_sink is not None:
                capture_sink.connect("new-sample", self.on_new_capture_sample)
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            self.main_loop = GLib.MainLoop()

            self.source_ids = [
                GLib.timeout_add(10, self.process_commands),
                GLib.timeout_add(10, self.display_latest_frame),
                GLib.timeout_add(100, self.check_first_frame),
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
                f"Could not build the {self.current_decoder_backend.name} camera pipeline: {error}"
            )
            print(f"[DEBUG][CAMERA] {self.last_pipeline_error}")
            if self.decoder_preference != "auto":
                self._report_pipeline_error(self.last_pipeline_error)
            return _RESULT_FAILURE, self.first_frame_received
        finally:
            self._remove_sources()
            if bus is not None:
                bus.remove_signal_watch()
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
            self._destroy_window()
            self.pipeline = None
            self.main_loop = None


def gstreamer_main(connection, pool, shutdown_event, transposition_channel=None, include_display_sink=True):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
    pipeline = GStreamerPipeline(
        connection,
        pool,
        shutdown_event,
        transposition_channel,
        include_display_sink=include_display_sink,
    )
    failed_attempts = 0

    try:
        while not shutdown_event.is_set():
            pipeline.process_commands()
            if pipeline.channel_changed:
                failed_attempts = 0
                pipeline.channel_changed = False
            if not pipeline.connected:
                failed_attempts = 0
                shutdown_event.wait(0.05)
                continue

            result, received_frame = pipeline.run()
            if result in (_RESULT_RESTART, _RESULT_CLOSED):
                failed_attempts = 0
                continue
            if shutdown_event.is_set() or not pipeline.connected:
                continue

            if received_frame:
                failed_attempts = 0
            failed_attempts += 1
            if failed_attempts >= MAX_PIPELINE_ATTEMPTS:
                print(f"[DEBUG][CAMERA] Camera channel {pipeline.channel} failed after {MAX_PIPELINE_ATTEMPTS} attempts")
                pipeline._report_pipeline_error(
                    pipeline.last_pipeline_error
                    or (
                        f"Camera channel {pipeline.channel} failed after "
                        f"{MAX_PIPELINE_ATTEMPTS} attempts with the "
                        f"{pipeline.current_decoder_backend.name} pipeline"
                    )
                )
                pipeline.connected = False
                if pipeline.snapshot_recorder.active:
                    pipeline._stop_snapshot_recording()
                pipeline._put_status("change_cam", False)
                failed_attempts = 0
                continue

            shutdown_event.wait(PIPELINE_RETRY_DELAY_SECONDS)
    finally:
        pipeline._stop_snapshot_recording()
        pipeline.snapshot_recorder.close()
        pipeline._remove_sources()
        if pipeline.pipeline:
            pipeline.pipeline.set_state(Gst.State.NULL)
        cv.destroyAllWindows()
