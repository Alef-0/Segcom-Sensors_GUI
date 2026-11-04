"""Capture-only RTSP stream that records one camera group.

The live camera window shows a single DVR channel at a time. Recording must
not depend on which channel is on screen, so every recorded group gets its own
capture pipeline and its own image recorder while a recording is active.
"""

from __future__ import annotations

import time
from typing import Callable

import gi
import numpy as np

from sensors.camera.camera_pipeline import CameraDecoderBackend, build_capture_pipeline
from sensors.camera.camera_reference_clock import ReferenceClockObserver
from sensors.camera.camera_timebase import FrameTimestampPolicy

gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

FIRST_FRAME_TIMEOUT_SECONDS = 5.0
RETRY_DELAY_SECONDS = 1.0
FAILED_ATTEMPTS_BEFORE_WARNING = 3


def sample_to_frame(sample):
    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return None
    try:
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        return np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3).copy()
    finally:
        buffer.unmap(map_info)


class ChannelCaptureStream:
    """Decode one DVR channel and submit timestamped frames to its recorder."""

    def __init__(
        self,
        channel: int,
        recorder,
        *,
        url: str,
        decoder_backend: CameraDecoderBackend,
        latency_ms: int,
        report_error: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.channel = int(channel)
        self.recorder = recorder
        self.url = url
        self.decoder_backend = decoder_backend
        self.latency_ms = int(latency_ms)
        self.report_error = report_error
        self.clock = clock
        self.pipeline = None
        self.bus = None
        self.timestamp_policy = FrameTimestampPolicy()
        self.reference_clock = ReferenceClockObserver()
        self.stream_epoch = 0
        self.first_frame_received = False
        self.attempt_started = 0.0
        self.failed_attempts = 0
        self.next_retry_at: float | None = None
        self.last_error: str | None = None
        self._error_reported = False

    @property
    def running(self) -> bool:
        return self.pipeline is not None

    def start(self) -> None:
        self.next_retry_at = None
        if self.pipeline is None:
            self._open()

    def stop(self) -> None:
        self.next_retry_at = None
        self._close()

    def restart(self) -> None:
        self._close()
        self.start()

    def _open(self) -> None:
        self.first_frame_received = False
        self.attempt_started = self.clock()
        try:
            pipeline = Gst.parse_launch(
                build_capture_pipeline(self.decoder_backend, latency_ms=self.latency_ms)
            )
            source = pipeline.get_by_name("source")
            capture_sink = pipeline.get_by_name("capture_sink")
            source.set_property("location", self.url)
            self.stream_epoch += 1
            self.timestamp_policy.reset(pipeline, stream_epoch=self.stream_epoch)
            self.reference_clock.reset(self.stream_epoch)
            self.reference_clock.attach_rtsp_source(source)
            capture_sink.connect("new-sample", self.on_new_sample)
            self.pipeline = pipeline
            self.bus = pipeline.get_bus()
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                self._fail(
                    f"GStreamer could not start camera channel {self.channel} "
                    f"with the {self.decoder_backend.name} pipeline"
                )
        except Exception as error:
            self._fail(f"Could not build the camera channel {self.channel} pipeline: {error}")

    def _close(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        self.bus = None
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)

    def _fail(self, reason: str) -> None:
        had_frames = self.first_frame_received
        self._close()
        self.last_error = reason
        self.failed_attempts = 1 if had_frames else self.failed_attempts + 1
        print(f"[DEBUG][CAMERA] Recording stream for channel {self.channel}: {reason}")
        if (
            self.failed_attempts >= FAILED_ATTEMPTS_BEFORE_WARNING
            and not self._error_reported
            and self.report_error is not None
        ):
            self._error_reported = True
            self.report_error(
                f"Camera group channel {self.channel} is not producing frames for the "
                f"recording ({reason}). Retrying until the recording stops."
            )
        self.next_retry_at = self.clock() + RETRY_DELAY_SECONDS

    def poll(self) -> None:
        """Drain bus messages, detect stalls, and retry failed streams."""

        now = self.clock()
        if self.pipeline is None:
            if self.next_retry_at is not None and now >= self.next_retry_at:
                self.next_retry_at = None
                self._open()
            return

        bus = self.bus
        while bus is not None:
            message = bus.pop()
            if message is None:
                break
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                self._fail(f"GStreamer error: {error}; {debug}")
                return
            if message.type == Gst.MessageType.EOS:
                self._fail("stream ended")
                return

        if (
            not self.first_frame_received
            and now - self.attempt_started >= FIRST_FRAME_TIMEOUT_SECONDS
        ):
            self._fail(
                f"no frame within {FIRST_FRAME_TIMEOUT_SECONDS:.1f} seconds"
            )
            return

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        frame = sample_to_frame(sample)
        if frame is None:
            return Gst.FlowReturn.ERROR
        if not self.first_frame_received:
            self.first_frame_received = True
            self.failed_attempts = 0
            self._error_reported = False
            print(
                f"[DEBUG][CAMERA] Recording camera channel {self.channel} with the "
                f"{self.decoder_backend.name} decoder"
            )

        timestamp = self.timestamp_policy.timestamp_for_sample(sample)
        if not timestamp.valid:
            self.recorder.note_invalid_timing_frame()
            return Gst.FlowReturn.OK

        timing = dict(timestamp.timing or {})
        timing.update({
            "timestamp_source": timestamp.source,
            "reference_clock_offset_seconds": timestamp.reference_clock_offset_seconds,
        })
        self.recorder.submit(frame, captured_at=timestamp.captured_at, timing=timing)
        return Gst.FlowReturn.OK
