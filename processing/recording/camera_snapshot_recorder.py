"""Bounded asynchronous JPEG writer for live camera and calibration capture."""

from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Mapping

import cv2 as cv
import numpy as np

from processing.recording.camera_telemetry import (
    CAMERA_RECORDING_SUMMARY_NAME, SCHEMA_VERSION, CameraTelemetryWriter,
)
from processing.recording.paths import IMAGE_DIRECTORY_NAME, image_path, image_reference

CAMERA_FRAME_RATE = 30
DEFAULT_RECORDED_FRAMES_PER_30 = 30
_STOP = object()


class CameraSnapshotRecorder:
    """Select camera frames on the capture thread and write them off-thread."""

    def __init__(self, saved_callback: Callable[[dict], None] | None = None,
                 dropped_callback: Callable[[dict], None] | None = None,
                 queue_size: int = 8):
        if queue_size < 1:
            raise ValueError("Camera writer queue size must be positive")
        self.saved_callback, self.dropped_callback = saved_callback, dropped_callback
        self.queue_size = queue_size
        self.error: Exception | None = None
        self.active = False
        self._folders: dict[int, Path] = {}
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._frame_number = 0
        self._telemetry = CameraTelemetryWriter()
        self._latency_adjustment_ms = 0.0
        self._calibration = False
        self.recorded_frames_per_30 = DEFAULT_RECORDED_FRAMES_PER_30
        self._sampling_accumulator = 0
        self.frames_observed = self.frames_selected = self.frames_dropped = 0
        self.unusual_pts_gap_candidates = self.frames_rejected_invalid_timing = 0
        self._recording_started_at = ""
        self._recording_started_unix_ns = self._recording_started_monotonic_ns = 0
        self._transport_stats: dict[str, int] = {}
        self._transport_stats_by_epoch: dict[int, dict[str, int]] = {}

    def prepare(self) -> None:
        with self._lock:
            if self._thread:
                if not self._thread.is_alive():
                    raise RuntimeError("Camera snapshot writer stopped unexpectedly")
                return
            self._queue = queue.Queue(self.queue_size)
            self._thread = threading.Thread(target=self._write_loop, name="camera-snapshot-writer", daemon=True)
            self._thread.start()

    def start(self, folders: Mapping[int, str], *, calibration: bool = False,
              latency_adjustment_ms: float = 0.0, timing_session: Mapping | None = None) -> None:
        with self._lock:
            if self.active:
                raise RuntimeError("Camera snapshot recording is already active")
            selected = {int(channel): Path(folder).expanduser() for channel, folder in folders.items()}
            if not selected:
                raise ValueError("No recording folders were supplied")
            for folder in selected.values():
                if not folder.is_dir():
                    raise ValueError(f"Recording folder does not exist: {folder}")
                (folder / IMAGE_DIRECTORY_NAME).mkdir(exist_ok=True)
            self.prepare()
            self._folders = selected
            self._sampling_accumulator = CAMERA_FRAME_RATE - self.recorded_frames_per_30
            self._frame_number = 0
            self.frames_observed = self.frames_selected = self.frames_dropped = 0
            self.unusual_pts_gap_candidates = self.frames_rejected_invalid_timing = 0
            self._transport_stats, self._transport_stats_by_epoch = {}, {}
            self.error = None
            self._latency_adjustment_ms, self._calibration = float(latency_adjustment_ms), bool(calibration)
            started = datetime.now().astimezone()
            self._recording_started_at = started.isoformat(timespec="microseconds")
            self._recording_started_unix_ns = time.time_ns()
            self._recording_started_monotonic_ns = time.monotonic_ns()
            self._telemetry.start(tuple(selected.values()), {
                "recording_started_at": self._recording_started_at,
                "recording_started_unix_ns": self._recording_started_unix_ns,
                "recording_started_monotonic_ns": self._recording_started_monotonic_ns,
                "recorded_frames_per_30": self.recorded_frames_per_30,
                "image_adjustment_ns": round(self._latency_adjustment_ms * 1_000_000),
                **dict(timing_session or {}),
            })
            self.active = True
            self._write_summary()

    def submit(self, frame: np.ndarray, captured_at: datetime | None = None,
               timing: Mapping | None = None) -> bool:
        with self._lock:
            if not self.active or self.error or self._queue is None:
                return False
            self.frames_observed += 1
            timing = dict(timing or {})
            self.unusual_pts_gap_candidates += int(bool(timing.get("large_pts_gap_candidate")))
            self._sampling_accumulator += self.recorded_frames_per_30
            if self._sampling_accumulator < CAMERA_FRAME_RATE:
                return False
            self._sampling_accumulator -= CAMERA_FRAME_RATE
            self.frames_selected += 1
            timestamp = (captured_at or datetime.now().astimezone()).isoformat(timespec="microseconds")
            try:
                self._queue.put_nowait((frame.copy(), timestamp, timing))
            except queue.Full:
                self.frames_dropped += 1
                self._telemetry.append_event({
                    "event": "frame_dropped_writer_queue", "observed_frame": self.frames_observed,
                    "selected_frame": self.frames_selected, "reason": "image writer queue is full",
                    "timing": timing,
                })
                if self.dropped_callback:
                    self.dropped_callback({
                        "reason": "image writer queue is full", "dropped": self.frames_dropped,
                        "selected": self.frames_selected, "queue_size": self.queue_size, "timing": timing,
                    })
                return False
            return True

    def poll_error(self):
        return self.error

    def note_invalid_timing_frame(self, *, reason="invalid timing", timing=None) -> None:
        with self._lock:
            if not self.active:
                return
            self.frames_rejected_invalid_timing += 1
            self._telemetry.append_event({
                "event": "frame_rejected_invalid_timing",
                "rejected_frame": self.frames_rejected_invalid_timing,
                "reason": str(reason), "timing": dict(timing or {}),
            })

    def record_timing_events(self, events) -> None:
        if self.active:
            for event in events:
                self._telemetry.append_event(event)

    def update_transport_stats(self, stats: Mapping, *, stream_epoch: int = 0) -> None:
        epoch_stats = self._transport_stats_by_epoch.setdefault(int(stream_epoch), {})
        for key, value in stats.items():
            epoch_stats[str(key)] = max(epoch_stats.get(str(key), 0), int(value))
        keys = {key for row in self._transport_stats_by_epoch.values() for key in row}
        self._transport_stats = {key: sum(row.get(key, 0) for row in self._transport_stats_by_epoch.values()) for key in keys}

    def set_latency_adjustment_ms(self, value: float) -> None:
        self._latency_adjustment_ms = float(value)

    def set_recorded_frames_per_30(self, value: int) -> None:
        number = float(value)
        if not number.is_integer() or not 1 <= number <= CAMERA_FRAME_RATE:
            raise ValueError("Recorded frames must be a whole number from 1 to 30")
        with self._lock:
            self.recorded_frames_per_30 = int(number)
            self._sampling_accumulator = CAMERA_FRAME_RATE - self.recorded_frames_per_30

    def stop(self) -> int:
        with self._lock:
            if not self.active:
                return self._frame_number
            self.active = False
            writer, work = self._thread, self._queue
        if writer and work and writer.is_alive():
            drained = threading.Event()
            while writer.is_alive():
                try:
                    work.put(drained, timeout=0.1)
                    break
                except queue.Full:
                    continue
            while writer.is_alive() and not drained.wait(0.1):
                pass
        self._write_summary(final=True)
        self._telemetry.stop()
        if self.error:
            raise self.error
        return self._frame_number

    def close(self) -> None:
        try:
            if self.active:
                self.stop()
        finally:
            if self._queue is not None and self._thread is not None:
                while self._thread.is_alive():
                    try:
                        self._queue.put(_STOP, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                self._thread.join()
                self._thread, self._queue = None, None

    def _write_loop(self):
        assert self._queue is not None
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    if isinstance(item, threading.Event):
                        item.set()
                        continue
                    frame, captured_at, timing = item
                    number = self._frame_number + 1
                    filename = f"camera_{number:06d}.jpg"
                    reference = image_reference(filename)
                    files = {}
                    for channel, folder in self._folders.items():
                        path = image_path(folder, filename)
                        if not cv.imwrite(str(path), frame):
                            raise RuntimeError(f"Could not save camera snapshot: {path}")
                        files[channel] = reference
                    self._frame_number = number
                    self._record_timing(reference, captured_at, timing, time.time_ns())
                    if self.saved_callback:
                        self.saved_callback({"files": files, "captured_at": captured_at, "calibration": self._calibration})
                finally:
                    self._queue.task_done()
        except Exception as error:
            self.error = error

    def _record_timing(self, filename: str, captured_at: str, timing: Mapping, saved_at_ns: int):
        if not self._telemetry.active:
            return
        data = dict(timing)
        epoch_fields = (
            "stream_epoch", "mapping_revision", "segment_epoch", "pipeline_zero_unix_ns",
            "pipeline_zero_monotonic_ns", "pipeline_clock_type", "pipeline_base_time_ns",
        )
        self._telemetry.update_epoch({key: data[key] for key in epoch_fields if data.get(key) is not None})
        media_ns = data.get("media_time_ns")
        if media_ns is None:
            media_ns = round(datetime.fromisoformat(captured_at).timestamp() * 1e9)
        adjustment_ns = round(self._latency_adjustment_ms * 1e6)
        fields = (
            "timestamp_schema_version", "stream_epoch", "mapping_revision", "segment_epoch",
            "segment", "pts_ns", "pts_delta_ns", "running_time_ns", "running_time_delta_ns",
            "media_monotonic_ns", "application_arrival_monotonic_ns", "application_arrival_unix_ns",
            "application_arrival_delta_ns", "arrival_boundary", "sample_pulled_monotonic_ns",
            "sample_pulled_unix_ns", "frame_converted_monotonic_ns", "frame_converted_unix_ns",
            "timestamped_monotonic_ns", "timestamped_unix_ns", "pipeline_running_time_observed_ns",
            "pipeline_clock_mapping_monotonic_ns", "pipeline_clock_mapping_uncertainty_ns",
            "capture_queue_level_buffers", "capture_queue_level_bytes", "capture_queue_level_time_ns",
            "reference_timestamp_raw_ns", "reference_clock", "capture_estimator_model",
            "capture_estimator_version", "capture_calibration_version", "capture_estimator_correction_ns",
            "capture_estimator_status", "capture_estimator_uncertainty_ns",
            "capture_estimator_uncertainty_meaning", "capture_time_reference",
        )
        record = {key: data.get(key) for key in fields}
        record.update({
            "frame": filename,
            "received_monotonic_ns": data.get("host_monotonic_received_ns"),
            "received_unix_ns": data.get("host_realtime_received_ns"),
            "reference_ntp_ns": data.get("camera_ntp_ns"),
            "media_unix_ns": int(media_ns),
            "estimated_exposure_unix_ns": int(media_ns) - adjustment_ns,
            "estimated_capture_unix_ns": data.get("estimated_capture_unix_ns"),
            "estimated_capture_monotonic_ns": data.get("estimated_capture_monotonic_ns"),
            "observable_arrival_minus_media_ns": data.get("observable_arrival_minus_media_ns"),
            "estimated_arrival_delay_ns": data.get("estimated_arrival_delay_ns"),
            "saved_unix_ns": int(saved_at_ns), "flags": list(data.get("flags") or ()),
        })
        self._telemetry.append_frame(record)

    def _write_summary(self, final: bool = False):
        if not self._telemetry.active:
            return
        summary = {
            "schema_version": SCHEMA_VERSION, "started_at": self._recording_started_at,
            "started_unix_ns": self._recording_started_unix_ns,
            "started_monotonic_ns": self._recording_started_monotonic_ns,
            "recorded_frames_per_30": self.recorded_frames_per_30,
            "frames_observed": self.frames_observed, "frames_selected": self.frames_selected,
            "frames_dropped_writer_queue": self.frames_dropped,
            "unusual_pts_gap_candidates": self.unusual_pts_gap_candidates,
            "frames_rejected_invalid_timing": self.frames_rejected_invalid_timing,
            "confirmed_frames_not_saved": self.frames_dropped + self.frames_rejected_invalid_timing,
            "frames_saved": self._frame_number, **self._transport_stats,
            "transport_stats_by_epoch": [
                {"stream_epoch": epoch, **stats}
                for epoch, stats in sorted(self._transport_stats_by_epoch.items())
            ],
        }
        if final:
            summary["stopped_at"] = datetime.now().astimezone().isoformat(timespec="microseconds")
        self._telemetry.write_summary(summary)
