from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Mapping

import cv2 as cv
import numpy as np

from processing.recording.camera_telemetry import (
    CAMERA_RECORDING_SUMMARY_NAME,
    CAMERA_TIMESTAMPS_JOURNAL_NAME,
    CAMERA_TIMING_EVENTS_NAME,
    CAMERA_TIMING_SESSION_NAME,
    SCHEMA_VERSION,
    CameraTelemetryWriter,
)
from processing.recording.paths import (
    IMAGE_DIRECTORY_NAME,
    image_path,
    image_reference,
)


CAMERA_FRAME_RATE = 30
DEFAULT_RECORDED_FRAMES_PER_30 = 30
_STOP = object()


class CameraSnapshotRecorder:
    def __init__(
        self,
        saved_callback: Callable[[dict], None] | None = None,
        dropped_callback: Callable[[dict], None] | None = None,
        queue_size: int = 8,
    ):
        self.saved_callback = saved_callback
        self.dropped_callback = dropped_callback
        self.queue_size = queue_size
        self.error: Exception | None = None
        self.active = False
        self._folders: dict[int, Path] = {}
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._frame_number = 0
        self._telemetry = CameraTelemetryWriter()
        self._latency_adjustment_ms = 0.0
        self._calibration = False
        self.recorded_frames_per_30 = DEFAULT_RECORDED_FRAMES_PER_30
        self._sampling_accumulator = 0
        self.frames_observed = 0
        self.frames_selected = 0
        self.frames_dropped = 0
        self.unusual_pts_gap_candidates = 0
        self.frames_rejected_invalid_timing = 0
        self._recording_started_at = ""
        self._recording_started_unix_ns = 0
        self._recording_started_monotonic_ns = 0
        self._transport_stats: dict[str, int] = {}
        self._transport_stats_by_epoch: dict[int, dict[str, int]] = {}

    def start(
        self,
        folders: Mapping[int, str],
        *,
        calibration: bool = False,
        latency_adjustment_ms: float = 0.0,
        timing_session: Mapping | None = None,
    ) -> None:
        if self.active:
            raise RuntimeError("Camera snapshot recording is already active")
        selected = {int(channel): Path(folder).expanduser() for channel, folder in folders.items()}
        if not selected:
            raise ValueError("No radar recording folders were supplied")
        for folder in selected.values():
            if not folder.is_dir():
                raise ValueError(f"Recording folder does not exist: {folder}")
            (folder / IMAGE_DIRECTORY_NAME).mkdir(exist_ok=True)

        self._folders = selected
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._sampling_accumulator = CAMERA_FRAME_RATE - self.recorded_frames_per_30
        self._frame_number = 0
        self.frames_observed = 0
        self.frames_selected = 0
        self.frames_dropped = 0
        self.unusual_pts_gap_candidates = 0
        self.frames_rejected_invalid_timing = 0
        self._transport_stats = {}
        self._transport_stats_by_epoch = {}
        self.error = None
        self._latency_adjustment_ms = float(latency_adjustment_ms)
        self._calibration = calibration
        started = datetime.now().astimezone()
        self._recording_started_at = started.isoformat(
            timespec="microseconds"
        )
        self._recording_started_unix_ns = time.time_ns()
        self._recording_started_monotonic_ns = time.monotonic_ns()
        if calibration:
            self._telemetry.start(
                tuple(selected.values()),
                {
                    "recording_started_at": self._recording_started_at,
                    "recording_started_unix_ns": self._recording_started_unix_ns,
                    "recording_started_monotonic_ns": self._recording_started_monotonic_ns,
                    "recorded_frames_per_30": self.recorded_frames_per_30,
                    "image_adjustment_ns": round(
                        self._latency_adjustment_ms * 1_000_000
                    ),
                    **dict(timing_session or {}),
                },
            )
        self._write_calibration_summary()
        self.active = True
        self._thread = threading.Thread(
            target=self._write_loop,
            name="camera-snapshot-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        frame: np.ndarray,
        captured_at: datetime | None = None,
        timing: Mapping | None = None,
    ) -> bool:
        if not self.active or self.error is not None or self._queue is None:
            return False
        self.frames_observed += 1
        timing = dict(timing or {})
        self.unusual_pts_gap_candidates += int(
            bool(timing.get("large_pts_gap_candidate"))
        )
        self._sampling_accumulator += self.recorded_frames_per_30
        if self._sampling_accumulator < CAMERA_FRAME_RATE:
            return False
        self._sampling_accumulator -= CAMERA_FRAME_RATE
        self.frames_selected += 1
        timestamp = (captured_at or datetime.now().astimezone()).isoformat(
            timespec="microseconds"
        )
        try:
            self._queue.put_nowait((
                frame,
                timestamp,
                timing,
            ))
        except queue.Full:
            self.frames_dropped += 1
            if self._calibration:
                self._telemetry.append_event({
                    "event": "frame_dropped_writer_queue",
                    "observed_frame": self.frames_observed,
                    "selected_frame": self.frames_selected,
                    "reason": "image writer queue is full",
                    "timing": timing,
                })
            if self.dropped_callback is not None:
                self.dropped_callback({
                    "reason": "image writer queue is full",
                    "dropped": self.frames_dropped,
                    "selected": self.frames_selected,
                    "queue_size": self.queue_size,
                    "timing": timing,
                })
            return False
        return True

    def poll_error(self) -> Exception | None:
        return self.error

    def note_invalid_timing_frame(
        self,
        *,
        reason: str = "invalid timing",
        timing: Mapping | None = None,
    ) -> None:
        if self.active:
            self.frames_rejected_invalid_timing += 1
            if self._calibration:
                self._telemetry.append_event({
                    "event": "frame_rejected_invalid_timing",
                    "rejected_frame": self.frames_rejected_invalid_timing,
                    "reason": str(reason),
                    "timing": dict(timing or {}),
                })

    def record_timing_events(self, events) -> None:
        if not self.active or not self._calibration:
            return
        for event in events:
            self._telemetry.append_event(event)

    def update_transport_stats(
        self,
        stats: Mapping,
        *,
        stream_epoch: int = 0,
    ) -> None:
        current = {
            str(key): int(value)
            for key, value in stats.items()
        }
        previous = self._transport_stats_by_epoch.setdefault(
            int(stream_epoch),
            {},
        )
        for key, value in current.items():
            previous[key] = max(previous.get(key, 0), value)
        fields = {
            key
            for epoch_stats in self._transport_stats_by_epoch.values()
            for key in epoch_stats
        }
        self._transport_stats = {
            key: sum(
                epoch_stats.get(key, 0)
                for epoch_stats in self._transport_stats_by_epoch.values()
            )
            for key in fields
        }

    def set_latency_adjustment_ms(self, value: float) -> None:
        self._latency_adjustment_ms = float(value)

    def set_recorded_frames_per_30(self, value: int) -> None:
        numeric_value = float(value)
        if not numeric_value.is_integer():
            raise ValueError("Recorded frames must be a whole number")
        frames_per_30 = int(numeric_value)
        if not 1 <= frames_per_30 <= CAMERA_FRAME_RATE:
            raise ValueError("Recorded frames must be between 1 and 30")
        self.recorded_frames_per_30 = frames_per_30
        self._sampling_accumulator = CAMERA_FRAME_RATE - frames_per_30

    def stop(self) -> int:
        if not self.active:
            return self._frame_number
        self.active = False
        if self._queue is not None and self._thread is not None:
            while self._thread.is_alive():
                try:
                    self._queue.put(_STOP, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
        self._write_calibration_summary(final=True)
        self._telemetry.stop()
        if self.error is not None:
            raise self.error
        return self._frame_number

    def _write_loop(self) -> None:
        assert self._queue is not None
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    frame, captured_at, timing = item
                    frame_number = self._frame_number + 1
                    filename = f"camera_{frame_number:06d}.jpg"
                    reference = image_reference(filename)
                    files = {}
                    for channel, folder in self._folders.items():
                        path = image_path(folder, filename)
                        if not cv.imwrite(str(path), frame):
                            raise RuntimeError(f"Could not save camera snapshot: {path}")
                        files[channel] = reference
                    self._frame_number = frame_number
                    saved_at_ns = time.time_ns()
                    self._record_calibration_timestamp(
                        reference,
                        captured_at,
                        timing,
                        saved_at_ns,
                    )
                    if self.saved_callback:
                        self.saved_callback({
                            "files": files,
                            "captured_at": captured_at,
                            "calibration": self._calibration,
                        })
                finally:
                    self._queue.task_done()
        except Exception as error:
            self.error = error

    def _record_calibration_timestamp(
        self,
        filename: str,
        captured_at: str,
        timing: Mapping,
        saved_at_ns: int,
    ) -> None:
        if not self._telemetry.active:
            return
        timing = dict(timing)
        self._telemetry.update_epoch({
            key: timing.get(key)
            for key in (
                "stream_epoch",
                "mapping_revision",
                "segment_epoch",
                "pipeline_zero_unix_ns",
                "pipeline_zero_monotonic_ns",
                "pipeline_clock_type",
                "pipeline_base_time_ns",
            )
            if timing.get(key) is not None
        })
        media_time_ns = timing.get("media_time_ns")
        if media_time_ns is None:
            media_time_ns = round(
                datetime.fromisoformat(captured_at).timestamp() * 1_000_000_000
            )
        adjustment_ns = round(self._latency_adjustment_ms * 1_000_000)
        estimated_exposure_ns = int(media_time_ns) - adjustment_ns
        record = {
            "timestamp_schema_version": timing.get(
                "timestamp_schema_version", SCHEMA_VERSION
            ),
            "frame": filename,
            "stream_epoch": timing.get("stream_epoch"),
            "mapping_revision": timing.get("mapping_revision"),
            "segment_epoch": timing.get("segment_epoch"),
            "segment": timing.get("segment"),
            "pts_ns": timing.get("pts_ns"),
            "running_time_ns": timing.get("running_time_ns"),
            "running_time_delta_ns": timing.get("running_time_delta_ns"),
            "media_monotonic_ns": timing.get("media_monotonic_ns"),
            "application_arrival_monotonic_ns": timing.get(
                "application_arrival_monotonic_ns"
            ),
            "application_arrival_unix_ns": timing.get(
                "application_arrival_unix_ns"
            ),
            "application_arrival_delta_ns": timing.get(
                "application_arrival_delta_ns"
            ),
            "arrival_boundary": timing.get("arrival_boundary"),
            "sample_pulled_monotonic_ns": timing.get(
                "sample_pulled_monotonic_ns"
            ),
            "sample_pulled_unix_ns": timing.get("sample_pulled_unix_ns"),
            "frame_converted_monotonic_ns": timing.get(
                "frame_converted_monotonic_ns"
            ),
            "frame_converted_unix_ns": timing.get("frame_converted_unix_ns"),
            "timestamped_monotonic_ns": timing.get("timestamped_monotonic_ns"),
            "timestamped_unix_ns": timing.get("timestamped_unix_ns"),
            # Legacy fields retain their pre-v3 meaning: the time sampled by the
            # timestamp policy after image conversion, not callback arrival.
            "received_monotonic_ns": timing.get("host_monotonic_received_ns"),
            "received_unix_ns": timing.get("host_realtime_received_ns"),
            "pipeline_running_time_observed_ns": timing.get(
                "pipeline_running_time_observed_ns"
            ),
            "pipeline_clock_mapping_monotonic_ns": timing.get(
                "pipeline_clock_mapping_monotonic_ns"
            ),
            "pipeline_clock_mapping_uncertainty_ns": timing.get(
                "pipeline_clock_mapping_uncertainty_ns"
            ),
            "capture_queue_level_buffers": timing.get(
                "capture_queue_level_buffers"
            ),
            "capture_queue_level_bytes": timing.get("capture_queue_level_bytes"),
            "capture_queue_level_time_ns": timing.get(
                "capture_queue_level_time_ns"
            ),
            "reference_timestamp_raw_ns": timing.get("reference_timestamp_raw_ns"),
            "reference_clock": timing.get("reference_clock"),
            "reference_ntp_ns": timing.get("camera_ntp_ns"),
            "media_unix_ns": int(media_time_ns),
            "estimated_exposure_unix_ns": estimated_exposure_ns,
            "observable_arrival_minus_media_ns": timing.get(
                "observable_arrival_minus_media_ns"
            ),
            "estimated_capture_unix_ns": timing.get("estimated_capture_unix_ns"),
            "estimated_capture_monotonic_ns": timing.get(
                "estimated_capture_monotonic_ns"
            ),
            "estimated_arrival_delay_ns": timing.get("estimated_arrival_delay_ns"),
            "capture_estimator_model": timing.get("capture_estimator_model"),
            "capture_estimator_version": timing.get("capture_estimator_version"),
            "capture_calibration_version": timing.get("capture_calibration_version"),
            "capture_estimator_correction_ns": timing.get(
                "capture_estimator_correction_ns"
            ),
            "capture_estimator_status": timing.get("capture_estimator_status"),
            "capture_estimator_uncertainty_ns": timing.get(
                "capture_estimator_uncertainty_ns"
            ),
            "capture_estimator_uncertainty_meaning": timing.get(
                "capture_estimator_uncertainty_meaning"
            ),
            "capture_time_reference": timing.get("capture_time_reference"),
            "saved_unix_ns": int(saved_at_ns),
            "flags": list(timing.get("flags") or ()),
        }
        self._telemetry.append_frame(record)

    def _write_calibration_summary(self, *, final: bool = False) -> None:
        if not self._telemetry.active:
            return
        summary = {
            "schema_version": SCHEMA_VERSION,
            "started_at": self._recording_started_at,
            "started_unix_ns": self._recording_started_unix_ns,
            "started_monotonic_ns": self._recording_started_monotonic_ns,
            "recorded_frames_per_30": self.recorded_frames_per_30,
            "frames_observed": self.frames_observed,
            "frames_selected": self.frames_selected,
            "frames_dropped_writer_queue": self.frames_dropped,
            "unusual_pts_gap_candidates": self.unusual_pts_gap_candidates,
            "frames_rejected_invalid_timing": self.frames_rejected_invalid_timing,
            "confirmed_frames_not_saved": (
                self.frames_dropped + self.frames_rejected_invalid_timing
            ),
            "frames_saved": self._frame_number,
            **self._transport_stats,
            "transport_stats_by_epoch": [
                {"stream_epoch": epoch, **stats}
                for epoch, stats in sorted(self._transport_stats_by_epoch.items())
            ],
        }
        if final:
            summary["stopped_at"] = datetime.now().astimezone().isoformat(
                timespec="microseconds"
            )
        self._telemetry.write_summary(summary)
