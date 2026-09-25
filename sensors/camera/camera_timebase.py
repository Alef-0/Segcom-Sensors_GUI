"""Map camera PTS to a stable host timebase without changing it from DVR NTP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sensors.camera.camera_reference_clock import reference_timestamp_for_buffer


CAMERA_FRAME_RATE = 30
FRAME_PERIOD_NS = Gst.SECOND // CAMERA_FRAME_RATE
UNUSUAL_PTS_GAP_NS = FRAME_PERIOD_NS * 7 // 4
SYSTEM_CLOCK_STEP_NS = Gst.MSECOND
LONG_APPLICATION_ARRIVAL_GAP_NS = Gst.SECOND
TIMESTAMP_SCHEMA_VERSION = 3
CAPTURE_ESTIMATOR_VERSION = 1


@dataclass(frozen=True)
class FrameTimestampResult:
    captured_at: datetime | None
    media_time_ns: int | None = None
    source: str | None = None
    reason: str | None = None
    receipt_offset_seconds: float | None = None
    reference_clock_offset_seconds: float | None = None
    reference_warning: str | None = None
    camera_ntp_ns: int | None = None
    timing: dict | None = None

    @property
    def valid(self) -> bool:
        return self.captured_at is not None


class FrameTimestampPolicy:
    """Validate PTS and map segment running time onto the host Unix clock."""

    def __init__(
        self,
        max_clock_offset_seconds: float = 5.0,
        *,
        capture_correction_ms: float = 0.0,
        capture_calibration_version: str = "unversioned",
    ) -> None:
        self.max_clock_offset_seconds = max_clock_offset_seconds
        self.capture_correction_ms = float(capture_correction_ms)
        self.capture_calibration_version = str(capture_calibration_version)
        self.pipeline = None
        self.stream_epoch = 0
        self._last_pts: int | None = None
        self._last_running_time_ns: int | None = None
        self._last_application_arrival_monotonic_ns: int | None = None
        self._pipeline_zero_unix_ns: int | None = None
        self._pipeline_zero_monotonic_ns: int | None = None
        self._clock = None
        self._pipeline_base_time_ns: int | None = None
        self._segment_signature: tuple | None = None
        self._segment_epoch = 0
        self._mapping_revision = 0

    def reset(self, pipeline=None, *, stream_epoch: int = 0) -> None:
        self.pipeline = pipeline
        self.stream_epoch = int(stream_epoch)
        self._last_pts = None
        self._last_running_time_ns = None
        self._last_application_arrival_monotonic_ns = None
        self._pipeline_zero_unix_ns = None
        self._pipeline_zero_monotonic_ns = None
        self._clock = None
        self._pipeline_base_time_ns = None
        self._segment_signature = None
        self._segment_epoch = 0
        self._mapping_revision = 0

    def set_capture_correction_ms(self, value: float) -> None:
        numeric = float(value)
        if numeric != self.capture_correction_ms:
            self.capture_calibration_version = f"runtime-configured-{numeric:g}ms"
        self.capture_correction_ms = numeric

    @staticmethod
    def _valid_clock_time(value) -> bool:
        return isinstance(value, int) and 0 <= value < Gst.CLOCK_TIME_NONE

    @staticmethod
    def _segment_metadata(segment) -> tuple[tuple, dict]:
        fields = (
            "start", "stop", "time", "base", "offset", "rate", "applied_rate",
        )
        values = tuple(getattr(segment, field, None) for field in fields)
        signature = (type(segment).__name__, *values)
        metadata = {
            field: value
            for field, value in zip(fields, values)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return signature, metadata

    def _running_time_ns(self, sample, pts: int) -> tuple[int | None, tuple | None, dict]:
        try:
            segment = sample.get_segment()
            running_time = segment.to_running_time(Gst.Format.TIME, pts)
        except (AttributeError, TypeError):
            return None, None, {}
        signature, metadata = self._segment_metadata(segment)
        value = int(running_time) if self._valid_clock_time(running_time) else None
        return value, signature, metadata

    def _clock_mapping_sample(
        self, before_ns: int
    ) -> tuple[int, int, int, object, int] | None:
        if self.pipeline is None:
            return None
        clock = self.pipeline.get_clock()
        if clock is None:
            return None
        clock_time_ns = int(clock.get_time())
        base_time_ns = int(self.pipeline.get_base_time())
        after_ns = time.monotonic_ns()
        current_running_time = clock_time_ns - base_time_ns
        if current_running_time < 0:
            return None
        return current_running_time, before_ns, after_ns, clock, base_time_ns

    def _ensure_anchor(
        self,
        application_arrival_unix_ns: int,
        application_arrival_monotonic_ns: int,
        mapping_monotonic_ns: int,
        current_running_time_ns: int,
    ) -> None:
        if self._pipeline_zero_unix_ns is not None:
            return
        self._pipeline_zero_monotonic_ns = mapping_monotonic_ns - current_running_time_ns
        realtime_minus_monotonic_ns = (
            application_arrival_unix_ns - application_arrival_monotonic_ns
        )
        self._pipeline_zero_unix_ns = (
            self._pipeline_zero_monotonic_ns + realtime_minus_monotonic_ns
        )

    @property
    def epoch_metadata(self) -> dict | None:
        if self._pipeline_zero_unix_ns is None:
            return None
        clock_name = None
        if self.pipeline is not None:
            clock = self.pipeline.get_clock()
            if clock is not None:
                clock_name = type(clock).__name__
        return {
            "stream_epoch": self.stream_epoch,
            "mapping_revision": self._mapping_revision,
            "segment_epoch": self._segment_epoch,
            "pipeline_zero_unix_ns": self._pipeline_zero_unix_ns,
            "pipeline_zero_monotonic_ns": self._pipeline_zero_monotonic_ns,
            "pipeline_clock_type": clock_name,
            "pipeline_base_time_ns": self._pipeline_base_time_ns,
        }

    def timestamp_for_sample(
        self,
        sample,
        *,
        application_arrival_monotonic_ns: int | None = None,
        application_arrival_unix_ns: int | None = None,
        sample_pulled_monotonic_ns: int | None = None,
        sample_pulled_unix_ns: int | None = None,
        frame_converted_monotonic_ns: int | None = None,
        frame_converted_unix_ns: int | None = None,
        capture_queue_levels: dict | None = None,
    ) -> FrameTimestampResult:
        timestamped_monotonic_ns = time.monotonic_ns()
        timestamped_unix_ns = time.time_ns()
        if application_arrival_monotonic_ns is None:
            application_arrival_monotonic_ns = timestamped_monotonic_ns
        if application_arrival_unix_ns is None:
            application_arrival_unix_ns = timestamped_unix_ns
        initial_timing = {
            "timestamp_schema_version": TIMESTAMP_SCHEMA_VERSION,
            "arrival_boundary": "appsink_new_sample_callback_entry",
            "application_arrival_monotonic_ns": int(application_arrival_monotonic_ns),
            "application_arrival_unix_ns": int(application_arrival_unix_ns),
            "sample_pulled_monotonic_ns": sample_pulled_monotonic_ns,
            "sample_pulled_unix_ns": sample_pulled_unix_ns,
            "frame_converted_monotonic_ns": frame_converted_monotonic_ns,
            "frame_converted_unix_ns": frame_converted_unix_ns,
            "timestamped_monotonic_ns": timestamped_monotonic_ns,
            "timestamped_unix_ns": timestamped_unix_ns,
            **dict(capture_queue_levels or {}),
        }
        buffer = sample.get_buffer()
        if buffer is None:
            return FrameTimestampResult(
                None, reason="sample has no buffer", timing=initial_timing,
            )

        pts = buffer.pts
        if not self._valid_clock_time(pts):
            return FrameTimestampResult(
                None,
                reason="frame has invalid PTS",
                timing=dict(initial_timing, pts_ns=None),
            )

        running_time_ns, segment_signature, segment_metadata = self._running_time_ns(
            sample, pts
        )
        if running_time_ns is None:
            return FrameTimestampResult(
                None,
                reason="frame PTS cannot be mapped to running time",
                timing=dict(initial_timing, pts_ns=int(pts)),
            )

        flags = []
        segment_changed = (
            self._segment_signature is not None
            and segment_signature != self._segment_signature
        )
        if self._segment_signature is None or segment_changed:
            self._segment_epoch += 1
            self._segment_signature = segment_signature
            self._last_pts = None
            self._last_running_time_ns = None
            if segment_changed:
                flags.append("segment_changed")
        if self._last_pts is not None and pts <= self._last_pts:
            return FrameTimestampResult(
                None,
                reason="frame PTS is duplicated or moved backwards",
                timing=dict(
                    initial_timing,
                    pts_ns=int(pts),
                    running_time_ns=running_time_ns,
                    segment_epoch=self._segment_epoch,
                    flags=flags,
                ),
            )

        mapping = self._clock_mapping_sample(timestamped_monotonic_ns)
        if mapping is None:
            return FrameTimestampResult(
                None,
                reason="pipeline clock is unavailable",
                timing=dict(initial_timing, pts_ns=int(pts), running_time_ns=running_time_ns),
            )
        current_running_time_ns, mapping_before_ns, mapping_after_ns, clock, base_time_ns = mapping
        # Retain the GObject and compare its native identity. Temporary Python
        # wrappers can have different ids while referring to the same clock.
        clock_changed = self._clock is not None and clock != self._clock
        base_time_changed = (
            self._pipeline_base_time_ns is not None
            and base_time_ns != self._pipeline_base_time_ns
        )
        if clock_changed or base_time_changed:
            self._mapping_revision += 1
            self._pipeline_zero_unix_ns = None
            self._pipeline_zero_monotonic_ns = None
            self._last_pts = None
            self._last_running_time_ns = None
            flags.append("pipeline_clock_changed" if clock_changed else "pipeline_base_time_changed")
        self._clock = clock
        self._pipeline_base_time_ns = base_time_ns
        mapping_monotonic_ns = (mapping_before_ns + mapping_after_ns) // 2
        mapping_uncertainty_ns = max(1, (mapping_after_ns - mapping_before_ns + 1) // 2)

        self._ensure_anchor(
            int(application_arrival_unix_ns),
            int(application_arrival_monotonic_ns),
            mapping_monotonic_ns,
            current_running_time_ns,
        )
        assert self._pipeline_zero_unix_ns is not None
        assert self._pipeline_zero_monotonic_ns is not None
        media_time_ns = self._pipeline_zero_unix_ns + running_time_ns
        media_monotonic_ns = self._pipeline_zero_monotonic_ns + running_time_ns
        stable_receipt_unix_ns = (
            self._pipeline_zero_unix_ns
            + timestamped_monotonic_ns
            - self._pipeline_zero_monotonic_ns
        )
        system_clock_error_ns = timestamped_unix_ns - stable_receipt_unix_ns

        reference = reference_timestamp_for_buffer(buffer)
        reference_offset_seconds = None
        reference_warning = None
        if reference.unix_ns is not None:
            reference_offset_seconds = (
                reference.unix_ns - media_time_ns
            ) / Gst.SECOND
            if abs(reference_offset_seconds) > self.max_clock_offset_seconds:
                reference_warning = (
                    "DVR reference clock offset is "
                    f"{reference_offset_seconds:+.3f} seconds"
                )

        previous_pts = self._last_pts
        previous_running_time_ns = self._last_running_time_ns
        previous_application_arrival_ns = (
            self._last_application_arrival_monotonic_ns
        )
        self._last_pts = int(pts)
        self._last_running_time_ns = running_time_ns
        self._last_application_arrival_monotonic_ns = int(
            application_arrival_monotonic_ns
        )
        pts_delta_ns = None if previous_pts is None else int(pts - previous_pts)
        running_time_delta_ns = (
            None
            if previous_running_time_ns is None
            else running_time_ns - previous_running_time_ns
        )
        application_arrival_delta_ns = (
            None
            if previous_application_arrival_ns is None
            else int(application_arrival_monotonic_ns)
            - previous_application_arrival_ns
        )
        unusual_pts_gap = (
            running_time_delta_ns is not None
            and running_time_delta_ns > UNUSUAL_PTS_GAP_NS
        )
        if unusual_pts_gap:
            flags.append("unusual_pts_gap")
        if (
            application_arrival_delta_ns is not None
            and application_arrival_delta_ns > LONG_APPLICATION_ARRIVAL_GAP_NS
        ):
            flags.append("long_application_arrival_gap")
        if abs(system_clock_error_ns) > SYSTEM_CLOCK_STEP_NS:
            flags.append("system_clock_step")
        if reference.raw_ns is not None and reference.unix_ns is None:
            flags.append("unknown_reference_clock")
        queue_level = (capture_queue_levels or {}).get("capture_queue_level_buffers")
        if isinstance(queue_level, int) and queue_level > 0:
            flags.append("capture_queue_backlog")

        captured_at = datetime.fromtimestamp(
            media_time_ns / Gst.SECOND,
            timezone.utc,
        ).astimezone()
        correction_ns = round(self.capture_correction_ms * Gst.MSECOND)
        estimated_capture_monotonic_ns = media_monotonic_ns - correction_ns
        estimated_capture_unix_ns = media_time_ns - correction_ns
        observable_arrival_minus_media_ns = (
            int(application_arrival_monotonic_ns) - media_monotonic_ns
        )
        timing = {
            **initial_timing,
            "stream_epoch": self.stream_epoch,
            "pts_ns": int(pts),
            "pts_delta_ns": pts_delta_ns,
            "running_time_ns": running_time_ns,
            "running_time_delta_ns": running_time_delta_ns,
            "application_arrival_delta_ns": application_arrival_delta_ns,
            "pipeline_age_ns": current_running_time_ns - running_time_ns,
            "pipeline_running_time_observed_ns": current_running_time_ns,
            "pipeline_clock_mapping_monotonic_ns": mapping_monotonic_ns,
            "pipeline_clock_mapping_uncertainty_ns": mapping_uncertainty_ns,
            "segment_epoch": self._segment_epoch,
            "segment": segment_metadata,
            "host_realtime_received_ns": timestamped_unix_ns,
            "host_monotonic_received_ns": timestamped_monotonic_ns,
            "reference_timestamp_raw_ns": reference.raw_ns,
            "reference_clock": reference.reference,
            "camera_ntp_ns": reference.unix_ns,
            "media_time_ns": media_time_ns,
            "media_monotonic_ns": media_monotonic_ns,
            "observable_arrival_minus_media_ns": observable_arrival_minus_media_ns,
            "estimated_capture_unix_ns": estimated_capture_unix_ns,
            "estimated_capture_monotonic_ns": estimated_capture_monotonic_ns,
            "estimated_arrival_delay_ns": (
                int(application_arrival_monotonic_ns)
                - estimated_capture_monotonic_ns
            ),
            "capture_estimator_model": "fixed_calibrated_correction",
            "capture_estimator_version": CAPTURE_ESTIMATOR_VERSION,
            "capture_calibration_version": self.capture_calibration_version,
            "capture_estimator_correction_ns": correction_ns,
            "capture_estimator_status": (
                "degraded_timing_observation" if flags else "provisional_observation"
            ),
            "capture_estimator_uncertainty_ns": mapping_uncertainty_ns,
            "capture_estimator_uncertainty_meaning": (
                "host-to-pipeline clock sampling only; excludes camera processing, "
                "display scanout, and exposure-reference uncertainty"
            ),
            "capture_time_reference": (
                "host-anchored segment running time minus configured correction; "
                "physical exposure reference is unverified"
            ),
            "large_pts_gap_candidate": unusual_pts_gap,
            "system_clock_error_ns": system_clock_error_ns,
            "flags": flags,
            **(self.epoch_metadata or {}),
        }
        return FrameTimestampResult(
            captured_at,
            media_time_ns=media_time_ns,
            source="host-anchored-segment-running-time",
            receipt_offset_seconds=(
                int(application_arrival_unix_ns) - media_time_ns
            ) / Gst.SECOND,
            reference_clock_offset_seconds=reference_offset_seconds,
            reference_warning=reference_warning,
            camera_ntp_ns=reference.unix_ns,
            timing=timing,
        )
