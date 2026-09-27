"""Tests for camera decoder backend selection, pipeline construction, and capture callback."""

from datetime import datetime, timezone
import unittest
from unittest.mock import Mock

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sensors.camera.camera_pipeline import (
    CPU_BACKEND,
    ORIN_BACKEND,
    RTX_BACKEND,
    available_decoder_backends,
    build_camera_pipeline,
)
from sensors.camera.camera_gstreamer import GStreamerPipeline


class FakeBuffer:
    def __init__(self, pts, reference_timestamp=None, reference_clock="timestamp/x-unix"):
        self.pts = pts
        self._reference_timestamp = reference_timestamp
        self._reference_clock = reference_clock

    def get_reference_timestamp_meta(self, _caps):
        if self._reference_timestamp is None:
            return None
        reference = type("ReferenceCaps", (), {"to_string": lambda _self: self._reference_clock})()
        return type("ReferenceMeta", (), {"timestamp": self._reference_timestamp, "reference": reference})()


class FakeSample:
    def __init__(self, buffer):
        self._buffer = buffer

    def get_buffer(self):
        return self._buffer

    def get_caps(self):
        return type(
            "Caps",
            (),
            {"get_structure": lambda _self, _idx: type(
                "Structure",
                (),
                {"get_value": lambda _s, key: 1280 if key == "width" else 720},
            )()},
        )()


class FakeSink:
    def __init__(self, sample):
        self._sample = sample

    def emit(self, action):
        if action == "pull-sample":
            return self._sample
        raise AssertionError(f"unexpected action: {action}")


class CameraPipelineTests(unittest.TestCase):
    def test_desktop_prefers_rtx_and_keeps_cpu_fallback(self):
        elements = {*RTX_BACKEND.required_elements, *CPU_BACKEND.required_elements}
        backends = available_decoder_backends(
            factory_find=lambda name: object() if name in elements else None,
            factory_make=lambda name: object() if name in elements else None,
            jetson=False,
        )
        self.assertEqual([backend.name for backend in backends], ["rtx", "cpu"])

    def test_jetson_prefers_orin_and_keeps_cpu_fallback(self):
        elements = {*ORIN_BACKEND.required_elements, *CPU_BACKEND.required_elements}
        backends = available_decoder_backends(
            factory_find=lambda name: object() if name in elements else None,
            factory_make=lambda name: object() if name in elements else None,
            jetson=True,
        )
        self.assertEqual([backend.name for backend in backends], ["orin", "cpu"])

    def test_missing_requested_hardware_decoder_falls_back_to_cpu(self):
        elements = set(CPU_BACKEND.required_elements)
        backends = available_decoder_backends(
            "rtx",
            factory_find=lambda name: object() if name in elements else None,
            factory_make=lambda name: object() if name in elements else None,
        )
        self.assertEqual(backends, (CPU_BACKEND,))

    def test_strict_requested_hardware_decoder_reports_missing_elements(self):
        elements = set(CPU_BACKEND.required_elements)
        with self.assertRaisesRegex(RuntimeError, "Missing GStreamer element"):
            available_decoder_backends(
                "rtx",
                factory_find=lambda name: object() if name in elements else None,
                factory_make=lambda name: object() if name in elements else None,
                jetson=False,
                strict=True,
            )

    def test_strict_arm_decoder_is_rejected_off_jetson(self):
        elements = set(ORIN_BACKEND.required_elements)
        with self.assertRaisesRegex(RuntimeError, "requires an NVIDIA Jetson"):
            available_decoder_backends(
                "orin",
                factory_find=lambda name: object() if name in elements else None,
                factory_make=lambda name: object() if name in elements else None,
                jetson=False,
                strict=True,
            )

    def test_available_hardware_decoder_does_not_require_cpu_plugin(self):
        elements = set(RTX_BACKEND.required_elements)
        backends = available_decoder_backends(
            factory_find=lambda name: object() if name in elements else None,
            factory_make=lambda name: object() if name in elements else None,
            jetson=False,
        )
        self.assertEqual(backends, (RTX_BACKEND,))

    def test_pipeline_contains_selected_decoder_and_low_latency_sinks(self):
        description = build_camera_pipeline(
            ORIN_BACKEND,
            display_width=1280,
            display_height=720,
            latency_ms=250,
        )
        self.assertIn("nvv4l2decoder ! nvvidconv", description)
        self.assertIn("rtph264depay ! h264parse !", description)
        self.assertIn("protocols=tcp+udp", description)
        self.assertIn("buffer-mode=1", description)
        self.assertIn("do-retransmission=true", description)
        self.assertIn("latency=250", description)
        self.assertIn("width=1280,height=720", description)
        self.assertEqual(description.count("max-buffers=1 drop=true"), 1)
        self.assertIn("queue name=capture_queue", description)
        capture_sink = description.split("appsink name=capture_sink", 1)[1]
        self.assertNotIn("max-buffers", capture_sink)
        self.assertNotIn("drop=true", capture_sink)

    def test_pipeline_without_display_sink(self):
        description = build_camera_pipeline(
            CPU_BACKEND,
            display_width=1280,
            display_height=720,
            latency_ms=145,
            include_display_sink=False,
        )
        self.assertNotIn("appsink name=display_sink", description)
        self.assertNotIn("tee name=video", description)
        self.assertIn("appsink name=capture_sink", description)

    def test_capture_callback_skips_recording_when_pts_is_invalid(self):
        pipeline = object.__new__(GStreamerPipeline)
        pipeline.first_frame_received = True
        pipeline._last_pts = None
        pipeline._last_timestamp_warning = 0.0
        pipeline._sample_to_frame = lambda _sample: object()
        submitted = []
        pipeline.snapshot_recorder = Mock()
        pipeline.snapshot_recorder.submit = lambda *args, **kwargs: submitted.append((args, kwargs))
        pipeline._put_status = Mock()
        pipeline._emit_manual_snapshot = Mock()

        # Buffer with invalid clock time
        sample = FakeSample(FakeBuffer(Gst.CLOCK_TIME_NONE))
        result = pipeline.on_new_capture_sample(FakeSink(sample))

        self.assertEqual(result, Gst.FlowReturn.OK)
        self.assertEqual(submitted, [])
        pipeline.snapshot_recorder.note_invalid_timing_frame.assert_called_once()
        call_kwargs = pipeline.snapshot_recorder.note_invalid_timing_frame.call_args.kwargs
        self.assertEqual(call_kwargs["reason"], "invalid PTS")
        self.assertIn("application_arrival_monotonic_ns", call_kwargs["timing"])

    def test_capture_callback_submits_valid_frame(self):
        frame_obj = object()
        pipeline = object.__new__(GStreamerPipeline)
        pipeline.first_frame_received = True
        pipeline._last_pts = None
        pipeline._last_ntp_ui_update = 0.0
        pipeline._last_timestamp_warning = 0.0
        pipeline.decoder_backends = (CPU_BACKEND,)
        pipeline.decoder_backend_index = 0
        pipeline.channel = 2
        pipeline._sample_to_frame = lambda _sample: frame_obj
        submitted = []
        pipeline.snapshot_recorder = Mock()
        pipeline.snapshot_recorder.submit = lambda *args, **kwargs: submitted.append((args, kwargs))
        pipeline._put_status = Mock()
        pipeline._emit_manual_snapshot = Mock()

        pipeline.snapshot_recorder.active = True
        pipeline.snapshot_request = None

        sample = FakeSample(FakeBuffer(pts=1_000_000_000))
        result = pipeline.on_new_capture_sample(FakeSink(sample))

        self.assertEqual(result, Gst.FlowReturn.OK)
        self.assertEqual(len(submitted), 1)
        args, kwargs = submitted[0]
        self.assertEqual(args[0], frame_obj)
        self.assertEqual(kwargs["timing"]["pts_ns"], 1_000_000_000)
        self.assertIsInstance(kwargs["timing"]["application_arrival_monotonic_ns"], int)

    def test_capture_callback_toggles_snapshot_frame(self):
        import numpy as np
        frame_obj = np.zeros((720, 1280, 3), dtype=np.uint8)
        pipeline = object.__new__(GStreamerPipeline)
        pipeline.first_frame_received = True
        pipeline._last_pts = None
        pipeline._last_ntp_ui_update = 0.0
        pipeline._last_timestamp_warning = 0.0
        pipeline.decoder_backends = (CPU_BACKEND,)
        pipeline.decoder_backend_index = 0
        pipeline.channel = 2
        pipeline._sample_to_frame = lambda _sample: frame_obj
        pipeline.snapshot_recorder = Mock(active=False)
        pipeline._put_status = Mock()
        pipeline.snapshot_request = {"request_id": "test_req", "folder": "/tmp"}

        sample = FakeSample(FakeBuffer(pts=1_000_000_000))
        result = pipeline.on_new_capture_sample(FakeSink(sample))

        self.assertEqual(result, Gst.FlowReturn.OK)
        # Snapshot request should now be consumed (toggled off)
        self.assertIsNone(pipeline.snapshot_request)
        pipeline._put_status.assert_called_once()
        status_msg, payload = pipeline._put_status.call_args[0]
        self.assertEqual(status_msg, "manual_snapshot_frame")
        self.assertEqual(payload["request_id"], "test_req")
        self.assertIn("image_bytes", payload)

    def test_capture_callback_records_dropped_frame_on_backward_pts(self):
        frame_obj = object()
        pipeline = object.__new__(GStreamerPipeline)
        pipeline.first_frame_received = True
        pipeline._last_pts = 2_000_000_000
        pipeline._last_timestamp_warning = 0.0
        pipeline._sample_to_frame = lambda _sample: frame_obj
        pipeline.snapshot_recorder = Mock()
        pipeline._put_status = Mock()
        pipeline._emit_manual_snapshot = Mock()

        # Older or equal PTS
        sample = FakeSample(FakeBuffer(pts=1_500_000_000))
        result = pipeline.on_new_capture_sample(FakeSink(sample))

        self.assertEqual(result, Gst.FlowReturn.OK)
        pipeline.snapshot_recorder.submit.assert_not_called()
        pipeline.snapshot_recorder.note_invalid_timing_frame.assert_called_once()


if __name__ == "__main__":
    unittest.main()
