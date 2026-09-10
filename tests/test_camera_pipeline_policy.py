from datetime import datetime, timezone
import time
import unittest
from unittest.mock import patch

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sensors.camera.camera_pipeline_policy import (
    CPU_BACKEND,
    FrameTimestampResult,
    FrameTimestampPolicy,
    ORIN_BACKEND,
    RTX_BACKEND,
    available_decoder_backends,
    build_camera_pipeline,
)
from sensors.camera.camera_gstreamer import GStreamerPipeline
from sensors.camera.camera_reference_clock import (
    NTP_UNIX_EPOCH_DELTA_SECONDS,
    ReferenceClockObserver,
    reference_timestamp_for_buffer,
)


class FakeBuffer:
    def __init__(
        self,
        pts,
        reference_timestamp=None,
        reference_clock="timestamp/x-unix",
    ):
        self.pts = pts
        self._reference_timestamp = reference_timestamp
        self._reference_clock = reference_clock

    def get_reference_timestamp_meta(self, _caps):
        if self._reference_timestamp is None:
            return None
        reference = type(
            "ReferenceCaps",
            (),
            {"to_string": lambda _self: self._reference_clock},
        )()
        return type(
            "ReferenceMeta",
            (),
            {"timestamp": self._reference_timestamp, "reference": reference},
        )()


class FakeSegment:
    @staticmethod
    def to_running_time(_format, pts):
        return pts


class FakeSample:
    def __init__(self, buffer):
        self._buffer = buffer

    def get_buffer(self):
        return self._buffer

    @staticmethod
    def get_segment():
        return FakeSegment()


class FakeClock:
    def __init__(self, value):
        self.value = value

    def get_time(self):
        return self.value


class FakePipeline:
    def __init__(self, running_time):
        self.clock = FakeClock(running_time)

    def get_clock(self):
        return self.clock

    @staticmethod
    def get_base_time():
        return 0


class FakeSink:
    def __init__(self, sample):
        self.sample = sample

    def emit(self, _signal):
        return self.sample


class FakeStats:
    def __init__(self, values):
        self.values = values

    def get_value(self, name):
        return self.values[name]


class FakeJitterBuffer:
    def __init__(self, values):
        self.stats = FakeStats(values)
        self.properties = {}

    @staticmethod
    def find_property(_name):
        return object()

    def set_property(self, name, value):
        self.properties[name] = value

    def get_property(self, name):
        if name != "stats":
            raise KeyError(name)
        return self.stats


class FakeManager:
    def __init__(self):
        self.connections = []

    def connect(self, signal, callback):
        self.connections.append((signal, callback))


class CameraPipelinePolicyTests(unittest.TestCase):
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
        self.assertNotIn("name=depay", description)
        self.assertNotIn("name=parser", description)
        self.assertIn("width=1280,height=720", description)
        self.assertEqual(description.count("max-buffers=1 drop=true"), 1)
        self.assertIn("queue name=capture_queue", description)
        capture_sink = description.split("appsink name=capture_sink", 1)[1]
        self.assertNotIn("max-buffers", capture_sink)
        self.assertNotIn("drop=true", capture_sink)

    def test_invalid_pts_is_rejected(self):
        policy = FrameTimestampPolicy()
        result = policy.timestamp_for_sample(FakeSample(FakeBuffer(Gst.CLOCK_TIME_NONE)))
        self.assertFalse(result.valid)
        self.assertIn("invalid PTS", result.reason)

    def test_declared_ntp_reference_is_converted_to_unix(self):
        unix_ns = 1_800_000_000 * Gst.SECOND
        raw_ntp_ns = unix_ns + NTP_UNIX_EPOCH_DELTA_SECONDS * Gst.SECOND

        reference = reference_timestamp_for_buffer(FakeBuffer(
            Gst.SECOND,
            reference_timestamp=raw_ntp_ns,
            reference_clock="timestamp/x-ntp,host=(string)camera",
        ))

        self.assertEqual(reference.raw_ns, raw_ntp_ns)
        self.assertEqual(reference.unix_ns, unix_ns)

    def test_rtp_manager_exposes_jitterbuffer_counters(self):
        observer = ReferenceClockObserver()
        manager = FakeManager()
        observer._on_new_manager(None, manager)
        self.assertEqual(
            [signal for signal, _callback in manager.connections],
            ["on-new-ssrc", "new-jitterbuffer"],
        )

        jitterbuffer = FakeJitterBuffer({
            "num-pushed": 100,
            "num-lost": 2,
            "num-late": 1,
            "num-duplicates": 3,
            "rtx-count": 4,
            "rtx-success-count": 2,
        })
        observer._on_new_jitterbuffer(None, jitterbuffer, 0, 123)

        self.assertTrue(jitterbuffer.properties["do-lost"])
        self.assertTrue(jitterbuffer.properties["post-drop-messages"])
        self.assertEqual(observer.transport_stats()["num_lost"], 2)
        events = observer.poll()
        self.assertEqual(events[0]["event"], "transport_counters")
        self.assertEqual(events[0]["num_late"], 1)

    def test_zero_ntp_reference_falls_back_to_pts(self):
        policy = FrameTimestampPolicy()
        policy.reset(FakePipeline(Gst.SECOND))
        result = policy.timestamp_for_sample(
            FakeSample(FakeBuffer(Gst.SECOND, reference_timestamp=0))
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.source, "host-anchored-pts")
        self.assertIsNone(result.camera_ntp_ns)

    def test_valid_pts_uses_pipeline_clock_when_ntp_meta_is_absent(self):
        running_time = 10 * Gst.SECOND
        policy = FrameTimestampPolicy()
        policy.reset(FakePipeline(running_time))
        result = policy.timestamp_for_sample(
            FakeSample(FakeBuffer(running_time - 100 * Gst.MSECOND))
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.source, "host-anchored-pts")
        self.assertLess(abs(result.captured_at.timestamp() - (time.time() - 0.1)), 0.1)

    def test_reference_timestamp_is_observational_only(self):
        running_time = Gst.SECOND
        host_time_ns = 100 * Gst.SECOND
        reference_time_ns = 125 * Gst.SECOND
        policy = FrameTimestampPolicy()
        policy.reset(FakePipeline(running_time))
        with patch(
            "sensors.camera.camera_timebase.time.time_ns",
            return_value=host_time_ns,
        ):
            result = policy.timestamp_for_sample(
                FakeSample(FakeBuffer(
                    running_time,
                    reference_timestamp=reference_time_ns,
                ))
            )
        self.assertTrue(result.valid)
        self.assertEqual(result.source, "host-anchored-pts")
        self.assertEqual(result.media_time_ns, host_time_ns)
        self.assertEqual(result.reference_clock_offset_seconds, 25.0)
        self.assertEqual(result.camera_ntp_ns, reference_time_ns)
        self.assertNotIn("filtered_ntp_correction_ns", result.timing)

    def test_large_stable_camera_clock_offset_is_not_rejected(self):
        policy = FrameTimestampPolicy(max_clock_offset_seconds=1.0)
        policy.reset(FakePipeline(Gst.SECOND))
        host_time_ns = 100 * Gst.SECOND
        reference_time_ns = host_time_ns + 25 * Gst.SECOND
        with patch(
            "sensors.camera.camera_timebase.time.time_ns",
            return_value=host_time_ns,
        ):
            result = policy.timestamp_for_sample(
                FakeSample(FakeBuffer(Gst.SECOND, reference_timestamp=reference_time_ns))
            )
        self.assertTrue(result.valid)
        self.assertEqual(result.source, "host-anchored-pts")
        self.assertAlmostEqual(result.reference_clock_offset_seconds, 25.0)
        self.assertIn("DVR reference clock offset", result.reference_warning)

    def test_pipeline_clock_mapping_uses_a_fixed_anchor(self):
        pipeline = FakePipeline(10 * Gst.SECOND)
        policy = FrameTimestampPolicy()
        policy.reset(pipeline)
        with patch(
            "sensors.camera.camera_timebase.time.time_ns",
            side_effect=(100 * Gst.SECOND, 500 * Gst.SECOND),
        ):
            first = policy.timestamp_for_sample(
                FakeSample(FakeBuffer(9 * Gst.SECOND))
            )
            pipeline.clock.value = 20 * Gst.SECOND
            second = policy.timestamp_for_sample(
                FakeSample(FakeBuffer(10 * Gst.SECOND))
            )
        self.assertEqual(
            second.captured_at.timestamp() - first.captured_at.timestamp(),
            1.0,
        )

    def test_ntp_progression_does_not_move_pts_time(self):
        pipeline = FakePipeline(Gst.SECOND)
        policy = FrameTimestampPolicy()
        policy.reset(pipeline)
        with (
            patch(
                "sensors.camera.camera_timebase.time.time_ns",
                side_effect=(100 * Gst.SECOND, 101 * Gst.SECOND),
            ),
            patch(
                "sensors.camera.camera_timebase.time.monotonic_ns",
                side_effect=(10 * Gst.SECOND, 11 * Gst.SECOND),
            ),
        ):
            policy.timestamp_for_sample(FakeSample(FakeBuffer(
                Gst.SECOND,
                reference_timestamp=125 * Gst.SECOND,
            )))
            pipeline.clock.value = 2 * Gst.SECOND
            result = policy.timestamp_for_sample(FakeSample(FakeBuffer(
                2 * Gst.SECOND,
                reference_timestamp=126 * Gst.SECOND + 100 * Gst.MSECOND,
            )))

        self.assertEqual(result.media_time_ns, 101 * Gst.SECOND)
        self.assertEqual(result.camera_ntp_ns, 126 * Gst.SECOND + 100 * Gst.MSECOND)
        self.assertEqual(result.source, "host-anchored-pts")

    def test_unknown_reference_clock_is_preserved_without_conversion(self):
        policy = FrameTimestampPolicy()
        policy.reset(FakePipeline(Gst.SECOND))
        result = policy.timestamp_for_sample(FakeSample(FakeBuffer(
            Gst.SECOND,
            reference_timestamp=125 * Gst.SECOND,
            reference_clock="timestamp/x-driver-clock",
        )))
        self.assertIsNone(result.camera_ntp_ns)
        self.assertEqual(
            result.timing["reference_timestamp_raw_ns"],
            125 * Gst.SECOND,
        )
        self.assertIn("unknown_reference_clock", result.timing["flags"])

    def test_normal_fifty_millisecond_pts_gap_is_not_a_loss_candidate(self):
        pipeline = FakePipeline(Gst.SECOND)
        policy = FrameTimestampPolicy()
        policy.reset(pipeline)
        policy.timestamp_for_sample(FakeSample(FakeBuffer(Gst.SECOND)))
        pipeline.clock.value += 50 * Gst.MSECOND
        result = policy.timestamp_for_sample(FakeSample(FakeBuffer(
            Gst.SECOND + 50 * Gst.MSECOND,
        )))
        self.assertFalse(result.timing["large_pts_gap_candidate"])

    def test_large_pts_gap_is_only_a_candidate(self):
        pipeline = FakePipeline(Gst.SECOND)
        policy = FrameTimestampPolicy()
        policy.reset(pipeline)
        policy.timestamp_for_sample(FakeSample(FakeBuffer(Gst.SECOND)))
        pipeline.clock.value += 70 * Gst.MSECOND
        result = policy.timestamp_for_sample(FakeSample(FakeBuffer(
            Gst.SECOND + 70 * Gst.MSECOND,
        )))
        self.assertTrue(result.timing["large_pts_gap_candidate"])
        self.assertIn("unusual_pts_gap", result.timing["flags"])

    def test_duplicate_pts_is_rejected_after_a_valid_frame(self):
        running_time = 10 * Gst.SECOND
        policy = FrameTimestampPolicy()
        policy.reset(FakePipeline(running_time))
        sample = FakeSample(FakeBuffer(Gst.SECOND))
        self.assertTrue(policy.timestamp_for_sample(sample).valid)
        result = policy.timestamp_for_sample(sample)
        self.assertFalse(result.valid)
        self.assertIn("duplicated", result.reason)

    def test_capture_callback_skips_recording_when_timestamp_is_invalid(self):
        pipeline = object.__new__(GStreamerPipeline)
        pipeline.first_frame_received = True
        pipeline._sample_to_frame = lambda _sample: object()
        pipeline.timestamp_policy = type(
            "Policy",
            (),
            {
                "timestamp_for_sample": lambda _self, _sample: FrameTimestampResult(
                    None, reason="frame has invalid PTS"
                )
            },
        )()
        submitted = []
        pipeline.snapshot_recorder = type(
            "Recorder",
            (),
            {"submit": lambda _self, *args, **kwargs: submitted.append((args, kwargs))},
        )()
        rejected = []
        pipeline._reject_synchronized_frame = rejected.append
        pipeline._emit_manual_snapshot = lambda *_args: self.fail(
            "manual snapshot must not receive an invalid timestamp"
        )

        result = pipeline.on_new_capture_sample(FakeSink(object()))

        self.assertEqual(result, Gst.FlowReturn.OK)
        self.assertEqual(submitted, [])
        self.assertEqual(rejected, ["frame has invalid PTS"])

    def test_capture_callback_sends_only_valid_timestamp_to_both_savers(self):
        captured_at = datetime.now(timezone.utc)
        frame = object()
        pipeline = object.__new__(GStreamerPipeline)
        pipeline.first_frame_received = True
        pipeline._sample_to_frame = lambda _sample: frame
        pipeline.timestamp_policy = type(
            "Policy",
            (),
            {
                "timestamp_for_sample": lambda _self, _sample: FrameTimestampResult(
                    captured_at,
                    source="pipeline-clock",
                    receipt_offset_seconds=0.25,
                )
            },
        )()
        submitted = []
        pipeline.snapshot_recorder = type(
            "Recorder",
            (),
            {"submit": lambda _self, *args, **kwargs: submitted.append((args, kwargs))},
        )()
        observed_ntp = []
        pipeline._observe_camera_ntp = lambda *args: observed_ntp.append(args)
        pipeline._report_unusual_pts_gap = lambda _timing: None
        manual = []
        pipeline._emit_manual_snapshot = lambda *args: manual.append(args)

        sample = FakeSample(FakeBuffer(Gst.SECOND))
        result = pipeline.on_new_capture_sample(FakeSink(sample))

        self.assertEqual(result, Gst.FlowReturn.OK)
        self.assertEqual(submitted, [((frame,), {
            "captured_at": captured_at,
            "timing": {
                "timestamp_source": "pipeline-clock",
                "reference_clock_offset_seconds": None,
            },
        })])
        self.assertEqual(observed_ntp, [(None, None)])
        self.assertEqual(manual, [(frame, captured_at)])

    def test_ntp_observation_only_publishes_reference_clock(self):
        pipeline = object.__new__(GStreamerPipeline)
        pipeline._has_frame_ntp = False
        published = []
        pipeline._publish_camera_ntp = lambda *args: published.append(args)

        pipeline._observe_camera_ntp(Gst.SECOND, 0.25)

        self.assertTrue(pipeline._has_frame_ntp)
        self.assertEqual(published, [(Gst.SECOND, 250.0)])


if __name__ == "__main__":
    unittest.main()
