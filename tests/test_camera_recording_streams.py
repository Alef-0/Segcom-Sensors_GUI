"""Each recorded camera group must be captured from its own DVR channel."""

from datetime import datetime, timezone
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import sensors.camera.camera_capture_stream as stream_module
import sensors.camera.camera_gstreamer as gstreamer_module
from processing import CameraSnapshotRecorder
from sensors.camera.camera_capture_stream import ChannelCaptureStream
from sensors.camera.camera_gstreamer import GStreamerPipeline
from sensors.camera.camera_pipeline import CPU_BACKEND, build_capture_pipeline
from sensors.camera.camera_timebase import FrameTimestampResult


class FakeSink:
    def __init__(self, sample):
        self.sample = sample

    def emit(self, _signal):
        return self.sample


class FakeRecorder:
    def __init__(self):
        self.submitted = []
        self.invalid = 0

    def submit(self, frame, captured_at=None, timing=None):
        self.submitted.append((frame, captured_at, timing))
        return True

    def note_invalid_timing_frame(self):
        self.invalid += 1


class FakeStream:
    created = []

    def __init__(self, channel, recorder, *, url, decoder_backend, latency_ms, report_error):
        self.channel = channel
        self.recorder = recorder
        self.url = url
        self.latency_ms = latency_ms
        self.running = False
        self.next_retry_at = None
        self.polls = 0
        FakeStream.created.append(self)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def poll(self):
        self.polls += 1


class FakeConnection:
    @staticmethod
    def poll():
        return False


def make_camera_worker(channel=2, connected=True):
    worker = GStreamerPipeline.__new__(GStreamerPipeline)
    worker.pool = queue.Queue()
    worker.communicate = FakeConnection()
    worker.channel = channel
    worker.connected = connected
    worker.latency_adjustment_ms = 65.0
    worker.pipeline_latency_ms = 145
    worker.recording_frames_per_30 = 30
    worker.decoder_backends = (CPU_BACKEND,)
    worker.decoder_backend_index = 0
    worker._last_writer_drop_warning = 0.0
    worker.recording_active = False
    worker.recording_recorders = {}
    worker.recording_streams = {}
    return worker


def drain(pool):
    messages = []
    while True:
        try:
            messages.append(pool.get_nowait())
        except queue.Empty:
            return messages


class CaptureStreamTests(unittest.TestCase):
    def test_capture_only_pipeline_has_no_display_branch(self):
        description = build_capture_pipeline(CPU_BACKEND, latency_ms=145)
        self.assertIn("latency=145", description)
        self.assertIn("appsink name=capture_sink", description)
        self.assertNotIn("display_sink", description)
        self.assertNotIn("tee", description)

    def test_valid_frames_are_submitted_with_timing(self):
        recorder = FakeRecorder()
        stream = ChannelCaptureStream(
            1, recorder, url="rtsp://x", decoder_backend=CPU_BACKEND, latency_ms=145,
        )
        captured_at = datetime.now(timezone.utc)
        frame = object()
        stream.timestamp_policy = type(
            "Policy",
            (),
            {
                "timestamp_for_sample": lambda _self, _sample: FrameTimestampResult(
                    captured_at, source="host-anchored-pts", timing={"pts_ns": 1},
                )
            },
        )()
        with patch.object(stream_module, "sample_to_frame", lambda _sample: frame):
            result = stream.on_new_sample(FakeSink(object()))

        self.assertEqual(result, Gst.FlowReturn.OK)
        self.assertTrue(stream.first_frame_received)
        self.assertEqual(recorder.submitted, [(frame, captured_at, {
            "pts_ns": 1,
            "timestamp_source": "host-anchored-pts",
            "reference_clock_offset_seconds": None,
        })])

    def test_invalid_timestamps_are_counted_not_recorded(self):
        recorder = FakeRecorder()
        stream = ChannelCaptureStream(
            3, recorder, url="rtsp://x", decoder_backend=CPU_BACKEND, latency_ms=145,
        )
        stream.timestamp_policy = type(
            "Policy",
            (),
            {
                "timestamp_for_sample": lambda _self, _sample: FrameTimestampResult(
                    None, reason="frame has invalid PTS",
                )
            },
        )()
        with patch.object(stream_module, "sample_to_frame", lambda _sample: object()):
            stream.on_new_sample(FakeSink(object()))

        self.assertEqual(recorder.submitted, [])
        self.assertEqual(recorder.invalid, 1)

    def test_failed_stream_retries_and_reports_once(self):
        now = [100.0]
        errors = []
        opens = []
        stream = ChannelCaptureStream(
            1,
            FakeRecorder(),
            url="rtsp://x",
            decoder_backend=CPU_BACKEND,
            latency_ms=145,
            report_error=errors.append,
            clock=lambda: now[0],
        )
        stream._open = lambda: opens.append(now[0])

        for _ in range(stream_module.FAILED_ATTEMPTS_BEFORE_WARNING + 1):
            stream._fail("boom")
            stream.poll()  # still inside the retry delay
            now[0] += stream_module.RETRY_DELAY_SECONDS
            stream.poll()

        self.assertEqual(len(opens), stream_module.FAILED_ATTEMPTS_BEFORE_WARNING + 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("channel 1", errors[0])


class CameraWorkerRecordingTests(unittest.TestCase):
    def setUp(self):
        FakeStream.created = []
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.folders = {1: root / "recording_A", 3: root / "recording_C"}
        for folder in self.folders.values():
            folder.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_each_selected_group_records_its_own_channel(self):
        worker = make_camera_worker(channel=2)
        with patch.object(gstreamer_module, "ChannelCaptureStream", FakeStream):
            worker._start_snapshot_recording({
                "folders": {channel: str(folder) for channel, folder in self.folders.items()}
            })
        try:
            self.assertTrue(worker.recording_active)
            self.assertEqual(sorted(worker.recording_streams), [1, 3])
            urls = {stream.channel: stream.url for stream in FakeStream.created}
            self.assertIn("channel=1&", urls[1])
            self.assertIn("channel=3&", urls[3])
            self.assertTrue(all(stream.running for stream in FakeStream.created))
            for channel, recorder in worker.recording_recorders.items():
                self.assertEqual(list(recorder._folders), [channel])
                self.assertEqual(recorder._folders[channel], self.folders[channel])
        finally:
            worker._stop_snapshot_recording()

        self.assertFalse(worker.recording_active)
        self.assertFalse(any(stream.running for stream in FakeStream.created))
        states = [payload for message, payload in drain(worker.pool)
                  if message == "camera_recording_state"]
        self.assertTrue(states[0]["active"])
        self.assertFalse(states[-1]["active"])
        self.assertEqual(states[-1]["counts"], {1: 0, 3: 0})

    def test_frames_from_one_group_are_not_written_to_another(self):
        saved = []
        recorder_a = CameraSnapshotRecorder(saved.append)
        recorder_c = CameraSnapshotRecorder(saved.append)
        recorder_a.start({1: self.folders[1]})
        recorder_c.start({3: self.folders[3]})
        frame_a = np.full((4, 4, 3), 10, dtype=np.uint8)
        recorder_a.submit(frame_a, captured_at=datetime.now(timezone.utc))
        self.assertEqual(recorder_a.stop(), 1)
        self.assertEqual(recorder_c.stop(), 0)

        self.assertTrue((self.folders[1] / "images" / "camera_000001.jpg").is_file())
        self.assertFalse((self.folders[3] / "images" / "camera_000001.jpg").exists())
        self.assertEqual([entry["files"] for entry in saved],
                         [{1: "images/camera_000001.jpg"}])

    def test_streams_stop_while_camera_is_disconnected_and_resume(self):
        worker = make_camera_worker()
        with patch.object(gstreamer_module, "ChannelCaptureStream", FakeStream):
            worker._start_snapshot_recording({"folders": {1: str(self.folders[1])}})
        stream = worker.recording_streams[1]
        try:
            worker.connected = False
            worker._poll_recording()
            self.assertFalse(stream.running)
            self.assertTrue(worker.recording_active)

            worker.connected = True
            worker._poll_recording()
            self.assertTrue(stream.running)
        finally:
            worker._stop_snapshot_recording()

    def test_live_channel_change_does_not_touch_recording_streams(self):
        worker = make_camera_worker(channel=2)
        with patch.object(gstreamer_module, "ChannelCaptureStream", FakeStream):
            worker._start_snapshot_recording({"folders": {1: str(self.folders[1])}})
        try:
            worker.channel = 3
            worker._poll_recording()
            self.assertEqual(list(worker.recording_streams), [1])
            self.assertIn("channel=1&", worker.recording_streams[1].url)
        finally:
            worker._stop_snapshot_recording()

    def test_groups_b_and_c_record_independent_camera_streams_and_radar_pairs(self):
        """Groups B and C must save independent camera images into their respective folders."""
        folders = {
            2: Path(self.temp.name) / "recording_B_test",
            3: Path(self.temp.name) / "recording_C_test",
        }
        for folder in folders.values():
            folder.mkdir()

        saved_snapshots = []
        worker = make_camera_worker(channel=2)
        worker._report_snapshot = saved_snapshots.append

        with patch.object(gstreamer_module, "ChannelCaptureStream", FakeStream):
            worker._start_snapshot_recording({
                "folders": {ch: str(f) for ch, f in folders.items()}
            })

        try:
            stream_b = worker.recording_streams[2]
            stream_c = worker.recording_streams[3]
            self.assertEqual(stream_b.channel, 2)
            self.assertEqual(stream_c.channel, 3)
            self.assertIn("channel=2&", stream_b.url)
            self.assertIn("channel=3&", stream_c.url)
            self.assertIs(stream_b.recorder, worker.recording_recorders[2])
            self.assertIs(stream_c.recorder, worker.recording_recorders[3])

            frame_b = np.full((10, 10, 3), 22, dtype=np.uint8)
            frame_c = np.full((10, 10, 3), 33, dtype=np.uint8)

            worker.recording_recorders[2].submit(
                frame_b, captured_at=datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)
            )
            worker.recording_recorders[3].submit(
                frame_c, captured_at=datetime(2026, 11, 4, 10, 0, 1, tzinfo=timezone.utc)
            )

            # Wait for writers to finish
            worker.recording_recorders[2]._queue.join()
            worker.recording_recorders[3]._queue.join()

            img_b_path = folders[2] / "images" / "camera_000001.jpg"
            img_c_path = folders[3] / "images" / "camera_000001.jpg"
            self.assertTrue(img_b_path.is_file())
            self.assertTrue(img_c_path.is_file())

            # Verify files are different and contain respective channel data
            read_b = gstreamer_module.cv.imread(str(img_b_path))
            read_c = gstreamer_module.cv.imread(str(img_c_path))
            self.assertTrue(np.all(read_b == 22))
            self.assertTrue(np.all(read_c == 33))
            self.assertFalse(np.array_equal(read_b, read_c))

            # Verify saved callback payloads only route to their own channel
            channels_reported = [list(item["files"].keys())[0] for item in saved_snapshots]
            self.assertIn(2, channels_reported)
            self.assertIn(3, channels_reported)
            for item in saved_snapshots:
                self.assertEqual(len(item["files"]), 1)
                self.assertIn("pts", item)
                self.assertIn("ntp", item)
                self.assertIn("arrival_time", item)
        finally:
            worker._stop_snapshot_recording()


if __name__ == "__main__":
    unittest.main()
