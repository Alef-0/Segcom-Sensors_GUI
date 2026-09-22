"""Exercise recording continuity without cameras, sockets, or display windows."""

import queue
import threading
import unittest
from unittest.mock import Mock

from sensors.camera.camera_gstreamer import GStreamerPipeline
from sensors.camera.camera_pipeline_policy import CPU_BACKEND


class CameraRecordingRestartTests(unittest.TestCase):
    def fixture(self):
        camera = GStreamerPipeline.__new__(GStreamerPipeline)
        camera.pipeline = None
        camera.main_loop = None
        camera.connected = True
        camera.channel = camera.normal_channel = 4
        camera.calibration_mode = True
        camera.calibration_recording = False
        camera.stream_epoch = 5
        camera.shutdown_event = threading.Event()
        camera.communicate = Mock()
        camera.communicate.poll.return_value = False
        camera.snapshot_recorder = Mock(active=False, frames_dropped=0,
                                        frames_rejected_invalid_timing=0,
                                        unusual_pts_gap_candidates=0)
        camera.snapshot_recorder.poll_error.return_value = None
        camera.reference_clock = Mock()
        camera.timestamp_policy = Mock()
        camera.calibration_scheduler_priority = Mock()
        camera._put_status = Mock()
        camera.frames = queue.Queue()
        camera.decoder_backends = (CPU_BACKEND,)
        camera.decoder_backend_index = 0
        camera.decoder_preference = "cpu"
        camera.display_width, camera.display_height = 1280, 720
        camera.pipeline_latency_ms = 145
        camera.latency_adjustment_ms = 87.348
        camera.source_ids = []
        camera._reset_camera_ntp_observation = Mock()
        camera._destroy_window = Mock()
        camera._remove_sources = Mock()
        return camera

    def command(self, camera, event, value=None):
        camera.communicate.poll.side_effect = (True, False)
        camera.communicate.recv.return_value = (event, value)
        camera.process_commands()
        camera.communicate.poll.side_effect = None

    def test_recordings_reuse_pipeline_epoch_and_decoder_state(self):
        for calibration in (False, True):
            with self.subTest(calibration=calibration):
                camera = self.fixture()
                camera.pipeline = Mock()
                camera.main_loop = Mock()
                payload = {"folders": {4: "/unused"}, "calibration": calibration}
                for _ in range(2):
                    self.command(camera, "record_start", payload)
                    session = camera.snapshot_recorder.start.call_args.kwargs["timing_session"]
                    self.assertEqual(session["stream_epoch_at_start"], 5)
                    self.assertFalse(session["pipeline_restarted_for_recording"])
                    self.command(camera, "record_stop")
                self.assertEqual(camera.snapshot_recorder.start.call_count, 2)
                camera.pipeline.set_state.assert_not_called()
                camera.main_loop.quit.assert_not_called()
                camera.timestamp_policy.reset.assert_not_called()

    def test_duplicate_request_does_not_replace_active_recording(self):
        camera = self.fixture()
        camera.snapshot_recorder.active = True
        self.command(camera, "record_start", {"folders": {4: "/second"}})
        camera.snapshot_recorder.start.assert_not_called()
        camera._put_status.assert_called_with(
            "camera_recording_error", "Camera recording is already active")

    def test_disconnected_camera_rejects_recording(self):
        camera = self.fixture()
        camera.connected = False
        self.command(camera, "record_start", {"folders": {4: "/unused"}, "calibration": True})
        camera.snapshot_recorder.start.assert_not_called()
        camera._put_status.assert_any_call("calibration_recording_state", {"active": False})

    def test_writer_failure_does_not_restart_camera(self):
        camera = self.fixture()
        camera.pipeline = Mock()
        camera.main_loop = Mock()
        camera.snapshot_recorder.start.side_effect = OSError("writer unavailable")
        self.command(camera, "record_start", {"folders": {4: "/unused"}, "calibration": True})
        camera.pipeline.set_state.assert_not_called()
        camera.main_loop.quit.assert_not_called()
        camera._put_status.assert_any_call("calibration_recording_state", {"active": False})


if __name__ == "__main__":
    unittest.main()
