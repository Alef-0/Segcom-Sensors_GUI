from datetime import datetime, timezone
import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from sensors.radar.connection_packages import MISSING_QUALITY, RadarPoint
import sensors.recording as camera_module
import sensors.auxiliary as reader_module
import sensors.recording as recorder_module
from sensors.playback import load_recording_entries, PlaybackController
from sensors.filter import Graph_radar
import menu_functions
from menu_layout import build_record_tab


class FakeWriterCloud:
    calls = []

    def __init__(self, values, fields, types):
        self.values = values
        self.fields = tuple(fields)
        self.types = tuple(types)

    @classmethod
    def from_points(cls, values, fields, types):
        cloud = cls(values.copy(), fields, types)
        cls.calls.append(cloud)
        return cloud

    def save(self, path):
        Path(path).write_bytes(b"pcd")


class FakeReaderCloud:
    current = None

    def __init__(self, fields, values):
        self.fields = tuple(fields)
        self.values = np.asarray(values)

    @classmethod
    def from_path(cls, _path):
        return cls.current

    def numpy(self, fields):
        indexes = [self.fields.index(field) for field in fields]
        return self.values[:, indexes]


class RecordingChangesTests(unittest.TestCase):
    def setUp(self):
        self.original_writer = recorder_module.PointCloud
        self.original_reader = reader_module.PointCloud
        recorder_module.PointCloud = FakeWriterCloud
        reader_module.PointCloud = FakeReaderCloud
        FakeWriterCloud.calls.clear()

    def tearDown(self):
        recorder_module.PointCloud = self.original_writer
        reader_module.PointCloud = self.original_reader

    @staticmethod
    def point():
        return RadarPoint(
            cluster_id=3,
            dist_long=25.5,
            dist_latitude=-1.25,
            velocity_longitude=4.0,
            velocity_latitude=0.0,
            dynamic_property=2,
            rcs=-8.5,
            pdh=3,
            ambiguity_state=4,
            invalid_flag=7,
        )

    def test_camera_snapshot_is_grouped_with_point_cloud(self):
        recorded_at = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
        with TemporaryDirectory() as folder:
            session = recorder_module.RadarRecordingSession()
            paths = session.start(folder, (1,))
            self.assertTrue(session.submit(1, (self.point(),), recorded_at))
            self.assertTrue(
                session.add_camera_snapshot(
                    1,
                    "camera_000001.jpg",
                    recorded_at.isoformat(),
                )
            )
            session.stop()
            metadata = json.loads(
                (Path(paths[1]) / recorder_module.RECORDING_METADATA_NAME).read_text()
            )
            timestamps = json.loads(
                (Path(paths[1]) / recorder_module.TIMESTAMPS_METADATA_NAME).read_text()
            )

        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0]["point_cloud"], "point_cloud/frame_000001.pcd")
        self.assertEqual(metadata[0]["frame_type"], "cluster")
        self.assertEqual(metadata[0]["camera_frame"], "images/camera_000001.jpg")
        self.assertEqual(timestamps["point_cloud/frame_000001.pcd"], recorded_at.isoformat(timespec="microseconds"))

    def test_point_cloud_reader_distinguishes_cluster_schema(self):
        FakeReaderCloud.current = FakeReaderCloud(
            recorder_module.CLUSTER_PCD_FIELDS,
            [[3, 25.5, -1.25, 4.0, 0.0, 2, -8.5, 3, 4, 7]],
        )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "frame.pcd"
            path.write_bytes(b"pcd")
            reader = reader_module.PointCloudReader(path)

        self.assertEqual(reader.frame_type, "cluster")
        self.assertEqual(reader.clusters[0].cluster_id, 3)
        self.assertEqual(reader.objects, ())

    def test_point_cloud_reader_restores_extended_object_fields(self):
        values = [[
            8, 40.0, -2.0, 3.5, 0.25, 1, -6.0,
            np.nan, 0.063, 0.105, 0.288, np.nan, 2.187, 180.0,
            MISSING_QUALITY, 7,
            1.25, -0.25, 4, 12.0, 4.2, 1.8, 0x81,
        ]]
        FakeReaderCloud.current = FakeReaderCloud(recorder_module.OBJECT_PCD_FIELDS, values)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "frame.pcd"
            path.write_bytes(b"pcd")
            reader = reader_module.PointCloudReader(path)

        obj = reader.objects[0]
        self.assertEqual(reader.frame_type, "object")
        self.assertIsNone(obj.dist_long_rms)
        self.assertIsNone(obj.acceleration_latitude_rms)
        self.assertIsNone(obj.measurement_state)
        self.assertEqual(obj.probability_of_existence, 7)
        self.assertAlmostEqual(obj.acceleration_longitude, 1.25)
        self.assertEqual(obj.object_class, 4)
        self.assertEqual(obj.collision_detection_regions, 0x81)

    def test_point_cloud_reader_supports_legacy_object_schema(self):
        values = [[
            8, 40.0, -2.0, 3.5, 0.25, 1, -6.0,
            np.nan, 0.063, 0.105, 0.288, np.nan, 2.187, 180.0,
            MISSING_QUALITY, 7,
        ]]
        FakeReaderCloud.current = FakeReaderCloud(
            recorder_module.LEGACY_OBJECT_PCD_FIELDS,
            values,
        )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pcd"
            path.write_bytes(b"pcd")
            reader = reader_module.PointCloudReader(path)

        obj = reader.objects[0]
        self.assertEqual(reader.frame_type, "object")
        self.assertIsNone(obj.measurement_state)
        self.assertEqual(obj.probability_of_existence, 7)
        self.assertIsNone(obj.object_class)
        self.assertIsNone(obj.collision_detection_regions)

    def test_camera_recorder_selects_configured_frames_out_of_thirty(self):
        original_imwrite = camera_module.cv.imwrite
        saved = []

        def fake_imwrite(path, _frame):
            Path(path).write_bytes(b"jpg")
            return True

        camera_module.cv.imwrite = fake_imwrite
        try:
            with TemporaryDirectory() as folder:
                recorder = camera_module.CameraSnapshotRecorder(saved.append)
                recorder.set_recorded_frames_per_30(4)
                recorder.start({1: folder})
                frame = np.zeros((2, 2, 3), dtype=np.uint8)
                selected = [recorder.submit(frame) for _ in range(30)]
                count = recorder.stop()
                files = sorted((Path(folder) / "images").glob("camera_*.jpg"))
        finally:
            camera_module.cv.imwrite = original_imwrite

        self.assertEqual(sum(selected), 4)
        self.assertTrue(selected[0])
        self.assertEqual(count, 4)
        self.assertEqual(len(files), 4)
        self.assertEqual(len(saved), 4)

    def test_calibration_camera_recording_writes_compact_timing_telemetry(self):
        original_imwrite = camera_module.cv.imwrite

        def fake_imwrite(path, _frame):
            Path(path).write_bytes(b"jpg")
            return True

        camera_module.cv.imwrite = fake_imwrite
        try:
            with TemporaryDirectory() as folder:
                recorder = camera_module.CameraRecorder()
                recorder.start({4: folder}, latency_adjustment_ms=250)
                captured_at = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
                recorder.submit(
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    captured_at=captured_at,
                    timing={
                        "pts_ns": 1_000_000_000,
                        "camera_ntp_ns": 1_787_745_628_000_000_000,
                        "application_arrival_monotonic_ns": 100_145_000_000,
                    },
                )
                recorder.stop()
                csv_records = (
                    Path(folder) / "camera_timestamps.csv"
                ).read_text().splitlines()
        finally:
            camera_module.cv.imwrite = original_imwrite

        self.assertEqual(csv_records[0], "index,pts,ntp,monotonic")
        self.assertEqual(csv_records[1], "1,1000000000,1787745628000000000,100145000000")

    def test_camera_recording_invalid_pts_writes_negative_one_in_csv(self):
        with TemporaryDirectory() as folder:
            recorder = camera_module.CameraRecorder()
            recorder.start({2: folder})
            recorder.note_invalid_timing_frame(
                reason="frame has invalid PTS",
                timing={"application_arrival_monotonic_ns": 123456789},
            )
            recorder.stop()
            csv_lines = (
                Path(folder) / "camera_timestamps.csv"
            ).read_text().splitlines()
            self.assertEqual(csv_lines[0], "index,pts,ntp,monotonic")
            self.assertEqual(csv_lines[1], "-1,-1,-1,123456789")

    def test_camera_recording_rate_is_limited_to_one_through_thirty(self):
        recorder = camera_module.CameraSnapshotRecorder()
        recorder.set_recorded_frames_per_30(1)
        self.assertEqual(recorder.recorded_frames_per_30, 1)
        recorder.set_recorded_frames_per_30(30)
        self.assertEqual(recorder.recorded_frames_per_30, 30)
        for invalid_value in (0, 31, 1.5):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    recorder.set_recorded_frames_per_30(invalid_value)

    def test_camera_recorder_drops_and_reports_when_writer_queue_is_full(self):
        drops = []
        recorder = camera_module.CameraSnapshotRecorder(
            dropped_callback=drops.append,
            queue_size=1,
        )
        recorder.active = True
        recorder._queue = queue.Queue(maxsize=1)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        self.assertTrue(recorder.submit(frame, timing={"pts_ns": 1}))
        self.assertFalse(recorder.submit(frame, timing={"pts_ns": 2}))

        self.assertEqual(recorder.frames_dropped, 1)
        self.assertEqual(drops[0]["reason"], "image writer queue is full")
        self.assertEqual(drops[0]["timing"]["pts_ns"], 2)

    def test_camera_writer_waits_and_survives_multiple_recordings(self):
        recorder = camera_module.CameraSnapshotRecorder()
        recorder.prepare()
        writer = recorder._thread
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertTrue(writer.is_alive())
        self.assertFalse(recorder.submit(frame))
        original_imwrite = camera_module.cv.imwrite

        def fake_imwrite(path, _frame):
            Path(path).write_bytes(b"jpg")
            return True

        camera_module.cv.imwrite = fake_imwrite
        try:
            with TemporaryDirectory() as folder:
                for _ in range(2):
                    recorder.start({4: folder})
                    self.assertIs(recorder._thread, writer)
                    self.assertTrue(recorder.submit(frame))
                    self.assertEqual(recorder.stop(), 1)
                    self.assertFalse(recorder.submit(frame))
                    self.assertTrue(writer.is_alive())
        finally:
            recorder.close()
            camera_module.cv.imwrite = original_imwrite

        self.assertFalse(writer.is_alive())


    def test_playback_loader_supports_new_and_legacy_metadata(self):
        timestamp = "2026-07-29T12:00:00+00:00"
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "point_cloud").mkdir()
            (root / "images").mkdir()
            (root / "point_cloud" / "frame_000001.pcd").write_bytes(b"pcd")
            (root / "images" / "camera_000001.jpg").write_bytes(b"jpg")
            (root / recorder_module.RECORDING_METADATA_NAME).write_text(json.dumps([{
                "point_cloud": "point_cloud/frame_000001.pcd",
                "recorded_at": timestamp,
                "frame_type": "cluster",
                "camera_frame": "images/camera_000001.jpg",
                "camera_recorded_at": timestamp,
            }]))
            entries = load_recording_entries(root)
            self.assertEqual(entries[0].camera_frame.name, "camera_000001.jpg")

            (root / recorder_module.RECORDING_METADATA_NAME).unlink()
            (root / recorder_module.TIMESTAMPS_METADATA_NAME).write_text(
                json.dumps({"frame_000001.pcd": timestamp})
            )
            legacy_entries = load_recording_entries(root)
            self.assertIsNone(legacy_entries[0].camera_frame)

    def test_playback_loader_resolves_legacy_bare_names_in_new_subfolders(self):
        timestamp = "2026-07-29T12:00:00+00:00"
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "point_cloud").mkdir()
            (root / "images").mkdir()
            (root / "point_cloud" / "frame_000001.pcd").write_bytes(b"pcd")
            (root / "images" / "camera_000001.jpg").write_bytes(b"jpg")
            (root / recorder_module.RECORDING_METADATA_NAME).write_text(json.dumps([{
                "point_cloud": "frame_000001.pcd",
                "recorded_at": timestamp,
                "camera_frame": "camera_000001.jpg",
                "camera_recorded_at": timestamp,
            }]))

            entries = load_recording_entries(root)

        self.assertEqual(entries[0].point_cloud.name, "frame_000001.pcd")
        self.assertEqual(entries[0].camera_frame.name, "camera_000001.jpg")

    def test_playback_loader_discovers_legacy_loose_files(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "frame_000001.pcd").write_bytes(b"pcd")
            (root / "camera_000001.jpg").write_bytes(b"jpg")
            (root / recorder_module.TIMESTAMPS_METADATA_NAME).write_text(
                json.dumps({"frame_000001.pcd": "2026-07-29T12:00:00+00:00"})
            )

            entries = load_recording_entries(root)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].point_cloud.name, "frame_000001.pcd")
        self.assertEqual(entries[1].camera_frame.name, "camera_000001.jpg")

    def test_gui_accepts_recording_with_nested_point_cloud_files(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "point_cloud").mkdir()
            (root / "point_cloud" / "frame_000001.pcd").write_bytes(b"pcd")
            (root / recorder_module.TIMESTAMPS_METADATA_NAME).write_text("{}")

            self.assertTrue(menu_functions.is_recording_folder(root))


class RequestedControlChangesTests(unittest.TestCase):
    class FakePipe:
        def __init__(self):
            self.sent = []

        def send(self, message):
            self.sent.append(message)

    class FakeConfig:
        connected_radar = False
        connected_cam = True

        def __init__(self):
            self.pending = False

        def set_recording_pending(self, starting):
            self.pending = starting

    class StuckProcess:
        def __init__(self):
            self.alive = True
            self.join_timeouts = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1
            self.alive = False

    def test_record_groups_default_to_unchecked_and_transport_controls_exist(self):
        layout = build_record_tab()
        elements = {
            element.Key: element
            for row in layout
            for element in row
            if getattr(element, "Key", None)
        }

        for channel in range(1, 4):
            self.assertFalse(elements[f"record_radar_{channel}"].InitialState)
        self.assertEqual(elements["snapshot_capture"].ButtonText, "SNAPSHOT")
        self.assertEqual(elements["playback_toggle"].ButtonText, "START PLAYBACK")
        self.assertEqual(elements["playback_previous"].ButtonText, "PREVIOUS")
        self.assertEqual(elements["playback_pause"].ButtonText, "PAUSE")
        self.assertEqual(elements["playback_next"].ButtonText, "NEXT")
        self.assertEqual(elements["playback_snapshot"].ButtonText, "SNAPSHOT CURRENT")

    def test_partial_recording_requires_confirmation(self):
        config = self.FakeConfig()
        pipe = self.FakePipe()
        with TemporaryDirectory() as folder:
            values = {"record_folder": folder, "record_radar_1": True}
            with patch.object(menu_functions.sg, "popup_ok_cancel", return_value="Cancel"):
                menu_functions.start_recording(values, config, pipe)
            self.assertFalse(config.pending)
            self.assertEqual(pipe.sent, [])

            with patch.object(menu_functions.sg, "popup_ok_cancel", return_value="OK"):
                menu_functions.start_recording(values, config, pipe)

        self.assertTrue(config.pending)
        self.assertEqual(pipe.sent[0][0], "record_start")

    def test_graph_resolution_and_ranges_can_be_changed(self):
        graph = Graph_radar(20, width=640, height=480, x_range=30, y_range=50)
        self.assertEqual(graph.base_image.shape, (480, 640, 3))
        self.assertEqual(graph.graph_to_pixel(-30, 0)[0], graph.margin)
        self.assertEqual(graph.graph_to_pixel(30, 0)[0], 640 - graph.margin)

        graph.set_resolution(900, 700)
        graph.set_range(40, 80)
        self.assertEqual(graph.base_image.shape, (700, 900, 3))
        self.assertEqual((graph.x_range, graph.y_range), (40.0, 80.0))

    def test_shutdown_kills_a_child_that_ignores_terminate(self):
        process = self.StuckProcess()

        menu_functions.join_processes([process], timeout=0.01, terminate_timeout=0.02)

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertEqual(process.join_timeouts, [0.01, 0.02, 0.02])

if __name__ == "__main__":
    unittest.main()
