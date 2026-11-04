from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sensors.radar.connection_packages import RadarPoint
from processing.recording.manual_snapshot import ManualSnapshotWriter
import processing.recording.point_cloud_recorder as recorder_module


class FakePointCloud:
    @classmethod
    def from_points(cls, values, fields, types):
        cloud = cls()
        cloud.values = values
        cloud.fields = tuple(fields)
        cloud.types = tuple(types)
        return cloud

    def save(self, path):
        Path(path).write_bytes(b"pcd")


class ManualSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.original_point_cloud = recorder_module.PointCloud
        recorder_module.PointCloud = FakePointCloud

    def tearDown(self):
        recorder_module.PointCloud = self.original_point_cloud

    @staticmethod
    def point():
        return RadarPoint(
            cluster_id=1,
            dist_long=10.0,
            dist_latitude=1.0,
            velocity_longitude=0.0,
            velocity_latitude=0.0,
            dynamic_property=1,
            rcs=-5.0,
            pdh=3,
            ambiguity_state=0,
            invalid_flag=0,
        )

    def test_snapshot_uses_recording_structure_without_group_metadata(self):
        radar_time = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
        camera_time = radar_time + timedelta(
            seconds=recorder_module.CAMERA_DELAY_SECONDS
        )
        with TemporaryDirectory() as folder:
            writer = ManualSnapshotWriter(folder)
            first = writer.save(
                (self.point(),), radar_time, "cluster", b"jpg-1", camera_time
            )
            second = writer.save(
                (self.point(),),
                radar_time + timedelta(seconds=1),
                "cluster",
                b"jpg-2",
                camera_time + timedelta(seconds=1),
            )
            root = Path(folder)
            records = recorder_module.load_recording_metadata(
                root / recorder_module.RECORDING_METADATA_NAME
            )
            timestamps = recorder_module.load_timestamps_metadata(
                root / recorder_module.TIMESTAMPS_METADATA_NAME
            )
            group_exists = (root / "group.json").exists()

        self.assertEqual(first["point_cloud"], "point_cloud/frame_000001.pcd")
        self.assertEqual(second["point_cloud"], "point_cloud/frame_000002.pcd")
        self.assertFalse(group_exists)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(timestamps), 2)
        self.assertEqual(records[0]["camera_frame"], "images/camera_000001.jpg")
        self.assertIn("arrival_time", records[0])
        self.assertIn("pts", records[0])
        self.assertIn("ntp", records[0])

    def test_snapshot_appends_to_legacy_json_metadata(self):
        recorded_at = datetime.now(timezone.utc)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "recording.json").write_text("[]\n")
            (root / "timestamps.json").write_text("{}\n")
            ManualSnapshotWriter(folder).save(
                (self.point(),),
                recorded_at,
                "cluster",
                b"jpg",
                recorded_at + timedelta(
                    seconds=recorder_module.CAMERA_DELAY_SECONDS
                ),
            )
            records_json = recorder_module.load_recording_metadata(root / "recording.json")
            records_csv = recorder_module.load_recording_metadata(root / "recording.csv")

        self.assertEqual(len(records_json), 1)
        self.assertEqual(len(records_csv), 1)
        self.assertEqual(records_json[0]["point_cloud"], "point_cloud/frame_000001.pcd")
        self.assertEqual(records_json[0]["camera_frame"], "images/camera_000001.jpg")
        self.assertEqual(records_csv[0]["point_cloud"], "point_cloud/frame_000001.pcd")
        self.assertEqual(records_csv[0]["camera_frame"], "images/camera_000001.jpg")
        self.assertNotIn("group", first)
        self.assertNotIn("channel", first)

    def test_snapshot_appends_to_existing_recording_metadata(self):
        recorded_at = datetime.now(timezone.utc)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / recorder_module.RECORDING_METADATA_NAME).write_text(
                "point_cloud,recorded_at,frame_type,camera_frame,pts,ntp,arrival_time\n"
            )
            (root / recorder_module.TIMESTAMPS_METADATA_NAME).write_text(
                "point_cloud,recorded_at\n"
            )
            ManualSnapshotWriter(folder).save(
                (self.point(),),
                recorded_at,
                "cluster",
                b"jpg",
                recorded_at + timedelta(
                    seconds=recorder_module.CAMERA_DELAY_SECONDS
                ),
            )
            records = recorder_module.load_recording_metadata(
                root / recorder_module.RECORDING_METADATA_NAME
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["point_cloud"], "point_cloud/frame_000001.pcd")
        self.assertEqual(records[0]["camera_frame"], "images/camera_000001.jpg")

    def test_legacy_group_metadata_is_ignored(self):
        recorded_at = datetime.now(timezone.utc)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "group.json").write_text('{"channel": 2, "group": "B"}\n')
            (root / recorder_module.RECORDING_METADATA_NAME).write_text(
                "point_cloud,recorded_at,frame_type,camera_frame,pts,ntp,arrival_time\n"
            )
            (root / recorder_module.TIMESTAMPS_METADATA_NAME).write_text(
                "point_cloud,recorded_at\n"
            )
            result = ManualSnapshotWriter(folder).save(
                (self.point(),),
                recorded_at,
                "cluster",
                b"jpg",
                recorded_at + timedelta(
                    seconds=recorder_module.CAMERA_DELAY_SECONDS
                ),
            )

        self.assertEqual(result["point_cloud"], "point_cloud/frame_000001.pcd")

    def test_recording_pairs_camera_to_the_configured_earlier_radar_frame(self):
        first_time = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
        second_time = first_time + timedelta(
            seconds=recorder_module.CAMERA_DELAY_SECONDS
        )
        with TemporaryDirectory() as folder:
            session = recorder_module.RadarRecordingSession()
            folders = session.start(folder, (1,))
            session.submit(1, (self.point(),), first_time)
            session.submit(1, (self.point(),), second_time)
            session.add_camera_snapshot(1, "camera_000001.jpg", second_time)
            session.stop()
            records = recorder_module.load_recording_metadata(
                Path(folders[1]) / recorder_module.RECORDING_METADATA_NAME
            )

        self.assertEqual(records[0]["camera_frame"], "images/camera_000001.jpg")
        self.assertIsNone(records[1]["camera_frame"])
        self.assertIn("arrival_time", records[0])
        self.assertIn("pts", records[0])
        self.assertIn("ntp", records[0])

    def test_snapshot_appends_to_legacy_json_metadata(self):
        recorded_at = datetime.now(timezone.utc)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "recording.json").write_text("[]\n")
            (root / "timestamps.json").write_text("{}\n")
            ManualSnapshotWriter(folder).save(
                (self.point(),),
                recorded_at,
                "cluster",
                b"jpg",
                recorded_at + timedelta(
                    seconds=recorder_module.CAMERA_DELAY_SECONDS
                ),
            )
            records_json = recorder_module.load_recording_metadata(root / "recording.json")
            records_csv = recorder_module.load_recording_metadata(root / "recording.csv")

        self.assertEqual(len(records_json), 1)
        self.assertEqual(len(records_csv), 1)
        self.assertEqual(records_json[0]["point_cloud"], "point_cloud/frame_000001.pcd")
        self.assertEqual(records_json[0]["camera_frame"], "images/camera_000001.jpg")
        self.assertEqual(records_csv[0]["point_cloud"], "point_cloud/frame_000001.pcd")
        self.assertEqual(records_csv[0]["camera_frame"], "images/camera_000001.jpg")


if __name__ == "__main__":
    unittest.main()
