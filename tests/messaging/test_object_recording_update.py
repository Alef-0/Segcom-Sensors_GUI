from datetime import datetime, timezone
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sensors.radar.connection_packages import MISSING_QUALITY, RadarObject
import processing.recording.point_cloud_reader as reader_module
import processing.recording.point_cloud_recorder as recorder_module


class FakePointCloud:
    calls = []

    def __init__(self, values, fields, types):
        self.values = values
        self.fields = fields
        self.types = types

    @classmethod
    def from_points(cls, values, fields, types):
        cloud = cls(values.copy(), tuple(fields), tuple(types))
        cls.calls.append(cloud)
        return cloud

    def save(self, path):
        self.path = Path(path)
        self.path.write_bytes(b"pcd")


class FakeReaderCloud:
    current = None

    def __init__(self, fields, values):
        import numpy as np
        self.fields = tuple(fields)
        self.values = np.asarray(values)

    @classmethod
    def from_path(cls, _path):
        return cls.current

    def numpy(self, fields):
        indexes = [self.fields.index(field) for field in fields]
        return self.values[:, indexes]


class ObjectRecordingUpdateTests(unittest.TestCase):
    def setUp(self):
        self.original_point_cloud = recorder_module.PointCloud
        self.original_reader_point_cloud = reader_module.PointCloud
        recorder_module.PointCloud = FakePointCloud
        reader_module.PointCloud = FakeReaderCloud
        FakePointCloud.calls.clear()

    def tearDown(self):
        recorder_module.PointCloud = self.original_point_cloud
        reader_module.PointCloud = self.original_reader_point_cloud

    def test_object_recording_includes_60d_and_60e_fields(self):
        obj = RadarObject(
            object_id=8,
            dist_long=40.0,
            dist_latitude=-2.0,
            dynamic_property=1,
            rcs=-6.0,
            acceleration_longitude=1.25,
            acceleration_latitude=-0.25,
            object_class=4,
            orientation_angle=12.0,
            length=4.2,
            width=1.8,
            collision_detection_regions=0x81,
        )
        with TemporaryDirectory() as folder:
            session = recorder_module.RadarRecordingSession()
            session.start(folder, (1,))
            self.assertTrue(
                session.submit(
                    1,
                    (obj,),
                    datetime.now(timezone.utc),
                    frame_type="object",
                )
            )
            session.stop()

        cloud, = FakePointCloud.calls
        values = dict(zip(cloud.fields, cloud.values[0]))
        self.assertAlmostEqual(values["acceleration_longitude"], 1.25)
        self.assertAlmostEqual(values["acceleration_latitude"], -0.25)
        self.assertEqual(values["object_class"], 4)
        self.assertAlmostEqual(values["orientation_angle"], 12.0)
        self.assertAlmostEqual(values["length"], 4.2)
        self.assertAlmostEqual(values["width"], 1.8)
        self.assertEqual(values["collision_detection_regions"], 0x81)

    def test_missing_extended_fields_use_existing_missing_conventions(self):
        obj = RadarObject(
            object_id=2,
            dist_long=10.0,
            dist_latitude=1.0,
        )
        with TemporaryDirectory() as folder:
            session = recorder_module.RadarRecordingSession()
            session.start(folder, (2,))
            session.submit(2, (obj,), datetime.now(timezone.utc), frame_type="object")
            session.stop()

        cloud, = FakePointCloud.calls
        values = dict(zip(cloud.fields, cloud.values[0]))
        for field in (
            "acceleration_longitude",
            "acceleration_latitude",
            "orientation_angle",
            "length",
            "width",
        ):
            self.assertTrue(math.isnan(values[field]))
        self.assertEqual(values["object_class"], MISSING_QUALITY)
        self.assertEqual(values["collision_detection_regions"], MISSING_QUALITY)

    def test_camera_snapshot_metadata_stays_grouped_after_schema_extension(self):
        import json
        obj = RadarObject(
            object_id=3,
            dist_long=12.0,
            dist_latitude=1.0,
            dynamic_property=0,
            rcs=-5.0,
        )
        recorded_at = datetime.now(timezone.utc)
        with TemporaryDirectory() as folder:
            session = recorder_module.RadarRecordingSession()
            paths = session.start(folder, (1,))
            session.submit(1, (obj,), recorded_at, frame_type="object")
            session.add_camera_snapshot(1, "camera_000001.jpg", recorded_at)
            session.stop()
            records = json.loads(
                (Path(paths[1]) / recorder_module.RECORDING_METADATA_NAME).read_text()
            )

        self.assertEqual(records[0]["frame_type"], "object")
        self.assertEqual(records[0]["camera_frame"], "images/camera_000001.jpg")

    def test_reader_restores_extended_object_fields(self):
        import numpy as np
        values = [[
            8, 40.0, -2.0, np.nan, np.nan, MISSING_QUALITY, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            MISSING_QUALITY, MISSING_QUALITY,
            1.25, -0.25, 4, 12.0, 4.2, 1.8, 0x81,
        ]]
        FakeReaderCloud.current = FakeReaderCloud(recorder_module.OBJECT_PCD_FIELDS, values)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "frame.pcd"
            path.write_bytes(b"pcd")
            obj = reader_module.PointCloudReader(path).objects[0]

        self.assertIsNone(obj.dynamic_property)
        self.assertIsNone(obj.rcs)
        self.assertAlmostEqual(obj.acceleration_longitude, 1.25)
        self.assertEqual(obj.object_class, 4)
        self.assertEqual(obj.collision_detection_regions, 0x81)

    def test_reader_supports_legacy_object_schema(self):
        import numpy as np
        values = [[
            8, 40.0, -2.0, 3.5, 0.25, 1, -6.0,
            np.nan, 0.063, 0.105, 0.288, np.nan, 2.187, 180.0,
            MISSING_QUALITY, 7,
        ]]
        FakeReaderCloud.current = FakeReaderCloud(
            recorder_module.LEGACY_OBJECT_PCD_FIELDS, values
        )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pcd"
            path.write_bytes(b"pcd")
            obj = reader_module.PointCloudReader(path).objects[0]

        self.assertEqual(obj.object_id, 8)
        self.assertIsNone(obj.measurement_state)
        self.assertEqual(obj.probability_of_existence, 7)
        self.assertIsNone(obj.object_class)
        self.assertIsNone(obj.collision_detection_regions)


if __name__ == "__main__":
    unittest.main()
