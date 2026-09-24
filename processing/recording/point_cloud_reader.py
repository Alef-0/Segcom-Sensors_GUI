"""Read supported PCD schemas into the application's radar data classes."""

from pathlib import Path
from typing import Iterator

import numpy as np

from processing.recording.point_cloud_recorder import (
    CLUSTER_PCD_FIELDS, LEGACY_OBJECT_PCD_FIELDS, OBJECT_PCD_FIELDS, PointCloud,
)
from sensors.radar.connection_packages import MISSING_QUALITY, RadarObject, RadarPoint


def _float(value):
    return None if np.isnan(value) else float(value)


def _integer(value):
    if np.isnan(value):
        return None
    result = int(value)
    return None if result == MISSING_QUALITY else result


class PointCloudReader:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if PointCloud is None:
            raise RuntimeError("pypcd4 is required to read point clouds")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        cloud = PointCloud.from_path(str(self.path))
        fields = set(cloud.fields)
        if set(OBJECT_PCD_FIELDS) <= fields:
            self.frame_type = "object"
            self._points = self._objects(cloud, OBJECT_PCD_FIELDS)
        elif set(LEGACY_OBJECT_PCD_FIELDS) <= fields:
            self.frame_type = "object"
            self._points = self._objects(cloud, LEGACY_OBJECT_PCD_FIELDS)
        elif set(CLUSTER_PCD_FIELDS) <= fields:
            self.frame_type = "cluster"
            self._points = self._clusters(cloud)
        else:
            raise ValueError(f"Unsupported PCD schema in {self.path.name}")

    @property
    def points(self):
        return self._points

    @property
    def clusters(self):
        return self._points if self.frame_type == "cluster" else ()

    @property
    def objects(self):
        return self._points if self.frame_type == "object" else ()

    def __iter__(self) -> Iterator[RadarPoint | RadarObject]:
        return iter(self._points)

    @staticmethod
    def _clusters(cloud):
        return tuple(RadarPoint(
            cluster_id=int(row[0]), dist_long=float(row[1]), dist_latitude=float(row[2]),
            velocity_longitude=_float(row[3]), velocity_latitude=_float(row[4]),
            dynamic_property=_integer(row[5]), rcs=_float(row[6]), pdh=int(row[7]),
            ambiguity_state=int(row[8]), invalid_flag=int(row[9]),
        ) for row in cloud.numpy(CLUSTER_PCD_FIELDS))

    @staticmethod
    def _objects(cloud, fields):
        result = []
        for row in cloud.numpy(fields):
            values = dict(zip(fields, row))
            result.append(RadarObject(
                object_id=int(values["ID"]),
                dist_long=float(values["dist_long"]),
                dist_latitude=float(values["dist_latitude"]),
                velocity_longitude=_float(values["velocity_longitude"]),
                velocity_latitude=_float(values["velocity_latitude"]),
                dynamic_property=_integer(values["dynamic_property"]),
                rcs=_float(values["rcs"]),
                dist_long_rms=_float(values["dist_long_rms"]),
                velocity_longitude_rms=_float(values["velocity_longitude_rms"]),
                dist_latitude_rms=_float(values["dist_latitude_rms"]),
                velocity_latitude_rms=_float(values["velocity_latitude_rms"]),
                acceleration_latitude_rms=_float(values["acceleration_latitude_rms"]),
                acceleration_longitude_rms=_float(values["acceleration_longitude_rms"]),
                orientation_rms=_float(values["orientation_rms"]),
                measurement_state=_integer(values["measurement_state"]),
                probability_of_existence=_integer(values["probability_of_existence"]),
                acceleration_longitude=_float(values["acceleration_longitude"]) if "acceleration_longitude" in values else None,
                acceleration_latitude=_float(values["acceleration_latitude"]) if "acceleration_latitude" in values else None,
                object_class=_integer(values["object_class"]) if "object_class" in values else None,
                orientation_angle=_float(values["orientation_angle"]) if "orientation_angle" in values else None,
                length=_float(values["length"]) if "length" in values else None,
                width=_float(values["width"]) if "width" in values else None,
                collision_detection_regions=_integer(values["collision_detection_regions"]) if "collision_detection_regions" in values else None,
            ))
        return tuple(result)
