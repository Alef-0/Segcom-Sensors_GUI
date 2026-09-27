"""Auxiliary utilities for PCD point cloud operations, image encoding, and paths."""

from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import cv2 as cv
import numpy as np

from sensors.radar.connection_packages import MISSING_QUALITY, RadarObject, RadarPoint

try:
    from pypcd4 import PointCloud
except ImportError:
    PointCloud = None

IMAGE_DIRECTORY_NAME = "images"
POINT_CLOUD_DIRECTORY_NAME = "point_cloud"
RECORDING_METADATA_NAME = "recording.json"
TIMESTAMPS_METADATA_NAME = "timestamps.json"
CAMERA_DELAY_SECONDS = 0.087348

CLUSTER_PCD_FIELDS = (
    "ID", "dist_long", "dist_latitude", "velocity_longitude", "velocity_latitude",
    "dynamic_property", "rcs", "pdh", "ambiguity_state", "invalid_flag",
)
CLUSTER_PCD_TYPES = (
    np.uint32, np.float32, np.float32, np.float32, np.float32,
    np.uint32, np.float32, np.uint32, np.uint32, np.uint32,
)
LEGACY_OBJECT_PCD_FIELDS = (
    "ID", "dist_long", "dist_latitude", "velocity_longitude", "velocity_latitude",
    "dynamic_property", "rcs", "dist_long_rms", "velocity_longitude_rms",
    "dist_latitude_rms", "velocity_latitude_rms", "acceleration_latitude_rms",
    "acceleration_longitude_rms", "orientation_rms", "measurement_state",
    "probability_of_existence",
)
LEGACY_OBJECT_PCD_TYPES = (
    np.uint32, np.float32, np.float32, np.float32, np.float32, np.uint32,
    np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
    np.float32, np.float32, np.uint32, np.uint32,
)
OBJECT_PCD_FIELDS = LEGACY_OBJECT_PCD_FIELDS + (
    "acceleration_longitude", "acceleration_latitude", "object_class",
    "orientation_angle", "length", "width", "collision_detection_regions",
)
OBJECT_PCD_TYPES = LEGACY_OBJECT_PCD_TYPES + (
    np.float32, np.float32, np.uint32, np.float32, np.float32, np.float32, np.uint32,
)


def _reference(directory: str, filename: str | Path) -> str:
    return (Path(directory) / Path(str(filename)).name).as_posix()


def image_reference(filename: str | Path) -> str:
    return _reference(IMAGE_DIRECTORY_NAME, filename)


def point_cloud_reference(filename: str | Path) -> str:
    return _reference(POINT_CLOUD_DIRECTORY_NAME, filename)


def image_path(folder: str | Path, filename: str | Path) -> Path:
    return Path(folder) / image_reference(filename)


def point_cloud_path(folder: str | Path, filename: str | Path) -> Path:
    return Path(folder) / point_cloud_reference(filename)


def resolve_recording_file(folder: Path, reference: str | Path | None, directory_name: str) -> Path | None:
    if not reference:
        return None
    root = Path(folder).expanduser().resolve()
    raw = Path(str(reference))
    candidates = [root / raw]
    if len(raw.parts) == 1:
        candidates.append(root / directory_name / raw.name)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file():
                return resolved
        except (OSError, RuntimeError):
            pass
    return None


def encode_jpeg(frame: np.ndarray, quality: int = 95) -> bytes:
    success, encoded = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Could not encode frame to JPEG")
    return encoded.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    frame = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG data")
    return frame


from sensors.filter.filter import (
    DYNAMIC_COLORS_BGR,
    UNKNOWN_DYNAMIC_COLOR_BGR,
    filter_point_cutoff,
    filter_radar_points,
    filter_rcs,
)


def _float(value):
    return np.nan if value is None else float(value)


def _integer(value):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return MISSING_QUALITY
    return int(value)


def _point_row(point: RadarPoint) -> tuple:
    return (
        point.cluster_id, _float(point.dist_long), _float(point.dist_latitude),
        _float(point.velocity_longitude), _float(point.velocity_latitude),
        _integer(point.dynamic_property), _float(point.rcs), _integer(point.pdh),
        _integer(point.ambiguity_state), _integer(point.invalid_flag),
    )


def _object_row(obj: RadarObject) -> tuple:
    return (
        obj.object_id, _float(obj.dist_long), _float(obj.dist_latitude),
        _float(obj.velocity_longitude), _float(obj.velocity_latitude),
        _integer(obj.dynamic_property), _float(obj.rcs), _float(obj.dist_long_rms),
        _float(obj.velocity_longitude_rms), _float(obj.dist_latitude_rms),
        _float(obj.velocity_latitude_rms), _float(obj.acceleration_latitude_rms),
        _float(obj.acceleration_longitude_rms), _float(obj.orientation_rms),
        _integer(obj.measurement_state), _integer(obj.probability_of_existence),
        _float(obj.acceleration_longitude), _float(obj.acceleration_latitude),
        _integer(obj.object_class), _float(obj.orientation_angle), _float(obj.length),
        _float(obj.width), _integer(obj.collision_detection_regions),
    )


def save_point_cloud(path: Path, points: Iterable[RadarPoint | RadarObject], frame_type: str, point_cloud_cls=None) -> None:
    if point_cloud_cls is None:
        try:
            import sensors.recording as rec
            if getattr(rec, "PointCloud", None) is not None:
                point_cloud_cls = rec.PointCloud
        except Exception:
            pass
    cloud_cls = point_cloud_cls or PointCloud
    if cloud_cls is None:
        raise RuntimeError("pypcd4 is required to record point clouds")
    if frame_type == "cluster":
        fields, types, rows = CLUSTER_PCD_FIELDS, CLUSTER_PCD_TYPES, [_point_row(p) for p in points]
    elif frame_type == "object":
        fields, types, rows = OBJECT_PCD_FIELDS, OBJECT_PCD_TYPES, [_object_row(p) for p in points]
    else:
        raise ValueError(f"Unsupported radar frame type: {frame_type}")
    values = np.asarray(rows, dtype=object).reshape((-1, len(fields)))
    cloud_cls.from_points(values, fields, types).save(str(path))


class PointCloudReader:
    """Reads saved PCD files back into RadarPoint and RadarObject structures."""

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
    def _parse_float(value):
        return None if np.isnan(value) else float(value)

    @staticmethod
    def _parse_int(value):
        if np.isnan(value):
            return None
        res = int(value)
        return None if res == MISSING_QUALITY else res

    @classmethod
    def _clusters(cls, cloud):
        return tuple(RadarPoint(
            cluster_id=int(row[0]), dist_long=float(row[1]), dist_latitude=float(row[2]),
            velocity_longitude=cls._parse_float(row[3]), velocity_latitude=cls._parse_float(row[4]),
            dynamic_property=cls._parse_int(row[5]), rcs=cls._parse_float(row[6]), pdh=int(row[7]),
            ambiguity_state=int(row[8]), invalid_flag=int(row[9]),
        ) for row in cloud.numpy(CLUSTER_PCD_FIELDS))

    @classmethod
    def _objects(cls, cloud, fields):
        return tuple(RadarObject(
            object_id=int(row[0]), dist_long=float(row[1]), dist_latitude=float(row[2]),
            velocity_longitude=cls._parse_float(row[3]), velocity_latitude=cls._parse_float(row[4]),
            dynamic_property=cls._parse_int(row[5]), rcs=cls._parse_float(row[6]),
            dist_long_rms=cls._parse_float(row[7]), velocity_longitude_rms=cls._parse_float(row[8]),
            dist_latitude_rms=cls._parse_float(row[9]), velocity_latitude_rms=cls._parse_float(row[10]),
            acceleration_latitude_rms=cls._parse_float(row[11]),
            acceleration_longitude_rms=cls._parse_float(row[12]), orientation_rms=cls._parse_float(row[13]),
            measurement_state=cls._parse_int(row[14]), probability_of_existence=cls._parse_int(row[15]),
            acceleration_longitude=cls._parse_float(row[16]) if len(fields) > 16 else None,
            acceleration_latitude=cls._parse_float(row[17]) if len(fields) > 17 else None,
            object_class=cls._parse_int(row[18]) if len(fields) > 18 else None,
            orientation_angle=cls._parse_float(row[19]) if len(fields) > 19 else None,
            length=cls._parse_float(row[20]) if len(fields) > 20 else None,
            width=cls._parse_float(row[21]) if len(fields) > 21 else None,
            collision_detection_regions=cls._parse_int(row[22]) if len(fields) > 22 else None,
        ) for row in cloud.numpy(fields))
