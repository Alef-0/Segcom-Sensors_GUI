"""Radar point-cloud persistence and camera-to-radar frame association."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Iterable

import numpy as np

from processing.recording.paths import (
    POINT_CLOUD_DIRECTORY_NAME,
    image_reference,
    point_cloud_reference,
)
from sensors.camera.timing_defaults import DEFAULT_CAMERA_TIMESTAMP_CORRECTION_SECONDS
from sensors.radar.connection_packages import MISSING_QUALITY, RadarObject, RadarPoint

try:
    from pypcd4 import PointCloud
except ImportError:
    PointCloud = None

CLUSTER_PCD_FIELDS = (
    "ID", "dist_long", "dist_latitude", "velocity_longitude",
    "velocity_latitude", "dynamic_property", "rcs", "pdh",
    "ambiguity_state", "invalid_flag",
)
CLUSTER_PCD_TYPES = (
    np.uint32, np.float32, np.float32, np.float32, np.float32,
    np.uint32, np.float32, np.uint32, np.uint32, np.uint32,
)
LEGACY_OBJECT_PCD_FIELDS = (
    "ID", "dist_long", "dist_latitude", "velocity_longitude",
    "velocity_latitude", "dynamic_property", "rcs", "dist_long_rms",
    "velocity_longitude_rms", "dist_latitude_rms", "velocity_latitude_rms",
    "acceleration_latitude_rms", "acceleration_longitude_rms",
    "orientation_rms", "measurement_state", "probability_of_existence",
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
PCD_FIELDS = CLUSTER_PCD_FIELDS
PCD_TYPES = CLUSTER_PCD_TYPES
RADAR_LETTERS = {1: "A", 2: "B", 3: "C"}
CAMERA_DELAY_SECONDS = DEFAULT_CAMERA_TIMESTAMP_CORRECTION_SECONDS
RECORDING_METADATA_NAME = "recording.json"
TIMESTAMPS_METADATA_NAME = "timestamps.json"
METADATA_FLUSH_SECONDS = 1.0
_STOP = object()


def _float(value):
    return np.nan if value is None else float(value)


def _integer(value):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return MISSING_QUALITY
    return int(value)


def _timestamp(value: datetime | str) -> str:
    return value.isoformat(timespec="microseconds") if isinstance(value, datetime) else str(value)


def _as_datetime(value: datetime | str) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


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


def save_point_cloud(path: Path, points: tuple[RadarPoint | RadarObject, ...], frame_type: str) -> None:
    if PointCloud is None:
        raise RuntimeError("pypcd4 is required to record point clouds")
    if frame_type == "cluster":
        fields, types, rows = CLUSTER_PCD_FIELDS, CLUSTER_PCD_TYPES, [_point_row(p) for p in points]
    elif frame_type == "object":
        fields, types, rows = OBJECT_PCD_FIELDS, OBJECT_PCD_TYPES, [_object_row(p) for p in points]
    else:
        raise ValueError(f"Unsupported radar frame type: {frame_type}")
    values = np.asarray(rows, dtype=object).reshape((-1, len(fields)))
    PointCloud.from_points(values, fields, types).save(str(path))


class PointCloudRecorder:
    """Write radar frames off the receive loop and keep pairing metadata bounded."""

    def __init__(self, root: Path, channel: int, timestamp: str,
                 progress_callback: Callable[[int, int], None] | None = None,
                 queue_size: int = 64,
                 camera_delay_seconds: float = CAMERA_DELAY_SECONDS):
        if PointCloud is None:
            raise RuntimeError("pypcd4 is required to record point clouds")
        if channel not in RADAR_LETTERS:
            raise ValueError(f"Unsupported radar channel: {channel}")
        self.channel = channel
        self.folder = root / f"recording_{RADAR_LETTERS[channel]}_{timestamp}"
        self.folder.mkdir(parents=True, exist_ok=False)
        self.point_cloud_folder = self.folder / POINT_CLOUD_DIRECTORY_NAME
        self.point_cloud_folder.mkdir()
        self.timestamps_path = self.folder / TIMESTAMPS_METADATA_NAME
        self.metadata_path = self.folder / RECORDING_METADATA_NAME
        self.progress_callback = progress_callback
        self.camera_delay_seconds = float(camera_delay_seconds)
        self.frames_written = 0
        self.error: Exception | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.RLock()
        self._records: list[dict] = []
        self._timestamps: dict[str, str] = {}
        self._pending_cameras: deque[dict] = deque()
        self._dirty = False
        self._last_flush = time.monotonic()
        self._write_metadata_locked()
        self._thread = threading.Thread(target=self._write_loop, name=f"pcd-writer-{channel}", daemon=True)
        self._thread.start()

    def submit(self, points: Iterable[RadarPoint | RadarObject], recorded_at: datetime,
               frame_type: str = "cluster") -> bool:
        if self.error is not None:
            return False
        if frame_type not in ("cluster", "object"):
            self.error = ValueError(f"Unsupported radar frame type: {frame_type}")
            return False
        try:
            self._queue.put_nowait((tuple(points), _timestamp(recorded_at), frame_type))
            return True
        except queue.Full:
            self.error = RuntimeError(f"Recording queue for radar {self.channel} is full")
            return False

    def add_camera_snapshot(self, filename: str, recorded_at: datetime | str) -> None:
        camera_time = _as_datetime(recorded_at)
        target = camera_time - timedelta(seconds=self.camera_delay_seconds)
        with self._lock:
            self._pending_cameras.append({
                "camera_frame": image_reference(filename),
                "camera_recorded_at": _timestamp(camera_time),
                "target_recorded_at": _timestamp(target),
                "camera_delay_ms": self.camera_delay_seconds * 1000.0,
            })
            self._match_cameras_locked()
            self._flush_locked()

    def set_camera_delay_seconds(self, value: float) -> None:
        self.camera_delay_seconds = float(value)

    def stop(self) -> None:
        while self._thread.is_alive():
            try:
                self._queue.put(_STOP, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        with self._lock:
            self._match_cameras_locked(force=True)
            self._flush_locked(force=True)
        if self.error:
            raise self.error

    def _match_cameras_locked(self, force: bool = False) -> None:
        while self._pending_cameras:
            available = [row for row in self._records if row["camera_frame"] is None]
            if not available:
                return
            pending = self._pending_cameras[0]
            target = _as_datetime(pending["target_recorded_at"])
            if not force and _as_datetime(available[-1]["recorded_at"]) < target:
                return
            chosen = min(available, key=lambda row: abs(
                (_as_datetime(row["recorded_at"]) - target).total_seconds()
            ))
            chosen.update(
                camera_frame=pending["camera_frame"],
                camera_recorded_at=pending["camera_recorded_at"],
                camera_delay_ms=round(pending["camera_delay_ms"], 3),
                synchronization_error_ms=round((
                    _as_datetime(chosen["recorded_at"]) - target
                ).total_seconds() * 1000, 3),
            )
            self._pending_cameras.popleft()
            self._dirty = True

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    points, timestamp, frame_type = item
                    number = self.frames_written + 1
                    filename = f"frame_{number:06d}.pcd"
                    save_point_cloud(self.point_cloud_folder / filename, points, frame_type)
                    with self._lock:
                        self.frames_written = number
                        reference = point_cloud_reference(filename)
                        self._timestamps[reference] = timestamp
                        self._records.append({
                            "point_cloud": reference,
                            "recorded_at": timestamp,
                            "frame_type": frame_type,
                            "camera_frame": None,
                            "camera_recorded_at": None,
                            "camera_delay_ms": None,
                            "synchronization_error_ms": None,
                        })
                        self._dirty = True
                        self._match_cameras_locked()
                        self._flush_locked()
                    if self.progress_callback:
                        self.progress_callback(self.channel, number)
                finally:
                    self._queue.task_done()
        except Exception as error:
            self.error = error

    def _flush_locked(self, force: bool = False) -> None:
        if not self._dirty or (not force and time.monotonic() - self._last_flush < METADATA_FLUSH_SECONDS):
            return
        self._replace_json(self.timestamps_path, self._timestamps)
        self._replace_json(self.metadata_path, self._records)
        self._dirty = False
        self._last_flush = time.monotonic()

    def _write_metadata_locked(self) -> None:
        self._replace_json(self.timestamps_path, self._timestamps)
        self._replace_json(self.metadata_path, self._records)

    @staticmethod
    def _replace_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


class RadarRecordingSession:
    def __init__(self, progress_callback=None, camera_delay_seconds: float = CAMERA_DELAY_SECONDS):
        self.recorders: dict[int, PointCloudRecorder] = {}
        self.progress_callback = progress_callback
        self.camera_delay_seconds = float(camera_delay_seconds)

    @property
    def active(self) -> bool:
        return bool(self.recorders)

    @property
    def channels(self) -> frozenset[int]:
        return frozenset(self.recorders)

    def start(self, root: str, channels: Iterable[int]) -> dict[int, str]:
        if self.active:
            raise RuntimeError("A recording session is already active")
        folder = Path(root).expanduser()
        if not folder.is_dir():
            raise ValueError("The recording destination must be an existing folder")
        selected = sorted(set(channels))
        if not selected:
            raise ValueError("Select at least one radar to record")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        made = {}
        try:
            for channel in selected:
                made[channel] = PointCloudRecorder(
                    folder, channel, stamp, self.progress_callback,
                    camera_delay_seconds=self.camera_delay_seconds,
                )
        except Exception:
            for recorder in made.values():
                try:
                    recorder.stop()
                except Exception:
                    pass
            raise
        self.recorders = made
        return {channel: str(rec.folder) for channel, rec in made.items()}

    def submit(self, channel, points, recorded_at, frame_type="cluster") -> bool:
        recorder = self.recorders.get(channel)
        return recorder.submit(points, recorded_at, frame_type) if recorder else False

    def add_camera_snapshot(self, channel, filename, recorded_at) -> bool:
        recorder = self.recorders.get(channel)
        if recorder is None:
            return False
        recorder.add_camera_snapshot(filename, recorded_at)
        return True

    def set_camera_delay_seconds(self, value: float) -> None:
        self.camera_delay_seconds = float(value)
        for recorder in self.recorders.values():
            recorder.set_camera_delay_seconds(value)

    def poll_error(self):
        return next((rec.error for rec in self.recorders.values() if rec.error), None)

    def stop(self) -> dict[int, int]:
        recorders, self.recorders = self.recorders, {}
        error = None
        for recorder in recorders.values():
            try:
                recorder.stop()
            except Exception as caught:
                error = error or caught
        counts = {channel: rec.frames_written for channel, rec in recorders.items()}
        if error:
            raise error
        return counts
