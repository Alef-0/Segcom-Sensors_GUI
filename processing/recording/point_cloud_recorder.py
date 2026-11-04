from collections import deque
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Iterable, Mapping

import numpy as np

from sensors.radar.connection_packages import MISSING_QUALITY, RadarObject, RadarPoint
from processing.recording.paths import (
    POINT_CLOUD_DIRECTORY_NAME,
    image_reference,
    point_cloud_reference,
)

try:
    from pypcd4 import PointCloud
except ImportError:  # Reported through the GUI when recording starts.
    PointCloud = None


CLUSTER_PCD_FIELDS = (
    "ID",
    "dist_long",
    "dist_latitude",
    "velocity_longitude",
    "velocity_latitude",
    "dynamic_property",
    "rcs",
    "pdh",
    "ambiguity_state",
    "invalid_flag",
)
CLUSTER_PCD_TYPES = (
    np.uint32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.uint32,
    np.float32,
    np.uint32,
    np.uint32,
    np.uint32,
)
LEGACY_OBJECT_PCD_FIELDS = (
    "ID",
    "dist_long",
    "dist_latitude",
    "velocity_longitude",
    "velocity_latitude",
    "dynamic_property",
    "rcs",
    "dist_long_rms",
    "velocity_longitude_rms",
    "dist_latitude_rms",
    "velocity_latitude_rms",
    "acceleration_latitude_rms",
    "acceleration_longitude_rms",
    "orientation_rms",
    "measurement_state",
    "probability_of_existence",
)
LEGACY_OBJECT_PCD_TYPES = (
    np.uint32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.uint32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.uint32,
    np.uint32,
)
OBJECT_PCD_FIELDS = LEGACY_OBJECT_PCD_FIELDS + (
    "acceleration_longitude",
    "acceleration_latitude",
    "object_class",
    "orientation_angle",
    "length",
    "width",
    "collision_detection_regions",
)
OBJECT_PCD_TYPES = LEGACY_OBJECT_PCD_TYPES + (
    np.float32,
    np.float32,
    np.uint32,
    np.float32,
    np.float32,
    np.float32,
    np.uint32,
)

# Backwards-compatible names for code that expects the cluster schema.
PCD_FIELDS = CLUSTER_PCD_FIELDS
PCD_TYPES = CLUSTER_PCD_TYPES
RADAR_LETTERS = {1: "A", 2: "B", 3: "C"}
CAMERA_DELAY_SECONDS = 0.065
METADATA_FLUSH_SECONDS = 1.0
RECORDING_METADATA_NAME = "recording.csv"
TIMESTAMPS_METADATA_NAME = "timestamps.csv"
LEGACY_RECORDING_METADATA_NAME = "recording.json"
LEGACY_TIMESTAMPS_METADATA_NAME = "timestamps.json"
RECORDING_CSV_COLUMNS = (
    "point_cloud",
    "recorded_at",
    "frame_type",
    "camera_frame",
    "pts",
    "ntp",
    "arrival_time",
)
_STOP = object()


def save_recording_csv(path: Path, records: Iterable[Mapping]) -> None:
    """Save records to CSV using a temporary file for atomic replacement."""
    records_list = list(records)
    standard_keys = list(RECORDING_CSV_COLUMNS)
    extra_keys = []
    for record in records_list:
        for key in record:
            if key not in standard_keys and key not in extra_keys:
                extra_keys.append(key)
    fieldnames = standard_keys + extra_keys
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records_list:
            row = {}
            for key in fieldnames:
                value = record.get(key)
                row[key] = "" if value is None else value
            writer.writerow(row)
    temporary_path.replace(path)


def load_recording_csv(path: Path) -> list[dict]:
    """Load recording CSV records into structured dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        records = []
        for row in reader:
            record = {}
            for key, value in row.items():
                if value == "" or value is None:
                    record[key] = None
                elif key in ("pts", "ntp"):
                    try:
                        record[key] = int(value)
                    except (ValueError, TypeError):
                        record[key] = value
                elif key in ("camera_delay_ms", "synchronization_error_ms"):
                    try:
                        record[key] = float(value)
                    except (ValueError, TypeError):
                        record[key] = value
                else:
                    record[key] = value
            records.append(record)
        return records


def save_timestamps_csv(path: Path, timestamps: Mapping[str, str]) -> None:
    """Save point cloud to timestamp mappings as CSV."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["point_cloud", "recorded_at"])
        for point_cloud, recorded_at in timestamps.items():
            writer.writerow([point_cloud, recorded_at])
    temporary_path.replace(path)


def load_timestamps_csv(path: Path) -> dict[str, str]:
    """Load point cloud to timestamp mappings from CSV."""
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        timestamps = {}
        header_skipped = False
        for row in reader:
            if not row or not any(row):
                continue
            if not header_skipped:
                header_skipped = True
                if row[0].strip().lower() in ("point_cloud", "file", "filename"):
                    continue
            if len(row) >= 2:
                timestamps[row[0].strip()] = row[1].strip()
        return timestamps


def save_recording_metadata(path: Path, records: Iterable[Mapping]) -> None:
    """Save recording metadata as CSV or JSON based on file extension."""
    if path.suffix.lower() == ".json":
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(list(records), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    else:
        save_recording_csv(path, records)


def load_recording_metadata(source: str | Path) -> list[dict]:
    """Load recording metadata transparently from CSV or JSON."""
    path = Path(source).expanduser()
    if path.is_dir():
        csv_path = path / RECORDING_METADATA_NAME
        json_path = path / LEGACY_RECORDING_METADATA_NAME
        if csv_path.is_file():
            return load_recording_metadata(csv_path)
        if json_path.is_file():
            return load_recording_metadata(json_path)
        raise FileNotFoundError(f"No recording metadata found in {path}")
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.lstrip()
    if stripped.startswith(("[", "{")):
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError(f"Invalid {path.name} format")
        return data
    return load_recording_csv(path)


def save_timestamps_metadata(path: Path, timestamps: Mapping[str, str]) -> None:
    """Save timestamps metadata as CSV or JSON based on file extension."""
    if path.suffix.lower() == ".json":
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(dict(timestamps), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    else:
        save_timestamps_csv(path, timestamps)


def load_timestamps_metadata(source: str | Path) -> dict[str, str]:
    """Load timestamps metadata transparently from CSV or JSON."""
    path = Path(source).expanduser()
    if path.is_dir():
        csv_path = path / TIMESTAMPS_METADATA_NAME
        json_path = path / LEGACY_TIMESTAMPS_METADATA_NAME
        if csv_path.is_file():
            return load_timestamps_metadata(csv_path)
        if json_path.is_file():
            return load_timestamps_metadata(json_path)
        raise FileNotFoundError(f"No timestamps metadata found in {path}")
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.lstrip()
    if stripped.startswith(("{", "[")):
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid {path.name} format")
        return data
    return load_timestamps_csv(path)


def _float_or_nan(value: float | None) -> float:
    return np.nan if value is None else value


def _int_or_missing(value: int | None) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return MISSING_QUALITY
    return int(value)


def _timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    return str(value)


def _datetime(value: datetime | str) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def save_point_cloud(
    path: Path,
    points: tuple[RadarPoint | RadarObject, ...],
    frame_type: str,
) -> None:
    if PointCloud is None:
        raise RuntimeError(
            "pypcd4 is required to record point clouds; install it with "
            "python -m pip install pypcd4"
        )

    if frame_type == "cluster":
        fields, types = CLUSTER_PCD_FIELDS, CLUSTER_PCD_TYPES
        values = np.empty((len(points), len(fields)), dtype=object)
        for row, point in enumerate(points):
            values[row] = (
                point.cluster_id,
                point.dist_long,
                point.dist_latitude,
                _float_or_nan(point.velocity_longitude),
                _float_or_nan(point.velocity_latitude),
                _int_or_missing(point.dynamic_property),
                _float_or_nan(point.rcs),
                _int_or_missing(point.pdh),
                _int_or_missing(point.ambiguity_state),
                _int_or_missing(point.invalid_flag),
            )
    elif frame_type == "object":
        fields, types = OBJECT_PCD_FIELDS, OBJECT_PCD_TYPES
        values = np.empty((len(points), len(fields)), dtype=object)
        for row, obj in enumerate(points):
            values[row] = (
                obj.object_id,
                obj.dist_long,
                obj.dist_latitude,
                _float_or_nan(obj.velocity_longitude),
                _float_or_nan(obj.velocity_latitude),
                _int_or_missing(obj.dynamic_property),
                _float_or_nan(obj.rcs),
                _float_or_nan(obj.dist_long_rms),
                _float_or_nan(obj.velocity_longitude_rms),
                _float_or_nan(obj.dist_latitude_rms),
                _float_or_nan(obj.velocity_latitude_rms),
                _float_or_nan(obj.acceleration_latitude_rms),
                _float_or_nan(obj.acceleration_longitude_rms),
                _float_or_nan(obj.orientation_rms),
                _int_or_missing(obj.measurement_state),
                _int_or_missing(obj.probability_of_existence),
                _float_or_nan(obj.acceleration_longitude),
                _float_or_nan(obj.acceleration_latitude),
                _int_or_missing(obj.object_class),
                _float_or_nan(obj.orientation_angle),
                _float_or_nan(obj.length),
                _float_or_nan(obj.width),
                _int_or_missing(obj.collision_detection_regions),
            )
    else:
        raise ValueError(f"Unsupported radar frame type: {frame_type}")
    PointCloud.from_points(values, fields, types).save(str(path))


class PointCloudRecorder:
    def __init__(
        self,
        root: Path,
        channel: int,
        timestamp: str,
        progress_callback: Callable[[int, int], None] | None = None,
        queue_size: int = 64,
        camera_delay_seconds: float = CAMERA_DELAY_SECONDS,
    ):
        if PointCloud is None:
            raise RuntimeError(
                "pypcd4 is required to record point clouds; install it with "
                "python -m pip install pypcd4"
            )
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
        self._timestamps: dict[str, str] = {}
        self._records: list[dict[str, str | None]] = []
        self._pending_camera_frames: deque[dict[str, str]] = deque()
        self._metadata_dirty = False
        self._last_metadata_flush = time.monotonic()
        self._metadata_lock = threading.Lock()
        self._write_metadata()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._write_loop,
            name=f"pcd-writer-{RADAR_LETTERS[channel]}",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        points: Iterable[RadarPoint | RadarObject],
        recorded_at: datetime,
        frame_type: str = "cluster",
    ) -> bool:
        if self.error is not None:
            return False
        if frame_type not in ("cluster", "object"):
            self.error = ValueError(f"Unsupported radar frame type: {frame_type}")
            return False
        try:
            timestamp = recorded_at.isoformat(timespec="microseconds")
            self._queue.put_nowait((tuple(points), timestamp, frame_type))
            return True
        except queue.Full:
            self.error = RuntimeError(
                f"Recording queue for radar {RADAR_LETTERS[self.channel]} is full"
            )
            return False

    def add_camera_snapshot(
        self,
        filename: str,
        recorded_at: datetime | str,
        pts: int | None = None,
        ntp: int | None = None,
        arrival_time: str | None = None,
    ) -> None:
        camera_time = _datetime(recorded_at)
        delay_seconds = self.camera_delay_seconds
        target_time = camera_time - timedelta(seconds=delay_seconds)
        snapshot = {
            "camera_frame": image_reference(filename),
            "target_recorded_at": _timestamp(target_time),
            "pts": pts,
            "ntp": ntp,
            "arrival_time": arrival_time or _timestamp(camera_time),
        }
        with self._metadata_lock:
            self._pending_camera_frames.append(snapshot)
            self._match_camera_frames_locked()
            self._flush_metadata_locked()

    def stop(self) -> None:
        while self._thread.is_alive():
            try:
                self._queue.put(_STOP, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        with self._metadata_lock:
            self._match_camera_frames_locked(force=True)
        self._flush_metadata(force=True)
        if self.error is not None:
            raise self.error

    def _match_camera_frames_locked(self, *, force: bool = False) -> None:
        while self._pending_camera_frames:
            available = [record for record in self._records if record["camera_frame"] is None]
            if not available:
                return

            snapshot = self._pending_camera_frames[0]
            target = _datetime(snapshot["target_recorded_at"])
            newest = _datetime(available[-1]["recorded_at"])
            if not force and newest < target:
                return

            record = min(
                available,
                key=lambda item: abs((_datetime(item["recorded_at"]) - target).total_seconds()),
            )
            record_time = _datetime(record["recorded_at"])
            record.update({
                "camera_frame": snapshot["camera_frame"],
                "pts": snapshot.get("pts"),
                "ntp": snapshot.get("ntp"),
                "arrival_time": snapshot.get("arrival_time"),
            })
            self._pending_camera_frames.popleft()
            self._metadata_dirty = True

    def set_camera_delay_seconds(self, value: float) -> None:
        self.camera_delay_seconds = float(value)

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        self._flush_metadata(force=True)
                        return
                    points, recorded_at, frame_type = item
                    frame_number = self.frames_written + 1
                    filename = f"frame_{frame_number:06d}.pcd"
                    path = self.point_cloud_folder / filename
                    save_point_cloud(path, points, frame_type)
                    self.frames_written = frame_number
                    with self._metadata_lock:
                        reference = point_cloud_reference(filename)
                        self._timestamps[reference] = recorded_at
                        self._records.append({
                            "point_cloud": reference,
                            "recorded_at": recorded_at,
                            "frame_type": frame_type,
                            "camera_frame": None,
                            "pts": None,
                            "ntp": None,
                            "arrival_time": None,
                        })
                        self._metadata_dirty = True
                        self._match_camera_frames_locked()
                        self._flush_metadata_locked()
                    if self.progress_callback:
                        self.progress_callback(self.channel, self.frames_written)
                finally:
                    self._queue.task_done()
        except Exception as error:
            self.error = error
            try:
                self._flush_metadata(force=True)
            except Exception:
                pass

    def _flush_metadata(self, *, force: bool = False) -> None:
        with self._metadata_lock:
            self._flush_metadata_locked(force=force)

    def _flush_metadata_locked(self, *, force: bool = False) -> None:
        if not self._metadata_dirty:
            return
        now = time.monotonic()
        if not force and now - self._last_metadata_flush < METADATA_FLUSH_SECONDS:
            return
        self._write_metadata_locked()
        self._metadata_dirty = False
        self._last_metadata_flush = now

    def _write_metadata(self) -> None:
        with self._metadata_lock:
            self._write_metadata_locked()

    def _write_metadata_locked(self) -> None:
        save_timestamps_metadata(self.timestamps_path, self._timestamps)
        save_recording_metadata(self.metadata_path, self._records)

    @staticmethod
    def _replace_json(path: Path, value: object) -> None:
        if path.suffix.lower() == ".csv":
            if isinstance(value, list):
                save_recording_csv(path, value)
            else:
                save_timestamps_csv(path, value)
            return
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(value, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @staticmethod
    def _save(
        path: Path,
        points: tuple[RadarPoint | RadarObject, ...],
        frame_type: str,
    ) -> None:
        save_point_cloud(path, points, frame_type)


class RadarRecordingSession:
    def __init__(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
        camera_delay_seconds: float = CAMERA_DELAY_SECONDS,
    ):
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

        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            raise ValueError("The recording destination must be an existing folder")

        selected = sorted(set(channels))
        if not selected:
            raise ValueError("Select at least one radar to record")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        created: dict[int, PointCloudRecorder] = {}
        try:
            for channel in selected:
                created[channel] = PointCloudRecorder(
                    root_path,
                    channel,
                    timestamp,
                    self.progress_callback,
                    camera_delay_seconds=self.camera_delay_seconds,
                )
        except Exception:
            for recorder in created.values():
                try:
                    recorder.stop()
                except Exception:
                    pass
            raise

        self.recorders = created
        return {channel: str(recorder.folder) for channel, recorder in created.items()}

    def submit(
        self,
        channel: int,
        points: Iterable[RadarPoint | RadarObject],
        recorded_at: datetime,
        frame_type: str = "cluster",
    ) -> bool:
        recorder = self.recorders.get(channel)
        return recorder.submit(points, recorded_at, frame_type) if recorder else False

    def add_camera_snapshot(
        self,
        channel: int,
        filename: str,
        recorded_at: datetime | str,
        pts: int | None = None,
        ntp: int | None = None,
        arrival_time: str | None = None,
    ) -> bool:
        recorder = self.recorders.get(channel)
        if recorder is None:
            return False
        recorder.add_camera_snapshot(
            filename,
            recorded_at,
            pts=pts,
            ntp=ntp,
            arrival_time=arrival_time,
        )
        return True

    def set_camera_delay_seconds(self, value: float) -> None:
        self.camera_delay_seconds = float(value)
        for recorder in self.recorders.values():
            recorder.set_camera_delay_seconds(value)

    def poll_error(self) -> Exception | None:
        for recorder in self.recorders.values():
            if recorder.error is not None:
                return recorder.error
        return None

    def stop(self) -> dict[int, int]:
        recorders, self.recorders = self.recorders, {}
        first_error = None
        for recorder in recorders.values():
            try:
                recorder.stop()
            except Exception as error:
                first_error = first_error or error
        counts = {channel: recorder.frames_written for channel, recorder in recorders.items()}
        if first_error is not None:
            raise first_error
        return counts
