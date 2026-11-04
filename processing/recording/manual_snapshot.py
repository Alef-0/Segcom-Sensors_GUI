from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from typing import Iterable

from sensors.radar.connection_packages import RadarObject, RadarPoint
from processing.recording.point_cloud_recorder import (
    CAMERA_DELAY_SECONDS,
    LEGACY_RECORDING_METADATA_NAME,
    LEGACY_TIMESTAMPS_METADATA_NAME,
    RECORDING_METADATA_NAME,
    TIMESTAMPS_METADATA_NAME,
    load_recording_metadata,
    load_timestamps_metadata,
    save_point_cloud,
    save_recording_metadata,
    save_timestamps_metadata,
)
from processing.recording.paths import (
    image_path,
    image_reference,
    point_cloud_path,
    point_cloud_reference,
)

# Older snapshot folders may contain this file. It is ignored and never created.
_LEGACY_GROUP_METADATA_NAME = "group.json"
_EXPECTED_METADATA_FILES = {
    _LEGACY_GROUP_METADATA_NAME,
    RECORDING_METADATA_NAME,
    TIMESTAMPS_METADATA_NAME,
    LEGACY_RECORDING_METADATA_NAME,
    LEGACY_TIMESTAMPS_METADATA_NAME,
}
_INDEX_PATTERN = re.compile(r"^(?:frame|camera)_(\d+)\.(?:pcd|jpg)$", re.IGNORECASE)


def _replace_json(path: Path, value: object) -> None:
    save_recording_metadata(path, value) if isinstance(value, list) else save_timestamps_metadata(path, value)


def _load_json(path: Path, expected_type: type):
    try:
        value = load_recording_metadata(path) if expected_type is list else load_timestamps_metadata(path)
    except Exception as error:
        raise ValueError(f"Could not read {path.name}: {error}") from error
    if not isinstance(value, expected_type):
        raise ValueError(f"Invalid {path.name} format")
    return value


class ManualSnapshotWriter:
    """Append a radar/image pair using the normal recording folder structure."""

    def __init__(
        self,
        folder: str | Path,
        camera_delay_seconds: float = CAMERA_DELAY_SECONDS,
    ):
        self.folder = Path(folder).expanduser()
        self.camera_delay_seconds = float(camera_delay_seconds)
        self.metadata_path = self.folder / RECORDING_METADATA_NAME
        self.timestamps_path = self.folder / TIMESTAMPS_METADATA_NAME
        self.legacy_metadata_path = self.folder / LEGACY_RECORDING_METADATA_NAME
        self.legacy_timestamps_path = self.folder / LEGACY_TIMESTAMPS_METADATA_NAME

    def _load_or_initialize(self) -> tuple[list[dict], dict[str, str]]:
        if not self.folder.is_dir():
            raise ValueError("The snapshot destination must be an existing folder")

        json_names = {path.name for path in self.folder.glob("*.json")}
        csv_names = {path.name for path in self.folder.glob("*.csv")}
        unexpected = (json_names | csv_names) - _EXPECTED_METADATA_FILES
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError("The selected folder contains unrelated metadata: " + names)

        has_csv = self.metadata_path.is_file() and self.timestamps_path.is_file()
        has_json = self.legacy_metadata_path.is_file() and self.legacy_timestamps_path.is_file()

        if not has_csv and not has_json:
            if any(
                path.exists()
                for path in (
                    self.metadata_path,
                    self.timestamps_path,
                    self.legacy_metadata_path,
                    self.legacy_timestamps_path,
                )
            ):
                raise ValueError(
                    f"The folder must contain both {RECORDING_METADATA_NAME} and "
                    f"{TIMESTAMPS_METADATA_NAME}, or contain neither"
                )
            records: list[dict] = []
            timestamps: dict[str, str] = {}
            save_recording_metadata(self.metadata_path, records)
            save_timestamps_metadata(self.timestamps_path, timestamps)
            return records, timestamps

        if has_csv:
            records = load_recording_metadata(self.metadata_path)
            timestamps = load_timestamps_metadata(self.timestamps_path)
            return records, timestamps

        records = load_recording_metadata(self.legacy_metadata_path)
        timestamps = load_timestamps_metadata(self.legacy_timestamps_path)
        return records, timestamps

    def _next_index(self, records: list[dict]) -> int:
        indexes = []
        for path in self.folder.rglob("*"):
            if not path.is_file():
                continue
            match = _INDEX_PATTERN.match(path.name)
            if match:
                indexes.append(int(match.group(1)))
        for record in records:
            for key in ("point_cloud", "camera_frame"):
                match = _INDEX_PATTERN.match(Path(str(record.get(key, ""))).name)
                if match:
                    indexes.append(int(match.group(1)))
        return max(indexes, default=0) + 1

    def save(
        self,
        points: Iterable[RadarPoint | RadarObject],
        radar_recorded_at: datetime,
        frame_type: str,
        image_bytes: bytes,
        camera_recorded_at: datetime,
        pts: int | None = None,
        ntp: int | None = None,
        arrival_time: str | None = None,
    ) -> dict:
        records, timestamps = self._load_or_initialize()
        index = self._next_index(records)
        point_cloud_name = point_cloud_reference(f"frame_{index:06d}.pcd")
        camera_name = image_reference(f"camera_{index:06d}.jpg")
        point_cloud_file = point_cloud_path(self.folder, point_cloud_name)
        camera_file = image_path(self.folder, camera_name)

        created = []
        try:
            point_cloud_file.parent.mkdir(exist_ok=True)
            camera_file.parent.mkdir(exist_ok=True)
            save_point_cloud(point_cloud_file, tuple(points), frame_type)
            created.append(point_cloud_file)
            camera_file.write_bytes(image_bytes)
            created.append(camera_file)

            radar_timestamp = radar_recorded_at.isoformat(timespec="microseconds")
            camera_timestamp = camera_recorded_at.isoformat(timespec="microseconds")
            timestamps[point_cloud_name] = radar_timestamp
            records.append({
                "point_cloud": point_cloud_name,
                "recorded_at": radar_timestamp,
                "frame_type": frame_type,
                "camera_frame": camera_name,
                "pts": pts,
                "ntp": ntp,
                "arrival_time": arrival_time or camera_timestamp,
            })
            save_timestamps_metadata(self.timestamps_path, timestamps)
            save_recording_metadata(self.metadata_path, records)
            if self.legacy_metadata_path.is_file() or self.legacy_timestamps_path.is_file():
                save_timestamps_metadata(self.legacy_timestamps_path, timestamps)
                save_recording_metadata(self.legacy_metadata_path, records)
        except Exception:
            for path in created:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise

        return {
            "folder": str(self.folder.resolve()),
            "point_cloud": point_cloud_name,
            "camera_frame": camera_name,
        }
