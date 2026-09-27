"""Unified manual snapshot writer for radar point clouds and camera images."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from typing import Iterable

from sensors.auxiliary import (
    CAMERA_DELAY_SECONDS,
    IMAGE_DIRECTORY_NAME,
    POINT_CLOUD_DIRECTORY_NAME,
    RECORDING_METADATA_NAME,
    TIMESTAMPS_METADATA_NAME,
    image_path,
    image_reference,
    point_cloud_path,
    point_cloud_reference,
    save_point_cloud,
)
from sensors.radar.connection_packages import RadarObject, RadarPoint

_INDEX = re.compile(r"^(?:frame|camera)_(\d+)\.(?:pcd|jpg)$", re.IGNORECASE)


def _replace_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


class SnapshotWriter:
    """Saves one paired radar frame (.pcd) and camera image (.jpg) to disk."""

    def __init__(self, folder: str | Path, camera_delay_seconds: float = CAMERA_DELAY_SECONDS):
        self.folder = Path(folder).expanduser()
        self.camera_delay_seconds = float(camera_delay_seconds)
        self.metadata_path = self.folder / RECORDING_METADATA_NAME
        self.timestamps_path = self.folder / TIMESTAMPS_METADATA_NAME

    def _metadata(self):
        if not self.folder.is_dir():
            raise ValueError("The snapshot destination must be an existing folder")
        if not self.metadata_path.exists() and not self.timestamps_path.exists():
            return [], {}
        if not self.metadata_path.is_file() or not self.timestamps_path.is_file():
            raise ValueError("Both recording.json and timestamps.json must exist, or neither")
        records = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        timestamps = json.loads(self.timestamps_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not isinstance(timestamps, dict):
            raise ValueError("Invalid recording metadata format")
        return records, timestamps

    def _next_index(self, records) -> int:
        numbers = [int(match.group(1)) for path in self.folder.rglob("*")
                   if path.is_file() and (match := _INDEX.match(path.name))]
        for row in records:
            for key in ("point_cloud", "camera_frame"):
                match = _INDEX.match(Path(str(row.get(key, ""))).name)
                if match:
                    numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def save(self, points: Iterable[RadarPoint | RadarObject], radar_recorded_at: datetime,
             frame_type: str, image_bytes: bytes, camera_recorded_at: datetime) -> dict:
        records, timestamps = self._metadata()
        index = self._next_index(records)
        pcd_filename = f"frame_{index:06d}.pcd"
        image_filename = f"camera_{index:06d}.jpg"
        pcd_ref = point_cloud_reference(pcd_filename)
        image_ref = image_reference(image_filename)

        target = camera_recorded_at - timedelta(seconds=self.camera_delay_seconds)
        sync_error_ms = round((radar_recorded_at - target).total_seconds() * 1000, 3)
        created = []
        try:
            (self.folder / POINT_CLOUD_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)
            (self.folder / IMAGE_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)
            full_pcd_path = point_cloud_path(self.folder, pcd_filename)
            full_image_path = image_path(self.folder, image_filename)
            save_point_cloud(full_pcd_path, tuple(points), frame_type)
            created.append(full_pcd_path)
            full_image_path.write_bytes(image_bytes)
            created.append(full_image_path)

            radar_iso = radar_recorded_at.isoformat(timespec="microseconds")
            camera_iso = camera_recorded_at.isoformat(timespec="microseconds")

            timestamps[pcd_ref] = radar_iso
            records.append({
                "point_cloud": pcd_ref,
                "recorded_at": radar_iso,
                "frame_type": frame_type,
                "camera_frame": image_ref,
                "camera_recorded_at": camera_iso,
                "camera_delay_ms": round(self.camera_delay_seconds * 1000.0, 3),
                "synchronization_error_ms": sync_error_ms,
            })

            _replace_json(self.timestamps_path, timestamps)
            _replace_json(self.metadata_path, records)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise

        return {
            "folder": str(self.folder.resolve()),
            "point_cloud": pcd_ref,
            "camera_frame": image_ref,
            "synchronization_error_ms": sync_error_ms,
        }


ManualSnapshotWriter = SnapshotWriter  # Backward-compatibility alias
