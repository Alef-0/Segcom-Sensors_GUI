from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import signal
import time

import cv2 as cv

from processing.visualization.graph_draw import Graph_radar
from processing.visualization.graph_filter import Filter_graph
from processing.recording.point_cloud_reader import PointCloudReader
from processing.recording.point_cloud_recorder import (
    LEGACY_RECORDING_METADATA_NAME,
    LEGACY_TIMESTAMPS_METADATA_NAME,
    RECORDING_METADATA_NAME,
    TIMESTAMPS_METADATA_NAME,
    load_recording_metadata,
    load_timestamps_metadata,
)
from processing.recording.paths import (
    IMAGE_DIRECTORY_NAME,
    POINT_CLOUD_DIRECTORY_NAME,
    resolve_recording_file,
)

DEFAULT_PLAYBACK_WIDTH = 1280
DEFAULT_PLAYBACK_HEIGHT = 720
_CAMERA_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PlaybackEntry:
    point_cloud: Path | None
    recorded_at: datetime
    camera_frame: Path | None = None
    camera_recorded_at: datetime | None = None


def _path_if_file(root: Path, filename, directory_name: str) -> Path | None:
    return resolve_recording_file(root, filename, directory_name)


def _file_time(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def _parse_time(value, fallback: Path) -> datetime:
    if value:
        return datetime.fromisoformat(str(value))
    return _file_time(fallback)


def load_recording_entries(folder: str | Path) -> tuple[PlaybackEntry, ...]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The playback source must be an existing recording folder")

    metadata_csv_path = root / RECORDING_METADATA_NAME
    metadata_json_path = root / LEGACY_RECORDING_METADATA_NAME
    timestamps_csv_path = root / TIMESTAMPS_METADATA_NAME
    timestamps_json_path = root / LEGACY_TIMESTAMPS_METADATA_NAME
    entries = []
    referenced_point_clouds = set()
    referenced_camera_frames = set()

    timestamps = {}
    if timestamps_csv_path.is_file():
        timestamps = load_timestamps_metadata(timestamps_csv_path)
    elif timestamps_json_path.is_file():
        timestamps = load_timestamps_metadata(timestamps_json_path)

    metadata = []
    if metadata_csv_path.is_file():
        metadata = load_recording_metadata(metadata_csv_path)
    elif metadata_json_path.is_file():
        metadata = load_recording_metadata(metadata_json_path)

    for item in metadata:
        if not isinstance(item, dict):
            continue
        point_name = item.get("point_cloud")
        camera_name = item.get("camera_frame")

        point_cloud = _path_if_file(root, point_name, POINT_CLOUD_DIRECTORY_NAME)
        camera_frame = _path_if_file(root, camera_name, IMAGE_DIRECTORY_NAME)
        if point_cloud is not None:
            referenced_point_clouds.add(point_cloud)
        if camera_frame is not None:
            referenced_camera_frames.add(camera_frame)
        if point_cloud is None and camera_frame is None:
            continue

        if point_cloud is not None:
            recorded_at = _parse_time(item.get("recorded_at"), point_cloud)
        else:
            recorded_at = _parse_time(
                item.get("arrival_time") or item.get("camera_recorded_at") or item.get("recorded_at"),
                camera_frame,
            )

        camera_recorded_at = None
        if camera_frame is not None:
            camera_recorded_at = _parse_time(
                item.get("arrival_time") or item.get("camera_recorded_at"),
                camera_frame,
            )

        entries.append(PlaybackEntry(
            point_cloud=point_cloud,
            recorded_at=recorded_at,
            camera_frame=camera_frame,
            camera_recorded_at=camera_recorded_at,
        ))

    for filename, recorded_at in timestamps.items():
        point_cloud = _path_if_file(root, filename, POINT_CLOUD_DIRECTORY_NAME)
        if point_cloud is None:
            continue
        if point_cloud in referenced_point_clouds:
            continue
        referenced_point_clouds.add(point_cloud)
        entries.append(PlaybackEntry(
            point_cloud=point_cloud,
            recorded_at=_parse_time(recorded_at, point_cloud),
        ))

    for point_cloud in root.rglob("*.pcd"):
        if point_cloud in referenced_point_clouds:
            continue
        referenced_point_clouds.add(point_cloud)
        entries.append(PlaybackEntry(
            point_cloud=point_cloud,
            recorded_at=_parse_time(timestamps.get(point_cloud.name), point_cloud),
        ))

    for camera_frame in root.rglob("*"):
        if (
            not camera_frame.is_file()
            or not camera_frame.name.lower().startswith("camera_")
            or camera_frame.suffix.lower() not in _CAMERA_SUFFIXES
            or camera_frame in referenced_camera_frames
        ):
            continue
        referenced_camera_frames.add(camera_frame)
        camera_time = _file_time(camera_frame)
        entries.append(PlaybackEntry(
            point_cloud=None,
            recorded_at=camera_time,
            camera_frame=camera_frame,
            camera_recorded_at=camera_time,
        ))

    entries.sort(key=lambda entry: (
        entry.recorded_at.timestamp(),
        entry.point_cloud.name if entry.point_cloud else "",
        entry.camera_frame.name if entry.camera_frame else "",
    ))
    if not entries:
        raise ValueError("The recording folder contains no playable point-cloud or camera frames")
    return tuple(entries)


def _put_status(pool, message, payload):
    try:
        pool.put((message, payload), timeout=0.2)
    except Exception:
        pass


def _destroy_windows():
    try:
        cv.destroyAllWindows()
        cv.waitKey(1)
    except cv.error:
        pass


class PlaybackController:
    def __init__(self, connection, pool, shutdown_event, initial_values):
        self.connection = connection
        self.pool = pool
        self.shutdown_event = shutdown_event
        self.filters = Filter_graph(initial_values)
        self.graph = Graph_radar(
            initial_values.get("point_cutoff", 15.0),
            initial_values.get("graph_width", 800),
            initial_values.get("graph_height", 600),
            initial_values.get("graph_x_range", 15.0),
            initial_values.get("graph_y_range", 15.0),
        )
        self.stop_requested = False
        self.transport_request = None
        self.width = DEFAULT_PLAYBACK_WIDTH
        self.height = DEFAULT_PLAYBACK_HEIGHT

    def run(self):
        while not self.shutdown_event.is_set():
            if not self.connection.poll(0.05):
                continue
            try:
                event, value = self.connection.recv()
            except (EOFError, OSError):
                self.shutdown_event.set()
                break
            if event == "STOP":
                self.shutdown_event.set()
            elif event == "playback_start":
                self._play(value)
            elif event == "playback_resolution":
                self._set_resolution(value)
            elif event == "point_cutoff":
                self.graph.set_distance_cutoff(value.get("distance", 15.0))
            elif event == "graph_resolution":
                self.graph.set_resolution(
                    value.get("width", 800), value.get("height", 600)
                )
            elif event == "graph_range":
                self.graph.set_range(
                    value.get("x_range", 15.0), value.get("y_range", 15.0)
                )
            elif isinstance(event, str) and event.startswith("filter"):
                self.filters.update_values(event, value)
        self._close_windows()

    def _set_resolution(self, value):
        width = int(value.get("width", DEFAULT_PLAYBACK_WIDTH))
        height = int(value.get("height", DEFAULT_PLAYBACK_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Playback image dimensions must be positive")
        self.width, self.height = width, height

    def _play(self, value):
        folder = value["folder"] if isinstance(value, dict) else value
        if isinstance(value, dict):
            self._set_resolution(value)
        try:
            entries = load_recording_entries(folder)
            self.stop_requested = False
            self.transport_request = None
            timestamps = [entry.recorded_at.timestamp() for entry in entries]
            started_at = timestamps[0]
            duration = max(0.0, timestamps[-1] - started_at)
            _put_status(self.pool, "playback_state", {
                "active": True, "folder": str(Path(folder).expanduser()),
                "current": 0, "total": len(entries), "mode": "record",
            })
            index = 0
            while index < len(entries):
                self._process_controls()
                if self.stop_requested or self.shutdown_event.is_set():
                    break
                index = self._apply_transport_request(index, timestamps)
                entry = entries[index]

                if entry.point_cloud is not None:
                    reader = PointCloudReader(entry.point_cloud)
                    if reader.frame_type == "cluster":
                        x, y, colors = self.filters.filter_point_sequence(reader.clusters)
                    else:
                        x, y, colors = self.filters.filter_object_sequence(reader.objects)
                    self.graph.show_points(x, y, colors, self.filters.last_points)

                if entry.camera_frame is not None:
                    image = cv.imread(str(entry.camera_frame))
                    if image is not None:
                        image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_AREA)
                        cv.imshow("CAMERA PLAYBACK", image)
                        cv.waitKey(1)

                current_file = (
                    entry.point_cloud.name
                    if entry.point_cloud is not None
                    else entry.camera_frame.name
                )
                _put_status(self.pool, "playback_progress", {
                    "current": index + 1,
                    "total": len(entries),
                    "file": current_file,
                    "point_cloud": entry.point_cloud.name if entry.point_cloud else None,
                    "image": entry.camera_frame.name if entry.camera_frame else None,
                    "elapsed": max(0.0, timestamps[index] - started_at),
                    "duration": duration,
                    "mode": "record",
                })
                if index + 1 < len(entries):
                    delay = max(
                        0.0,
                        timestamps[index + 1] - timestamps[index],
                    )
                    self._wait(delay)
                if self.transport_request is None:
                    index += 1
            completed = not self.stop_requested and not self.shutdown_event.is_set()
            _put_status(self.pool, "playback_state", {
                "active": False, "completed": completed,
                "current": len(entries) if completed else 0,
                "total": len(entries), "mode": "record",
            })
        except Exception as error:
            _put_status(self.pool, "playback_error", {"mode": "record", "message": str(error)})
            _put_status(self.pool, "playback_state", {
                "active": False, "completed": False, "current": 0, "total": 0, "mode": "record",
            })
        finally:
            self._close_windows()

    def _apply_transport_request(self, index, timestamps):
        request = self.transport_request
        self.transport_request = None
        if request is None:
            return index
        action, value = request
        if action == "restart":
            return 0
        target = timestamps[index] + float(value)
        return min(len(timestamps) - 1, bisect_left(timestamps, target))

    def _handle_control(self, event, value):
        if event == "STOP":
            self.shutdown_event.set()
        elif event == "playback_stop":
            self.stop_requested = True
        elif event == "playback_restart":
            self.transport_request = ("restart", 0.0)
        elif event == "playback_seek":
            self.transport_request = ("seek", float(value.get("seconds", 0.0)))
        elif event == "playback_resolution":
            self._set_resolution(value)
        elif event == "point_cutoff":
            self.graph.set_distance_cutoff(value.get("distance", 15.0))
        elif event == "graph_resolution":
            self.graph.set_resolution(
                value.get("width", 800), value.get("height", 600)
            )
        elif event == "graph_range":
            self.graph.set_range(
                value.get("x_range", 15.0), value.get("y_range", 15.0)
            )
        elif isinstance(event, str) and event.startswith("filter"):
            self.filters.update_values(event, value)

    def _process_controls(self):
        while self.connection.poll():
            try:
                event, value = self.connection.recv()
            except (EOFError, OSError):
                self.shutdown_event.set()
                return
            self._handle_control(event, value)

    def _wait(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if (
                self.shutdown_event.is_set()
                or self.stop_requested
                or self.transport_request is not None
            ):
                return
            remaining = max(0.0, deadline - time.monotonic())
            if not self.connection.poll(min(0.05, remaining)):
                continue
            try:
                event, value = self.connection.recv()
            except (EOFError, OSError):
                self.shutdown_event.set()
                return
            self._handle_control(event, value)
            if self.transport_request is not None:
                return

    def _close_windows(self):
        _destroy_windows()


def playback_main(connection, pool, shutdown_event, initial_values):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
    PlaybackController(connection, pool, shutdown_event, initial_values).run()
