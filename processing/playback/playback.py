"""Timed playback for paired radar and camera recordings."""

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import signal
import time

import cv2 as cv

from processing.recording.paths import (
    IMAGE_DIRECTORY_NAME, POINT_CLOUD_DIRECTORY_NAME, resolve_recording_file,
)
from processing.recording.point_cloud_reader import PointCloudReader
from processing.recording.point_cloud_recorder import RECORDING_METADATA_NAME, TIMESTAMPS_METADATA_NAME
from processing.visualization.graph_draw import Graph_radar
from processing.visualization.graph_filter import Filter_graph

DEFAULT_PLAYBACK_WIDTH, DEFAULT_PLAYBACK_HEIGHT = 1280, 720
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PlaybackEntry:
    point_cloud: Path | None
    recorded_at: datetime
    camera_frame: Path | None = None
    camera_recorded_at: datetime | None = None


def _time(value, fallback: Path) -> datetime:
    return datetime.fromisoformat(str(value)) if value else datetime.fromtimestamp(fallback.stat().st_mtime).astimezone()


def load_recording_entries(folder: str | Path) -> tuple[PlaybackEntry, ...]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The playback source must be an existing recording folder")
    metadata_path, timestamps_path = root / RECORDING_METADATA_NAME, root / TIMESTAMPS_METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else []
    timestamps = json.loads(timestamps_path.read_text(encoding="utf-8")) if timestamps_path.is_file() else {}
    if not isinstance(metadata, list) or not isinstance(timestamps, dict):
        raise ValueError("Invalid recording metadata")
    entries, pcd_seen, image_seen = [], set(), set()
    for row in metadata:
        if not isinstance(row, dict):
            continue
        pcd = resolve_recording_file(root, row.get("point_cloud"), POINT_CLOUD_DIRECTORY_NAME)
        image = resolve_recording_file(root, row.get("camera_frame"), IMAGE_DIRECTORY_NAME)
        if pcd:
            pcd_seen.add(pcd)
        if image:
            image_seen.add(image)
        if not pcd and not image:
            continue
        stamp_file = pcd or image
        stamp = row.get("recorded_at") or row.get("camera_recorded_at")
        camera_stamp = row.get("camera_recorded_at")
        entries.append(PlaybackEntry(
            pcd, _time(stamp, stamp_file), image,
            _time(camera_stamp, image) if image else None,
        ))
    for reference, stamp in timestamps.items():
        pcd = resolve_recording_file(root, reference, POINT_CLOUD_DIRECTORY_NAME)
        if pcd and pcd not in pcd_seen:
            pcd_seen.add(pcd)
            entries.append(PlaybackEntry(pcd, _time(stamp, pcd)))
    for pcd in root.rglob("*.pcd"):
        if pcd not in pcd_seen:
            pcd_seen.add(pcd)
            entries.append(PlaybackEntry(pcd, _time(timestamps.get(pcd.name), pcd)))
    for image in root.rglob("*"):
        if (image.is_file() and image.name.lower().startswith("camera_")
                and image.suffix.lower() in _IMAGE_SUFFIXES and image not in image_seen):
            image_seen.add(image)
            stamp = _time(None, image)
            entries.append(PlaybackEntry(None, stamp, image, stamp))
    entries.sort(key=lambda item: (item.recorded_at, str(item.point_cloud or ""), str(item.camera_frame or "")))
    if not entries:
        raise ValueError("The recording folder contains no playable sensor frames")
    return tuple(entries)


def _send(pool, event, payload):
    try:
        pool.put((event, payload), timeout=0.2)
    except Exception:
        pass


def _close_windows():
    try:
        cv.destroyAllWindows()
        cv.waitKey(1)
    except cv.error:
        pass


class PlaybackController:
    """Own the OpenCV playback windows and consume GUI control messages."""

    def __init__(self, connection, pool, shutdown_event, initial_values):
        self.connection, self.pool, self.shutdown_event = connection, pool, shutdown_event
        self.filters = Filter_graph(initial_values)
        self.graph = Graph_radar(
            initial_values.get("point_cutoff", 15.0),
            initial_values.get("graph_width", 800), initial_values.get("graph_height", 600),
            initial_values.get("graph_x_range", 15.0), initial_values.get("graph_y_range", 15.0),
        )
        self.width, self.height = DEFAULT_PLAYBACK_WIDTH, DEFAULT_PLAYBACK_HEIGHT
        self.stop_requested = False
        self.transport_request = None

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
            else:
                self._control(event, value)
        _close_windows()

    def _set_resolution(self, value):
        width, height = int(value.get("width", DEFAULT_PLAYBACK_WIDTH)), int(value.get("height", DEFAULT_PLAYBACK_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Playback image dimensions must be positive")
        self.width, self.height = width, height

    def _control(self, event, value):
        if event == "playback_stop":
            self.stop_requested = True
        elif event == "playback_restart":
            self.transport_request = ("restart", 0.0)
        elif event == "playback_seek":
            self.transport_request = ("seek", float(value.get("seconds", 0)))
        elif event == "playback_resolution":
            self._set_resolution(value)
        elif event == "point_cutoff":
            self.graph.set_distance_cutoff(value.get("distance", 15.0))
        elif event == "graph_resolution":
            self.graph.set_resolution(value.get("width", 800), value.get("height", 600))
        elif event == "graph_range":
            self.graph.set_range(value.get("x_range", 15.0), value.get("y_range", 15.0))
        elif isinstance(event, str) and event.startswith("filter"):
            self.filters.update_values(event, value)

    def _take_transport(self, index, times):
        request, self.transport_request = self.transport_request, None
        if request is None:
            return index
        action, seconds = request
        return 0 if action == "restart" else min(len(times) - 1, bisect_left(times, times[index] + seconds))

    def _play(self, value):
        folder = value.get("folder") if isinstance(value, dict) else value
        try:
            if isinstance(value, dict):
                self._set_resolution(value)
            entries = load_recording_entries(folder)
            times = [item.recorded_at.timestamp() for item in entries]
            start, index = times[0], 0
            self.stop_requested = False
            self.transport_request = None
            _send(self.pool, "playback_state", {"active": True, "folder": str(folder), "current": 0, "total": len(entries), "mode": "record"})
            while index < len(entries) and not self.stop_requested and not self.shutdown_event.is_set():
                index = self._take_transport(index, times)
                item = entries[index]
                if item.point_cloud:
                    reader = PointCloudReader(item.point_cloud)
                    coords = self.filters.filter_point_sequence(reader.clusters) if reader.frame_type == "cluster" else self.filters.filter_object_sequence(reader.objects)
                    self.graph.show_points(*coords, self.filters.last_points)
                if item.camera_frame:
                    image = cv.imread(str(item.camera_frame))
                    if image is None:
                        raise RuntimeError(f"Could not read {item.camera_frame.name}")
                    cv.imshow("CAMERA PLAYBACK", cv.resize(image, (self.width, self.height), interpolation=cv.INTER_AREA))
                    cv.waitKey(1)
                _send(self.pool, "playback_progress", {
                    "current": index + 1, "total": len(entries),
                    "file": (item.point_cloud or item.camera_frame).name,
                    "point_cloud": item.point_cloud.name if item.point_cloud else None,
                    "image": item.camera_frame.name if item.camera_frame else None,
                    "elapsed": max(0.0, times[index] - start),
                    "duration": max(0.0, times[-1] - start), "mode": "record",
                })
                if self.transport_request is not None:
                    continue
                if index + 1 < len(entries):
                    self._wait(times[index + 1] - times[index])
                if self.transport_request is None:
                    index += 1
            complete = not self.stop_requested and not self.shutdown_event.is_set()
            _send(self.pool, "playback_state", {"active": False, "completed": complete, "current": len(entries) if complete else 0, "total": len(entries), "mode": "record"})
        except Exception as error:
            _send(self.pool, "playback_error", {"mode": "record", "message": str(error)})
            _send(self.pool, "playback_state", {"active": False, "completed": False, "current": 0, "total": 0, "mode": "record"})
        finally:
            _close_windows()

    def _wait(self, seconds):
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline and not self.stop_requested and not self.shutdown_event.is_set() and self.transport_request is None:
            if not self.connection.poll(min(0.05, deadline - time.monotonic())):
                continue
            try:
                event, value = self.connection.recv()
            except (EOFError, OSError):
                self.shutdown_event.set()
                return
            if event == "STOP":
                self.shutdown_event.set()
            else:
                self._control(event, value)


def playback_main(connection, pool, shutdown_event, initial_values):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
    PlaybackController(connection, pool, shutdown_event, initial_values).run()
