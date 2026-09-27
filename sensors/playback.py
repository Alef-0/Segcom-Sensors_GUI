"""Unified playback controller for recorded sensor sessions and manual snapshots."""

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import signal
import time

import cv2 as cv

from processing.visualization.graph_draw import Graph_radar
from processing.visualization.graph_filter import Filter_graph
from sensors.auxiliary import (
    CAMERA_DELAY_SECONDS,
    IMAGE_DIRECTORY_NAME,
    POINT_CLOUD_DIRECTORY_NAME,
    RECORDING_METADATA_NAME,
    TIMESTAMPS_METADATA_NAME,
    PointCloudReader,
    resolve_recording_file,
)
from sensors.snapshot import SnapshotWriter

DEFAULT_PLAYBACK_WIDTH, DEFAULT_PLAYBACK_HEIGHT = 1280, 720
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PlaybackEntry:
    """Represents a playable recording frame with point cloud and/or camera frame."""

    point_cloud: Path | None
    recorded_at: datetime
    camera_frame: Path | None = None
    camera_recorded_at: datetime | None = None


def _time(value, fallback: Path) -> datetime:
    if value:
        return datetime.fromisoformat(str(value))
    return datetime.fromtimestamp(fallback.stat().st_mtime).astimezone()


def load_recording_entries(folder: str | Path) -> tuple[PlaybackEntry, ...]:
    """Loads and sorts point cloud and camera entries from recording metadata or disk."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The playback source must be an existing recording folder")

    entries, pcd_seen, image_seen = [], set(), set()
    metadata_dirs = []
    if (root / RECORDING_METADATA_NAME).is_file() or (root / TIMESTAMPS_METADATA_NAME).is_file():
        metadata_dirs.append(root)
    else:
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and ((sub / RECORDING_METADATA_NAME).is_file() or (sub / TIMESTAMPS_METADATA_NAME).is_file()):
                metadata_dirs.append(sub)

    for base_dir in metadata_dirs:
        meta_path = base_dir / RECORDING_METADATA_NAME
        ts_path = base_dir / TIMESTAMPS_METADATA_NAME
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else []
        timestamps = json.loads(ts_path.read_text(encoding="utf-8")) if ts_path.is_file() else {}
        if not isinstance(metadata, list) or not isinstance(timestamps, dict):
            raise ValueError("Invalid recording metadata")

        for row in metadata:
            if not isinstance(row, dict):
                continue
            pcd = resolve_recording_file(base_dir, row.get("point_cloud"), POINT_CLOUD_DIRECTORY_NAME)
            image = resolve_recording_file(base_dir, row.get("camera_frame"), IMAGE_DIRECTORY_NAME)
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
            pcd = resolve_recording_file(base_dir, reference, POINT_CLOUD_DIRECTORY_NAME)
            if pcd and pcd not in pcd_seen:
                pcd_seen.add(pcd)
                entries.append(PlaybackEntry(pcd, _time(stamp, pcd)))

    for pcd in root.rglob("*.pcd"):
        if pcd not in pcd_seen:
            pcd_seen.add(pcd)
            entries.append(PlaybackEntry(pcd, _time(None, pcd)))

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


def detect_playback_structure(folder: str | Path, entries: tuple[PlaybackEntry, ...] | list[PlaybackEntry]) -> str:
    """Identifies whether a folder holds a manual snapshot or continuous recording structure."""
    path = Path(folder).expanduser()
    if "snapshot" in path.name.lower():
        return "snapshot"
    if entries and all(e.point_cloud and e.camera_frame for e in entries):
        return "snapshot"
    return "recording"


def load_snapshot_entries(folder: str | Path, synced_only: bool = True, loader=None) -> tuple[Path, list[PlaybackEntry]]:
    """Loads recording entries filtered to synchronized snapshot pairs if requested."""
    root = Path(folder).expanduser()
    read_entries = loader or load_recording_entries
    entries = list(read_entries(root))
    if not synced_only:
        return root, entries
    selected = [entry for entry in entries if entry.point_cloud and entry.camera_frame]
    if not selected:
        raise ValueError("No synced image + PCD pairs were found")
    return root, selected


_load_entries = load_snapshot_entries


def _send(pool, event: str, payload) -> None:
    try:
        pool.put((event, payload), timeout=0.2)
    except Exception:
        pass


def _close_windows() -> None:
    try:
        cv.destroyAllWindows()
        cv.waitKey(1)
    except cv.error:
        pass


class PlaybackController:
    """Unified playback controller using snapshot stepping and pause playback logic."""

    def __init__(self, connection, pool, shutdown_event, initial_values):
        self.connection = connection
        self.pool = pool
        self.shutdown_event = shutdown_event
        self.filters = Filter_graph(initial_values)
        self.graph = Graph_radar(
            initial_values.get("point_cutoff", 15.0),
            initial_values.get("graph_width", 800), initial_values.get("graph_height", 600),
            initial_values.get("graph_x_range", 15.0), initial_values.get("graph_y_range", 15.0),
        )
        self.width = DEFAULT_PLAYBACK_WIDTH
        self.height = DEFAULT_PLAYBACK_HEIGHT
        self.camera_delay_seconds = float(initial_values.get(
            "camera_latency_adjustment", CAMERA_DELAY_SECONDS * 1000,
        )) / 1000
        self.active = False
        self.paused = False
        self.stop_requested = False
        self.entries: list[PlaybackEntry] = []
        self.index = 0
        self.current_reader: PointCloudReader | None = None
        self.snapshot_folder = None

    def run(self) -> None:
        while not self.shutdown_event.is_set():
            if not self.connection.poll(0.05):
                continue
            try:
                event, value = self.connection.recv()
            except (EOFError, OSError):
                self.shutdown_event.set()
                break
            if event in ("playback_start", "snapshot_playback_start"):
                self._play(value)
            else:
                self._handle(event, value)
        _close_windows()

    def _set_resolution(self, value: dict) -> None:
        width = int(value.get("width", DEFAULT_PLAYBACK_WIDTH))
        height = int(value.get("height", DEFAULT_PLAYBACK_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Playback image dimensions must be positive")
        self.width, self.height = width, height

    def _frame_delay(self) -> float:
        if self.index + 1 >= len(self.entries):
            return 0.05
        diff = (self.entries[self.index + 1].recorded_at - self.entries[self.index].recorded_at).total_seconds()
        return max(0.05, min(5.0, diff))

    def _play(self, value: dict | str) -> None:
        try:
            folder = value.get("folder") if isinstance(value, dict) else value
            synced_only = value.get("synced_only", True) if isinstance(value, dict) else True
            root, self.entries = load_snapshot_entries(folder, synced_only)
            if isinstance(value, dict):
                self._set_resolution(value)
                self.snapshot_folder = value.get("snapshot_folder")
            self.active, self.paused, self.stop_requested, self.index = True, False, False, 0
            self._state(folder=str(root))
            self._render()

            while self.active and not self.stop_requested and not self.shutdown_event.is_set():
                deadline = time.monotonic() + self._frame_delay()
                while self.active and not self.stop_requested and not self.shutdown_event.is_set():
                    wait = 0.05 if self.paused else min(0.05, max(0.0, deadline - time.monotonic()))
                    if self.connection.poll(wait):
                        cmd, payload = self.connection.recv()
                        self._handle(cmd, payload)
                        if not self.paused:
                            deadline = time.monotonic() + self._frame_delay()
                    elif not self.paused and time.monotonic() >= deadline:
                        if self.index + 1 >= len(self.entries):
                            self.paused = True
                            self._state()
                        else:
                            self.index += 1
                            self._render()
                        break
                    cv.waitKey(1)
            self._state(active=False, paused=False, completed=not self.stop_requested)
        except Exception as error:
            self._error(str(error))
            self._state(active=False, paused=False, completed=False)
        finally:
            self.active = False
            self.entries, self.current_reader = [], None
            _close_windows()

    def _render(self) -> None:
        if not self.entries or self.index >= len(self.entries):
            return
        entry = self.entries[self.index]
        self.current_reader = None
        if entry.point_cloud:
            try:
                self.current_reader = PointCloudReader(entry.point_cloud)
                reader = self.current_reader
                points = reader.clusters if reader.frame_type == "cluster" else reader.objects
                coords = self.filters.filter_point_sequence(points) if reader.frame_type == "cluster" else self.filters.filter_object_sequence(points)
                self.graph.show_points(*coords, self.filters.last_points)
            except Exception:
                pass
        if entry.camera_frame:
            image = cv.imread(str(entry.camera_frame))
            if image is not None:
                cv.imshow(
                    "CAMERA PLAYBACK",
                    cv.resize(image, (self.width, self.height), interpolation=cv.INTER_AREA),
                )
                cv.waitKey(1)
        pcd_name = entry.point_cloud.name if entry.point_cloud else None
        image_name = entry.camera_frame.name if entry.camera_frame else None
        prog = {
            "current": self.index + 1, "total": len(self.entries),
            "file": pcd_name or image_name, "point_cloud": pcd_name, "image": image_name,
        }
        _send(self.pool, "playback_progress", prog)
        _send(self.pool, "snapshot_playback_progress", prog)

    def _state(self, **extra) -> None:
        payload = {
            "active": self.active,
            "paused": self.paused,
            "current": self.index + 1 if self.entries else 0,
            "total": len(self.entries),
            **extra,
        }
        _send(self.pool, "playback_state", payload)
        _send(self.pool, "snapshot_playback_state", payload)

    def _error(self, message: str) -> None:
        _send(self.pool, "playback_error", message)
        _send(self.pool, "snapshot_playback_error", message)

    def _handle(self, event: str, value) -> None:
        if event == "STOP":
            self.shutdown_event.set()
        elif event in ("playback_stop", "snapshot_playback_stop"):
            self.stop_requested = True
        elif event in ("playback_pause", "snapshot_playback_pause"):
            self.paused = not self.paused
            self._state()
        elif event in ("playback_previous", "playback_next", "snapshot_playback_previous", "snapshot_playback_next") and self.entries:
            self.paused = True
            step = 1 if event.endswith("next") else -1
            self.index = min(max(0, self.index + step), len(self.entries) - 1)
            self._render()
            self._state()
        elif event in ("playback_snapshot", "snapshot_playback_snapshot"):
            try:
                target = value.get("folder") if isinstance(value, dict) else None
                self._save_snapshot(target)
            except Exception as error:
                _send(self.pool, "playback_snapshot_error", str(error))
                _send(self.pool, "snapshot_playback_snapshot_error", str(error))
        elif event == "playback_resolution":
            try:
                self._set_resolution(value)
                if self.active and self.entries:
                    self._render()
            except Exception as error:
                _send(self.pool, "playback_resolution_error", str(error))
        elif event == "point_cutoff":
            self.graph.set_distance_cutoff(value.get("distance", 15.0))
            if self.active and self.entries:
                self._render()
        elif event == "graph_resolution":
            self.graph.set_resolution(value.get("width", 800), value.get("height", 600))
        elif event == "graph_range":
            self.graph.set_range(value.get("x_range", 15.0), value.get("y_range", 15.0))
        elif event == "camera_latency_adjustment":
            self.camera_delay_seconds = float(value.get("latency_adjustment_ms")) / 1000
        elif isinstance(event, str) and event.startswith("filter"):
            self.filters.update_values(event, value)
            if self.active and self.entries:
                self._render()

    def _save_snapshot(self, destination: str | Path | None) -> None:
        if not self.entries:
            raise RuntimeError("No playback frame is displayed")
        entry = self.entries[self.index]
        if self.current_reader is None or not entry.camera_frame or not entry.point_cloud:
            raise RuntimeError("The current entry does not contain a synchronized pair")
        destination = destination or self.snapshot_folder
        if not destination:
            raise ValueError("Select a snapshot destination folder")
        camera_time = (
            entry.camera_recorded_at
            or entry.recorded_at + timedelta(seconds=self.camera_delay_seconds)
        )
        result = SnapshotWriter(destination, self.camera_delay_seconds).save(
            self.current_reader.points, entry.recorded_at, self.current_reader.frame_type,
            entry.camera_frame.read_bytes(), camera_time,
        )
        _send(self.pool, "playback_snapshot_saved", result)
        _send(self.pool, "snapshot_playback_snapshot_saved", result)


SnapshotPlaybackController = PlaybackController  # Backward-compatibility alias


def playback_main(connection, pool, shutdown_event, initial_values):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
    PlaybackController(connection, pool, shutdown_event, initial_values).run()


snapshot_playback_main = playback_main
