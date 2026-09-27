"""Unified recording module for camera JPEG streams and radar PCD point clouds."""

from collections import deque
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Iterable, Mapping

import cv2 as cv
import numpy as np

from sensors.auxiliary import (
    CAMERA_DELAY_SECONDS,
    CLUSTER_PCD_FIELDS,
    CLUSTER_PCD_TYPES,
    IMAGE_DIRECTORY_NAME,
    LEGACY_OBJECT_PCD_FIELDS,
    LEGACY_OBJECT_PCD_TYPES,
    OBJECT_PCD_FIELDS,
    OBJECT_PCD_TYPES,
    POINT_CLOUD_DIRECTORY_NAME,
    RECORDING_METADATA_NAME,
    TIMESTAMPS_METADATA_NAME,
    PointCloud,
    image_path,
    image_reference,
    point_cloud_path,
    point_cloud_reference,
)
from sensors.radar.connection_packages import RadarObject, RadarPoint


def save_point_cloud(path: Path, points: Iterable[RadarPoint | RadarObject], frame_type: str) -> None:
    import sensors.auxiliary as aux
    return aux.save_point_cloud(path, points, frame_type, point_cloud_cls=PointCloud)


CAMERA_FRAME_RATE = 30
RADAR_LETTERS = {1: "A", 2: "B", 3: "C"}
_STOP = object()


class CameraRecorder:
    """Off-thread JPEG and CSV timestamp writer for camera frames."""

    def __init__(self, saved_callback: Callable[[dict], None] | None = None,
                 dropped_callback: Callable[[dict], None] | None = None,
                 queue_size: int = 8):
        self.saved_callback, self.dropped_callback = saved_callback, dropped_callback
        self.queue_size = queue_size
        self.error: Exception | None = None
        self.active = False
        self._folders: dict[int, Path] = {}
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._frame_number = 0
        self._latency_adjustment_ms = 0.0
        self.recorded_frames_per_30 = CAMERA_FRAME_RATE
        self._sampling_accumulator = 0
        self.frames_observed = self.frames_selected = self.frames_dropped = 0
        self.frames_rejected_invalid_timing = 0
        self._csv_files = []

    def prepare(self) -> None:
        with self._lock:
            if self._queue is None:
                self._queue = queue.Queue(maxsize=self.queue_size)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._write_loop, name="camera-writer", daemon=True)
                self._thread.start()

    def start(self, folders: Mapping[int, str | Path], latency_adjustment_ms: float = 0.0, **_kwargs) -> None:
        with self._lock:
            if self.active:
                raise RuntimeError("Camera recording is already active")
            selected = {int(ch): Path(folder).expanduser() for ch, folder in folders.items()}
            if not selected:
                raise ValueError("Select at least one destination folder")
            for folder in selected.values():
                if not folder.is_dir():
                    raise ValueError(f"Recording folder does not exist: {folder}")
                (folder / IMAGE_DIRECTORY_NAME).mkdir(exist_ok=True)
            self.prepare()
            self._folders = selected
            self._sampling_accumulator = CAMERA_FRAME_RATE - self.recorded_frames_per_30
            self._frame_number = 0
            self.frames_observed = self.frames_selected = self.frames_dropped = 0
            self.frames_rejected_invalid_timing = 0
            self.error = None
            self._latency_adjustment_ms = float(latency_adjustment_ms)
            self._open_csv_writers()
            self.active = True

    def _open_csv_writers(self):
        self._csv_files = []
        for folder in self._folders.values():
            csv_path = folder / "camera_timestamps.csv"
            f = csv_path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["index", "pts", "ntp", "monotonic"])
            f.flush()
            self._csv_files.append((f, writer))

    def _close_csv_writers(self):
        for f, _ in self._csv_files:
            try:
                f.close()
            except Exception:
                pass
        self._csv_files = []

    def append_timestamp_csv(self, index: int, pts: int, ntp: int, monotonic: int):
        with self._lock:
            for f, writer in self._csv_files:
                writer.writerow([index, pts, ntp, monotonic])
                f.flush()

    def submit(self, frame: np.ndarray, captured_at: datetime | None = None, timing: Mapping | None = None) -> bool:
        with self._lock:
            if not self.active or self.error or self._queue is None:
                return False
            self.frames_observed += 1
            timing = dict(timing or {})
            self._sampling_accumulator += self.recorded_frames_per_30
            if self._sampling_accumulator < CAMERA_FRAME_RATE:
                return False
            self._sampling_accumulator -= CAMERA_FRAME_RATE
            self.frames_selected += 1
            timestamp = (captured_at or datetime.now().astimezone()).isoformat(timespec="microseconds")
            try:
                self._queue.put_nowait((frame.copy(), timestamp, timing))
            except queue.Full:
                self.frames_dropped += 1
                if self.dropped_callback:
                    self.dropped_callback({
                        "reason": "image writer queue is full", "dropped": self.frames_dropped,
                        "selected": self.frames_selected, "queue_size": self.queue_size,
                        "timing": timing,
                    })
                return False
            return True

    def poll_error(self):
        return self.error

    def note_invalid_timing_frame(self, *, reason="invalid timing", timing=None) -> None:
        with self._lock:
            if not self.active:
                return
            self.frames_rejected_invalid_timing += 1
            timing_dict = dict(timing or {})
            mono = (
                timing_dict.get("application_arrival_monotonic_ns")
                or timing_dict.get("monotonic")
                or time.monotonic_ns()
            )
            self.append_timestamp_csv(-1, -1, -1, int(mono))

    def set_latency_adjustment_ms(self, value: float) -> None:
        self._latency_adjustment_ms = float(value)

    def set_recorded_frames_per_30(self, value: int) -> None:
        number = float(value)
        if not number.is_integer() or not 1 <= number <= CAMERA_FRAME_RATE:
            raise ValueError(f"Recording rate must be an integer between 1 and {CAMERA_FRAME_RATE}")
        with self._lock:
            self.recorded_frames_per_30 = int(number)
            self._sampling_accumulator = 0

    def stop(self) -> int:
        with self._lock:
            if not self.active:
                return self._frame_number
            self.active = False
            drain_marker = threading.Event()
            if self._queue is not None:
                self._queue.put(drain_marker)
        drain_marker.wait()
        self._close_csv_writers()
        if self.error:
            raise self.error
        return self._frame_number

    def close(self) -> None:
        try:
            if self.active:
                self.stop()
        finally:
            if self._queue is not None and self._thread is not None:
                while self._thread.is_alive():
                    try:
                        self._queue.put(_STOP, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                self._thread.join()
                self._thread, self._queue = None, None

    def _write_loop(self):
        assert self._queue is not None
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    if isinstance(item, threading.Event):
                        item.set()
                        continue
                    frame, captured_at, timing = item
                    number = self._frame_number + 1
                    filename = f"camera_{number:06d}.jpg"
                    files = {}
                    for channel, folder in self._folders.items():
                        path = image_path(folder, filename)
                        if not cv.imwrite(str(path), frame):
                            raise RuntimeError(f"Could not save camera snapshot: {path}")
                        files[channel] = image_reference(filename)
                    self._frame_number = number
                    pts = timing.get("pts_ns", -1)
                    ntp = timing.get("camera_ntp_ns", -1)
                    mono = timing.get("application_arrival_monotonic_ns", time.monotonic_ns())
                    self.append_timestamp_csv(number, int(pts), int(ntp), int(mono))
                    if self.saved_callback:
                        self.saved_callback({"files": files, "captured_at": captured_at})
                finally:
                    self._queue.task_done()
        except Exception as error:
            self.error = error


CameraSnapshotRecorder = CameraRecorder  # Backward-compatibility alias


class PointCloudRecorder:
    """Writes radar point cloud frames off the receive loop and tracks pairing metadata."""

    def __init__(self, root: Path, channel: int, timestamp: str,
                 progress_callback: Callable[[int, int], None] | None = None,
                 queue_size: int = 64, camera_delay_seconds: float = CAMERA_DELAY_SECONDS):
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
            ts = recorded_at.isoformat(timespec="microseconds") if isinstance(recorded_at, datetime) else str(recorded_at)
            self._queue.put_nowait((tuple(points), ts, frame_type))
            return True
        except queue.Full:
            self.error = RuntimeError(f"Recording queue for radar {self.channel} is full")
            return False

    def add_camera_snapshot(self, filename: str, recorded_at: datetime | str) -> None:
        camera_time = recorded_at if isinstance(recorded_at, datetime) else datetime.fromisoformat(str(recorded_at))
        target = camera_time - timedelta(seconds=self.camera_delay_seconds)
        with self._lock:
            self._pending_cameras.append({
                "camera_frame": image_reference(filename),
                "camera_recorded_at": camera_time.isoformat(timespec="microseconds"),
                "target_recorded_at": target.isoformat(timespec="microseconds"),
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
            target = datetime.fromisoformat(pending["target_recorded_at"])
            if not force and datetime.fromisoformat(available[-1]["recorded_at"]) < target:
                return
            chosen = min(available, key=lambda row: abs((datetime.fromisoformat(row["recorded_at"]) - target).total_seconds()))
            chosen.update(
                camera_frame=pending["camera_frame"],
                camera_recorded_at=pending["camera_recorded_at"],
                camera_delay_ms=round(pending["camera_delay_ms"], 3),
                synchronization_error_ms=round((datetime.fromisoformat(chosen["recorded_at"]) - target).total_seconds() * 1000, 3),
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
                            "point_cloud": reference, "recorded_at": timestamp,
                            "frame_type": frame_type, "camera_frame": None,
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
        now = time.monotonic()
        if not self._dirty:
            return
        if force or (now - self._last_flush) >= 1.0:
            self._write_metadata_locked()
            self._last_flush = now
            self._dirty = False

    def _write_metadata_locked(self) -> None:
        _replace_json(self.timestamps_path, self._timestamps)
        _replace_json(self.metadata_path, self._records)


def _replace_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


class RadarRecorder:
    """Manages multi-channel PointCloudRecorder sessions."""

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


RadarRecordingSession = RadarRecorder  # Backward-compatibility alias
