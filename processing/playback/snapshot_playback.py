from datetime import timedelta
from pathlib import Path
import signal
import time

import cv2 as cv

from processing.visualization.graph_draw import Graph_radar
from processing.visualization.graph_filter import Filter_graph
from processing.recording.manual_snapshot import ManualSnapshotWriter
from processing.playback.playback import load_recording_entries
from processing.recording.point_cloud_reader import PointCloudReader
from processing.recording.point_cloud_recorder import CAMERA_DELAY_SECONDS

DEFAULT_PLAYBACK_WIDTH = 1280
DEFAULT_PLAYBACK_HEIGHT = 720


def _put_status(pool, message, payload):
    try:
        pool.put((message, payload), timeout=0.2)
    except Exception:
        pass


def _load_entries(folder, synced_only=True):
    root = Path(folder).expanduser()
    entries = list(load_recording_entries(root))
    if not synced_only:
        return root, entries
    synced_entries = [
        entry
        for entry in entries
        if entry.point_cloud is not None and entry.camera_frame is not None
    ]
    if not synced_entries:
        raise ValueError(
            "No synced image + PCD pairs were found. Uncheck "
            "'Synced image + PCD only' to play single-modality entries."
        )
    return root, synced_entries


class SnapshotPlaybackController:
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
        self.active = False
        self.paused = False
        self.stop_requested = False
        self.width = DEFAULT_PLAYBACK_WIDTH
        self.height = DEFAULT_PLAYBACK_HEIGHT
        try:
            self.camera_delay_seconds = float(
                initial_values.get(
                    "camera_latency_adjustment",
                    CAMERA_DELAY_SECONDS * 1000.0,
                )
            ) / 1000.0
        except (TypeError, ValueError):
            self.camera_delay_seconds = CAMERA_DELAY_SECONDS
        self.entries = []
        self.index = 0
        self.current_reader = None
        self.snapshot_folder = None

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
            elif event == "snapshot_playback_start":
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
            elif event == "camera_latency_adjustment":
                self._set_camera_latency_adjustment(value)
            elif isinstance(event, str) and event.startswith("filter"):
                self.filters.update_values(event, value)
        self._close_windows()

    def _set_resolution(self, value):
        width = int(value.get("width", DEFAULT_PLAYBACK_WIDTH))
        height = int(value.get("height", DEFAULT_PLAYBACK_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Playback image dimensions must be positive")
        self.width, self.height = width, height
        if self.active and self.entries:
            self._render()

    def _set_camera_latency_adjustment(self, value):
        self.camera_delay_seconds = float(value.get("latency_adjustment_ms")) / 1000.0

    def _play(self, value):
        try:
            root, self.entries = _load_entries(
                value["folder"], value.get("synced_only", True)
            )
            self.snapshot_folder = value.get("snapshot_folder")
            self._set_resolution(value)
            self.active = True
            self.paused = False
            self.stop_requested = False
            self.index = 0
            _put_status(self.pool, "snapshot_playback_state", {
                "active": True,
                "paused": False,
                "current": 1,
                "total": len(self.entries),
                "folder": str(root.resolve()),
            })
            self._render()

            while (
                self.active
                and not self.stop_requested
                and not self.shutdown_event.is_set()
            ):
                delay = self._frame_delay()
                deadline = time.monotonic() + delay
                while (
                    self.active
                    and not self.stop_requested
                    and not self.shutdown_event.is_set()
                ):
                    if self.connection.poll(0.05):
                        event, payload = self.connection.recv()
                        self._handle(event, payload)
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
                    self._process_window_events()

            _put_status(self.pool, "snapshot_playback_state", {
                "active": False,
                "paused": False,
                "current": 0,
                "total": len(self.entries),
                "completed": not self.stop_requested,
            })
        except Exception as error:
            _put_status(self.pool, "snapshot_playback_error", str(error))
            _put_status(self.pool, "snapshot_playback_state", {
                "active": False,
                "paused": False,
                "current": 0,
                "total": 0,
            })
        finally:
            self.active = False
            self.entries = []
            self.current_reader = None
            self._close_windows()

    def _frame_delay(self):
        if self.index + 1 >= len(self.entries):
            return 0.05
        return max(
            0.05,
            self.entries[self.index + 1].recorded_at.timestamp()
            - self.entries[self.index].recorded_at.timestamp(),
        )

    def _handle(self, event, value):
        if event == "STOP":
            self.shutdown_event.set()
        elif event == "snapshot_playback_stop":
            self.stop_requested = True
        elif event == "snapshot_playback_pause":
            self.paused = not self.paused
            self._state()
        elif event == "snapshot_playback_next":
            self.paused = True
            self.index = min(self.index + 1, len(self.entries) - 1)
            self._render()
            self._state()
        elif event == "snapshot_playback_previous":
            self.paused = True
            self.index = max(self.index - 1, 0)
            self._render()
            self._state()
        elif event == "snapshot_playback_snapshot":
            try:
                destination = value.get("folder") if isinstance(value, dict) else None
                self._save_snapshot(destination)
            except Exception as error:
                _put_status(self.pool, "snapshot_playback_snapshot_error", str(error))
        elif event == "playback_resolution":
            try:
                self._set_resolution(value)
            except Exception as error:
                _put_status(self.pool, "playback_resolution_error", str(error))
        elif event == "point_cutoff":
            try:
                self.graph.set_distance_cutoff(value.get("distance", 15.0))
                if self.active and self.entries:
                    self._render()
            except Exception as error:
                _put_status(self.pool, "point_cutoff_error", str(error))
        elif event == "graph_resolution":
            try:
                self.graph.set_resolution(
                    value.get("width", 800), value.get("height", 600)
                )
            except (AttributeError, TypeError, ValueError) as error:
                _put_status(self.pool, "graph_resolution_error", str(error))
        elif event == "graph_range":
            try:
                self.graph.set_range(
                    value.get("x_range", 15.0), value.get("y_range", 15.0)
                )
            except (AttributeError, TypeError, ValueError) as error:
                _put_status(self.pool, "graph_range_error", str(error))
        elif event == "camera_latency_adjustment":
            try:
                self._set_camera_latency_adjustment(value)
            except Exception as error:
                _put_status(self.pool, "camera_latency_error", str(error))
        elif isinstance(event, str) and event.startswith("filter"):
            self.filters.update_values(event, value)
            self._render()

    def _render(self):
        entry = self.entries[self.index]
        self.current_reader = None

        if entry.point_cloud is not None:
            reader = PointCloudReader(entry.point_cloud)
            self.current_reader = reader
            if reader.frame_type == "cluster":
                x, y, colors = self.filters.filter_point_sequence(reader.clusters)
            else:
                x, y, colors = self.filters.filter_object_sequence(reader.objects)
            self.graph.show_points(x, y, colors, self.filters.last_points)

        if entry.camera_frame is not None:
            image = cv.imread(str(entry.camera_frame))
            if image is None:
                raise RuntimeError(f"Could not read image: {entry.camera_frame.name}")
            image = cv.resize(
                image,
                (self.width, self.height),
                interpolation=cv.INTER_AREA,
            )
            cv.imshow("CAMERA PLAYBACK", image)
            cv.waitKey(1)

        current_file = (
            entry.point_cloud.name
            if entry.point_cloud is not None
            else entry.camera_frame.name
        )
        _put_status(self.pool, "snapshot_playback_progress", {
            "current": self.index + 1,
            "total": len(self.entries),
            "file": current_file,
            "point_cloud": entry.point_cloud.name if entry.point_cloud else None,
            "image": entry.camera_frame.name if entry.camera_frame else None,
        })

    def _state(self):
        _put_status(self.pool, "snapshot_playback_state", {
            "active": self.active,
            "paused": self.paused,
            "current": self.index + 1 if self.entries else 0,
            "total": len(self.entries),
        })

    @staticmethod
    def _process_window_events():
        # OpenCV dispatches mouse callbacks from waitKey. Keep pumping its event
        # queue even while playback is paused so radar clicks are handled now.
        cv.waitKey(1)

    def _save_snapshot(self, folder):
        if not self.entries:
            raise RuntimeError("No playback frame is currently displayed")
        entry = self.entries[self.index]
        if (
            self.current_reader is None
            or entry.point_cloud is None
            or entry.camera_frame is None
        ):
            raise RuntimeError(
                "The current playback entry does not contain both radar and camera data"
            )
        destination = folder or self.snapshot_folder
        if not destination:
            raise ValueError("Select a snapshot destination folder")

        camera_time = entry.camera_recorded_at or (
            entry.recorded_at + timedelta(seconds=self.camera_delay_seconds)
        )
        result = ManualSnapshotWriter(
            destination,
            camera_delay_seconds=self.camera_delay_seconds,
        ).save(
            self.current_reader.points,
            entry.recorded_at,
            self.current_reader.frame_type,
            entry.camera_frame.read_bytes(),
            camera_time,
        )
        _put_status(self.pool, "snapshot_playback_snapshot_saved", result)

    @staticmethod
    def _close_windows():
        try:
            cv.destroyAllWindows()
            cv.waitKey(1)
        except cv.error:
            pass


def snapshot_playback_main(connection, pool, shutdown_event, initial_values):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
    SnapshotPlaybackController(connection, pool, shutdown_event, initial_values).run()
