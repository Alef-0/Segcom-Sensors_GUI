from datetime import datetime
from pathlib import Path
import queue
import threading
from typing import Callable, Mapping

import cv2 as cv
import numpy as np

from processing.recording.paths import (
    IMAGE_DIRECTORY_NAME,
    image_path,
    image_reference,
)


CAMERA_FRAME_RATE = 30
DEFAULT_RECORDED_FRAMES_PER_30 = 30
_STOP = object()


class CameraSnapshotRecorder:
    """Asynchronously writes camera JPEG frames and emits NTP, PTS, and arrival timing."""

    def __init__(
        self,
        saved_callback: Callable[[dict], None] | None = None,
        dropped_callback: Callable[[dict], None] | None = None,
        queue_size: int = 8,
    ):
        self.saved_callback = saved_callback
        self.dropped_callback = dropped_callback
        self.queue_size = queue_size
        self.error: Exception | None = None
        self.active = False
        self._folders: dict[int, Path] = {}
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._frame_number = 0
        self.recorded_frames_per_30 = DEFAULT_RECORDED_FRAMES_PER_30
        self._sampling_accumulator = 0
        self.frames_observed = 0
        self.frames_selected = 0
        self.frames_dropped = 0
        self.unusual_pts_gap_candidates = 0
        self.frames_rejected_invalid_timing = 0

    def start(
        self,
        folders: Mapping[int, str],
        *,
        latency_adjustment_ms: float = 65.0,
        **kwargs,
    ) -> None:
        """Initialize image folders and launch worker thread."""
        if self.active:
            raise RuntimeError("Camera snapshot recording is already active")
        selected = {int(channel): Path(folder).expanduser() for channel, folder in folders.items()}
        if not selected:
            raise ValueError("No radar recording folders were supplied")
        for folder in selected.values():
            if not folder.is_dir():
                raise ValueError(f"Recording folder does not exist: {folder}")
            (folder / IMAGE_DIRECTORY_NAME).mkdir(exist_ok=True)

        self._folders = selected
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._sampling_accumulator = CAMERA_FRAME_RATE - self.recorded_frames_per_30
        self._frame_number = 0
        self.frames_observed = 0
        self.frames_selected = 0
        self.frames_dropped = 0
        self.unusual_pts_gap_candidates = 0
        self.frames_rejected_invalid_timing = 0
        self.error = None
        self.active = True
        self._thread = threading.Thread(
            target=self._write_loop,
            name="camera-snapshot-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        frame: np.ndarray,
        captured_at: datetime | None = None,
        timing: Mapping | None = None,
    ) -> bool:
        """Enqueue frame if sampling accumulator permits."""
        if not self.active or self.error is not None or self._queue is None:
            return False
        self.frames_observed += 1
        timing = dict(timing or {})
        self._sampling_accumulator += self.recorded_frames_per_30
        if self._sampling_accumulator < CAMERA_FRAME_RATE:
            return False
        self._sampling_accumulator -= CAMERA_FRAME_RATE
        self.frames_selected += 1
        timestamp = (captured_at or datetime.now().astimezone()).isoformat(
            timespec="microseconds"
        )
        try:
            self._queue.put_nowait((
                frame,
                timestamp,
                timing,
            ))
        except queue.Full:
            self.frames_dropped += 1
            if self.dropped_callback is not None:
                self.dropped_callback({
                    "reason": "image writer queue is full",
                    "dropped": self.frames_dropped,
                    "selected": self.frames_selected,
                    "queue_size": self.queue_size,
                    "timing": timing,
                })
            return False
        return True

    def poll_error(self) -> Exception | None:
        """Return the first fatal worker error if one occurred."""
        return self.error

    def note_invalid_timing_frame(self) -> None:
        """Track invalid frames dropped prior to submission."""
        if self.active:
            self.frames_rejected_invalid_timing += 1

    def record_timing_events(self, events) -> None:
        """Compatibility no-op for transport timing events."""

    def update_transport_stats(
        self,
        stats: Mapping,
        *,
        stream_epoch: int = 0,
    ) -> None:
        """Compatibility no-op for RTP transport stats."""

    def set_latency_adjustment_ms(self, value: float) -> None:
        """Compatibility no-op for software latency adjustments."""

    def set_recorded_frames_per_30(self, value: int) -> None:
        """Adjust decimated capture rate between 1 and 30 frames."""
        numeric_value = float(value)
        if not numeric_value.is_integer():
            raise ValueError("Recorded frames must be a whole number")
        frames_per_30 = int(numeric_value)
        if not 1 <= frames_per_30 <= CAMERA_FRAME_RATE:
            raise ValueError("Recorded frames must be between 1 and 30")
        self.recorded_frames_per_30 = frames_per_30
        self._sampling_accumulator = CAMERA_FRAME_RATE - frames_per_30

    def stop(self) -> int:
        """Drain worker queue and wait for the thread to exit."""
        if not self.active:
            return self._frame_number
        self.active = False
        if self._queue is not None and self._thread is not None:
            while self._thread.is_alive():
                try:
                    self._queue.put(_STOP, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
        if self.error is not None:
            raise self.error
        return self._frame_number

    def _write_loop(self) -> None:
        """Worker loop writing JPEG files and notifying saved callback."""
        assert self._queue is not None
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    frame, captured_at, timing = item
                    frame_number = self._frame_number + 1
                    filename = f"camera_{frame_number:06d}.jpg"
                    reference = image_reference(filename)
                    files = {}
                    for channel, folder in self._folders.items():
                        path = image_path(folder, filename)
                        if not cv.imwrite(str(path), frame):
                            raise RuntimeError(f"Could not save camera snapshot: {path}")
                        files[channel] = reference
                    self._frame_number = frame_number
                    if self.saved_callback:
                        timing_dict = dict(timing or {})
                        self.saved_callback({
                            "files": files,
                            "captured_at": captured_at,
                            "pts": timing_dict.get("pts_ns"),
                            "ntp": timing_dict.get("camera_ntp_ns"),
                            "arrival_time": captured_at,
                        })
                finally:
                    self._queue.task_done()
        except Exception as error:
            self.error = error
