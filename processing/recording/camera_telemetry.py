"""Thread-safe append-only camera timing journals."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Mapping

CAMERA_TIMESTAMPS_JOURNAL_NAME = "camera_timestamps.jsonl"
CAMERA_TIMING_EVENTS_NAME = "camera_timing_events.jsonl"
CAMERA_TIMING_SESSION_NAME = "camera_timing_session.json"
CAMERA_RECORDING_SUMMARY_NAME = "camera_recording_summary.json"
SCHEMA_VERSION = 3


class CameraTelemetryWriter:
    """Mirror one canonical timing session and its journals to recording folders."""

    def __init__(self) -> None:
        self._folders: tuple[Path, ...] = ()
        self._session: dict = {}
        self._lock = threading.RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._folders)

    def start(self, folders: tuple[Path, ...], session: Mapping) -> None:
        paths = tuple(Path(folder) for folder in folders)
        if not paths or any(not path.is_dir() for path in paths):
            raise ValueError("Camera telemetry needs existing recording folders")
        with self._lock:
            if self._folders:
                raise RuntimeError("Camera telemetry is already active")
            self._folders = paths
            self._session = {
                **dict(session),
                "schema_version": SCHEMA_VERSION,
                "epochs": [],
            }
            for folder in paths:
                (folder / CAMERA_TIMESTAMPS_JOURNAL_NAME).write_text("", encoding="utf-8")
                (folder / CAMERA_TIMING_EVENTS_NAME).write_text("", encoding="utf-8")
            self._write_session_locked()

    def stop(self) -> None:
        with self._lock:
            if self._folders:
                self._write_session_locked()
                self._folders = ()

    def update_epoch(self, epoch: Mapping | None) -> None:
        if not epoch:
            return
        value = dict(epoch)
        identity = tuple(value.get(key) for key in (
            "stream_epoch", "mapping_revision", "segment_epoch",
        ))
        with self._lock:
            if not self._folders:
                return
            epochs = self._session.setdefault("epochs", [])
            match = next((item for item in epochs if tuple(
                item.get(key) for key in (
                    "stream_epoch", "mapping_revision", "segment_epoch",
                )
            ) == identity), None)
            if match is None:
                epochs.append(value)
            else:
                match.update(value)
            self._write_session_locked()

    def append_frame(self, record: Mapping) -> None:
        self._append_jsonl(CAMERA_TIMESTAMPS_JOURNAL_NAME, record)

    def append_event(self, event: Mapping) -> None:
        self._append_jsonl(CAMERA_TIMING_EVENTS_NAME, event)

    def write_summary(self, summary: Mapping) -> None:
        with self._lock:
            for folder in self._folders:
                self._replace_json(folder / CAMERA_RECORDING_SUMMARY_NAME, dict(summary))

    def _append_jsonl(self, filename: str, value: Mapping) -> None:
        line = json.dumps(dict(value), separators=(",", ":")) + "\n"
        with self._lock:
            for folder in self._folders:
                with (folder / filename).open("a", encoding="utf-8") as stream:
                    stream.write(line)

    def _write_session_locked(self) -> None:
        for folder in self._folders:
            self._replace_json(folder / CAMERA_TIMING_SESSION_NAME, self._session)

    @staticmethod
    def _replace_json(path: Path, value: Mapping) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
