"""Compact, append-only camera calibration telemetry."""

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
    """Write one canonical frame journal plus sparse session and event data."""

    def __init__(self) -> None:
        self._folders: tuple[Path, ...] = ()
        self._session: dict = {}
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return bool(self._folders)

    def start(self, folders: tuple[Path, ...], session: Mapping) -> None:
        self._folders = folders
        self._session = {
            "schema_version": SCHEMA_VERSION,
            **dict(session),
            "epochs": [],
        }
        for folder in self._folders:
            (folder / CAMERA_TIMESTAMPS_JOURNAL_NAME).write_text("", encoding="utf-8")
            (folder / CAMERA_TIMING_EVENTS_NAME).write_text("", encoding="utf-8")
        self._write_session()

    def stop(self) -> None:
        with self._lock:
            self._write_session_locked()
            self._folders = ()

    def update_epoch(self, epoch: Mapping | None) -> None:
        if not self.active or not epoch:
            return
        value = dict(epoch)
        identity = tuple(
            value.get(key)
            for key in ("stream_epoch", "mapping_revision", "segment_epoch")
        )
        with self._lock:
            epochs = self._session.setdefault("epochs", [])
            for existing in epochs:
                existing_identity = tuple(
                    existing.get(key)
                    for key in ("stream_epoch", "mapping_revision", "segment_epoch")
                )
                if existing_identity == identity:
                    if all(existing.get(key) == item for key, item in value.items()):
                        return
                    existing.update(value)
                    break
            else:
                epochs.append(value)
            self._write_session_locked()

    def append_frame(self, record: Mapping) -> None:
        self._append_jsonl(CAMERA_TIMESTAMPS_JOURNAL_NAME, record)

    def append_event(self, event: Mapping) -> None:
        self._append_jsonl(CAMERA_TIMING_EVENTS_NAME, event)

    def write_summary(self, summary: Mapping) -> None:
        for folder in self._folders:
            self._replace_json(folder / CAMERA_RECORDING_SUMMARY_NAME, summary)

    def _append_jsonl(self, filename: str, value: Mapping) -> None:
        if not self.active:
            return
        serialized = json.dumps(dict(value), separators=(",", ":")) + "\n"
        with self._lock:
            for folder in self._folders:
                with (folder / filename).open("a", encoding="utf-8") as handle:
                    handle.write(serialized)

    def _write_session(self) -> None:
        with self._lock:
            self._write_session_locked()

    def _write_session_locked(self) -> None:
        for folder in self._folders:
            self._replace_json(folder / CAMERA_TIMING_SESSION_NAME, self._session)

    @staticmethod
    def _replace_json(path: Path, value: Mapping) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(dict(value), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
