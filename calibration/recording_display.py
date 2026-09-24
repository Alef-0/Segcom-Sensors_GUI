"""Review saved QR evidence or explicitly decode a camera calibration recording."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
import multiprocessing
from pathlib import Path
from queue import Empty, Queue
import statistics
import tempfile
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from calibration.display_common import DISPLAY_JOURNAL_NAME, timing_issues
from calibration.evidence import assess_evidence, detection_evidence, repeated_state_checks, screen_cell, screen_geometry
from calibration.qr import (
    DETECTION_BATCH_SIZE, cell_index_for, create_qreader, decode_qrs_with_grid_retries_batch,
    detect_contrast_cells, grid_cell_names, grid_positions, grid_shape, order_by_cell,
    timestamp_payload,
)
from processing.recording.paths import IMAGE_DIRECTORY_NAME, resolve_recording_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTRINSICS = PROJECT_ROOT / "calibration" / "intrinsics.json"
CAMERA_JOURNALS = ("camera_timestamps.jsonl", "camera_timestamps.json")
PRESENTATIONS_CSV = "display_presentations.csv"
PARALLEL_FRAME_WORKERS = 2


def read_json_rows(path: Path) -> list[dict]:
    """Read JSON or JSONL, ignoring only an incomplete final JSONL record."""
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                if index == len(lines) - 1 and not path.read_bytes().endswith(b"\n"):
                    break
                raise ValueError(f"Invalid JSON in {path.name}, line {index + 1}") from error
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    for key in ("frames", "camera_timestamps"):
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"{path.name} does not contain a frame list")


def payload_time(raw: str | None) -> str:
    if raw is None or not raw.isdigit() or len(raw) != 12:
        return "Unreadable"
    seconds, milliseconds = divmod(int(raw), 1000)
    return f"{seconds:,}".replace(",", " ") + f".{milliseconds:03d} s"


def editable_seconds(value: int | None) -> str:
    if value is None:
        return ""
    seconds, nanoseconds = divmod(int(value), 1_000_000_000)
    return f"{seconds}.{nanoseconds:09d}"


def parse_seconds(value: str, label: str) -> int | None:
    if not value.strip():
        return None
    try:
        return int(Decimal(value.strip()) * 1_000_000_000)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} must be seconds written as a number") from error


def parse_qr_value(value: str, cell: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if not text.isdigit() or len(text) > 12:
        raise ValueError(f"{cell} QR must be the integer milliseconds stored in the QR")
    return text.zfill(12)


def _bbox_iou(first: list[float], second: list[float]) -> float:
    """Return intersection-over-union for two x1, y1, x2, y2 boxes."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


class Undistorter:
    """Cache camera undistortion maps by image size and alpha."""

    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        self.distortion = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
        self.calibration_size = tuple(data["image_size"]) if data.get("image_size") else None
        if (self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all()
                or not np.isfinite(self.distortion).all()
                or self.distortion.size not in (4, 5, 8, 12, 14)):
            raise ValueError("Invalid camera_matrix or dist_coeffs in intrinsics.json")
        self._maps = {}

    def geometry(self, size: tuple[int, int], alpha: float):
        key = (size, float(alpha))
        if key not in self._maps:
            matrix = self.matrix.copy()
            if self.calibration_size:
                matrix[0, :] *= size[0] / self.calibration_size[0]
                matrix[1, :] *= size[1] / self.calibration_size[1]
            output, _ = cv2.getOptimalNewCameraMatrix(matrix, self.distortion, size, alpha, size)
            maps = cv2.initUndistortRectifyMap(matrix, self.distortion, None, output, size, cv2.CV_32FC1)
            self._maps[key] = matrix, output, maps
        return self._maps[key]

    def image(self, frame: np.ndarray, alpha: float) -> np.ndarray:
        size = (frame.shape[1], frame.shape[0])
        return cv2.remap(frame, *self.geometry(size, alpha)[2], cv2.INTER_LINEAR)

    def to_original(self, points: np.ndarray, size: tuple[int, int], alpha: float) -> np.ndarray:
        matrix, output, _ = self.geometry(size, alpha)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        rays = np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(output).T
        return cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix, self.distortion)[0].reshape(-1, 2)


_PROCESS_READER = None
_PROCESS_UNDISTORTER = None
_PROCESS_GRID_QRS = None


def _initialize_decode_process(intrinsics: str, grid_qrs: int) -> None:
    global _PROCESS_READER, _PROCESS_UNDISTORTER, _PROCESS_GRID_QRS
    _PROCESS_READER = None
    _PROCESS_UNDISTORTER = Undistorter(Path(intrinsics))
    _PROCESS_GRID_QRS = grid_qrs


def _decode_recording_frame(task: tuple[int, str, float]) -> tuple[int, list[dict]]:
    global _PROCESS_READER
    index, filename, alpha = task
    image = cv2.imread(filename)
    if image is None:
        raise ValueError("Could not read " + filename)
    undistorted = _PROCESS_UNDISTORTER.image(image, alpha)
    visual = detect_contrast_cells(undistorted, _PROCESS_GRID_QRS)
    decoded = []
    if visual:
        if _PROCESS_READER is None:
            _PROCESS_READER = create_qreader()
        decoded = decode_qrs_with_grid_retries_batch(
            _PROCESS_READER, [undistorted], _PROCESS_GRID_QRS, DETECTION_BATCH_SIZE
        )[0]
        decoded_cells = {
            cell_index_for(item["center"], (undistorted.shape[1], undistorted.shape[0]), _PROCESS_GRID_QRS)
            for item in decoded
        }
        decoded.extend(item for item in visual if item["cell"] not in decoded_cells)
    return index, decoded


class DisplayTimeline:
    """Validate the display journal and answer marker/presentation interval queries."""

    def __init__(self, folder: Path):
        path = folder / DISPLAY_JOURNAL_NAME
        if not path.is_file():
            raise ValueError(f"Missing {DISPLAY_JOURNAL_NAME}")
        rows = read_json_rows(path)
        sessions = [row for row in rows if row.get("kind") == "session"]
        self.frames = [row for row in rows if row.get("kind") == "frame"]
        if len(sessions) != 1 or len(self.frames) < 2:
            raise ValueError("Display journal needs one session row and at least two frames")
        self.metadata = sessions[0]
        self.grid_qrs = int(self.metadata.get("grid_qrs", 4))
        self.grid_rows, self.grid_columns = grid_shape(self.grid_qrs)
        self.visible_qrs = int(self.metadata.get("visible_qrs", min(2, self.grid_qrs)))
        if not 1 <= self.visible_qrs <= self.grid_qrs:
            raise ValueError("Display journal has an invalid visible QR count")
        self.cell_names, self.cell_positions = grid_cell_names(self.grid_qrs), grid_positions(self.grid_qrs)
        self.paused = {row.get("last_frame_index") for row in rows
                       if row.get("kind") == "pause" and row.get("paused")}
        self.by_payload, self.presentation_times = {}, []
        previous_marker = previous_return = -1
        for index, frame in enumerate(self.frames):
            cell = frame.get("cell", frame.get("corner"))
            marker = frame.get("marker_ns")
            returned = frame.get("presentation_return_ns", frame.get("flip_return_ns", marker))
            if (frame.get("index") != index or not isinstance(cell, int) or not 0 <= cell < self.grid_qrs
                    or not isinstance(marker, int) or marker <= previous_marker
                    or not isinstance(returned, int) or returned <= previous_return):
                raise ValueError("Display journal has invalid, reordered, or nonmonotonic frames")
            frame.update({
                "cell": cell, "presentation_return_ns": returned,
                "presentation_event_kind": frame.get("presentation_event_kind", "legacy_flip_return"),
                "physical_presentation_measured": bool(frame.get("physical_presentation_measured", False)),
            })
            self.by_payload.setdefault(timestamp_payload(marker), []).append(frame)
            self.presentation_times.append(returned)
            previous_marker, previous_return = marker, returned

    def match(self, raw: str | None, _reference_ns: int | None = None) -> dict | None:
        if not isinstance(raw, str) or len(raw) != 12 or not raw.isdigit():
            return None
        matches = self.by_payload.get(raw, [])
        # A repeated payload can wrap or recur; proximity is not identity evidence.
        return matches[0] if len(matches) == 1 else None

    def presentation_event(self, index: int) -> dict:
        frame = self.frames[index]
        return {"display_index": index, "cell": frame["cell"], "marker_ns": frame["marker_ns"],
                "predicted_flip_ns": frame.get("predicted_flip_ns"),
                "presentation_return_ns": frame["presentation_return_ns"],
                "presentation_event_kind": frame["presentation_event_kind"],
                "physical_presentation_measured": frame["physical_presentation_measured"],
                "interval_ns": frame.get("interval_ns"),
                "prediction_error_ns": frame.get("prediction_error_ns"),
                "timing_issues": timing_issues(frame)}

    def active_index_at(self, timestamp_ns: int) -> int | None:
        index = bisect_right(self.presentation_times, timestamp_ns) - 1
        return index if index >= 0 else None

    def events_between(self, start_ns: int, end_ns: int) -> list[dict]:
        if end_ns < start_ns:
            return []
        start = bisect_right(self.presentation_times, start_ns)
        stop = bisect_right(self.presentation_times, end_ns)
        return [self.presentation_event(index) for index in range(start, stop)]

    def observation_interval(self, index: int, reference_ns: int | None) -> dict:
        current = self.frames[index]
        following = self.frames[index + 1] if index + 1 < len(self.frames) else None
        start = current["presentation_return_ns"]
        end = following["presentation_return_ns"] if following else None
        lower = (reference_ns - end) / 1e6 if reference_ns is not None and end is not None else None
        upper = (reference_ns - start) / 1e6 if reference_ns is not None else None
        period = int(current.get("frame_period_ns") or self.metadata.get("prediction_period_ns") or
                     (end - start if end is not None else 0))
        margin_ns = max(1_000_000, abs(int(current.get("prediction_error_ns") or 0)),
                        abs(int(following.get("prediction_error_ns") or 0)) if following else 0)
        if period > 0:
            margin_ns = min(margin_ns, period // 2)
        issues = ["interval_start_" + value for value in timing_issues(current)]
        if following is None:
            issues.append("interval_end_evidence_missing")
            status = "Unknown"
        else:
            issues.extend("interval_end_" + value for value in timing_issues(following))
            status = "Timing suspect" if issues else "Clean"
        return {"observed_display_index": index, "active_interval_start_ns": start,
                "active_interval_end_ns": end, "offset_interval_lower_ms": lower,
                "offset_interval_upper_ms": upper,
                "offset_interval_width_ms": upper - lower if lower is not None and upper is not None else None,
                "software_transition_margin_ms": margin_ns / 1e6,
                "boundary_timing_status": status, "boundary_timing_issues": issues,
                "current_presentation": self.presentation_event(index),
                "next_presentation": self.presentation_event(index + 1) if following else None}

    def marker_status(self, index: int) -> tuple[str, list[str]]:
        replacement = index + self.visible_qrs
        issues = ["marker_" + value for value in timing_issues(self.frames[index])]
        if any(isinstance(paused, int) and index <= paused < replacement for paused in self.paused):
            issues.append("marker_held_for_pause")
        if replacement >= len(self.frames):
            issues.append("replacement_evidence_missing")
        else:
            issues.extend("replacement_" + value for value in timing_issues(self.frames[replacement]))
            if self.frames[replacement].get("interval_ns") is None:
                issues.append("replacement_interval_unavailable")
        if not issues:
            return "Clean", []
        unknown = {"replacement_evidence_missing", "replacement_interval_unavailable"}
        return ("Unknown" if all(value in unknown for value in issues) else "Timing suspect"), issues

    def totals(self) -> dict:
        return {"displayed": len(self.frames),
                "missed_period_candidates": sum(int(row.get("skipped_periods", 0)) for row in self.frames),
                "late_submissions": sum(bool(row.get("late_submit")) for row in self.frames),
                "irregular_intervals": sum(bool(row.get("irregular_interval")) for row in self.frames),
                "grid_qrs": self.grid_qrs, "grid_rows": self.grid_rows, "grid_columns": self.grid_columns,
                "visible_qrs": self.visible_qrs, "presentation_events": len(self.frames),
                "physical_presentation_measured": any(row["physical_presentation_measured"] for row in self.frames),
                "presentation_semantics": self.metadata.get("presentation_semantics",
                    "software presentation return; physical panel scanout is not measured")}


def _maximum_interval_consensus(intervals: list[tuple[float, float]]) -> dict | None:
    usable = sorted((float(a), float(b)) for a, b in intervals if a <= b)
    if not usable:
        return None
    edges = sorted({edge for pair in usable for edge in pair})
    candidates = []
    for index, point in enumerate(edges):
        candidates.append((point, point, sum(a <= point <= b for a, b in usable)))
        if index + 1 < len(edges):
            end = edges[index + 1]
            midpoint = (point + end) / 2
            candidates.append((point, end, sum(a <= midpoint <= b for a, b in usable)))
    maximum = max(item[2] for item in candidates)
    winners = sorted((a, b) for a, b, count in candidates if count == maximum)
    regions = []
    for start, end in winners:
        if regions and start <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], end)
        else:
            regions.append([start, end])
    median = statistics.median((a + b) / 2 for a, b in usable)
    region = min(regions, key=lambda r: (0 if r[0] <= median <= r[1] else min(abs(median-r[0]), abs(median-r[1])), -(r[1]-r[0])))
    estimate = min(max(median, region[0]), region[1])
    return {"method": "maximum overlapping presentation-offset intervals",
            "offset_range_lower_ms": region[0], "offset_range_upper_ms": region[1],
            "estimated_offset_ms": estimate, "contributing_frames": len(usable),
            "maximum_consistent_frames": maximum, "maximum_consistent_pct": 100 * maximum / len(usable)}


class RecordingAnalyzer:
    """Read saved evidence by default; load QReader only for an explicit fresh scan."""

    def __init__(self, folder: Path, intrinsics: Path = DEFAULT_INTRINSICS, reader=None):
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise ValueError("Select an existing calibration recording folder")
        journal = next((self.folder / name for name in CAMERA_JOURNALS if (self.folder / name).is_file()), None)
        if journal is None:
            raise ValueError("Recording has no camera timestamp journal")
        self.rows, seen = [], set()
        for row in read_json_rows(journal):
            name = row.get("frame") or row.get("camera_frame")
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("Camera journal contains missing or duplicate frame names")
            if resolve_recording_file(self.folder, name, IMAGE_DIRECTORY_NAME) is None:
                raise ValueError("Missing image or image outside recording folder: " + name)
            self.rows.append({**row, "filename": name})
            seen.add(name)
        if not self.rows:
            raise ValueError("Camera timestamp journal is empty")
        self.timeline = DisplayTimeline(self.folder)
        self.grid_qrs, self.cell_names = self.timeline.grid_qrs, self.timeline.cell_names
        self.cell_positions = self.timeline.cell_positions
        session_path = self.folder / "camera_timing_session.json"
        self.session = json.loads(session_path.read_text(encoding="utf-8")) if session_path.is_file() else {}
        self.epochs = {row.get("stream_epoch"): row for row in self.session.get("epochs", []) if row.get("stream_epoch") is not None}
        if not Path(intrinsics).is_file():
            raise ValueError("Camera intrinsics file does not exist: " + str(intrinsics))
        self.intrinsics = Path(intrinsics).resolve()
        self.undistorter = Undistorter(self.intrinsics)
        self.reader = reader
        self._reader_was_provided = reader is not None
        self.cache: OrderedDict[tuple[int, float], dict] = OrderedDict()
        self.cache_limit = DETECTION_BATCH_SIZE * 2
        self.manual_values = {}
        geometry_path = self.folder / "analysis_screen_geometry.json"
        self.screen_geometry_config = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path.is_file() else None
        if self.screen_geometry_config is not None and not isinstance(self.screen_geometry_config, dict):
            raise ValueError("analysis_screen_geometry.json must contain an object")
        self.saved_frames = {}
        self.saved_analysis_alpha = 0.25
        self.saved_analysis_notice = "No saved QR results. Select GO to decode the recording."
        self.saved_analysis_complete = False
        self.review_saved = False
        self.fresh_analysis_requested = False
        if self.output_folder.joinpath("calibration_analysis.json").is_file():
            self._read_saved_analysis()

    def _decode_image_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        visual_batches = [detect_contrast_cells(image, self.grid_qrs) for image in images]
        if self._reader_was_provided:
            decode_indices = list(range(len(images)))
        else:
            decode_indices = [index for index, visual in enumerate(visual_batches) if visual]
        decoded_batches = [[] for _ in images]
        if decode_indices:
            if self.reader is None:
                self.reader = create_qreader()
            selected = [images[index] for index in decode_indices]
            decoded = decode_qrs_with_grid_retries_batch(
                self.reader, selected, self.grid_qrs, DETECTION_BATCH_SIZE
            )
            for index, detections in zip(decode_indices, decoded):
                decoded_batches[index] = detections
        for image, decoded, visual in zip(images, decoded_batches, visual_batches):
            height, width = image.shape[:2]
            decoded_cells = {
                cell_index_for(item["center"], (width, height), self.grid_qrs)
                for item in decoded
            }
            decoded.extend(item for item in visual if item["cell"] not in decoded_cells)
        return decoded_batches

    @property
    def output_folder(self) -> Path:
        return self.folder.with_name(self.folder.name + "_analysis")

    def _read_saved_analysis(self) -> None:
        path = self.output_folder / "calibration_analysis.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("grid", {}).get("qr_count", self.grid_qrs) != self.grid_qrs:
                raise ValueError("saved QR grid differs from this recording")
            alpha = float(report.get("analysis_alpha", 0.25))
            rows = report.get("frames")
            if not np.isfinite(alpha) or not 0 <= alpha <= 1 or not isinstance(rows, list):
                raise ValueError("saved report has an invalid analysis setting or frame list")
            names = {row["filename"] for row in self.rows}
            saved = {}
            for row in rows:
                filename, values = row.get("filename"), row.get("qr_values_ms")
                if filename not in names:
                    continue
                if (filename in saved or not isinstance(values, list) or len(values) != self.grid_qrs
                        or any(value is not None and (not isinstance(value, str) or len(value) != 12 or not value.isdigit()) for value in values)):
                    raise ValueError(f"invalid or duplicate saved values for {filename}")
                saved[filename] = row
            self.saved_frames, self.saved_analysis_alpha, self.review_saved = saved, alpha, True
            complete = (
                len(saved) == len(self.rows)
                and report.get("processed", len(saved)) == len(self.rows)
                and not report.get("cancelled", False)
                and report.get("stopped") is None
            )
            self.saved_analysis_complete = complete
            if complete:
                self.saved_analysis_notice = (
                    f"Complete decoded results found in {self.output_folder} "
                    f"({len(saved)} / {len(self.rows)} frames). No need to decode again; "
                    "GO is only for a fresh rerun."
                )
            else:
                self.saved_analysis_notice = (
                    f"Saved decoded results: {len(saved)} / {len(self.rows)} frames. "
                    "Review is read only; GO starts a fresh decode."
                )
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
            self.saved_analysis_complete = False
            self.saved_analysis_notice = f"Could not load saved analysis: {error}. GO starts a fresh decode."

    def _load_frame(self, index: int, alpha: float):
        row = self.rows[index]
        path = resolve_recording_file(self.folder, row["filename"], IMAGE_DIRECTORY_NAME)
        original = cv2.imread(str(path)) if path else None
        if original is None:
            raise ValueError("Could not read " + row["filename"])
        return row, original, self.undistorter.image(original, alpha)

    def pts_monotonic_ns(self, row: dict) -> int | None:
        direct = row.get("media_monotonic_ns")
        if direct is not None:
            return int(direct)
        epoch = self.epochs.get(row.get("stream_epoch"))
        running = row.get("running_time_ns")
        if epoch and running is not None and epoch.get("pipeline_zero_monotonic_ns") is not None:
            return int(epoch["pipeline_zero_monotonic_ns"]) + int(running)
        return None

    def pts_monotonic_for_value(self, row: dict, pts_ns: int | None) -> int | None:
        if pts_ns == row.get("pts_ns"):
            return self.pts_monotonic_ns(row)
        return None

    def _finish_analysis(self, index: int, row: dict, original: np.ndarray,
                         undistorted: np.ndarray, decoded: list[dict], alpha: float,
                         *, cache_result: bool = True) -> dict:
        size = (undistorted.shape[1], undistorted.shape[0])
        detections = order_by_cell(decoded, size, self.grid_qrs)
        geometry = screen_geometry(self.screen_geometry_config, row["filename"], size, alpha)
        reference = self.pts_monotonic_ns(row)
        observations = []
        for detection in detections:
            cell = detection["cell"]
            screen_index = screen_cell(detection["center"], geometry, self.cell_positions)
            if screen_index is not None:
                detection["cell"] = screen_index
            x1, y1, x2, y2 = detection["bbox"]
            corners = np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=float)
            original_points = self.undistorter.to_original(corners, size, alpha)
            marker = self.timeline.match(detection.get("raw"), reference)
            mismatch = bool(marker is not None and marker["cell"] != detection["cell"])
            status, issues = self.timeline.marker_status(marker["index"]) if marker else ("Unmatched", ["no_display_journal_match"])
            if mismatch:
                status, issues = "Grid cell mismatch", ["decoded_cell_does_not_match_journal", *issues]
            observations.append({
                **detection, "detected_cell": cell, "screen_cell": screen_index,
                "position_basis": "screen_geometry" if screen_index is not None else "camera_image_grid",
                "marker": marker, "display_index": marker["index"] if marker else None,
                "status": status, "timing_status": status if not mismatch else self.timeline.marker_status(marker["index"])[0],
                "issues": issues, "mismatch": mismatch,
                "offset_ms": (reference - marker["marker_ns"]) / 1e6 if reference is not None and marker else None,
                "undistorted_points": corners, "original_points": original_points,
            })
        latest = max((item for item in observations if item["display_index"] is not None),
                     key=lambda item: item["display_index"], default=None)
        result = {"index": index, "row": row, "original": original, "undistorted": undistorted,
                  "observations": observations,
                  "qr_evidence": detection_evidence(observations, size, geometry),
                  "latest": latest, "pts_monotonic_ns": reference}
        if cache_result:
            self.cache[(index, alpha)] = result
            self.cache.move_to_end((index, alpha))
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        return result

    def _confirm_temporal_successor(self, previous: dict, following: dict) -> list[dict]:
        """Infer a visible unreadable successor only when the next frame confirms its track."""
        if following.get("index") != previous.get("index", -2) + 1:
            return []
        if previous.get("index") in self.manual_values or following.get("index") in self.manual_values:
            return []
        if any(previous.get("row", {}).get(key) != following.get("row", {}).get(key)
               for key in ("stream_epoch", "mapping_revision", "segment_epoch")):
            return []
        old_evidence = previous.get("qr_evidence") or {}
        new_evidence = following.get("qr_evidence") or {}
        if old_evidence.get("image_size") != new_evidence.get("image_size"):
            return []
        old_rows, new_rows = old_evidence.get("detections", []), new_evidence.get("detections", [])
        if not old_rows or not new_rows:
            return []

        old_groups: dict[int, list[tuple[int, dict]]] = {}
        new_groups: dict[int, list[tuple[int, dict]]] = {}
        for index, row in enumerate(old_rows):
            old_groups.setdefault(int(row.get("group", index)), []).append((index, row))
        for index, row in enumerate(new_rows):
            new_groups.setdefault(int(row.get("group", index)), []).append((index, row))

        def identities(groups: dict[int, list[tuple[int, dict]]]) -> dict[int, list[tuple[int, dict]]]:
            found: dict[int, list[tuple[int, dict]]] = {}
            for members in groups.values():
                indices = {int(row["display_index"]) for _, row in members
                           if row.get("display_index") is not None}
                if len(indices) == 1:
                    found.setdefault(next(iter(indices)), []).append(members)
            return found

        old_ids, new_ids = identities(old_groups), identities(new_groups)
        old_direct = set(old_ids)
        if not old_direct:
            return []
        latest = max(old_direct)
        successor = latest + 1
        if latest < 1 or latest - 1 not in old_direct or successor >= len(self.timeline.frames):
            return []
        if len(old_ids.get(latest, [])) != 1 or len(old_ids.get(latest - 1, [])) != 1:
            return []
        if len(new_ids.get(latest, [])) != 1 or len(new_ids.get(successor, [])) != 1:
            return []

        def position(members: list[tuple[int, dict]]) -> tuple[str, int] | None:
            positions = set()
            for _, row in members:
                if isinstance(row.get("screen_cell"), int):
                    positions.add(("screen_geometry", row["screen_cell"]))
                elif (row.get("position_basis") == "camera_image_grid"
                      and isinstance(row.get("detected_cell"), int)):
                    positions.add(("camera_image_grid", row["detected_cell"]))
            return next(iter(positions)) if len(positions) == 1 else None

        def best_overlap(first: list[tuple[int, dict]], second: list[tuple[int, dict]]) -> float:
            return max((_bbox_iou(a["bbox"], b["bbox"])
                        for _, a in first for _, b in second
                        if isinstance(a.get("bbox"), list) and isinstance(b.get("bbox"), list)
                        and len(a["bbox"]) == len(b["bbox"]) == 4), default=0.0)

        old_predecessor = old_ids[latest][0]
        new_predecessor = new_ids[latest][0]
        new_successor = new_ids[successor][0]
        predecessor_cell = self.timeline.frames[latest]["cell"]
        successor_cell = self.timeline.frames[successor]["cell"]
        predecessor_position = position(old_predecessor)
        if (predecessor_position is None
                or predecessor_position != position(new_predecessor)
                or predecessor_position[1] != predecessor_cell
                or position(new_successor) != (predecessor_position[0], successor_cell)
                or best_overlap(old_predecessor, new_predecessor) < 0.35):
            return []

        unreadable = set(old_evidence.get("unreadable_groups", []))
        conflicting = set(old_evidence.get("conflicting_groups", []))
        candidate_groups = []
        expected_position = (predecessor_position[0], successor_cell)
        for group_id in unreadable - conflicting:
            members = old_groups.get(int(group_id), [])
            if (not members or any(row.get("raw") is not None or row.get("clipped") for _, row in members)
                    or position(members) != expected_position):
                continue
            candidate_groups.append(members)
        if len(candidate_groups) != 1 or len(new_ids.get(successor, [])) != 1:
            return []
        candidate = candidate_groups[0]
        if best_overlap(candidate, new_successor) < 0.35:
            return []

        marker = self.timeline.frames[successor]
        expected_raw = timestamp_payload(marker["marker_ns"])
        confirmed_raws = {row.get("raw") for _, row in new_successor
                          if row.get("display_index") == successor and row.get("raw") is not None}
        if confirmed_raws != {expected_raw}:
            return []

        best_candidate_index, best_candidate = max(
            candidate, key=lambda item: float(item[1].get("confidence", 0.0))
        )
        inference = {
            "status": "confirmed_by_next_frame", "method": "increasing_sequence_and_spatial_track",
            "raw": expected_raw, "display_index": successor, "marker_ns": marker["marker_ns"],
            "cell": successor_cell, "detected_cell": best_candidate.get("detected_cell"),
            "position_basis": best_candidate.get("position_basis"),
            "source_frame_number": previous["index"] + 1,
            "confirmation_frame_number": following["index"] + 1,
            "unreadable_group": int(best_candidate.get("group", -1)),
            "candidate_detection_indices": [index for index, _ in candidate],
            "candidate_detection_index": best_candidate_index,
            "candidate_bbox_iou": best_overlap(candidate, new_successor),
            "minimum_bbox_iou": 0.35,
            "supporting_readable_display_indices": [latest - 1, latest],
            "predecessor_display_index": latest,
            "predecessor_cell": predecessor_cell,
            "predecessor_bbox_iou": best_overlap(old_predecessor, new_predecessor),
            "confirmed_raw": expected_raw,
            "candidate_original_points": best_candidate.get("original_points"),
            "candidate_bbox": best_candidate.get("bbox"),
        }
        if (0 <= best_candidate_index < len(previous.get("observations", []))):
            points = previous["observations"][best_candidate_index].get("undistorted_points")
            inference["candidate_undistorted_points"] = (
                np.asarray(points, dtype=float).tolist() if points is not None else None
            )
        return [inference]

    def inspect(self, index: int, alpha: float = 0.25) -> dict:
        if not 0 <= index < len(self.rows):
            raise IndexError(index)
        row, original, undistorted = self._load_frame(index, alpha)
        saved = self.saved_frames.get(row["filename"]) if self.review_saved else None
        evidence = (saved or {}).get("qr_evidence") or {}
        boxes_match = (saved is not None and abs(alpha - self.saved_analysis_alpha) < 1e-9
                       and evidence.get("image_size") == [original.shape[1], original.shape[0]])
        decoded = []
        if not self.review_saved and (self.reader is not None or self.fresh_analysis_requested):
            decoded = self._decode_image_batch([undistorted])[0]
        elif boxes_match:
            for item in evidence.get("detections", []):
                box = np.asarray(item.get("bbox"), dtype=float)
                if box.shape == (4,) and np.isfinite(box).all():
                    decoded.append({"raw": item.get("raw"), "bbox": box,
                                    "center": ((box[0]+box[2])/2, (box[1]+box[3])/2),
                                    "confidence": item.get("confidence", 0.0),
                                    "detection_method": item.get("detection_method", "qr_reader")})
        result = self._finish_analysis(index, row, original, undistorted, decoded, alpha, cache_result=False)
        if self.review_saved:
            result["saved_values"] = {
                "pts_ns": saved.get("pts_ns", row.get("pts_ns")) if saved else row.get("pts_ns"),
                "ntp_ns": saved.get("ntp_ns", row.get("reference_ntp_ns")) if saved else row.get("reference_ntp_ns"),
                "qrs": tuple(saved["qr_values_ms"]) if saved else (None,) * self.grid_qrs,
                "manual": bool(saved and saved.get("manual_values")),
            }
            result["review_notice"] = ("Saved QR results — no decoding performed." if saved else
                                       "No saved QR results for this frame. GO starts a fresh decode.")
            if saved and not boxes_match:
                result["review_notice"] += " Saved boxes are unavailable or do not match the current image/undistortion setting."
            result["qr_evidence"] = evidence
            result["temporal_inferences"] = list((saved or {}).get("temporal_inferences", []))
        return result

    def begin_fresh_analysis(self) -> None:
        self.review_saved = False
        self.saved_analysis_complete = False
        self.fresh_analysis_requested = True
        self.cache.clear()

    def _analysis_batches(self, alpha: float, cancel: threading.Event,
                          parallel_scan: bool):
        batch_size = PARALLEL_FRAME_WORKERS if parallel_scan else DETECTION_BATCH_SIZE
        if parallel_scan:
            with ProcessPoolExecutor(
                max_workers=PARALLEL_FRAME_WORKERS,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_decode_process,
                initargs=(str(self.intrinsics), self.grid_qrs),
            ) as executor:
                for start in range(0, len(self.rows), batch_size):
                    if cancel.is_set():
                        break
                    yield self.analyze_batch(
                        range(start, min(start + batch_size, len(self.rows))), alpha, executor
                    )
        else:
            for start in range(0, len(self.rows), batch_size):
                if cancel.is_set():
                    break
                yield self.analyze_batch(
                    range(start, min(start + batch_size, len(self.rows))), alpha
                )

    def analyze_batch(self, indices, alpha: float = 0.25, executor=None) -> list[dict]:
        indices = tuple(indices)
        missing = [index for index in indices if (index, alpha) not in self.cache]
        if missing:
            if executor is None:
                loaded = [self._load_frame(index, alpha) for index in missing]
                batches = self._decode_image_batch([frame[2] for frame in loaded])
            else:
                tasks = []
                for index in missing:
                    row = self.rows[index]
                    image_path = resolve_recording_file(
                        self.folder, row["filename"], IMAGE_DIRECTORY_NAME
                    )
                    if image_path is None:
                        raise ValueError(
                            "Missing image or image outside recording folder: " + row["filename"]
                        )
                    tasks.append((index, str(image_path), alpha))
                decoded_by_index = dict(executor.map(_decode_recording_frame, tasks))
                loaded = [self._load_frame(index, alpha) for index in missing]
                batches = [decoded_by_index[index] for index in missing]
            for index, (row, original, image), decoded in zip(missing, loaded, batches):
                self._finish_analysis(index, row, original, image, decoded, alpha)
        return [self.cache[(index, alpha)] for index in indices]

    def analyze(self, index: int, alpha: float = 0.25) -> dict:
        return self.analyze_batch((index,), alpha)[0]

    def detected_qr_values(self, result: dict) -> tuple[str | None, ...]:
        cells = [[] for _ in range(self.grid_qrs)]
        for item in result["observations"]:
            if item.get("raw") is not None:
                cell = item.get("marker", {}).get("cell", item["cell"])
                cells[int(cell)].append(item)
        values = []
        for items in cells:
            matched = [item for item in items if item.get("display_index") is not None]
            selected = max(matched, key=lambda item: item["display_index"]) if matched else (items[0] if len(items) == 1 else None)
            values.append(selected["raw"] if selected else None)
        return tuple(values)

    def frame_values(self, result: dict) -> dict:
        if result["index"] in self.manual_values:
            return self.manual_values[result["index"]]
        if "saved_values" in result:
            return result["saved_values"]
        row = result["row"]
        return {"pts_ns": row.get("pts_ns"), "ntp_ns": row.get("reference_ntp_ns"),
                "qrs": self.detected_qr_values(result), "manual": False}

    def set_manual_values(self, index: int, values: dict) -> None:
        if len(values.get("qrs", ())) != self.grid_qrs:
            raise ValueError(f"Manual QR values must contain {self.grid_qrs} grid cells")
        self.manual_values[index] = {**values, "manual": True}

    def reset_manual_values(self, index: int) -> None:
        self.manual_values.pop(index, None)

    def check_frame(self, result: dict) -> dict:
        values = self.frame_values(result)
        reference = self.pts_monotonic_for_value(result["row"], values.get("pts_ns"))
        arrival = result["row"].get("application_arrival_monotonic_ns", result["row"].get("received_monotonic_ns"))
        candidates, ignored = [], []
        source = ([{"cell": cell, "raw": raw} for cell, raw in enumerate(values["qrs"]) if raw is not None]
                  if values.get("manual") or "saved_values" in result else result["observations"])
        for item in source:
            raw, cell = item.get("raw"), item["cell"]
            if raw is None:
                continue
            if not isinstance(raw, str) or len(raw) != 12 or not raw.isdigit():
                ignored.append({"cell": cell, "raw": raw, "reason": "invalid_payload"})
                continue
            marker = self.timeline.match(raw, reference)
            if marker is None:
                ignored.append({"cell": cell, "raw": raw, "reason": "not_in_display_journal"})
                continue
            if values.get("manual") and marker["cell"] != cell:
                ignored.append({"cell": cell, "raw": raw, "reason": "cell_mismatch"})
                continue
            candidates.append({"cell": marker["cell"], "detected_cell": cell, "raw": raw,
                               "marker": marker, "inferred": False})
        inferences = [] if values.get("manual") else list(result.get("temporal_inferences", []))
        inferred_candidates = []
        for inference in inferences:
            index = inference.get("display_index")
            raw = inference.get("raw")
            if not isinstance(index, int) or not 0 <= index < len(self.timeline.frames):
                continue
            marker = self.timeline.frames[index]
            if (self.timeline.match(raw, reference) is not marker
                    or marker["cell"] != inference.get("cell")):
                continue
            inferred_candidates.append({
                "cell": marker["cell"], "detected_cell": inference.get("cell"),
                "raw": raw, "marker": marker, "inferred": True, "inference": inference,
            })
        candidates.extend(inferred_candidates)
        base = {"values": values, "decode_issues": ignored, "ignored_readable_qrs": len(ignored),
                "camera_reference_monotonic_ns": reference,
                "received_monotonic_ns": int(arrival) if arrival is not None else None,
                "temporal_inferences": inferences}
        if not candidates:
            return {**base, "valid": False, "skippable": True,
                    "reason": "No readable QR matched the display journal", "indices": [],
                    "inferred_indices": [], "matched_readable_qrs": 0,
                    "latest_is_inferred": False}
        latest = max(candidates, key=lambda item: item["marker"]["index"])
        marker = latest["marker"]
        warnings = [{"raw": item["raw"], "detected_cell": item["detected_cell"],
                     "journal_cell": item["cell"], "reason": "camera_grid_position_differs_from_journal"}
                    for item in candidates if item["detected_cell"] != item["cell"]]
        status, issues = self.timeline.marker_status(marker["index"])
        interval = self.timeline.observation_interval(marker["index"], reference)
        arrival_interval = self.timeline.observation_interval(marker["index"], base["received_monotonic_ns"])
        if interval["boundary_timing_status"] == "Timing suspect":
            status = "Timing suspect"
        elif interval["boundary_timing_status"] == "Unknown" and status == "Clean":
            status = "Unknown"
        return {**base, "valid": True, "skippable": False, "reason": None,
                "indices": sorted({item["marker"]["index"] for item in candidates if not item["inferred"]}),
                "inferred_indices": sorted({item["marker"]["index"] for item in candidates if item["inferred"]}),
                "matched_readable_qrs": sum(not item["inferred"] for item in candidates),
                "position_warnings": warnings,
                "latest_cell": latest["cell"], "latest_raw": latest["raw"], "latest_marker": marker,
                "latest_is_inferred": bool(latest["inferred"]),
                "latest_inference": latest.get("inference"),
                "timing_status": status, "issues": [*issues, *interval["boundary_timing_issues"]],
                "offset_ms": (reference - marker["marker_ns"]) / 1e6 if reference is not None else None,
                "presentation_interval": interval, "arrival_presentation_interval": arrival_interval}

    def _annotate_interval_analysis(self, reports: list[dict]) -> dict:
        intervals = [(row["offset_interval_lower_ms"], row["offset_interval_upper_ms"])
                     for row in reports if row.get("validation") == "accepted_clean"
                     and row.get("offset_interval_lower_ms") is not None
                     and row.get("offset_interval_upper_ms") is not None]
        primary = _maximum_interval_consensus(intervals)
        classifications = Counter()
        previous = None
        for row in reports:
            reference, observed = row.get("camera_reference_monotonic_ns"), row.get("latest_display_index")
            estimate = primary["estimated_offset_ms"] if primary else None
            exposure = reference - round(estimate * 1e6) if reference is not None and estimate is not None else None
            expected = self.timeline.active_index_at(exposure) if exposure is not None else None
            events = self.timeline.events_between(previous, exposure) if previous is not None and exposure is not None else []
            if exposure is not None:
                previous = exposure
            visible = (list(range(max(0, expected - self.timeline.visible_qrs + 1), expected + 1))
                       if expected is not None else [])
            after = before = None
            expected_issues = []
            margin = float(row.get("software_transition_margin_ms") or 1.0)
            if expected is None:
                classification = "unavailable"
            elif observed is None:
                classification = "no_readable_qr"
            else:
                expected_ns = self.timeline.presentation_times[expected]
                after = (exposure - expected_ns) / 1e6
                if expected + 1 < len(self.timeline.presentation_times):
                    before = (self.timeline.presentation_times[expected + 1] - exposure) / 1e6
                expected_issues = timing_issues(self.timeline.frames[expected])
                if observed == expected:
                    nearest = min(value for value in (after, before) if value is not None)
                    classification = "expected_flip_boundary" if nearest <= margin else "stable_expected"
                elif observed == expected - 1:
                    classification = "expected_flip_transition" if abs(exposure - expected_ns) <= margin * 1e6 else "stale_after_expected_flip"
                elif observed == expected + 1:
                    boundary = self.timeline.presentation_times[observed]
                    classification = "expected_flip_transition" if abs(exposure - boundary) <= margin * 1e6 else "future_before_expected_flip"
                elif observed < expected:
                    classification = "stale_after_expected_flip"
                else:
                    classification = "future_before_expected_flip"
            row.update({
                "estimated_exposure_monotonic_ns": exposure,
                "expected_display_index": expected,
                "expected_visible_display_indices": visible,
                "display_events_since_previous_camera": events,
                "presentation_classification": classification,
                "phase_after_presentation_ms": after,
                "phase_before_next_presentation_ms": before,
                "expected_presentation_timing_issues": expected_issues,
            })
            classifications[classification] += 1
        return {"method": "conditional intervals from consecutive software presentation returns",
                "primary_reference": "host-anchored segment-running-time camera reference",
                "primary_pts": primary,
                "arrival_reference": "application arrival is diagnostic and includes transport and buffering",
                "arrival_diagnostic": None, "classification_counts": dict(classifications),
                "physical_boundary": "No physical scanout, photon, exposure, or rolling-shutter measurement is available."}

    def summarize(self, alpha: float, cancel: threading.Event, progress) -> dict:
        parallel_scan = self.reader is None
        self.begin_fresh_analysis()
        reports, counts, readable = [], Counter(), Counter()
        clean_offsets, stopped = [], None

        def add_result(result: dict) -> None:
            check = self.check_frame(result)
            observations, values = result["observations"], check["values"]
            inferences = check.get("temporal_inferences", [])
            counts["detections"] += len(observations)
            counts["unreadable"] += sum(item.get("raw") is None for item in observations)
            counts["mismatches"] += sum(bool(item.get("mismatch")) for item in observations)
            matched = check.get("matched_readable_qrs", 0)
            counts["journal_matched_readable"] += matched
            counts["temporal_inferred_qrs"] += len(inferences)
            if inferences:
                counts["frames_with_temporal_inference"] += 1
            readable[matched] += 1
            interval = check.get("presentation_interval") or {}
            arrival_interval = check.get("arrival_presentation_interval") or {}
            valid = check["valid"]
            suffix = check["timing_status"].lower().replace(" ", "_") if valid else None
            validation = (("accepted_inferred_" if check.get("latest_is_inferred") else "accepted_") + suffix
                          if valid else "skipped_no_readable_qr")
            inferred_values = [None] * self.grid_qrs
            for inference in inferences:
                cell = inference.get("cell")
                if isinstance(cell, int) and 0 <= cell < self.grid_qrs:
                    inferred_values[cell] = inference.get("raw")
            report_evidence = dict(result["qr_evidence"])
            if inferences:
                report_evidence["temporal_inferences"] = inferences
            report = {
                "frame_number": result["index"] + 1, "filename": result["row"]["filename"],
                "validation": validation, "reason": check.get("reason"),
                "manual_values": bool(values.get("manual")), "qr_evidence": report_evidence,
                "pts_ns": values.get("pts_ns"), "ntp_ns": values.get("ntp_ns"),
                "grid_qrs": self.grid_qrs, "qr_values_ms": list(values["qrs"]),
                "inferred_qr_values_ms": inferred_values,
                "temporal_inferences": inferences,
                "decoded_detections": len(observations), "matched_readable_qrs": matched,
                "ignored_readable_qrs": check.get("ignored_readable_qrs", 0),
                "decode_issues": check.get("decode_issues", []),
                "position_warnings": check.get("position_warnings", []),
                "display_indices": check.get("indices", []),
                "inferred_display_indices": check.get("inferred_indices", []),
                "latest_qr_ms": check.get("latest_raw"), "latest_cell": check.get("latest_cell"),
                "latest_cell_name": self.cell_names[check["latest_cell"]] if check.get("latest_cell") is not None else None,
                "latest_display_index": check["latest_marker"]["index"] if check.get("latest_marker") else None,
                "latest_readable_display_index": max(check.get("indices", []), default=None),
                "latest_is_inferred": check.get("latest_is_inferred", False),
                "latest_inference": check.get("latest_inference"),
                "latest_value_source": "next_frame_confirmation" if check.get("latest_is_inferred") else "decoded",
                "timing_status": check.get("timing_status"), "timing_issues": check.get("issues", []),
                "pts_minus_latest_qr_ms": check.get("offset_ms"),
                "camera_reference_monotonic_ns": check.get("camera_reference_monotonic_ns"),
                "received_monotonic_ns": check.get("received_monotonic_ns"),
                "active_interval_start_ns": interval.get("active_interval_start_ns"),
                "active_interval_end_ns": interval.get("active_interval_end_ns"),
                "offset_interval_lower_ms": interval.get("offset_interval_lower_ms"),
                "offset_interval_upper_ms": interval.get("offset_interval_upper_ms"),
                "offset_interval_width_ms": interval.get("offset_interval_width_ms"),
                "software_transition_margin_ms": interval.get("software_transition_margin_ms"),
                "presentation_boundary_timing_status": interval.get("boundary_timing_status"),
                "presentation_boundary_timing_issues": interval.get("boundary_timing_issues", []),
                "current_presentation": interval.get("current_presentation"),
                "next_presentation": interval.get("next_presentation"),
                "arrival_offset_interval_lower_ms": arrival_interval.get("offset_interval_lower_ms"),
                "arrival_offset_interval_upper_ms": arrival_interval.get("offset_interval_upper_ms"),
            }
            report["evidence_assessment"] = assess_evidence(
                report, report["display_indices"],
                transition=bool(report["display_indices"] and
                                max(report["display_indices"]) - min(report["display_indices"]) >= self.timeline.visible_qrs))
            if check.get("latest_is_inferred"):
                report["evidence_assessment"]["warnings"].append(
                    "latest_qr_identity_inferred_from_next_frame")
            if self.grid_qrs == 4:
                report.update(dict(zip(("qr_top_left_ms", "qr_top_right_ms", "qr_bottom_right_ms", "qr_bottom_left_ms"), values["qrs"])))
                report["latest_quadrant"] = report["latest_cell_name"]
            reports.append(report)
            progress(len(reports), len(self.rows), result)
            if valid:
                counts["accepted_frames"] += 1
                if (check["timing_status"] == "Clean" and check.get("offset_ms") is not None
                        and not check.get("latest_is_inferred")):
                    clean_offsets.append(check["offset_ms"])
            elif matched == 0:
                counts["frames_without_readable_qr"] += 1

        pending_result = None
        for batch in self._analysis_batches(alpha, cancel, parallel_scan):
            for result in batch:
                if cancel.is_set():
                    break
                if pending_result is not None:
                    inferences = self._confirm_temporal_successor(pending_result, result)
                    if inferences:
                        pending_result["temporal_inferences"] = inferences
                        pending_result["qr_evidence"]["temporal_inferences"] = inferences
                if pending_result is not None:
                    add_result(pending_result)
                pending_result = result
            if cancel.is_set():
                break
        if pending_result is not None:
            add_result(pending_result)

        temporal_inputs = [{"latest_display_index": row.get("latest_display_index"),
                            "media_reference_monotonic_ns": row.get("camera_reference_monotonic_ns"),
                            "segment": tuple(self.rows[row["frame_number"]-1].get(key) for key in
                                             ("stream_epoch", "mapping_revision", "segment_epoch"))}
                           for row in reports]
        for row, check in zip(reports, repeated_state_checks(temporal_inputs, self.timeline.presentation_times, self.timeline.paused)):
            row["temporal_evidence"] = check
            if check is not None:
                was_accepted = row["validation"].startswith("accepted")
                row["validation"], row["reason"] = "skipped_suspected_stale_visual_state", check["reason"]
                row["evidence_assessment"] = assess_evidence(row, row["display_indices"])
                counts["frames_suspected_stale_visual_state"] += 1
                if was_accepted:
                    counts["accepted_frames"] -= 1
        clean_offsets = [row["pts_minus_latest_qr_ms"] for row in reports
                         if row["validation"] == "accepted_clean" and row.get("pts_minus_latest_qr_ms") is not None]
        inferred_offsets = [row["pts_minus_latest_qr_ms"] for row in reports
                            if row["validation"] == "accepted_inferred_clean"
                            and row.get("pts_minus_latest_qr_ms") is not None]
        interval_analysis = self._annotate_interval_analysis(reports)
        return {
            "recording_directory": str(self.folder), "analysis_alpha": alpha,
            "detection_batch_size": DETECTION_BATCH_SIZE,
            "parallel_frame_workers": PARALLEL_FRAME_WORKERS if parallel_scan else 1,
            "processed": len(reports), "total": len(self.rows),
            "cancelled": cancel.is_set() and stopped is None, "stopped": stopped,
            "counts": dict(counts), "clean_offsets": len(clean_offsets),
            "median_offset_ms": statistics.median(clean_offsets) if clean_offsets else None,
            "temporally_inferred_clean_offsets": len(inferred_offsets),
            "median_temporally_inferred_offset_ms": statistics.median(inferred_offsets) if inferred_offsets else None,
            "temporal_inference_policy": {
                "requires_two_consecutive_readable_predecessors": True,
                "requires_stable_predecessor_and_successor_cells": True,
                "minimum_cross_frame_bbox_iou": 0.35,
                "raw_decoded_values_are_preserved": True,
                "inferred_offsets_are_excluded_from_direct_read_consensus": True,
            },
            "readability": {"frames_by_matched_readable_qr_count": {str(key): readable[key] for key in sorted(readable)},
                            "mean_matched_readable_qrs_per_frame": sum(k*v for k,v in readable.items()) / len(reports) if reports else 0,
                            "maximum_matched_readable_qrs_in_frame": max(readable, default=0)},
            "grid": {"qr_count": self.grid_qrs, "rows": self.timeline.grid_rows,
                     "columns": self.timeline.grid_columns, "visible_qrs": self.timeline.visible_qrs,
                     "cell_order": list(self.cell_names)},
            "display": self.timeline.totals(), "presentation_interval_analysis": interval_analysis,
            "frames": reports,
        }

    def save_report(self, report: dict) -> dict:
        output = self.output_folder
        output.mkdir(parents=True, exist_ok=True)
        saved = {**report, "generated_at": datetime.now().astimezone().isoformat(),
                 "output_directory": str(output),
                 "report_files": ["calibration_analysis.json", "calibration_frames.csv", PRESENTATIONS_CSV],
                 "presentation_timeline_file": PRESENTATIONS_CSV}
        files = {"calibration_analysis.json": json.dumps(saved, indent=2, ensure_ascii=False) + "\n"}
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            temp = Path(temporary)
            for name, content in files.items():
                (temp / name).write_text(content, encoding="utf-8")
            rows = saved["frames"]
            if rows:
                import csv
                with (temp / "calibration_frames.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                                      for key, value in row.items()} for row in rows)
            else:
                (temp / "calibration_frames.csv").write_text("", encoding="utf-8")
            events = [self.timeline.presentation_event(index) for index in range(len(self.timeline.frames))]
            import csv
            with (temp / PRESENTATIONS_CSV).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(events[0]))
                writer.writeheader()
                writer.writerows({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                                  for key, value in row.items()} for row in events)
            for path in temp.iterdir():
                path.replace(output / path.name)
        return saved


class AnalysisWorker(threading.Thread):
    """Load and analyze frames away from Tk's event loop."""

    def __init__(self, model: RecordingAnalyzer):
        super().__init__(daemon=True, name="qr-calibration-analysis")
        self.model = model
        self.jobs: Queue = Queue()
        self.results: Queue = Queue()
        self.cancel = threading.Event()

    def run(self) -> None:
        while True:
            kind, request, payload = self.jobs.get()
            if kind == "stop":
                return
            try:
                if kind == "frame":
                    result = self.model.inspect(payload["index"], payload["alpha"])
                    self.results.put((kind, request, result))
                elif kind == "scan":
                    report = self.model.summarize(
                        payload["alpha"],
                        self.cancel,
                        lambda done, total, frame: self.results.put(
                            ("progress", request, (done, total, frame))
                        ),
                    )
                    self.results.put(("complete", request, self.model.save_report(report)))
            except Exception as error:
                self.results.put(("error", request, str(error)))

    def submit(self, kind: str, request: int, **payload) -> None:
        if kind == "frame":
            self.cancel.set()
        elif kind == "scan":
            self.cancel.clear()
        self.jobs.put((kind, request, payload))

    def stop(self) -> None:
        self.cancel.set()
        self.jobs.put(("stop", 0, {}))


class CalibrationWindow:
    """Review frames and show the timing evidence as the recording is decoded."""

    def __init__(self, root: tk.Tk, model: RecordingAnalyzer, alpha: float = 0.25):
        self.root = root
        self.model = model
        self.worker = AnalysisWorker(model)
        self.worker.start()
        self.index = 0
        self.request = 0
        self.scan_request = 0
        self.scan_active = False
        self.current = None
        self.current_check = None
        self.photo = None
        self.resize_job = None
        self.saved_output: Path | None = None
        initial_alpha = model.saved_analysis_alpha if model.review_saved else alpha
        self.alpha = tk.StringVar(value=f"{initial_alpha:g}")
        self.variant = tk.StringVar(value="Undistorted")
        self.position = tk.StringVar(value="1")
        self.title = tk.StringVar(value="Loading first frame…")
        self.pts_edit = tk.StringVar()
        self.ntp_edit = tk.StringVar()
        self.monotonic_time = tk.StringVar()
        self.qr_edits = [tk.StringVar() for _ in range(model.grid_qrs)]
        self.exhibited = tk.StringVar(value="LATEST VALID DISPLAYED QR TIME\nLoading…")
        self.codes = tk.StringVar(value="")
        self.status = tk.StringVar(value=model.saved_analysis_notice)
        self.summary = tk.StringVar(
            value=model.saved_analysis_notice or "Press GO to decode the full folder and create analysis files."
        )
        self.scan_progress = tk.StringVar(value=f"Ready: 0 / {len(model.rows)}")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Left>", lambda event: self._keyboard_step(event, -1))
        self.root.bind("<Right>", lambda event: self._keyboard_step(event, 1))
        self.root.after(50, self._poll)
        self.show_frame()

    def _build(self) -> None:
        self.root.title("QR Calibration Analysis — " + self.model.folder.name)
        width = min(1480, self.root.winfo_screenwidth() - 60)
        height = min(980, self.root.winfo_screenheight() - 90)
        self.root.geometry(f"{max(900, width)}x{max(680, height)}")
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        navigation = ttk.Frame(outer)
        navigation.pack(fill="x")
        ttk.Button(navigation, text="Previous", command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(navigation, text="Next", command=lambda: self.step(1)).pack(side="left", padx=4)
        entry = ttk.Entry(navigation, textvariable=self.position, width=7)
        entry.pack(side="left")
        entry.bind("<Return>", lambda _event: self.go())
        ttk.Button(navigation, text="Show frame", command=self.go).pack(side="left", padx=4)
        self.slider = ttk.Scale(navigation, from_=1, to=len(self.model.rows))
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        self.slider.bind("<ButtonRelease-1>", lambda _event: self.go(round(self.slider.get())))
        self.scan_button = ttk.Button(
            navigation,
            text=("Fresh decode (optional)" if self.model.saved_analysis_complete
                  else "GO — DECODE FULL FOLDER"),
            command=self.scan,
        )
        self.scan_button.pack(side="left", padx=4)
        ttk.Button(navigation, text="Cancel", command=self.worker.cancel.set).pack(side="left")
        ttk.Label(
            navigation, textvariable=self.scan_progress, anchor="e",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side="right", padx=(12, 0))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(6, 4))
        ttk.Label(controls, text="Image").pack(side="left")
        image = ttk.Combobox(
            controls, textvariable=self.variant, values=("Undistorted", "Original"),
            state="readonly", width=12,
        )
        image.pack(side="left", padx=(4, 12))
        image.bind("<<ComboboxSelected>>", lambda _event: self.draw())
        ttk.Label(controls, text="Alpha").pack(side="left")
        alpha = ttk.Combobox(
            controls, textvariable=self.alpha, values=("0", "0.25", "0.5", "0.75", "1"),
            state="readonly", width=5,
        )
        alpha.pack(side="left", padx=(4, 12))
        alpha.bind("<<ComboboxSelected>>", lambda _event: self.show_frame())
        ttk.Label(outer, textvariable=self.title, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        timing = ttk.LabelFrame(outer, text="Editable timing values", padding=6)
        timing.pack(fill="x", pady=5)
        ttk.Label(timing, text="PTS (seconds)").grid(row=0, column=0, sticky="w")
        ttk.Entry(timing, textvariable=self.pts_edit).grid(
            row=1, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Label(timing, text="NTP Unix time (seconds)").grid(row=0, column=1, sticky="w")
        ttk.Entry(timing, textvariable=self.ntp_edit).grid(
            row=1, column=1, sticky="ew", padx=4
        )
        ttk.Label(timing, text="Computer monotonic time (seconds)").grid(
            row=0, column=2, sticky="w"
        )
        ttk.Entry(timing, textvariable=self.monotonic_time, state="readonly").grid(
            row=1, column=2, sticky="ew", padx=4
        )
        ttk.Button(timing, text="APPLY AND CONTINUE", command=self.apply_edits).grid(
            row=1, column=3, padx=(8, 4)
        )
        ttk.Button(timing, text="RESTORE DETECTED", command=self.restore_detected).grid(
            row=1, column=4, padx=(4, 0)
        )
        for column in range(3):
            timing.columnconfigure(column, weight=1)

        qr_information = ttk.LabelFrame(
            outer, text=f"QR grid values ({self.model.grid_qrs} cells)", padding=6
        )
        qr_information.pack(fill="x", pady=(0, 5))
        for name, variable, (row, column) in zip(
            self.model.cell_names, self.qr_edits, self.model.cell_positions
        ):
            ttk.Label(qr_information, text=name + " (ms)").grid(
                row=row * 2, column=column, sticky="w", padx=4
            )
            ttk.Entry(qr_information, textvariable=variable, width=15).grid(
                row=row * 2 + 1, column=column, sticky="ew", padx=4, pady=(0, 3)
            )
        for column in range(self.model.timeline.grid_columns):
            qr_information.columnconfigure(column, weight=1)

        style = ttk.Style(self.root)
        style.configure("Latest.TLabel", background="#ffe66d", foreground="#191600")
        ttk.Label(
            outer, textvariable=self.exhibited, padding=8, anchor="center", style="Latest.TLabel"
        ).pack(fill="x")
        ttk.Label(outer, textvariable=self.codes, wraplength=1350).pack(anchor="w", fill="x")
        self.canvas = tk.Canvas(outer, background="#15191e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=5)
        self.canvas.bind("<Configure>", self.schedule_draw)
        ttk.Label(outer, textvariable=self.summary, wraplength=1350).pack(anchor="w")
        ttk.Label(outer, textvariable=self.status, wraplength=1350).pack(anchor="w", pady=(3, 0))

    def _keyboard_step(self, event, delta: int):
        if event.widget.winfo_class() not in ("Entry", "TEntry", "TCombobox"):
            self.step(delta)
            return "break"

    def step(self, delta: int) -> None:
        self.go(self.index + 1 + delta)

    def go(self, number=None) -> None:
        try:
            index = int(self.position.get() if number is None else number) - 1
            if not 0 <= index < len(self.model.rows):
                raise ValueError
        except (TypeError, ValueError):
            self.status.set(f"Choose a frame from 1 to {len(self.model.rows)}.")
            return
        self.index = index
        self.show_frame()

    def show_frame(self) -> None:
        self.request += 1
        self.current = None
        self.current_check = None
        self.position.set(str(self.index + 1))
        self.slider.set(self.index + 1)
        filename = self.model.rows[self.index]["filename"]
        action = "loading saved results…" if self.model.review_saved else "decoding…"
        self.title.set(f"{filename} — {self.index + 1} / {len(self.model.rows)} — {action}")
        self.status.set("Loading saved QR results." if self.model.review_saved else "Decoding this frame.")
        self.canvas.delete("all")
        self.worker.submit("frame", self.request, index=self.index, alpha=float(self.alpha.get()))

    def scan(self) -> None:
        if self.scan_active:
            return
        self.scan_active = True
        self.scan_button.configure(state="disabled")
        self.scan_request += 1
        self.summary.set("Decoding the recording; each completed frame will appear here.")
        total = len(self.model.rows)
        self.scan_progress.set(f"Analysis: 0 / {total} · {total} remaining")
        self.worker.submit("scan", self.scan_request, alpha=float(self.alpha.get()))

    def apply_edits(self) -> None:
        if self.current is None:
            return
        try:
            values = {
                "pts_ns": parse_seconds(self.pts_edit.get(), "PTS"),
                "ntp_ns": parse_seconds(self.ntp_edit.get(), "NTP"),
                "qrs": tuple(
                    parse_qr_value(variable.get(), name)
                    for variable, name in zip(self.qr_edits, self.model.cell_names)
                ),
            }
        except ValueError as error:
            self.status.set(str(error))
            return
        self.model.set_manual_values(self.index, values)
        self.populate()

    def restore_detected(self) -> None:
        if self.current is None:
            return
        self.model.reset_manual_values(self.index)
        self.populate()

    def _poll(self) -> None:
        try:
            while True:
                kind, request, payload = self.worker.results.get_nowait()
                if kind == "frame" and request == self.request:
                    self.current = payload
                    self.populate()
                elif kind == "progress" and request == self.scan_request:
                    done, total, frame = payload
                    self.index = frame["index"]
                    self.current = frame
                    self.current_check = None
                    self.position.set(str(self.index + 1))
                    self.slider.set(self.index + 1)
                    self.populate()
                    progress = f"Analysis: {done} / {total} · {max(0, total - done)} remaining"
                    self.scan_progress.set(progress)
                    self.summary.set(progress + " · live QR and offset values are above")
                    break
                elif kind == "complete" and request == self.scan_request:
                    self.scan_active = False
                    self.scan_button.configure(state="normal")
                    self.model._read_saved_analysis()
                    self.scan_button.configure(
                        text=("Fresh decode (optional)" if self.model.saved_analysis_complete
                              else "GO — DECODE FULL FOLDER")
                    )
                    self.show_summary(payload)
                    self.show_frame()
                elif kind == "error" and request in (self.request, self.scan_request):
                    self.status.set("Analysis failed: " + str(payload))
                    if request == self.scan_request:
                        self.scan_active = False
                        self.scan_button.configure(state="normal")
                        self.scan_progress.set("Analysis stopped with an error")
        except Empty:
            pass
        self.root.after(50, self._poll)

    def populate(self) -> None:
        result = self.current
        if result is None:
            return
        row = result["row"]
        check = self.model.check_frame(result)
        values = check["values"]
        self.current_check = check
        self.title.set(
            f"{row['filename']} — {result['index'] + 1} / {len(self.model.rows)} — "
            f"{self.variant.get()}"
        )
        self.pts_edit.set(editable_seconds(values.get("pts_ns")))
        self.ntp_edit.set(editable_seconds(values.get("ntp_ns")))
        self.monotonic_time.set(
            editable_seconds(self.model.pts_monotonic_for_value(row, values.get("pts_ns")))
        )
        for variable, raw in zip(self.qr_edits, values["qrs"]):
            variable.set(raw or "")
        displayed_qrs = list(values["qrs"])
        inferred_cells = set()
        for inference in check.get("temporal_inferences", []):
            cell = inference.get("cell")
            if isinstance(cell, int) and 0 <= cell < len(displayed_qrs) and displayed_qrs[cell] is None:
                displayed_qrs[cell] = inference.get("raw")
                inferred_cells.add(cell)

        if not check["valid"]:
            prefix = "Frame skipped" if check.get("skippable") else "Validation stopped"
            self.exhibited.set(f"LATEST DISPLAYED QR TIME\n{prefix}: {check['reason']}")
        else:
            offset = "" if check["offset_ms"] is None else (
                f" · camera PTS minus latest QR {check['offset_ms']:.3f} ms"
            )
            interval = check.get("presentation_interval") or {}
            lower, upper = interval.get("offset_interval_lower_ms"), interval.get("offset_interval_upper_ms")
            bracket = (
                f" · presentation interval {lower:.3f} to {upper:.3f} ms"
                if lower is not None and upper is not None else ""
            )
            self.exhibited.set(
                f"LATEST VALID DISPLAYED QR TIME\n{payload_time(check['latest_raw'])} · "
                f"{self.model.cell_names[check['latest_cell']]} · {check['timing_status']}"
                f"{offset}{bracket}"
                + (f" · inferred from frame {check['latest_inference']['confirmation_frame_number']}"
                   if check.get("latest_is_inferred") else "")
            )

        self.codes.set("   |   ".join(
            f"{name}: {payload_time(raw)}" + (" (inferred)" if cell in inferred_cells else "")
            for cell, (name, raw) in enumerate(zip(self.model.cell_names, displayed_qrs))
        ) or "No QR code detected.")
        if not check["valid"]:
            source = "manual values" if values["manual"] else (
                "saved results" if "saved_values" in result else "QReader detections"
            )
            action = "Skipped" if check.get("skippable") else "Stopped"
            status = f"{action} on {source}: {check['reason']}"
        else:
            warnings = check.get("position_warnings", [])
            warning_text = f"{len(warnings)} camera-grid position warning(s). " if warnings else ""
            source_text = (
                "Manual values accepted. " if values["manual"] else
                (f"Latest QR identity inferred from and confirmed by frame "
                 f"{check['latest_inference']['confirmation_frame_number']}. "
                 if check.get("latest_is_inferred") else
                 f"{check['matched_readable_qrs']} journal-matched QR value(s) accepted; latest selected. ")
            )
            timing_text = ", ".join(check["issues"]) if check["issues"] else "clean"
            status = source_text + warning_text + "Latest display timing: " + timing_text + "."
        if result.get("review_notice"):
            status = result["review_notice"] + " " + status
        self.status.set(status)
        self.draw()

    def show_summary(self, report: dict) -> None:
        counts = report.get("counts", {})
        state = "Cancelled" if report.get("cancelled") else "Complete"
        stopped = report.get("stopped")
        if stopped:
            state = f"Stopped at frame {stopped['index'] + 1}: {stopped['reason']}"
        self.saved_output = (
            Path(report["output_directory"])
            if not report.get("cancelled") and stopped is None
            and report.get("processed") == report.get("total") else None
        )
        remaining = max(0, report.get("total", 0) - report.get("processed", 0))
        self.scan_progress.set(
            f"{state}: {report.get('processed', 0)} / {report.get('total', 0)} · {remaining} remaining"
        )
        median = report.get("median_offset_ms")
        offset_text = (
            "no clean offset" if median is None else
            f"median camera PTS minus latest valid displayed QR: {median:.3f} ms "
            f"(n={report.get('clean_offsets', 0)})"
        )
        inferred_median = report.get("median_temporally_inferred_offset_ms")
        inferred_text = (
            f"{counts.get('temporal_inferred_qrs', 0)} QR identity/identities inferred from the next frame; "
            + ("no clean inferred timing offset" if inferred_median is None else
               f"median inferred offset {inferred_median:.3f} ms "
               f"(n={report.get('temporally_inferred_clean_offsets', 0)}; excluded from direct-read consensus)")
        )
        interval_result = (report.get("presentation_interval_analysis") or {}).get("primary_pts")
        interval_text = (
            "no clean presentation-offset interval" if interval_result is None else
            f"presentation-offset range {interval_result['offset_range_lower_ms']:.3f}–"
            f"{interval_result['offset_range_upper_ms']:.3f} ms; representative "
            f"{interval_result['estimated_offset_ms']:.3f} ms; consistent "
            f"{interval_result['maximum_consistent_frames']} / {interval_result['contributing_frames']}"
        )
        display = report.get("display") or {}
        self.summary.set(
            f"{state}: {report.get('processed', 0)} / {report.get('total', 0)} frames; "
            f"{offset_text}; {interval_text}. QR detections {counts.get('detections', 0)}, "
            f"{inferred_text}. "
            f"unreadable {counts.get('unreadable', 0)}, journal-matched "
            f"{counts.get('journal_matched_readable', 0)}, frames without a readable match "
            f"{counts.get('frames_without_readable_qr', 0)}, grid-cell mismatches "
            f"{counts.get('mismatches', 0)}, timing-suspect markers "
            f"{counts.get('timing_suspect', 0)}. Display: "
            f"{display.get('late_submissions', 0)} late submissions, "
            f"{display.get('irregular_intervals', 0)} irregular intervals, "
            f"{display.get('missed_period_candidates', 0)} missed-period candidates. "
            f"Results saved in {report.get('output_directory', self.model.output_folder)}"
        )
        if report.get("output_directory"):
            self.status.set(
                "Saved calibration_analysis.json, calibration_frames.csv, and "
                f"{PRESENTATIONS_CSV} in {report['output_directory']}"
            )
        if stopped:
            self.root.after(0, lambda: self.go(stopped["index"] + 1))

    def schedule_draw(self, _event=None) -> None:
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.draw)

    def draw(self) -> None:
        self.resize_job = None
        self.canvas.delete("all")
        if self.current is None:
            return
        pixels = self.current["undistorted" if self.variant.get() == "Undistorted" else "original"]
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        scale = min(canvas_w / pixels.shape[1], canvas_h / pixels.shape[0])
        size = max(1, int(pixels.shape[1] * scale)), max(1, int(pixels.shape[0] * scale))
        preview = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).resize(
            size, Image.Resampling.BILINEAR
        )
        self.photo = ImageTk.PhotoImage(preview)
        left, top = (canvas_w - size[0]) / 2, (canvas_h - size[1]) / 2
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")

        check = self.current_check or self.model.check_frame(self.current)
        latest_marker = check.get("latest_marker") if check.get("valid") else None
        latest_index = latest_marker.get("index") if latest_marker else None
        if check.get("latest_is_inferred"):
            inference = check.get("latest_inference") or {}
            observation_index = inference.get("candidate_detection_index")
            observations = self.current.get("observations", [])
            expected_box = inference.get("candidate_bbox")
            selected = max(
                (item for item in observations
                 if isinstance(expected_box, list) and isinstance(item.get("bbox"), (list, np.ndarray))),
                key=lambda item: _bbox_iou(expected_box, np.asarray(item["bbox"], dtype=float).tolist()),
                default=None,
            )
            if (selected is None and isinstance(observation_index, int)
                    and 0 <= observation_index < len(observations)):
                selected = observations[observation_index]
            if selected is None:
                selected = {
                    "raw": inference.get("raw"),
                    "original_points": inference.get("candidate_original_points"),
                    "undistorted_points": inference.get("candidate_undistorted_points"),
                }
        else:
            selected = next(
                (item for item in self.current.get("observations", [])
                 if latest_index is not None and item.get("display_index") == latest_index),
                None,
            )
        if selected is None:
            selected = self.current.get("latest")
        if selected is not None:
            points = np.asarray(
                selected.get(
                    "undistorted_points" if self.variant.get() == "Undistorted" else "original_points"
                ),
                dtype=float,
            )
            if points.shape == (4, 2):
                points = points * scale + np.array([left, top])
                coordinates = points.ravel().tolist()
                self.canvas.create_polygon(
                    coordinates, fill="", outline="#000000", width=7
                )
                self.canvas.create_polygon(
                    coordinates, fill="", outline=("#00e5ff" if check.get("latest_is_inferred") else "#ffffff"), width=3
                )
                label = self.canvas.create_text(
                    points[:, 0].min() + 4,
                    max(4, points[:, 1].min() - 26),
                    text=(payload_time(selected.get("raw")) + " · inferred"
                          if check.get("latest_is_inferred") else payload_time(selected.get("raw"))),
                    fill="#ffffff",
                    anchor="nw",
                    font=("TkDefaultFont", 10, "bold"),
                )
                background = self.canvas.create_rectangle(
                    self.canvas.bbox(label), fill="#000000", outline="#ffffff"
                )
                self.canvas.tag_raise(label, background)

        for item in self.current.get("observations", []):
            if item.get("detection_method") != "screen_contrast":
                continue
            points = np.asarray(
                item.get(
                    "undistorted_points" if self.variant.get() == "Undistorted" else "original_points"
                ),
                dtype=float,
            )
            if points.shape != (4, 2):
                continue
            points = points * scale + np.array([left, top])
            coordinates = points.ravel().tolist()
            self.canvas.create_polygon(coordinates, fill="", outline="#000000", width=5)
            self.canvas.create_polygon(coordinates, fill="", outline="#00e5ff", width=2)
            label = self.canvas.create_text(
                points[:, 0].min() + 4,
                max(4, points[:, 1].min() - 22),
                text="Bright content · timestamp unreadable",
                fill="#00e5ff",
                anchor="nw",
                font=("TkDefaultFont", 9, "bold"),
            )
            background = self.canvas.create_rectangle(
                self.canvas.bbox(label), fill="#000000", outline="#00e5ff"
            )
            self.canvas.tag_raise(label, background)

    def close(self) -> None:
        self.worker.stop()
        self.root.destroy()


def run_recording_display(folder: Path, intrinsics: Path = DEFAULT_INTRINSICS,
                          *, alpha: float = 0.25) -> Path | None:
    model = RecordingAnalyzer(folder, intrinsics)
    root = tk.Tk()
    window = CalibrationWindow(root, model, alpha=alpha)
    root.mainloop()
    return window.saved_output


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Review QR calibration evidence")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    run_recording_display(args.folder, args.intrinsics, alpha=args.alpha)


if __name__ == "__main__":
    main()
