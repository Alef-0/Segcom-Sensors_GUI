#!/usr/bin/env python3
"""Estimate per-image display-observation time from QR/flip interval anchors.

The QR payload identifies a display generation; it is not treated as an exact
camera exposure timestamp.  Each usable image instead contributes the interval
between the newest observed marker's recorded presentation return and the next
presentation return.  An interval-aware arrival-delay model maps host frame
receipt time onto that hidden observation time and fills frames without a QR
anchor.  Host-anchored PTS remains an independent comparison baseline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np


DISPLAY_JOURNAL = "display_timestamps.jsonl"
CAMERA_JOURNALS = ("camera_timestamps.jsonl", "camera_timestamps.json")
CAMERA_SESSION = "camera_timing_session.json"
ANALYSIS_JSON = "calibration_analysis.json"

REPORT_JSON = "distance_analysis.json"
FRAMES_CSV = "distance_frames.csv"
MARKERS_CSV = "distance_markers.csv"
OVERVIEW_GRAPH = "distance_analysis_overview.png"

PAYLOAD_MODULUS_MS = 1_000_000_000_000
TRAIN_FRACTION = 0.70
MINIMUM_ANCHORS = 10
MAX_CAMERA_GAP_NS = 1_000_000_000
MAX_ANCHOR_PERIODS = 3.0
AFFINE_SLOPE_RIDGE = 1e-4
_PLOTTING = None


@dataclass(frozen=True, slots=True)
class DisplayEvent:
    index: int
    cell: int
    marker_ns: int
    predicted_flip_ns: int | None
    presentation_return_ns: int
    presentation_event_kind: str
    physical_presentation_measured: bool
    timing_issues: tuple[str, ...]


def _finite_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _read_json_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path.name}, line {line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path.name}, line {line_number} is not a JSON object"
                    )
                rows.append(value)
        return rows

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}") from error
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("frames", "camera_timestamps"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"{path.name} does not contain a frame list")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(marker_ns: int) -> str:
    milliseconds = marker_ns // 1_000_000 % PAYLOAD_MODULUS_MS
    return f"{milliseconds:012d}"


def _display_issues(row: dict) -> tuple[str, ...]:
    issues = []
    if row.get("late_submit"):
        issues.append("late_submission")
    if row.get("irregular_interval"):
        issues.append("irregular_interval")
    if int(row.get("skipped_periods") or 0) > 0:
        issues.append("missed_period_candidate")
    if row.get("resumed_after_pause"):
        issues.append("resumed_after_pause")
    return tuple(issues)


def _load_display_timeline(
    recording: Path,
) -> tuple[dict, list[DisplayEvent], dict[str, list[int]], list[int]]:
    path = recording / DISPLAY_JOURNAL
    if not path.is_file():
        raise ValueError(f"Recording is missing {DISPLAY_JOURNAL}")
    rows = _read_json_rows(path)
    metadata = next((row for row in rows if row.get("kind") == "session"), None)
    raw_events = [row for row in rows if row.get("kind") == "frame"]
    if metadata is None or not raw_events:
        raise ValueError("Display journal has no session or frame entries")
    grid_qrs = _finite_int(metadata.get("grid_qrs", 4))
    if grid_qrs is None or grid_qrs < 1:
        raise ValueError("Display journal has an invalid grid_qrs value")

    events = []
    previous_presentation = -1
    for expected_index, row in enumerate(raw_events):
        index = _finite_int(row.get("index"))
        cell = _finite_int(row.get("cell", row.get("corner")))
        marker_ns = _finite_int(row.get("marker_ns"))
        presentation_ns = _finite_int(
            row.get("presentation_return_ns", row.get("flip_return_ns"))
        )
        if index != expected_index:
            raise ValueError("Display journal indices are missing or reordered")
        if cell is None or not 0 <= cell < grid_qrs or marker_ns is None:
            raise ValueError(f"Display frame {expected_index} has invalid marker data")
        if presentation_ns is None:
            raise ValueError(
                f"Display frame {expected_index} has no flip/presentation return"
            )
        if presentation_ns <= previous_presentation:
            raise ValueError("Display presentation returns are not monotonic")
        previous_presentation = presentation_ns
        events.append(DisplayEvent(
            index=expected_index,
            cell=cell,
            marker_ns=marker_ns,
            predicted_flip_ns=_finite_int(row.get("predicted_flip_ns")),
            presentation_return_ns=presentation_ns,
            presentation_event_kind=str(
                row.get("presentation_event_kind")
                or ("legacy_flip_return" if row.get("flip_return_ns") is not None
                    else "presentation_return")
            ),
            physical_presentation_measured=bool(
                row.get("physical_presentation_measured", False)
            ),
            timing_issues=_display_issues(row),
        ))

    by_payload: dict[str, list[int]] = {}
    for event in events:
        by_payload.setdefault(_payload(event.marker_ns), []).append(event.index)
    pause_boundaries = sorted(
        value
        for row in rows
        if row.get("kind") == "pause"
        for value in (_finite_int(row.get("monotonic_ns")),)
        if value is not None
    )
    return metadata, events, by_payload, pause_boundaries


def _load_analysis(analysis_directory: Path) -> tuple[dict, dict[str, dict], Path]:
    path = analysis_directory / ANALYSIS_JSON
    if not path.is_file():
        raise ValueError(f"Analysis directory is missing {ANALYSIS_JSON}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}") from error
    raw_frames = report.get("frames") if isinstance(report, dict) else None
    if not isinstance(raw_frames, list):
        raise ValueError(f"{ANALYSIS_JSON} has no frame list")
    frames = {}
    for position, frame in enumerate(raw_frames, 1):
        if not isinstance(frame, dict):
            raise ValueError(f"Analysis frame {position} is not an object")
        filename = frame.get("filename")
        if not isinstance(filename, str) or not filename or filename in frames:
            raise ValueError("Analysis has missing or duplicate frame names")
        frames[filename] = frame
    return report, frames, path


def _load_camera_rows(recording: Path) -> tuple[list[dict], Path]:
    path = next(
        (recording / name for name in CAMERA_JOURNALS if (recording / name).is_file()),
        None,
    )
    if path is None:
        raise ValueError("Recording has no camera timestamp journal")
    raw_rows = _read_json_rows(path)
    rows = []
    names = set()
    for position, row in enumerate(raw_rows, 1):
        filename = row.get("frame") or row.get("camera_frame")
        if not isinstance(filename, str) or not filename or filename in names:
            raise ValueError(
                f"Camera journal row {position} has a missing or duplicate frame name"
            )
        names.add(filename)
        rows.append({**row, "filename": filename})
    if not rows:
        raise ValueError("Camera timestamp journal is empty")
    return rows, path


def _load_epoch_anchors(recording: Path) -> tuple[dict[int, int], Path | None]:
    path = recording / CAMERA_SESSION
    if not path.is_file():
        return {}, None
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}") from error
    anchors = {}
    for row in session.get("epochs", ()):
        epoch = _finite_int(row.get("stream_epoch"))
        zero = _finite_int(row.get("pipeline_zero_monotonic_ns"))
        if epoch is not None and zero is not None:
            anchors[epoch] = zero
    return anchors, path


def _pts_monotonic_ns(
    camera: dict, analysis: dict | None, epoch_anchors: dict[int, int]
) -> int | None:
    if analysis is not None:
        value = _finite_int(analysis.get("camera_reference_monotonic_ns"))
        if value is not None:
            return value
    for key in ("frame_monotonic_ns", "captured_monotonic_ns"):
        value = _finite_int(camera.get(key))
        if value is not None:
            return value
    epoch = _finite_int(camera.get("stream_epoch"))
    pts = _finite_int(camera.get("pts_ns"))
    if epoch is not None and pts is not None and epoch in epoch_anchors:
        return epoch_anchors[epoch] + pts
    return None


def _match_payload(
    raw: str,
    candidates: list[int],
    reported_indices: set[int],
    reported_latest: int | None,
) -> tuple[int | None, str]:
    if len(candidates) == 1:
        return candidates[0], "unique_payload"
    reported = [index for index in candidates if index in reported_indices]
    if len(reported) == 1:
        return reported[0], "analysis_index_match"
    choices = reported or candidates
    if reported_latest is not None and choices:
        distances = [abs(index - reported_latest) for index in choices]
        closest = min(distances)
        winners = [
            index for index, distance in zip(choices, distances) if distance == closest
        ]
        if len(winners) == 1:
            return winners[0], "nearest_analysis_index"
    return None, "ambiguous_wrapped_payload"


def _marker_rows_for_frame(
    frame_number: int,
    filename: str,
    received_ns: int | None,
    analysis: dict | None,
    events: list[DisplayEvent],
    by_payload: dict[str, list[int]],
    visible_qrs: int,
) -> tuple[list[dict], list[int]]:
    if analysis is None:
        return [], []
    raw_values = analysis.get("qr_values_ms")
    if not isinstance(raw_values, (list, tuple)):
        return [], []
    reported_values = analysis.get("display_indices")
    if not isinstance(reported_values, (list, tuple, set)):
        reported_values = ()
    reported_indices = {
        value
        for value in (_finite_int(item) for item in reported_values)
        if value is not None
    }
    reported_latest = _finite_int(analysis.get("latest_display_index"))
    markers = []
    mapped_indices = []
    for camera_cell, value in enumerate(raw_values):
        if value is None:
            continue
        raw = str(value)
        candidates = by_payload.get(raw, []) if len(raw) == 12 and raw.isdigit() else []
        if not candidates:
            markers.append({
                "frame_number": frame_number,
                "filename": filename,
                "received_monotonic_ns": received_ns,
                "camera_cell": camera_cell,
                "qr_payload": raw,
                "mapping_status": "invalid_payload" if not raw.isdigit() or len(raw) != 12
                else "not_in_display_journal",
                "display_index": None,
                "journal_cell": None,
                "marker_ns": None,
                "predicted_flip_ns": None,
                "presentation_return_ns": None,
                "replacement_display_index": None,
                "replacement_presentation_return_ns": None,
                "is_latest_observed": False,
                "timing_issues": [],
            })
            continue
        index, status = _match_payload(
            raw, candidates, reported_indices, reported_latest
        )
        if index is None:
            markers.append({
                "frame_number": frame_number,
                "filename": filename,
                "received_monotonic_ns": received_ns,
                "camera_cell": camera_cell,
                "qr_payload": raw,
                "mapping_status": status,
                "display_index": None,
                "journal_cell": None,
                "marker_ns": None,
                "predicted_flip_ns": None,
                "presentation_return_ns": None,
                "replacement_display_index": None,
                "replacement_presentation_return_ns": None,
                "is_latest_observed": False,
                "timing_issues": [],
            })
            continue
        event = events[index]
        replacement_index = index + visible_qrs
        replacement = (
            events[replacement_index] if replacement_index < len(events) else None
        )
        mapped_indices.append(index)
        markers.append({
            "frame_number": frame_number,
            "filename": filename,
            "received_monotonic_ns": received_ns,
            "camera_cell": camera_cell,
            "qr_payload": raw,
            "mapping_status": status,
            "display_index": index,
            "journal_cell": event.cell,
            "marker_ns": event.marker_ns,
            "predicted_flip_ns": event.predicted_flip_ns,
            "presentation_return_ns": event.presentation_return_ns,
            "replacement_display_index": (
                replacement.index if replacement is not None else None
            ),
            "replacement_presentation_return_ns": (
                replacement.presentation_return_ns
                if replacement is not None else None
            ),
            "is_latest_observed": False,
            "timing_issues": list(event.timing_issues),
        })
    markers.sort(key=lambda marker: (
        marker["display_index"] is None,
        marker["display_index"] if marker["display_index"] is not None else 0,
        marker["camera_cell"],
    ))
    unique_indices = sorted(set(mapped_indices))
    if unique_indices:
        latest = unique_indices[-1]
        for marker in markers:
            marker["is_latest_observed"] = marker["display_index"] == latest
    return markers, unique_indices


def _full_marker_interval(
    indices: list[int], events: list[DisplayEvent], visible_qrs: int
) -> tuple[int | None, int | None, str]:
    if not indices:
        return None, None, "no_mapped_markers"
    lower = max(events[index].presentation_return_ns for index in indices)
    replacements = [
        events[index + visible_qrs].presentation_return_ns
        for index in indices
        if index + visible_qrs < len(events)
    ]
    if len(replacements) != len(indices):
        return lower, None, "replacement_boundary_missing"
    upper = min(replacements)
    if upper <= lower:
        return lower, upper, "mixed_or_rolling_generations"
    return lower, upper, "compatible_global_state"


def _build_frames(
    camera_rows: list[dict],
    analysis_frames: dict[str, dict],
    events: list[DisplayEvent],
    by_payload: dict[str, list[int]],
    visible_qrs: int,
    epoch_anchors: dict[int, int],
) -> tuple[list[dict], list[dict]]:
    periods = np.diff(np.asarray(
        [event.presentation_return_ns for event in events], dtype=np.int64
    ))
    typical_period_ns = float(np.median(periods)) if len(periods) else math.nan
    maximum_anchor_width_ns = (
        typical_period_ns * MAX_ANCHOR_PERIODS
        if math.isfinite(typical_period_ns) else math.inf
    )
    frames = []
    all_markers = []
    for frame_number, camera in enumerate(camera_rows, 1):
        filename = camera["filename"]
        analysis = analysis_frames.get(filename)
        received_ns = _finite_int(
            camera.get("received_monotonic_ns", camera.get("host_monotonic_received_ns"))
        )
        if received_ns is None and analysis is not None:
            received_ns = _finite_int(analysis.get("received_monotonic_ns"))
        pts_reference_ns = _pts_monotonic_ns(camera, analysis, epoch_anchors)
        markers, indices = _marker_rows_for_frame(
            frame_number,
            filename,
            received_ns,
            analysis,
            events,
            by_payload,
            visible_qrs,
        )
        all_markers.extend(markers)
        latest_index = indices[-1] if indices else None
        if latest_index is None and analysis is not None:
            reported = _finite_int(analysis.get("latest_display_index"))
            if reported is not None and 0 <= reported < len(events):
                latest_index = reported

        latest_event = events[latest_index] if latest_index is not None else None
        next_event = (
            events[latest_index + 1]
            if latest_index is not None and latest_index + 1 < len(events)
            else None
        )
        anchor_start = (
            latest_event.presentation_return_ns if latest_event is not None else None
        )
        anchor_end = (
            next_event.presentation_return_ns if next_event is not None else None
        )
        full_start, full_end, state_status = _full_marker_interval(
            indices, events, visible_qrs
        )
        validation = str(
            (analysis.get("validation") or "unknown")
            if analysis is not None else "missing_analysis"
        )
        anchor_status = "usable"
        if received_ns is None:
            anchor_status = "missing_received_time"
        elif anchor_start is None:
            anchor_status = "missing_qr_anchor"
        elif anchor_end is None:
            anchor_status = "missing_next_flip"
        elif anchor_end <= anchor_start:
            anchor_status = "invalid_flip_interval"
        elif anchor_end - anchor_start > maximum_anchor_width_ns:
            anchor_status = "held_or_discontinuous_display_interval"
        elif not validation.startswith("accepted_"):
            anchor_status = "analysis_rejected"

        observed_values = analysis.get("qr_values_ms") if analysis is not None else ()
        if not isinstance(observed_values, (list, tuple)):
            observed_values = ()
        frames.append({
            "frame_number": frame_number,
            "filename": filename,
            "stream_epoch": _finite_int(camera.get("stream_epoch")),
            "received_monotonic_ns": received_ns,
            "pts_ns": _finite_int(camera.get("pts_ns")),
            "pts_reference_monotonic_ns": pts_reference_ns,
            "analysis_validation": validation,
            "analysis_timing_status": (
                analysis.get("timing_status") if analysis is not None else None
            ),
            "observed_qr_count": sum(value is not None for value in observed_values),
            "mapped_qr_count": len(indices),
            "observed_display_indices": indices,
            "observed_display_span": (
                indices[-1] - indices[0] + 1 if indices else None
            ),
            "latest_display_index": latest_index,
            "latest_qr_payload": (
                _payload(latest_event.marker_ns) if latest_event is not None else None
            ),
            "latest_marker_ns": (
                latest_event.marker_ns if latest_event is not None else None
            ),
            "latest_predicted_flip_ns": (
                latest_event.predicted_flip_ns if latest_event is not None else None
            ),
            "latest_presentation_return_ns": anchor_start,
            "next_presentation_return_ns": anchor_end,
            "all_markers_interval_start_ns": full_start,
            "all_markers_interval_end_ns": full_end,
            "all_markers_state_status": state_status,
            "anchor_interval_start_ns": anchor_start,
            "anchor_interval_end_ns": anchor_end,
            "anchor_interval_width_ms": (
                (anchor_end - anchor_start) / 1e6
                if anchor_start is not None and anchor_end is not None else None
            ),
            "anchor_status": anchor_status,
            "segment_id": None,
            "segment_break_reason": None,
            "estimated_observation_monotonic_ns": None,
            "predicted_arrival_delay_ms": None,
            "estimation_kind": "unavailable",
            "signed_distance_to_qr_interval_ms": None,
            "inside_qr_interval": None,
            "pts_baseline_correction_ms": None,
            "pts_baseline_observation_monotonic_ns": None,
            "pts_signed_distance_to_qr_interval_ms": None,
        })
    return frames, all_markers


def _assign_segments(frames: list[dict], pause_boundaries: list[int]) -> None:
    segment = -1
    previous_received = None
    previous_epoch = None
    for frame in frames:
        received = frame["received_monotonic_ns"]
        epoch = frame["stream_epoch"]
        reason = None
        if segment < 0:
            reason = "recording_start"
        elif epoch != previous_epoch:
            reason = "stream_epoch_changed"
        elif received is None or previous_received is None:
            reason = "missing_received_time_boundary"
        elif received <= previous_received:
            reason = "received_time_not_monotonic"
        elif received - previous_received > MAX_CAMERA_GAP_NS:
            reason = "large_camera_arrival_gap"
        elif any(previous_received < value <= received for value in pause_boundaries):
            reason = "display_pause_boundary"
        if reason is not None:
            segment += 1
            frame["segment_break_reason"] = reason
        frame["segment_id"] = segment
        previous_received = received
        previous_epoch = epoch


def _signed_interval_distance(
    prediction: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    return np.where(
        prediction < lower,
        lower - prediction,
        np.where(prediction > upper, upper - prediction, 0.0),
    )


def _fit_interval_least_squares(
    design: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    slope_ridge: float = 0.0,
) -> dict:
    target = (lower + upper) / 2
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    penalty = np.zeros(design.shape[1], dtype=float)
    if design.shape[1] > 1:
        penalty[1:] = slope_ridge
    converged = False
    gradient_norm = math.inf
    iterations = 0
    for iterations in range(1, 201):
        prediction = design @ coefficients
        residual = _signed_interval_distance(prediction, lower, upper)
        gradient = -(design.T @ residual) / len(design) + penalty * coefficients
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= 1e-9:
            converged = True
            break
        active = residual != 0
        hessian = design[active].T @ design[active] / len(design)
        hessian += np.diag(penalty) + np.eye(design.shape[1]) * 1e-12
        try:
            direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            direction = -gradient
        derivative = float(gradient @ direction)
        if derivative >= 0:
            direction = -gradient
            derivative = -float(gradient @ gradient)
        objective = (
            0.5 * float(np.mean(residual**2))
            + 0.5 * float(np.sum(penalty * coefficients**2))
        )
        step = 1.0
        while step >= 1e-10:
            updated = coefficients + step * direction
            updated_residual = _signed_interval_distance(
                design @ updated, lower, upper
            )
            updated_objective = (
                0.5 * float(np.mean(updated_residual**2))
                + 0.5 * float(np.sum(penalty * updated**2))
            )
            if updated_objective <= objective + 1e-4 * step * derivative:
                coefficients = updated
                break
            step *= 0.5
        else:
            break
    return {
        "coefficients": coefficients,
        "rank": int(rank),
        "iterations": iterations,
        "converged": converged,
        "final_gradient_max_abs": gradient_norm,
    }


def _metrics(
    prediction: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict | None:
    if len(prediction) == 0:
        return None
    residual = _signed_interval_distance(prediction, lower, upper)
    absolute = np.abs(residual)
    return {
        "frames": int(len(prediction)),
        "mae_ms": float(np.mean(absolute)),
        "median_absolute_ms": float(np.median(absolute)),
        "p95_absolute_ms": float(np.percentile(absolute, 95)),
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "signed_bias_ms": float(np.mean(residual)),
        "inside_interval_pct": float(100 * np.mean(absolute == 0)),
    }


def _arrival_arrays(
    frames: Iterable[dict], origin_ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(frames)
    elapsed = np.asarray([
        (frame["received_monotonic_ns"] - origin_ns) / 1e9 for frame in rows
    ])
    lower_delay = np.asarray([
        (frame["received_monotonic_ns"] - frame["anchor_interval_end_ns"]) / 1e6
        for frame in rows
    ])
    upper_delay = np.asarray([
        (frame["received_monotonic_ns"] - frame["anchor_interval_start_ns"]) / 1e6
        for frame in rows
    ])
    return elapsed, lower_delay, upper_delay


def _fit_arrival_model(frames: list[dict], origin_ns: int, affine: bool) -> dict:
    elapsed, lower, upper = _arrival_arrays(frames, origin_ns)
    design = (
        np.column_stack((np.ones(len(frames)), elapsed))
        if affine else np.ones((len(frames), 1))
    )
    fit = _fit_interval_least_squares(
        design,
        lower,
        upper,
        slope_ridge=AFFINE_SLOPE_RIDGE if affine else 0.0,
    )
    coefficients = fit.pop("coefficients")
    return {
        **fit,
        "origin_received_monotonic_ns": origin_ns,
        "delay_intercept_ms": float(coefficients[0]),
        "delay_drift_ms_per_second": (
            float(coefficients[1]) if affine else 0.0
        ),
    }


def _predict_arrival_delay(model: dict, received_ns: int) -> float:
    elapsed_seconds = (
        received_ns - model["origin_received_monotonic_ns"]
    ) / 1e9
    return (
        model["delay_intercept_ms"]
        + model["delay_drift_ms_per_second"] * elapsed_seconds
    )


def _arrival_model_metrics(model: dict, frames: list[dict]) -> dict | None:
    if not frames:
        return None
    delays = np.asarray([
        _predict_arrival_delay(model, frame["received_monotonic_ns"])
        for frame in frames
    ])
    predicted_time = np.asarray([
        frame["received_monotonic_ns"] / 1e6 for frame in frames
    ]) - delays
    lower = np.asarray([
        frame["anchor_interval_start_ns"] / 1e6 for frame in frames
    ])
    upper = np.asarray([
        frame["anchor_interval_end_ns"] / 1e6 for frame in frames
    ])
    return _metrics(predicted_time, lower, upper)


def _fit_pts_constant(frames: list[dict]) -> dict | None:
    rows = [frame for frame in frames if frame["pts_reference_monotonic_ns"] is not None]
    if not rows:
        return None
    lower = np.asarray([
        (frame["pts_reference_monotonic_ns"] - frame["anchor_interval_end_ns"]) / 1e6
        for frame in rows
    ])
    upper = np.asarray([
        (frame["pts_reference_monotonic_ns"] - frame["anchor_interval_start_ns"]) / 1e6
        for frame in rows
    ])
    fit = _fit_interval_least_squares(np.ones((len(rows), 1)), lower, upper)
    coefficients = fit.pop("coefficients")
    return {**fit, "correction_ms": float(coefficients[0])}


def _pts_metrics(model: dict | None, frames: list[dict]) -> dict | None:
    if model is None:
        return None
    rows = [frame for frame in frames if frame["pts_reference_monotonic_ns"] is not None]
    if not rows:
        return None
    prediction = np.asarray([
        frame["pts_reference_monotonic_ns"] / 1e6 - model["correction_ms"]
        for frame in rows
    ])
    lower = np.asarray([
        frame["anchor_interval_start_ns"] / 1e6 for frame in rows
    ])
    upper = np.asarray([
        frame["anchor_interval_end_ns"] / 1e6 for frame in rows
    ])
    return _metrics(prediction, lower, upper)


def _annotate_segment(
    segment_id: int,
    frames: list[dict],
    train_fraction: float,
    minimum_anchors: int,
) -> dict:
    anchors = [frame for frame in frames if frame["anchor_status"] == "usable"]
    summary = {
        "segment_id": segment_id,
        "stream_epoch": frames[0]["stream_epoch"] if frames else None,
        "frames": len(frames),
        "anchors": len(anchors),
        "first_frame_number": frames[0]["frame_number"] if frames else None,
        "last_frame_number": frames[-1]["frame_number"] if frames else None,
    }
    if not anchors:
        summary.update({
            "status": "no_usable_qr_anchors",
            "arrival_model": None,
            "arrival_constant_baseline": None,
            "pts_constant_baseline": None,
        })
        return summary

    if len(anchors) == 1:
        train_count = 1
    else:
        train_count = max(1, min(len(anchors) - 1, int(len(anchors) * train_fraction)))
    train = anchors[:train_count]
    holdout = anchors[train_count:]
    origin_ns = train[0]["received_monotonic_ns"]

    train_constant = _fit_arrival_model(train, origin_ns, affine=False)
    train_affine = (
        _fit_arrival_model(train, origin_ns, affine=True)
        if len(train) >= 2 else train_constant
    )
    pts_train = _fit_pts_constant(train)
    evaluation = {
        "chronological_train_anchors": len(train),
        "chronological_holdout_anchors": len(holdout),
        "train_last_frame_number": train[-1]["frame_number"],
        "holdout_first_frame_number": (
            holdout[0]["frame_number"] if holdout else None
        ),
        "arrival_affine_holdout": _arrival_model_metrics(train_affine, holdout),
        "arrival_constant_holdout": _arrival_model_metrics(train_constant, holdout),
        "pts_constant_holdout": _pts_metrics(pts_train, holdout),
    }

    full_origin_ns = anchors[0]["received_monotonic_ns"]
    final_affine = (
        _fit_arrival_model(anchors, full_origin_ns, affine=True)
        if len(anchors) >= 2 else _fit_arrival_model(anchors, full_origin_ns, affine=False)
    )
    final_constant = _fit_arrival_model(anchors, full_origin_ns, affine=False)
    final_pts = _fit_pts_constant(anchors)
    anchor_receipts = [frame["received_monotonic_ns"] for frame in anchors]

    for frame in frames:
        received = frame["received_monotonic_ns"]
        if received is None:
            continue
        delay_ms = _predict_arrival_delay(final_affine, received)
        estimate_ns = round(received - delay_ms * 1e6)
        frame["estimated_observation_monotonic_ns"] = estimate_ns
        frame["predicted_arrival_delay_ms"] = delay_ms
        if frame["anchor_status"] == "usable":
            frame["estimation_kind"] = "qr_anchored"
        elif min(anchor_receipts) <= received <= max(anchor_receipts):
            frame["estimation_kind"] = "interpolated_qr_gap"
        else:
            frame["estimation_kind"] = "extrapolated_outside_qr_anchors"
        start = frame["anchor_interval_start_ns"]
        end = frame["anchor_interval_end_ns"]
        if start is not None and end is not None:
            residual_ms = float(_signed_interval_distance(
                np.asarray([estimate_ns / 1e6]),
                np.asarray([start / 1e6]),
                np.asarray([end / 1e6]),
            )[0])
            frame["signed_distance_to_qr_interval_ms"] = residual_ms
            frame["inside_qr_interval"] = residual_ms == 0
        if final_pts is not None and frame["pts_reference_monotonic_ns"] is not None:
            correction = final_pts["correction_ms"]
            pts_estimate = round(
                frame["pts_reference_monotonic_ns"] - correction * 1e6
            )
            frame["pts_baseline_correction_ms"] = correction
            frame["pts_baseline_observation_monotonic_ns"] = pts_estimate
            if start is not None and end is not None:
                frame["pts_signed_distance_to_qr_interval_ms"] = float(
                    _signed_interval_distance(
                        np.asarray([pts_estimate / 1e6]),
                        np.asarray([start / 1e6]),
                        np.asarray([end / 1e6]),
                    )[0]
                )

    arrival_holdout = evaluation["arrival_affine_holdout"]
    pts_holdout = evaluation["pts_constant_holdout"]
    summary.update({
        "status": (
            "chronologically_validated"
            if len(anchors) >= minimum_anchors and holdout else
            "provisional_insufficient_anchors"
        ),
        "minimum_anchors_for_validated_status": minimum_anchors,
        "evaluation": evaluation,
        "arrival_model": {
            "kind": "affine_host_arrival_delay",
            "equation": (
                "observation_ns = received_ns - 1e6 * "
                "(delay_intercept_ms + delay_drift_ms_per_second * "
                "(received_ns - origin_received_monotonic_ns) / 1e9)"
            ),
            "fit_objective": "squared distance outside QR/flip time intervals",
            "final_fit_uses_all_segment_anchors": True,
            **final_affine,
        },
        "arrival_constant_baseline": {
            "kind": "constant_host_arrival_delay",
            **final_constant,
        },
        "pts_constant_baseline": (
            {"kind": "constant_host_anchored_pts_correction", **final_pts}
            if final_pts is not None else None
        ),
        "arrival_model_beats_pts_holdout": (
            arrival_holdout["mae_ms"] < pts_holdout["mae_ms"]
            if arrival_holdout is not None and pts_holdout is not None else None
        ),
        "deployment_recommendation": (
            "Offline gap-filling result only. Do not enable as a live correction "
            "without a later independent recording."
        ),
    })
    return summary


def _csv_value(value):
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    temporary.replace(path)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _plotting():
    """Load Matplotlib lazily after recording and decoding have finished."""
    global _PLOTTING
    if _PLOTTING is None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot
        _PLOTTING = pyplot
    return _PLOTTING


def _rolling_median(values: np.ndarray, window: int = 21) -> np.ndarray:
    result = np.full(len(values), np.nan)
    half_window = window // 2
    for index in range(len(values)):
        sample = values[
            max(0, index - half_window):min(len(values), index + half_window + 1)
        ]
        if np.isfinite(sample).any():
            result[index] = np.nanmedian(sample)
    return result


def _overview_segment(report: dict) -> dict | None:
    candidates = [
        segment
        for segment in report.get("segments", ())
        if isinstance(segment.get("evaluation"), dict)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda segment: segment["evaluation"].get(
            "chronological_holdout_anchors", 0
        ),
    )


def _write_overview_graph(report: dict, path: Path) -> None:
    """Summarize distance-model fit, holdout error, and QR-state evidence."""
    plt = _plotting()
    frames = report.get("frames", ())
    frame_number = np.asarray(
        [frame.get("frame_number", index) for index, frame in enumerate(frames, 1)],
        dtype=float,
    )

    def values(key: str) -> np.ndarray:
        return np.asarray([
            float(frame[key]) if frame.get(key) is not None else np.nan
            for frame in frames
        ])

    received_ns = values("received_monotonic_ns")
    arrival_distance = values("signed_distance_to_qr_interval_ms")
    pts_distance = values("pts_signed_distance_to_qr_interval_ms")
    arrival_delay = values("predicted_arrival_delay_ms")
    finite_received = received_ns[np.isfinite(received_ns)]
    elapsed_seconds = (
        (received_ns - np.min(finite_received)) / 1e9
        if len(finite_received) else np.full(len(frames), np.nan)
    )

    orange = "#d97706"
    blue = "#2563eb"
    teal = "#0f766e"
    slate = "#64748b"
    red = "#b91c1c"
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    figure.suptitle(
        "Calibration distance analysis overview",
        fontsize=17,
        fontweight="bold",
    )

    axis = axes[0, 0]
    axis.scatter(frame_number, arrival_distance, s=7, alpha=0.16, color=orange)
    axis.scatter(frame_number, pts_distance, s=7, alpha=0.14, color=blue)
    axis.plot(
        frame_number,
        _rolling_median(arrival_distance),
        linewidth=2.1,
        color=orange,
        label="Arrival affine, 21-frame median",
    )
    axis.plot(
        frame_number,
        _rolling_median(pts_distance),
        linewidth=2.1,
        color=blue,
        label="PTS constant, 21-frame median",
    )
    axis.axhline(0, color="#111827", linewidth=1, alpha=0.75)
    segment = _overview_segment(report)
    if segment is not None:
        evaluation = segment["evaluation"]
        boundary = evaluation.get("train_last_frame_number")
        if boundary is not None:
            axis.axvline(
                boundary + 0.5,
                color=slate,
                linewidth=1.4,
                linestyle="--",
                label="Train / holdout boundary",
            )
    axis.set(
        title="Full-recording fitted distance to QR interval",
        xlabel="Camera frame",
        ylabel="Signed interval distance (ms)",
    )
    axis.legend(loc="lower left", fontsize=8, frameon=False)
    axis.grid(axis="y", alpha=0.18)

    axis = axes[0, 1]
    if segment is None:
        axis.text(0.5, 0.5, "No chronological holdout", ha="center", va="center")
        axis.set_axis_off()
    else:
        evaluation = segment["evaluation"]
        models = [
            ("Arrival\naffine", evaluation.get("arrival_affine_holdout")),
            ("Arrival\nconstant", evaluation.get("arrival_constant_holdout")),
            ("PTS\nconstant", evaluation.get("pts_constant_holdout")),
        ]
        models = [(label, metrics) for label, metrics in models if metrics]
        metrics = (
            ("MAE", "mae_ms", blue),
            ("Median absolute", "median_absolute_ms", teal),
            ("P95 absolute", "p95_absolute_ms", orange),
        )
        positions = np.arange(len(models))
        width = 0.23
        for metric_index, (label, key, color) in enumerate(metrics):
            metric_values = [model[key] for _, model in models]
            bars = axis.bar(
                positions + (metric_index - 1) * width,
                metric_values,
                width,
                label=label,
                color=color,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.1f}" for value in metric_values],
                padding=2,
                fontsize=8,
            )
        axis.set_xticks(positions, [label for label, _ in models])
        axis.set(
            title=(
                "Chronological holdout error "
                f"({evaluation.get('chronological_holdout_anchors', 0)} frames)"
            ),
            ylabel="Holdout error (ms)",
        )
        axis.legend(frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=0.18)

    axis = axes[1, 0]
    axis.plot(
        elapsed_seconds,
        arrival_delay,
        color=orange,
        linewidth=2.2,
        label="Affine delay fitted on all anchors",
    )
    if segment is not None:
        constant_model = segment.get("arrival_constant_baseline") or {}
        constant = constant_model.get("delay_intercept_ms")
        if constant is not None:
            axis.axhline(
                constant,
                color=slate,
                linewidth=1.8,
                linestyle="--",
                label=f"Constant baseline: {constant:.1f} ms",
            )
        affine_model = segment.get("arrival_model") or {}
        drift = affine_model.get("delay_drift_ms_per_second")
        if drift is not None:
            axis.text(
                0.01,
                0.03,
                f"Affine drift: {drift:+.3f} ms/s",
                transform=axis.transAxes,
                fontsize=9,
                color=red,
            )
    axis.set(
        title="Arrival-delay estimate across the recording",
        xlabel="Elapsed receipt time (s)",
        ylabel="Predicted arrival delay (ms)",
    )
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.18)

    axis = axes[1, 1]
    state_counts = report.get("counts", {}).get("all_markers_state_status", {})
    state_order = (
        "mixed_or_rolling_generations",
        "replacement_boundary_missing",
        "compatible_global_state",
    )
    state_labels = (
        "Mixed / rolling generations",
        "Replacement boundary missing",
        "Compatible global state",
    )
    state_values = [state_counts.get(state, 0) for state in state_order]
    bars = axis.barh(state_labels, state_values, color=(orange, slate, teal))
    axis.bar_label(bars, labels=[str(value) for value in state_values], padding=4)
    axis.set(
        title="All-marker visibility-state diagnostic",
        xlabel="Camera frames",
    )
    axis.set_xlim(0, max(max(state_values, default=0) * 1.12, 1))
    axis.grid(axis="x", alpha=0.18)
    axis.invert_yaxis()

    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_overview_graph_compatibly(report: dict, output: Path) -> None:
    """Render locally or switch to the installed compatible plotting pair."""
    from calibration.quantitative_analysis import matplotlib_environment

    environment = matplotlib_environment()
    current_system_only = os.environ.get("PYTHONNOUSERSITE") == "1"
    selected_system_only = environment.get("PYTHONNOUSERSITE") == "1"
    if selected_system_only != current_system_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "calibration.distance_analysis",
                "--render-existing",
                str(output),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
        )
        return
    os.environ.setdefault("MPLCONFIGDIR", environment["MPLCONFIGDIR"])
    _write_overview_graph(report, output / OVERVIEW_GRAPH)


def analyze_recording(
    recording_directory: str | Path,
    *,
    analysis_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    train_fraction: float = TRAIN_FRACTION,
    minimum_anchors: int = MINIMUM_ANCHORS,
) -> dict:
    """Run the distance analysis and save its frame and marker maps."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be greater than 0 and less than 1")
    if minimum_anchors < 2:
        raise ValueError("minimum_anchors must be at least 2")

    recording = Path(recording_directory).expanduser().resolve()
    if not recording.is_dir():
        raise ValueError(f"Recording directory does not exist: {recording}")
    analysis = (
        Path(analysis_directory).expanduser().resolve()
        if analysis_directory is not None
        else recording.with_name(recording.name + "_analysis")
    )
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None else analysis
    )
    if not analysis.is_dir():
        raise ValueError(f"Analysis directory does not exist: {analysis}")
    output.mkdir(parents=True, exist_ok=True)

    display_metadata, events, by_payload, pause_boundaries = _load_display_timeline(
        recording
    )
    source_analysis, analysis_frames, analysis_path = _load_analysis(analysis)
    camera_rows, camera_path = _load_camera_rows(recording)
    epoch_anchors, session_path = _load_epoch_anchors(recording)
    visible_qrs = _finite_int(display_metadata.get("visible_qrs"))
    if visible_qrs is None or visible_qrs < 1:
        raise ValueError("Display journal has an invalid visible_qrs value")

    frames, markers = _build_frames(
        camera_rows,
        analysis_frames,
        events,
        by_payload,
        visible_qrs,
        epoch_anchors,
    )
    _assign_segments(frames, pause_boundaries)
    segment_reports = []
    for segment_id in sorted({frame["segment_id"] for frame in frames}):
        segment_frames = [
            frame for frame in frames if frame["segment_id"] == segment_id
        ]
        segment_reports.append(_annotate_segment(
            segment_id,
            segment_frames,
            train_fraction,
            minimum_anchors,
        ))

    provenance = {
        str(camera_path.relative_to(recording)): _sha256(camera_path),
        DISPLAY_JOURNAL: _sha256(recording / DISPLAY_JOURNAL),
        str(analysis_path): _sha256(analysis_path),
    }
    if session_path is not None:
        provenance[str(session_path.relative_to(recording))] = _sha256(session_path)

    anchor_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    estimation_counts: dict[str, int] = {}
    for frame in frames:
        for counts, key in (
            (anchor_counts, frame["anchor_status"]),
            (state_counts, frame["all_markers_state_status"]),
            (estimation_counts, frame["estimation_kind"]),
        ):
            counts[key] = counts.get(key, 0) + 1

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "recording_directory": str(recording),
        "analysis_directory": str(analysis),
        "output_directory": str(output),
        "target": (
            "Estimated newest-observed display-region time on the host monotonic "
            "clock, derived from frame receipt time and QR/flip interval anchors"
        ),
        "timestamp_semantics": {
            "qr_payload": "Identity of the predicted display generation; not exact exposure time",
            "presentation_return": display_metadata.get(
                "presentation_semantics",
                "Software flip/swap return boundary; not physical panel scanout",
            ),
            "estimated_observation": (
                "Arrival-regression estimate for the region containing the newest "
                "readable QR; a rolling-shutter image has no single physical exposure instant"
            ),
            "pts_baseline": (
                "Independent host-anchored PTS comparison; never used as the arrival model target"
            ),
        },
        "evidence_policy": {
            "anchor": (
                "Interval from the newest mapped QR's presentation return to the next "
                "presentation return"
            ),
            "all_markers": (
                "Every mapped marker is exported and its complete visibility intervals "
                "are intersected as a global-state diagnostic"
            ),
            "mixed_generations": (
                "An empty all-marker intersection is retained as rolling-shutter/mixed-state "
                "evidence; it does not turn a QR payload into an exact timestamp"
            ),
            "gap_fill": (
                "Frames without a usable QR interval receive an arrival-model interpolation "
                "or explicitly labeled extrapolation"
            ),
        },
        "configuration": {
            "visible_qrs": visible_qrs,
            "train_fraction": train_fraction,
            "minimum_anchors": minimum_anchors,
            "maximum_camera_gap_ms": MAX_CAMERA_GAP_NS / 1e6,
            "maximum_anchor_interval_periods": MAX_ANCHOR_PERIODS,
        },
        "counts": {
            "camera_frames": len(frames),
            "display_events": len(events),
            "marker_observations": len(markers),
            "mapped_marker_observations": sum(
                marker["display_index"] is not None for marker in markers
            ),
            "anchor_status": anchor_counts,
            "all_markers_state_status": state_counts,
            "estimation_kind": estimation_counts,
        },
        "segments": segment_reports,
        "source_analysis_summary": {
            "processed": source_analysis.get("processed"),
            "total": source_analysis.get("total"),
            "cancelled": source_analysis.get("cancelled"),
            "stopped": source_analysis.get("stopped"),
        },
        "source_provenance_sha256": provenance,
        "physical_boundary": (
            "This analysis does not measure monitor processing, panel scanout, photon "
            "output, camera exposure duration, or per-row rolling-shutter time"
        ),
        "output_files": [REPORT_JSON, FRAMES_CSV, MARKERS_CSV, OVERVIEW_GRAPH],
        "graph_formats": ["png"],
        "frames": frames,
        "markers": markers,
    }
    _write_csv(output / FRAMES_CSV, frames)
    _write_csv(output / MARKERS_CSV, markers)
    _write_json(output / REPORT_JSON, report)
    _write_overview_graph_compatibly(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_directory", type=Path, nargs="?")
    parser.add_argument(
        "--render-existing",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--analysis-directory",
        type=Path,
        help="Existing QR analysis directory (default: sibling <recording>_analysis)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Destination for distance outputs (default: the QR analysis directory)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=TRAIN_FRACTION,
        help=f"Chronological training fraction (default: {TRAIN_FRACTION:g})",
    )
    parser.add_argument(
        "--minimum-anchors",
        type=int,
        default=MINIMUM_ANCHORS,
        help=f"Minimum anchors for validated status (default: {MINIMUM_ANCHORS})",
    )
    arguments = parser.parse_args()
    if arguments.render_existing is not None:
        output = arguments.render_existing.expanduser().resolve()
        report = json.loads((output / REPORT_JSON).read_text(encoding="utf-8"))
        _write_overview_graph(report, output / OVERVIEW_GRAPH)
        return
    if arguments.recording_directory is None:
        parser.error("recording_directory is required")
    report = analyze_recording(
        arguments.recording_directory,
        analysis_directory=arguments.analysis_directory,
        output_directory=arguments.output_directory,
        train_fraction=arguments.train_fraction,
        minimum_anchors=arguments.minimum_anchors,
    )
    print(
        f"Estimated {report['counts']['camera_frames']} camera frame(s) across "
        f"{len(report['segments'])} segment(s)."
    )
    print(f"Saved {', '.join(report['output_files'])} in {report['output_directory']}")


if __name__ == "__main__":
    main()
