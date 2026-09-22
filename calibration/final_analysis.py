#!/usr/bin/env python3
"""Shared journal reconstruction and conditional QR interval scoring.

Model fitting and model selection were retired for the frozen-anchor experiment.
Use analyze_pts_anchor.py to calibrate an offset or evaluate independent streams.
This module retains evidence validation; it is not an analysis CLI.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration.evidence import assess_evidence, repeated_state_checks

TARGETS = ("newest_generation", "newest_visibility", "common_visibility")


@dataclass(frozen=True)
class Goals:
    median_ms: float = 5.0
    threshold_ms: float = 10.0
    coverage_pct: float = 95.0


@dataclass
class Session:
    name: str
    directory: Path
    recording: Path
    frames: list[dict]
    provenance: dict
    audit: dict
    stream_keys: set[str]
    identity_complete: bool


def _integer(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        number = int(value)
        return number if isinstance(value, str) or value == number else None
    except (ValueError, TypeError, OverflowError):
        return None


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _journal(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get(
            "frames", value.get("camera_timestamps", [])
        )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty journal rows: {path}")
    return rows


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounds(reference: int, start: int, end: int) -> list[float] | None:
    if start >= end:
        return None
    # Subtract integers BEFORE converting to floating point milliseconds.
    return [(reference - end) / 1e6, (reference - start) / 1e6]


def load_session(directory: str | Path) -> Session:
    """Reconstruct timing bounds independently of saved offsets/acceptance labels."""
    directory = Path(directory).expanduser().resolve()
    if not (directory / "calibration_analysis.json").is_file():
        directory = directory.with_name(directory.name + "_analysis")
    analysis_path = directory / "calibration_analysis.json"
    source = _json(analysis_path)
    # Prefer the sibling checkout when recordings have moved between machines.
    sibling = directory.with_name(directory.name.removesuffix("_analysis"))
    recorded_path = source.get("recording_directory")
    recording = (sibling if sibling != directory and (sibling / "display_timestamps.jsonl").is_file()
                 else Path(recorded_path).expanduser() if recorded_path else sibling)
    if not recording.is_absolute():
        recording = (Path.cwd() / recording).resolve()
    display_path = recording / "display_timestamps.jsonl"
    camera_path = next((recording / name for name in
                        ("camera_timestamps.jsonl", "camera_timestamps.json")
                        if (recording / name).is_file()), None)
    if camera_path is None:
        raise ValueError(f"No camera journal in {recording}")
    epoch_path = recording / "camera_timing_session.json"
    summary_path = recording / "camera_recording_summary.json"
    events_path = recording / "camera_timing_events.jsonl"
    recording_summary = _json(summary_path) if summary_path.is_file() else {}
    timing_events = (
        [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if events_path.is_file() else []
    )
    event_counts = Counter(
        str(row.get("event", "unknown"))
        for row in timing_events
        if isinstance(row, dict)
    )
    epochs = _json(epoch_path) if epoch_path.is_file() else {}
    epoch_map = {}
    legacy_epoch_map = {}
    stream_identity_anchor = {}
    stream_identity_keys = defaultdict(set)
    for epoch in epochs.get("epochs", []):
        stream_epoch = _integer(epoch.get("stream_epoch"))
        mapping_revision = _integer(epoch.get("mapping_revision"))
        segment_epoch = _integer(epoch.get("segment_epoch"))
        zero = _integer(epoch.get("pipeline_zero_monotonic_ns"))
        if stream_epoch is not None and zero is not None:
            key = (stream_epoch, mapping_revision, segment_epoch)
            if key in epoch_map and epoch_map[key] != zero:
                raise ValueError(f"Conflicting epoch anchors: {epoch_path}")
            epoch_map[key] = zero
            stream_identity_anchor.setdefault(stream_epoch, zero)
            base_time = _integer(epoch.get("pipeline_base_time_ns"))
            if base_time is not None:
                # Unlike sampled host anchors, the pipeline base time does not
                # jitter when a Python clock wrapper falsely triggers remapping.
                stream_identity_keys[stream_epoch].add(f"pipeline-base:{base_time}:{stream_epoch}")
            if mapping_revision is None and segment_epoch is None:
                legacy_epoch_map[stream_epoch] = zero
    journal = _journal(display_path)
    metadata = [row for row in journal if row.get("kind") == "session"]
    if len(metadata) != 1:
        raise ValueError(f"Expected exactly one display session in {display_path}")
    metadata = metadata[0]
    grid = _integer(metadata.get("grid_qrs", 4))
    visible = _integer(metadata.get("visible_qrs"))
    if grid is None or visible is None or not 1 <= visible <= grid:
        raise ValueError(f"Invalid grid/visible counts in {display_path}")
    events = [row for row in journal if row.get("kind") == "frame"]
    times, payloads = [], defaultdict(list)
    for index, row in enumerate(events):
        stamp = _integer(row.get("presentation_return_ns", row.get("flip_return_ns")))
        marker = _integer(row.get("marker_ns"))
        cell = _integer(row.get("cell", row.get("corner")))
        if (row.get("index") != index or stamp is None or marker is None
                or cell is None or not 0 <= cell < grid or (times and stamp <= times[-1])):
            raise ValueError(f"Invalid/reordered display event {index} in {display_path}")
        row = dict(row, cell=cell)
        events[index] = row
        times.append(stamp)
        payloads[f"{marker // 1_000_000 % 1_000_000_000_000:012d}"].append(index)
    if len(events) < 2:
        raise ValueError("At least two presentation events are required")
    period = float(np.median(np.diff(np.asarray(times, dtype=np.int64)))) / 1e6
    pauses = [_integer(row.get("monotonic_ns")) for row in journal
              if row.get("kind") == "pause"]
    pauses = [stamp for stamp in pauses if stamp is not None]
    # Simulate documented cell expiration/update, rather than adding a hardcoded
    # number of events to every marker, including across irregular cell orders.
    active, lifetimes, next_cell_update = {}, {}, {}
    last_update = {}
    for index, event in enumerate(events):
        cell = event["cell"]
        for expired_cell in {(cell - visible) % grid, cell}:
            if expired_cell in active:
                lifetimes[active.pop(expired_cell)] = index
        active[cell] = index
        if cell in last_update:
            next_cell_update[last_update[cell]] = index
        last_update[cell] = index
    decoded = {}
    for row in source.get("frames", []):
        filename = row.get("filename")
        if not filename or filename in decoded:
            raise ValueError(f"Missing/duplicate decoded filename in {analysis_path}")
        decoded[filename] = row
    cameras = _journal(camera_path)
    seen, output, counts = set(), [], Counter()
    previous = None
    segment = 0
    stream_keys = set()
    identity_complete = True
    for number, camera in enumerate(cameras, 1):
        filename = camera.get("frame") or camera.get("camera_frame")
        if not filename or filename in seen:
            raise ValueError(f"Missing/duplicate camera filename in {camera_path}")
        seen.add(filename)
        decoded_row = decoded.get(filename, {})
        epoch = _integer(camera.get("stream_epoch"))
        mapping_revision = _integer(camera.get("mapping_revision"))
        segment_epoch = _integer(camera.get("segment_epoch"))
        zero = epoch_map.get((epoch, mapping_revision, segment_epoch))
        if zero is None:
            zero = legacy_epoch_map.get(epoch)
        pts = _integer(camera.get("pts_ns"))
        running_time = _integer(camera.get("running_time_ns"))
        received = _integer(camera.get("application_arrival_monotonic_ns"))
        if received is None:
            received = _integer(
                camera.get(
                    "received_monotonic_ns",
                    camera.get("host_monotonic_received_ns"),
                )
            )
        saved_media_monotonic = _integer(camera.get("media_monotonic_ns"))
        if saved_media_monotonic is not None:
            reference = saved_media_monotonic
            reference_method = "saved_segment_running_time_mapping"
        elif zero is not None and running_time is not None:
            reference = zero + running_time
            reference_method = "replayed_segment_running_time_mapping"
        elif zero is not None and pts is not None:
            reference = zero + pts
            reference_method = "legacy_raw_pts_fallback"
        else:
            reference = None
            reference_method = "unavailable"
        counts[f"reference_{reference_method}"] += 1
        identity_zero = stream_identity_anchor.get(epoch)
        identity_keys = stream_identity_keys.get(epoch) or (
            {f"{identity_zero}:{epoch}"} if identity_zero is not None else set()
        )
        if identity_zero is None:
            identity_complete = False
        else:
            stream_keys.update(identity_keys)
        continuity_key = (epoch, mapping_revision, segment_epoch)
        continuous = (
            previous is not None
            and continuity_key == previous[0]
            and running_time is not None
            and previous[1] is not None
            and 0 < running_time - previous[1] <= 100_000_000
            and received is not None
            and previous[2] is not None
            and 0 < received - previous[2] <= 1_000_000_000
            and not any(previous[2] < pause <= received for pause in pauses)
        )
        reset_reasons = []
        if not continuous:
            if previous is None:
                reset_reasons.append("first_frame")
            else:
                if continuity_key != previous[0]:
                    reset_reasons.append("stream_mapping_or_segment_changed")
                if (running_time is None or previous[1] is None
                        or not 0 < running_time - previous[1] <= 100_000_000):
                    reset_reasons.append("invalid_or_large_running_time_gap")
                if (received is None or previous[2] is None
                        or not 0 < received - previous[2] <= 1_000_000_000):
                    reset_reasons.append("invalid_or_large_arrival_gap")
                if (received is not None and previous[2] is not None
                        and any(previous[2] < pause <= received for pause in pauses)):
                    reset_reasons.append("display_pause")
        if not continuous:
            segment += 1
        previous = (continuity_key, running_time, received, reference)
        raw_values = decoded_row.get("qr_values_ms", [])
        if not isinstance(raw_values, list):
            raw_values = []
        matched = []
        bad, wrong_cell = 0, 0
        for cell, value in enumerate(raw_values):
            if value is None or value == "":
                continue
            candidates = payloads.get(str(value), [])
            # Never disambiguate a payload by proximity to a fitted correction.
            if len(candidates) != 1:
                bad += 1
                continue
            index = candidates[0]
            if cell != events[index]["cell"]:
                wrong_cell += 1
            matched.append(index)
        indices = sorted(set(matched))
        latest = indices[-1] if indices else None
        targets = {target: None for target in TARGETS}
        reasons = []
        if not decoded_row:
            reasons.append("no_decoded_frame")
        if reference is None:
            reasons.append("missing_segment_running_time_mapping")
        if not indices:
            reasons.append("no_unique_qr_identity")
        if bad or wrong_cell or len(indices) != len(matched):
            reasons.append("ambiguous_unmapped_or_conflicting_qr_identity")
        newest_end = latest + 1 if latest is not None and latest + 1 < len(times) else None
        if latest is not None and newest_end is None:
            reasons.append("missing_next_presentation")
        usable = reference is not None and latest is not None and not bad and not wrong_cell
        usable = usable and len(indices) == len(matched)
        visibility_ends = [lifetimes.get(index) for index in indices]
        complete = bool(indices) and all(end is not None for end in visibility_ends)
        common_start = times[latest] if latest is not None else None
        common_end = min(times[end] for end in visibility_ends) if complete else None
        compatible = complete and common_start < common_end
        overwrite_ends = [next_cell_update.get(index) for index in indices]
        overwrite_complete = bool(indices) and all(end is not None for end in overwrite_ends)
        overwrite_compatible = (overwrite_complete and common_start < min(
            times[end] for end in overwrite_ends))
        if usable and newest_end is not None:
            if any(times[latest] <= pause <= times[newest_end] for pause in pauses):
                reasons.append("presentation_interval_crosses_pause")
            elif times[newest_end] - times[latest] > 3 * period * 1e6:
                reasons.append("presentation_interval_over_three_periods")
            else:
                targets["newest_generation"] = _bounds(reference, times[latest], times[newest_end])
        if usable and latest in lifetimes:
            targets["newest_visibility"] = _bounds(reference, times[latest], times[lifetimes[latest]])
        if usable and compatible:
            targets["common_visibility"] = _bounds(reference, common_start, common_end)
        diagnostic_interval = targets["newest_generation"]
        assessment = assess_evidence(
            dict(decoded_row, temporal_evidence=None), indices, transition=bool(complete and not compatible),
        )
        if not assessment["primary_usable"]:
            targets["newest_generation"] = None
            reasons.extend(assessment["reasons"])
        saved_reference = _integer(decoded_row.get("camera_reference_monotonic_ns"))
        flags = {
            "more_codes_than_configured_visible": len(indices) > visible,
            "all_grid_cells_decoded": len(indices) == grid,
            "visibility_contradiction": complete and not compatible,
            "visibility_end_missing": bool(indices) and not complete,
            "compatible_if_codes_persist_until_cell_update": bool(overwrite_compatible),
            "upstream_position_warnings": bool(decoded_row.get("position_warnings")),
            "manual_qr_values": bool(decoded_row.get("manual_values")),
            "saved_reference_disagrees": (reference is not None and saved_reference is not None
                                          and reference != saved_reference),
            "display_timing_flag": any(
                events[index].get("late_submit") or events[index].get("irregular_interval")
                or events[index].get("skipped_periods")
                for index in (latest, newest_end) if index is not None),
        }
        counts.update(key for key, enabled in flags.items() if enabled)
        counts.update(reasons)
        counts.update(assessment.get("warnings", []))
        counts["evidence_" + assessment["status"]] += 1
        counts.update("continuity_reset_" + reason for reason in reset_reasons)
        output.append({
            "recording": directory.name, "frame_number": number, "filename": filename,
            "segment": segment, "pts_ns": pts,
            "running_time_ns": running_time,
            "application_arrival_monotonic_ns": received,
            "legacy_received_monotonic_ns": _integer(camera.get("received_monotonic_ns")),
            "sample_pulled_monotonic_ns": _integer(camera.get("sample_pulled_monotonic_ns")),
            "frame_converted_monotonic_ns": _integer(camera.get("frame_converted_monotonic_ns")),
            "capture_queue_level_buffers": _integer(camera.get("capture_queue_level_buffers")),
            "stream_key": (
                sorted(identity_keys)[0] if identity_keys else None
            ),
            "media_reference_monotonic_ns": reference,
            "reference_reconstruction": reference_method,
            "observable_arrival_minus_media_ms": (
                (received - reference) / 1e6
                if received is not None and reference is not None else None
            ),
            "continuity_reset_reasons": reset_reasons,
            "recorded_timing_flags": camera.get("flags", []),
            "evidence_assessment": assessment,
            "qr_evidence": decoded_row.get("qr_evidence"),
            "decoded_display_indices": indices,
            "diagnostic_newest_decoded_interval_ms": diagnostic_interval,
            "image_path": str(recording / filename),
            "targets": targets, "flags": flags, "exclusion_reasons": reasons,
            "decoded_unique_qrs": len(indices), "grid_qrs": grid, "visible_qrs": visible,
            "latest_display_index": latest,
            "latest_journal_cell": events[latest]["cell"] if latest is not None else None,
            "all_marker_gap_ms": ((common_start - common_end) / 1e6 if complete else None),
            "display_period_ms": period,
        })
    # Recompute from raw timing and decoded identities, including legacy reports.
    paused_indices = {row.get("last_frame_index") for row in journal
                      if row.get("kind") == "pause" and row.get("paused")}
    for frame, temporal in zip(output, repeated_state_checks(output, times, paused_indices)):
        frame["temporal_evidence"] = temporal
        if temporal is None:
            continue
        assessment = frame["evidence_assessment"]
        counts["evidence_" + assessment["status"]] -= 1
        assessment["status"] = "suspected_stale_visual_state"
        assessment["primary_usable"] = False
        reason = temporal["reason"]
        if reason not in assessment["reasons"]:
            assessment["reasons"].append(reason)
        if reason not in frame["exclusion_reasons"]:
            frame["exclusion_reasons"].append(reason)
            counts[reason] += 1
        counts["evidence_suspected_stale_visual_state"] += 1
        # Keep the unfiltered interval above for diagnostics, never fit this run.
        frame["targets"] = {target: None for target in TARGETS}
    counts["camera_frames"] = len(output)
    counts["decoded_frames_not_in_camera_journal"] = len(set(decoded) - seen)
    counts["primary_scored_frames"] = sum(row["targets"]["newest_generation"] is not None
                                         for row in output)
    counts["common_visibility_frames"] = sum(row["targets"]["common_visibility"] is not None
                                            for row in output)
    counts["recording_summary_frames_rejected_invalid_timing"] = int(
        recording_summary.get("frames_rejected_invalid_timing", 0) or 0
    )
    counts["recording_summary_frames_dropped_writer_queue"] = int(
        recording_summary.get("frames_dropped_writer_queue", 0) or 0
    )
    counts["timing_event_rejected_invalid_timing"] = event_counts.get(
        "frame_rejected_invalid_timing", 0
    )
    counts["timing_event_dropped_writer_queue"] = event_counts.get(
        "frame_dropped_writer_queue", 0
    )
    paths = [analysis_path, camera_path, display_path]
    paths.extend(
        path
        for path in (epoch_path, summary_path, events_path)
        if path.is_file()
    )
    provenance = {str(path): _hash(path) for path in paths}
    return Session(directory.name, directory, recording, output, provenance,
                   {"counts": dict(counts), "display_period_ms": period,
                    "grid_qrs": grid, "visible_qrs": visible,
                    "presentation_semantics": metadata.get("presentation_semantics"),
                    "camera_journal_sha256": _hash(camera_path),
                    "unsaved_frame_accounting": {
                        "summary": {
                            key: recording_summary.get(key)
                            for key in (
                                "frames_observed",
                                "frames_selected",
                                "frames_saved",
                                "frames_rejected_invalid_timing",
                                "frames_dropped_writer_queue",
                                "confirmed_frames_not_saved",
                            )
                        },
                        "timing_event_counts": dict(event_counts),
                    },
                    "decoded_inputs_already_remapped_by_upstream_reader": True},
                   stream_keys, identity_complete)


def _residual(bounds: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Correction residual: positive needs more subtraction; zero is compatible."""
    return np.maximum(bounds[:, 0] - prediction, 0) - np.maximum(prediction - bounds[:, 1], 0)


def _score(bounds: np.ndarray, prediction: np.ndarray, goals: Goals) -> dict:
    absolute = np.abs(_residual(bounds, prediction))
    midpoint = np.mean(bounds, axis=1)
    worst = np.maximum(np.abs(bounds[:, 0] - prediction), np.abs(bounds[:, 1] - prediction))
    widths = bounds[:, 1] - bounds[:, 0]
    coverage = float(100 * np.mean(absolute < goals.threshold_ms))
    median = float(np.median(absolute))
    maximum = float(max(absolute))
    at_or_above = int(np.sum(absolute >= goals.threshold_ms))
    return {
        "n": len(absolute), "mae_ms": float(np.mean(absolute)),
        "median_absolute_ms": median, "p95_absolute_ms": float(np.percentile(absolute, 95)),
        "p99_absolute_ms": float(np.percentile(absolute, 99)), "maximum_absolute_ms": maximum,
        "errors_at_or_above_threshold": at_or_above,
        "errors_at_or_above_threshold_pct": float(100 * at_or_above / len(absolute)),
        "bias_ms": float(np.mean(_residual(bounds, prediction))),
        "inside_interval_pct": float(100 * np.mean(absolute == 0)),
        "below_threshold_pct": coverage,
        "below_median_goal_pct": float(100 * np.mean(absolute < goals.median_ms)),
        "interval_width_ms": {
            "minimum": float(np.min(widths)),
            "median": float(np.median(widths)),
            "p95": float(np.percentile(widths, 95)),
            "maximum": float(np.max(widths)),
        },
        "median_interval_width_ms": float(np.median(widths)),
        "midpoint_mae_ms": float(np.mean(np.abs(midpoint - prediction))),
        "worst_endpoint_median_ms": float(np.median(worst)),
        "worst_endpoint_p95_ms": float(np.percentile(worst, 95)),
        "worst_endpoint_maximum_ms": float(np.max(worst)),
        "worst_endpoint_below_threshold_pct": float(100 * np.mean(worst < goals.threshold_ms)),
        "strict_maximum_goal_passed": bool(maximum < goals.threshold_ms),
        "strict_median_goal_passed": bool(median < goals.median_ms),
        "conditional_goal_passed": bool(
            maximum < goals.threshold_ms and median < goals.median_ms
        ),
    }


def _stream_components(sessions: list[Session]) -> list[list[Session]]:
    """Keep overlapping epoch identities together, including multi-epoch files."""
    groups = []
    for session in sessions:
        if not session.identity_complete or not session.stream_keys:
            continue
        matching = [group for group in groups if any(
            member.stream_keys & session.stream_keys for member in group)]
        combined = [session]
        for group in matching:
            combined.extend(group)
            groups.remove(group)
        groups.append(combined)
    return groups
