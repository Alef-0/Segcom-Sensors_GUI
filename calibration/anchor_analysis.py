"""Rebuild conditional QR timing intervals from saved journals and evidence."""

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
    try:
        number = int(value)
        return number if isinstance(value, str) or number == value else None
    except (TypeError, ValueError, OverflowError):
        return None


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _journal(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("frames", value.get("camera_timestamps", []))
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty journal rows: {path}")
    return rows


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounds(reference_ns: int, start_ns: int, end_ns: int) -> list[float] | None:
    if start_ns >= end_ns:
        return None
    return [(reference_ns - end_ns) / 1e6, (reference_ns - start_ns) / 1e6]


def _display_windows(events: list[dict], visible: int, grid: int) -> tuple[dict, dict]:
    """Map each marker to its visible lifetime and next reuse of its cell."""
    active: dict[int, int] = {}
    lifetimes, cell_updates = {}, {}
    previous_update = {}
    for index, event in enumerate(events):
        cell = event["cell"]
        expired = (cell - visible) % grid
        if expired in active:
            lifetimes[active.pop(expired)] = index
        if cell in active:
            lifetimes[active.pop(cell)] = index
        active[cell] = index
        if cell in previous_update:
            cell_updates[previous_update[cell]] = index
        previous_update[cell] = index
    return lifetimes, cell_updates


def _epoch_maps(epoch_rows: list[dict], path: Path):
    anchors, legacy, identities = {}, {}, defaultdict(set)
    first_zero = {}
    for row in epoch_rows:
        stream = _integer(row.get("stream_epoch"))
        mapping = _integer(row.get("mapping_revision"))
        segment = _integer(row.get("segment_epoch"))
        zero = _integer(row.get("pipeline_zero_monotonic_ns"))
        if stream is None or zero is None:
            continue
        key = (stream, mapping, segment)
        if key in anchors and anchors[key] != zero:
            raise ValueError(f"Conflicting epoch anchors: {path}")
        anchors[key] = zero
        first_zero.setdefault(stream, zero)
        base_time = _integer(row.get("pipeline_base_time_ns"))
        identities[stream].add(
            f"pipeline-base:{base_time}:{stream}" if base_time is not None
            else f"{zero}:{stream}"
        )
        if mapping is None and segment is None:
            legacy[stream] = zero
    return anchors, legacy, first_zero, identities


def _event_counts(path: Path) -> Counter:
    if not path.is_file():
        return Counter()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Counter(str(row.get("event", "unknown")) for row in rows if isinstance(row, dict))


def load_session(directory: str | Path) -> Session:
    """Reconstruct intervals from raw camera clocks, display returns, and QR evidence."""
    directory = Path(directory).expanduser().resolve()
    if not (directory / "calibration_analysis.json").is_file():
        directory = directory.with_name(directory.name + "_analysis")
    analysis_path = directory / "calibration_analysis.json"
    source = _json(analysis_path)
    sibling = directory.with_name(directory.name.removesuffix("_analysis"))
    recorded = source.get("recording_directory")
    recording = (sibling if (sibling != directory and (sibling / "display_timestamps.jsonl").is_file())
                 else Path(recorded).expanduser() if recorded else sibling)
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
    event_path = recording / "camera_timing_events.jsonl"
    summary = _json(summary_path) if summary_path.is_file() else {}
    event_counts = _event_counts(event_path)
    epochs = _json(epoch_path).get("epochs", []) if epoch_path.is_file() else []
    anchors, legacy_anchors, first_zero, identity_by_epoch = _epoch_maps(epochs, epoch_path)

    display = _journal(display_path)
    metadata_rows = [row for row in display if row.get("kind") == "session"]
    if len(metadata_rows) != 1:
        raise ValueError(f"Expected one display session row in {display_path}")
    metadata = metadata_rows[0]
    grid = _integer(metadata.get("grid_qrs", 4))
    visible = _integer(metadata.get("visible_qrs"))
    if grid is None or visible is None or not 1 <= visible <= grid:
        raise ValueError(f"Invalid QR grid metadata in {display_path}")
    events = [row for row in display if row.get("kind") == "frame"]
    times, payload_index = [], defaultdict(list)
    for index, original in enumerate(events):
        stamp = _integer(original.get("presentation_return_ns", original.get("flip_return_ns")))
        marker = _integer(original.get("marker_ns"))
        cell = _integer(original.get("cell", original.get("corner")))
        if (original.get("index") != index or stamp is None or marker is None or cell is None
                or not 0 <= cell < grid or (times and stamp <= times[-1])):
            raise ValueError(f"Invalid or reordered display event {index} in {display_path}")
        events[index] = {**original, "cell": cell}
        times.append(stamp)
        payload_index[f"{marker // 1_000_000 % 1_000_000_000_000:012d}"].append(index)
    if len(events) < 2:
        raise ValueError("At least two presentation events are required")
    period_ns = int(np.median(np.diff(np.asarray(times, dtype=np.int64))))
    pauses = [_integer(row.get("monotonic_ns")) for row in display if row.get("kind") == "pause"]
    pauses = [stamp for stamp in pauses if stamp is not None]
    lifetimes, cell_updates = _display_windows(events, visible, grid)

    decoded = {}
    for row in source.get("frames", []):
        filename = row.get("filename")
        if not filename or filename in decoded:
            raise ValueError(f"Missing or duplicate decoded filename in {analysis_path}")
        decoded[filename] = row

    output, counts = [], Counter()
    seen, stream_keys, identity_complete = set(), set(), True
    previous, segment_id = None, 0
    camera_rows = _journal(camera_path)
    for number, camera in enumerate(camera_rows, 1):
        filename = camera.get("frame") or camera.get("camera_frame")
        if not filename or filename in seen:
            raise ValueError(f"Missing or duplicate camera filename in {camera_path}")
        seen.add(filename)
        saved = decoded.get(filename, {})
        stream = _integer(camera.get("stream_epoch"))
        mapping = _integer(camera.get("mapping_revision"))
        segment_epoch = _integer(camera.get("segment_epoch"))
        zero = anchors.get((stream, mapping, segment_epoch), legacy_anchors.get(stream))
        running = _integer(camera.get("running_time_ns"))
        pts = _integer(camera.get("pts_ns"))
        arrival = _integer(camera.get("application_arrival_monotonic_ns"))
        if arrival is None:
            arrival = _integer(camera.get("received_monotonic_ns", camera.get("host_monotonic_received_ns")))
        saved_reference = _integer(camera.get("media_monotonic_ns"))
        if saved_reference is not None:
            reference, reference_method = saved_reference, "saved_segment_running_time_mapping"
        elif zero is not None and running is not None:
            reference, reference_method = zero + running, "replayed_segment_running_time_mapping"
        elif zero is not None and pts is not None:
            reference, reference_method = zero + pts, "legacy_raw_pts_fallback"
        else:
            reference, reference_method = None, "unavailable"
        counts[f"reference_{reference_method}"] += 1

        identity_zero = first_zero.get(stream)
        stream_identity = identity_by_epoch.get(stream, set())
        if identity_zero is None:
            identity_complete = False
        else:
            stream_keys.update(stream_identity)
        continuity_key = (stream, mapping, segment_epoch)
        continuous = False
        reset_reasons = []
        if previous is None:
            reset_reasons.append("first_frame")
        else:
            old_key, old_running, old_arrival = previous
            if continuity_key != old_key:
                reset_reasons.append("stream_mapping_or_segment_changed")
            if running is None or old_running is None or not 0 < running - old_running <= 100_000_000:
                reset_reasons.append("invalid_or_large_running_time_gap")
            if arrival is None or old_arrival is None or not 0 < arrival - old_arrival <= 1_000_000_000:
                reset_reasons.append("invalid_or_large_arrival_gap")
            if arrival is not None and old_arrival is not None and any(old_arrival < pause <= arrival for pause in pauses):
                reset_reasons.append("display_pause")
            continuous = not reset_reasons
        if not continuous:
            segment_id += 1
        previous = continuity_key, running, arrival

        values = saved.get("qr_values_ms", [])
        if not isinstance(values, list):
            values = []
        matched, bad, wrong_cell = [], 0, 0
        for cell, value in enumerate(values):
            if value is None or value == "":
                continue
            candidates = payload_index.get(str(value), [])
            if len(candidates) != 1:
                bad += 1
                continue
            index = candidates[0]
            wrong_cell += events[index]["cell"] != cell
            matched.append(index)
        indices = sorted(set(matched))
        latest = indices[-1] if indices else None
        targets = {name: None for name in TARGETS}
        reasons = []
        if not saved:
            reasons.append("no_decoded_frame")
        if reference is None:
            reasons.append("missing_segment_running_time_mapping")
        if not indices:
            reasons.append("no_unique_qr_identity")
        if bad or wrong_cell or len(indices) != len(matched):
            reasons.append("ambiguous_unmapped_or_conflicting_qr_identity")

        next_presentation = latest + 1 if latest is not None and latest + 1 < len(times) else None
        usable = reference is not None and latest is not None and not bad and not wrong_cell and len(indices) == len(matched)
        visibility_ends = [lifetimes.get(index) for index in indices]
        visibility_complete = bool(indices) and all(end is not None for end in visibility_ends)
        common_start = times[latest] if latest is not None else None
        common_end = min(times[end] for end in visibility_ends) if visibility_complete else None
        common_ok = visibility_complete and common_start < common_end
        overwrite_ends = [cell_updates.get(index) for index in indices]
        overwrite_ok = bool(indices) and all(end is not None for end in overwrite_ends)
        overwrite_ok = overwrite_ok and common_start < min(times[end] for end in overwrite_ends)
        if usable and next_presentation is not None:
            if any(times[latest] <= pause <= times[next_presentation] for pause in pauses):
                reasons.append("presentation_interval_crosses_pause")
            elif times[next_presentation] - times[latest] > 3 * period_ns:
                reasons.append("presentation_interval_over_three_periods")
            else:
                targets["newest_generation"] = _bounds(reference, times[latest], times[next_presentation])
        elif latest is not None:
            reasons.append("missing_next_presentation")
        if usable and latest in lifetimes:
            targets["newest_visibility"] = _bounds(reference, times[latest], times[lifetimes[latest]])
        if usable and common_ok:
            targets["common_visibility"] = _bounds(reference, common_start, common_end)

        transition = bool(visibility_complete and not common_ok)
        assessment = assess_evidence({**saved, "temporal_evidence": None}, indices, transition=transition)
        diagnostic = targets["newest_generation"]
        if not assessment["primary_usable"]:
            targets["newest_generation"] = None
            reasons.extend(assessment["reasons"])
        flags = {
            "more_codes_than_configured_visible": len(indices) > visible,
            "all_grid_cells_decoded": len(indices) == grid,
            "visibility_contradiction": visibility_complete and not common_ok,
            "visibility_end_missing": bool(indices) and not visibility_complete,
            "compatible_if_codes_persist_until_cell_update": bool(overwrite_ok),
            "upstream_position_warnings": bool(saved.get("position_warnings")),
            "manual_qr_values": bool(saved.get("manual_values")),
            "saved_reference_disagrees": reference is not None and _integer(saved.get("camera_reference_monotonic_ns")) not in (None, reference),
            "display_timing_flag": any(events[index].get(key) for index in (latest, next_presentation) if index is not None
                                        for key in ("late_submit", "irregular_interval", "skipped_periods")),
        }
        counts.update(key for key, enabled in flags.items() if enabled)
        counts.update(reasons)
        counts.update(assessment.get("warnings", []))
        counts["evidence_" + assessment["status"]] += 1
        counts.update("continuity_reset_" + reason for reason in reset_reasons)
        output.append({
            "recording": directory.name, "frame_number": number, "filename": filename,
            "segment": segment_id, "pts_ns": pts, "running_time_ns": running,
            "application_arrival_monotonic_ns": arrival,
            "legacy_received_monotonic_ns": _integer(camera.get("received_monotonic_ns")),
            "sample_pulled_monotonic_ns": _integer(camera.get("sample_pulled_monotonic_ns")),
            "frame_converted_monotonic_ns": _integer(camera.get("frame_converted_monotonic_ns")),
            "capture_queue_level_buffers": _integer(camera.get("capture_queue_level_buffers")),
            "stream_key": sorted(stream_identity)[0] if stream_identity else None,
            "media_reference_monotonic_ns": reference, "reference_reconstruction": reference_method,
            "observable_arrival_minus_media_ms": (arrival - reference) / 1e6 if arrival is not None and reference is not None else None,
            "continuity_reset_reasons": reset_reasons,
            "recorded_timing_flags": camera.get("flags", []), "evidence_assessment": assessment,
            "qr_evidence": saved.get("qr_evidence"), "decoded_display_indices": indices,
            "diagnostic_newest_decoded_interval_ms": diagnostic,
            "image_path": str(recording / filename), "targets": targets, "flags": flags,
            "exclusion_reasons": reasons, "decoded_unique_qrs": len(indices),
            "grid_qrs": grid, "visible_qrs": visible, "latest_display_index": latest,
            "latest_journal_cell": events[latest]["cell"] if latest is not None else None,
            "all_marker_gap_ms": (common_start - common_end) / 1e6 if visibility_complete else None,
            "display_period_ms": period_ns / 1e6,
        })

    paused_indices = {row.get("last_frame_index") for row in display
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
        frame["targets"] = {name: None for name in TARGETS}

    counts.update({
        "camera_frames": len(output),
        "decoded_frames_not_in_camera_journal": len(set(decoded) - seen),
        "primary_scored_frames": sum(row["targets"]["newest_generation"] is not None for row in output),
        "common_visibility_frames": sum(row["targets"]["common_visibility"] is not None for row in output),
        "recording_summary_frames_rejected_invalid_timing": int(summary.get("frames_rejected_invalid_timing", 0) or 0),
        "recording_summary_frames_dropped_writer_queue": int(summary.get("frames_dropped_writer_queue", 0) or 0),
        "timing_event_rejected_invalid_timing": event_counts.get("frame_rejected_invalid_timing", 0),
        "timing_event_dropped_writer_queue": event_counts.get("frame_dropped_writer_queue", 0),
    })
    provenance_paths = [analysis_path, camera_path, display_path]
    provenance_paths.extend(path for path in (epoch_path, summary_path, event_path) if path.is_file())
    provenance = {str(path): _hash(path) for path in provenance_paths}
    audit = {
        "counts": dict(counts), "display_period_ms": period_ns / 1e6,
        "grid_qrs": grid, "visible_qrs": visible,
        "presentation_semantics": metadata.get("presentation_semantics"),
        "camera_journal_sha256": _hash(camera_path),
        "unsaved_frame_accounting": {
            "summary": {key: summary.get(key) for key in (
                "frames_observed", "frames_selected", "frames_saved",
                "frames_rejected_invalid_timing", "frames_dropped_writer_queue",
                "confirmed_frames_not_saved")},
            "timing_event_counts": dict(event_counts),
        },
        "decoded_inputs_already_remapped_by_upstream_reader": True,
    }
    return Session(directory.name, directory, recording, output, provenance, audit,
                   stream_keys, identity_complete)


def _residual(bounds: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.maximum(bounds[:, 0] - prediction, 0) - np.maximum(prediction - bounds[:, 1], 0)


def _score(bounds: np.ndarray, prediction: np.ndarray, goals: Goals) -> dict:
    bounds = np.asarray(bounds, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) != len(prediction) or not len(bounds):
        raise ValueError("Scoring needs matching, nonempty interval and prediction arrays")
    error = _residual(bounds, prediction)
    absolute = np.abs(error)
    widths = bounds[:, 1] - bounds[:, 0]
    worst = np.maximum(np.abs(bounds[:, 0] - prediction), np.abs(bounds[:, 1] - prediction))
    median, maximum = float(np.median(absolute)), float(np.max(absolute))
    misses = int(np.count_nonzero(absolute >= goals.threshold_ms))
    return {
        "n": len(absolute), "mae_ms": float(np.mean(absolute)),
        "median_absolute_ms": median, "p95_absolute_ms": float(np.percentile(absolute, 95)),
        "p99_absolute_ms": float(np.percentile(absolute, 99)), "maximum_absolute_ms": maximum,
        "errors_at_or_above_threshold": misses,
        "errors_at_or_above_threshold_pct": 100 * misses / len(absolute),
        "bias_ms": float(np.mean(error)), "inside_interval_pct": 100 * float(np.mean(absolute == 0)),
        "below_threshold_pct": 100 * float(np.mean(absolute < goals.threshold_ms)),
        "below_median_goal_pct": 100 * float(np.mean(absolute < goals.median_ms)),
        "interval_width_ms": {"minimum": float(np.min(widths)), "median": float(np.median(widths)),
                              "p95": float(np.percentile(widths, 95)), "maximum": float(np.max(widths))},
        "median_interval_width_ms": float(np.median(widths)),
        "midpoint_mae_ms": float(np.mean(np.abs(np.mean(bounds, axis=1) - prediction))),
        "worst_endpoint_median_ms": float(np.median(worst)),
        "worst_endpoint_p95_ms": float(np.percentile(worst, 95)),
        "worst_endpoint_maximum_ms": float(np.max(worst)),
        "worst_endpoint_below_threshold_pct": 100 * float(np.mean(worst < goals.threshold_ms)),
        "strict_maximum_goal_passed": maximum < goals.threshold_ms,
        "strict_median_goal_passed": median < goals.median_ms,
        "conditional_goal_passed": maximum < goals.threshold_ms and median < goals.median_ms,
    }


def _stream_components(sessions: list[Session]) -> list[list[Session]]:
    groups: list[list[Session]] = []
    for session in sessions:
        if not session.identity_complete or not session.stream_keys:
            continue
        matches = [group for group in groups if any(item.stream_keys & session.stream_keys for item in group)]
        combined = [session]
        for group in matches:
            combined.extend(group)
            groups.remove(group)
        groups.append(combined)
    return groups
