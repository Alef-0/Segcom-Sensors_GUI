#!/usr/bin/env python3
r"""Fresh calibration experiments from decoded QR identities and raw journals.

Run from the project root (no old verdict/model files are read)::

    python3 -m calibration.final_analysis \
        recordings/01_calibration_analysis recordings/02_calibration_analysis \
        recordings/03_calibration_analysis --output-directory recordings/final_analysis

Recording directories with a sibling ``_analysis`` directory also work. Inputs
are explicit: unrelated sibling recordings are never added automatically.
Only this module is needed; NumPy and optional Matplotlib are existing project
dependencies. ``--no-plots`` avoids Matplotlib. ``--self-test`` runs in memory.

The primary endpoint remains CONDITIONAL: the newest *decoded* code must also
be the newest displayed generation for its next-refresh interval to be valid.
Software swap returns are not physical exposure measurements. Wider visibility
intervals are separate sensitivity checks, never a source of better labels.
Saved QR identities have already been decoded/remapped by the upstream reader;
this module cannot recover omitted detections or establish their pixel position.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import csv
import subprocess
import sys
import tempfile

import numpy as np


VERSION = 1
HISTORY = 12
MIN_TRAIN = 40
MIN_TEST = 20
TARGETS = ("newest_generation", "newest_visibility", "common_visibility")
FEATURE_NAMES = ([f"pts_gap_lag_{index}_ms" for index in range(HISTORY)]
                 + [f"receipt_gap_lag_{index}_ms" for index in range(4)]
                 + [f"receipt_minus_pts_age_change_lag_{index}_ms" for index in range(4)])


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


def _features(steps: list[float], arrivals: list[float], ages: list[float]) -> list:
    def history(values, count):
        return list(reversed(values[-count:])) + [None] * max(0, count - len(values))
    return history(steps, HISTORY) + history(arrivals, 4) + history(ages, 4)


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
    epochs = _json(epoch_path) if epoch_path.is_file() else {}
    epoch_map = {}
    for epoch in epochs.get("epochs", []):
        key = _integer(epoch.get("stream_epoch"))
        zero = _integer(epoch.get("pipeline_zero_monotonic_ns"))
        if key is not None and zero is not None:
            if key in epoch_map and epoch_map[key] != zero:
                raise ValueError(f"Conflicting epoch anchors: {epoch_path}")
            epoch_map[key] = zero
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
    steps, arrivals, ages = [], [], []
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
        zero = epoch_map.get(epoch)
        pts = _integer(camera.get("pts_ns"))
        received = _integer(camera.get("received_monotonic_ns",
                                       camera.get("host_monotonic_received_ns")))
        reference = zero + pts if zero is not None and pts is not None else None
        if zero is None:
            identity_complete = False
        else:
            stream_keys.add(f"{zero}:{epoch}")
        continuous = (previous is not None and epoch == previous[0]
                      and pts is not None and previous[1] is not None
                      and 0 < pts - previous[1] <= 100_000_000
                      and received is not None and previous[2] is not None
                      and 0 < received - previous[2] <= 1_000_000_000
                      and not any(previous[2] < pause <= received for pause in pauses))
        if continuous:
            steps.append((pts - previous[1]) / 1e6)
            arrivals.append((received - previous[2]) / 1e6)
            if reference is not None and previous[3] is not None:
                ages.append(((received - reference) - (previous[2] - previous[3])) / 1e6)
        else:
            steps, arrivals, ages = [], [], []
            segment += 1
        features = _features(steps, arrivals, ages)
        previous = (epoch, pts, received, reference)
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
            reasons.append("missing_raw_pts_epoch_anchor")
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
        saved_reference = _integer(decoded_row.get("camera_reference_monotonic_ns"))
        flags = {
            "more_codes_than_configured_visible": len(indices) > visible,
            "all_grid_cells_decoded": len(indices) == grid,
            "visibility_contradiction": complete and not compatible,
            "visibility_end_missing": bool(indices) and not complete,
            "compatible_if_codes_persist_until_cell_update": bool(overwrite_compatible),
            "upstream_position_warnings": bool(decoded_row.get("position_warnings")),
            "manual_qr_values": bool(decoded_row.get("manual_values")),
            "history_warmup": len(steps) < HISTORY,
            "saved_reference_disagrees": (reference is not None and saved_reference is not None
                                          and reference != saved_reference),
            "display_timing_flag": any(
                events[index].get("late_submit") or events[index].get("irregular_interval")
                or events[index].get("skipped_periods")
                for index in (latest, newest_end) if index is not None),
        }
        counts.update(key for key, enabled in flags.items() if enabled)
        counts.update(reasons)
        output.append({
            "recording": directory.name, "frame_number": number, "filename": filename,
            "segment": segment, "pts_ns": pts, "received_monotonic_ns": received,
            "stream_key": f"{zero}:{epoch}" if zero is not None else None,
            "pts_reference_monotonic_ns": reference, "features": features,
            "targets": targets, "flags": flags, "exclusion_reasons": reasons,
            "decoded_unique_qrs": len(indices), "grid_qrs": grid, "visible_qrs": visible,
            "latest_display_index": latest,
            "latest_journal_cell": events[latest]["cell"] if latest is not None else None,
            "all_marker_gap_ms": ((common_start - common_end) / 1e6 if complete else None),
            "display_period_ms": period,
        })
    counts["camera_frames"] = len(output)
    counts["decoded_frames_not_in_camera_journal"] = len(set(decoded) - seen)
    counts["primary_scored_frames"] = sum(row["targets"]["newest_generation"] is not None
                                         for row in output)
    counts["common_visibility_frames"] = sum(row["targets"]["common_visibility"] is not None
                                            for row in output)
    paths = [analysis_path, camera_path, display_path] + ([epoch_path] if epoch_path.is_file() else [])
    provenance = {str(path): _hash(path) for path in paths}
    return Session(directory.name, directory, recording, output, provenance,
                   {"counts": dict(counts), "display_period_ms": period,
                    "grid_qrs": grid, "visible_qrs": visible,
                    "presentation_semantics": metadata.get("presentation_semantics"),
                    "camera_journal_sha256": _hash(camera_path),
                    "decoded_inputs_already_remapped_by_upstream_reader": True},
                   stream_keys, identity_complete)


def _residual(bounds: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Correction residual: positive needs more subtraction; zero is compatible."""
    return np.maximum(bounds[:, 0] - prediction, 0) - np.maximum(prediction - bounds[:, 1], 0)


def _score(bounds: np.ndarray, prediction: np.ndarray, goals: Goals) -> dict:
    absolute = np.abs(_residual(bounds, prediction))
    midpoint = np.mean(bounds, axis=1)
    worst = np.maximum(np.abs(bounds[:, 0] - prediction), np.abs(bounds[:, 1] - prediction))
    coverage = float(100 * np.mean(absolute < goals.threshold_ms))
    median = float(np.median(absolute))
    return {
        "n": len(absolute), "mae_ms": float(np.mean(absolute)),
        "median_absolute_ms": median, "p95_absolute_ms": float(np.percentile(absolute, 95)),
        "p99_absolute_ms": float(np.percentile(absolute, 99)), "maximum_absolute_ms": float(max(absolute)),
        "bias_ms": float(np.mean(_residual(bounds, prediction))),
        "inside_interval_pct": float(100 * np.mean(absolute == 0)),
        "below_threshold_pct": coverage,
        "below_median_goal_pct": float(100 * np.mean(absolute < goals.median_ms)),
        "median_interval_width_ms": float(np.median(bounds[:, 1] - bounds[:, 0])),
        "midpoint_mae_ms": float(np.mean(np.abs(midpoint - prediction))),
        "worst_endpoint_p95_ms": float(np.percentile(worst, 95)),
        "worst_endpoint_below_threshold_pct": float(100 * np.mean(worst < goals.threshold_ms)),
        "conditional_goal_passed": bool(median < goals.median_ms and coverage >= goals.coverage_pct),
    }


def _rank(metrics: dict, goals: Goals) -> tuple:
    # First minimize violations of the two requested goals, then the error tail.
    deficit = max(0, goals.coverage_pct - metrics["below_threshold_pct"]) / 100
    deficit += max(0, metrics["median_absolute_ms"] / goals.median_ms - 1)
    return (not metrics["conditional_goal_passed"], deficit,
            metrics["p95_absolute_ms"], metrics["median_absolute_ms"], metrics["mae_ms"])


def _eligible(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["targets"]["newest_generation"] is not None]


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray([row["features"] for row in rows], dtype=float),
            np.asarray([row["targets"]["newest_generation"] for row in rows], dtype=float))


def _coverage_shift(bounds: np.ndarray, prediction: np.ndarray, goals: Goals) -> float:
    shifted = bounds - prediction[:, None]
    # Candidate locations are derived only from training residuals. Uniform and
    # quantile grids avoid a privileged old correction or handpicked generation.
    center = np.mean(shifted, axis=1)
    candidates = np.unique(np.concatenate((np.quantile(center, np.linspace(0, 1, 65)),
        np.linspace(float(center.min()), float(center.max()), 65), [0.0])))
    best = min(candidates, key=lambda value: _rank(
        _score(bounds, prediction + value, goals), goals))
    return float(best)


def _design(x: np.ndarray, depth: int, receipt: bool) -> np.ndarray:
    values = x[:, :depth]
    # Missing history is explicit and later imputed from training data only.
    if receipt:
        values = np.column_stack((values, x[:, HISTORY:HISTORY + 8]))
    return values


def _preprocess(x: np.ndarray, parameters: dict | None = None) -> tuple[np.ndarray, dict]:
    if parameters is None:
        medians, lower, upper = [], [], []
        for column in x.T:
            finite = column[np.isfinite(column)]
            medians.append(float(np.median(finite)) if len(finite) else 0.0)
            lower.append(float(np.percentile(finite, 1)) if len(finite) else 0.0)
            upper.append(float(np.percentile(finite, 99)) if len(finite) else 0.0)
        scale = np.maximum(np.asarray(upper) - lower, 1.0)
        parameters = {"median": medians, "lower": lower, "upper": upper, "scale": scale.tolist()}
    missing = ~np.isfinite(x)
    filled = np.where(missing, np.asarray(parameters["median"]), x)
    scaled = (np.clip(filled, parameters["lower"], parameters["upper"])
              - parameters["median"]) / parameters["scale"]
    return np.column_stack((scaled, missing.astype(float))), parameters


def _recipes() -> dict[str, list[dict]]:
    return {
        "fixed_midpoint": [{"kind": "fixed_midpoint"}],
        "fixed_coverage": [{"kind": "fixed_coverage"}],
        "robust_pts_history": [dict(kind="huber", depth=depth, ridge=ridge, receipt=False)
                               for depth in (1, 3, 6, 12) for ridge in (0.1, 1.0)],
        "local_cadence_neighbors": [dict(kind="neighbors", depth=depth, neighbors=count, receipt=False)
                                    for depth in (3, 6, 12) for count in (15, 40)],
        "receipt_innovation_history": [dict(kind="huber", depth=depth, ridge=1.0, receipt=True)
                                       for depth in (3, 6, 12)],
    }


def _fit(rows: list[dict], recipe: dict, goals: Goals) -> dict:
    x, bounds = _arrays(rows)
    y = np.mean(bounds, axis=1)
    kind = recipe["kind"]
    model = {"recipe": recipe, "training_frames": len(rows)}
    if kind.startswith("fixed"):
        correction = float(np.median(y))
        if kind == "fixed_coverage":
            correction += _coverage_shift(bounds, np.full(len(y), correction), goals)
        return dict(model, correction_ms=correction)
    design, preprocessing = _preprocess(_design(x, recipe["depth"], recipe["receipt"]))
    model["preprocessing"] = preprocessing
    if kind == "neighbors":
        # This nonparametric model retains only training camera features/targets.
        model.update(training_features=design.tolist(), training_midpoints=y.tolist())
        model["shift_ms"] = 0.0
        return model
    design = np.column_stack((np.ones(len(design)), design))
    intercept = float(np.median(y))
    centered = y - intercept
    weights = np.ones(len(y))
    penalty = np.eye(design.shape[1]) * recipe["ridge"]
    penalty[0, 0] = 0
    coefficients = np.zeros(design.shape[1])
    for _ in range(40):
        updated = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                                  design.T @ (weights * centered))
        residual = centered - design @ updated
        robust_scale = max(1.0, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1, 1.345 * robust_scale / np.maximum(np.abs(residual), 1e-12))
        converged = np.max(np.abs(updated - coefficients)) < 1e-7
        coefficients = updated
        if converged:
            break
    coefficients[0] += intercept
    prediction = design @ coefficients
    model.update(coefficients=coefficients.tolist(),
                 shift_ms=_coverage_shift(bounds, prediction, goals))
    return model


def _predict(model: dict, rows: list[dict]) -> np.ndarray:
    recipe = model["recipe"]
    if "correction_ms" in model:
        return np.full(len(rows), model["correction_ms"])
    x = np.asarray([row["features"] for row in rows], dtype=float)
    design, _ = _preprocess(_design(x, recipe["depth"], recipe["receipt"]), model["preprocessing"])
    if recipe["kind"] == "neighbors":
        train = np.asarray(model["training_features"])
        labels = np.asarray(model["training_midpoints"])
        k = min(recipe["neighbors"], len(train))
        prediction = []
        for start in range(0, len(design), 256):
            distances = np.sum((design[start:start + 256, None] - train[None]) ** 2, axis=2)
            # Stable sorting makes equal-distance neighborhoods reproducible.
            nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
            prediction.extend(np.median(labels[nearest], axis=1))
        return np.asarray(prediction)
    return np.column_stack((np.ones(len(design)), design)) @ model["coefficients"] + model["shift_ms"]


def _select_models(groups: list[list[dict]], goals: Goals) -> tuple[dict, dict]:
    """Select settings/family on two forward inner folds; never see outer labels."""
    training = [row for group in groups for row in _eligible(group)]
    if len(training) < MIN_TRAIN:
        raise ValueError(f"Need at least {MIN_TRAIN} eligible training frames")
    folds = []
    for fraction, end_fraction in ((0.5, 0.75), (0.75, 1.0)):
        fit_rows, validation_groups = [], []
        for group in groups:
            cut, end = int(len(group) * fraction), int(len(group) * end_fraction)
            fit_rows.extend(_eligible(group[:max(0, cut - HISTORY)]))
            validation = _eligible(group[cut:end])
            if len(validation) >= MIN_TEST:
                validation_groups.append(validation)
        if len(fit_rows) >= MIN_TRAIN and validation_groups:
            folds.append((fit_rows, validation_groups))
    if not folds:
        raise ValueError("Insufficient chronological evidence for inner selection; use longer recordings")
    models, trials = {}, {}
    for family, recipes in _recipes().items():
        trials[family] = []
        for recipe in recipes:
            results = []
            for fit_rows, validation_groups in folds:
                model = _fit(fit_rows, recipe, goals)
                for validation_rows in validation_groups:
                    bounds = _arrays(validation_rows)[1]
                    score = _score(bounds, _predict(model, validation_rows), goals)
                    score["validation_recording"] = validation_rows[0].get("recording", "synthetic")
                    results.append(score)
            trials[family].append({"recipe": recipe, "fold_metrics": results,
                                   "worst_fold_rank": list(max(_rank(m, goals) for m in results))})
        choice = min(trials[family], key=lambda trial: tuple(trial["worst_fold_rank"]))
        models[family] = _fit(training, choice["recipe"], goals)
    # This choice is frozen before looking at the outer holdout/other recording.
    family = min(trials, key=lambda key: min(tuple(t["worst_fold_rank"]) for t in trials[key]))
    models["preselected"] = models[family]
    return models, {"selected_family": family, "inner_folds": len(folds), "trials": trials}


def _block_uncertainty(rows: list[dict], absolute: np.ndarray, goals: Goals) -> dict:
    """Moving-block sensitivity intervals; not independent-frame confidence."""
    candidates = []
    for start in range(len(rows)):
        block = [start]
        for stop in range(start + 1, min(len(rows), start + 20)):
            if (rows[stop]["recording"] != rows[start]["recording"]
                    or rows[stop]["segment"] != rows[start]["segment"]
                    or rows[stop]["frame_number"] != rows[stop - 1]["frame_number"] + 1):
                break
            block.append(stop)
        candidates.append(block)
    rng = np.random.default_rng(20260915)
    coverage, medians = [], []
    for _ in range(200):
        selected = []
        while len(selected) < len(rows):
            selected.extend(candidates[int(rng.integers(len(candidates)))])
        sample = absolute[selected[:len(rows)]]
        coverage.append(float(100 * np.mean(sample < goals.threshold_ms)))
        medians.append(float(np.median(sample)))
    return {"method": "200 moving-block resamples, up to 20 consecutive camera frames, seed 20260915",
            "coverage_95pct_resampling_range": np.percentile(coverage, [2.5, 97.5]).tolist(),
            "median_95pct_resampling_range_ms": np.percentile(medians, [2.5, 97.5]).tolist(),
            "interpretation": "Within-recording sampling sensitivity only; not physical or cross-session certainty"}


def _evaluate(model: dict, rows: list[dict], goals: Goals) -> tuple[dict, list[dict]]:
    eligible = _eligible(rows)
    if not eligible:
        return {"status": "no_evaluable_frames", "camera_frames": len(rows)}, []
    prediction = _predict(model, eligible)
    bounds = _arrays(eligible)[1]
    result = _score(bounds, prediction, goals)
    residual = _residual(bounds, prediction)
    result.update(status="evaluated", camera_frames=len(rows),
                  scored_camera_pct=100 * len(eligible) / len(rows),
                  below_threshold_all_camera_lower_bound_pct=100 * int(np.sum(
                      np.abs(residual) < goals.threshold_ms)) / len(rows),
                  enough_evaluation_frames=len(eligible) >= MIN_TEST,
                  uncertainty=_block_uncertainty(eligible, np.abs(residual), goals))
    result["cohorts"] = {}
    for flag in ("all_grid_cells_decoded", "more_codes_than_configured_visible",
                 "visibility_contradiction", "upstream_position_warnings",
                 "display_timing_flag", "history_warmup", "manual_qr_values"):
        for value in (False, True):
            mask = np.asarray([row["flags"][flag] == value for row in eligible])
            if mask.any():
                result["cohorts"][f"{flag}={value}"] = _score(bounds[mask], prediction[mask], goals)
    result["sensitivity"] = {}
    for target in TARGETS[1:]:
        subset = [row for row in rows if row["targets"][target] is not None]
        if subset:
            result["sensitivity"][target] = _score(np.asarray([row["targets"][target] for row in subset]),
                                                  _predict(model, subset), goals)
    records = []
    for row, predicted, error, interval in zip(eligible, prediction, residual, bounds):
        records.append({"recording": row["recording"], "frame_number": row["frame_number"],
                        "filename": row["filename"], "correction_ms": float(predicted),
                        "interval_lower_ms": float(interval[0]), "interval_upper_ms": float(interval[1]),
                        "interval_residual_ms": float(error),
                        "midpoint_residual_ms": float(np.mean(interval) - predicted),
                        "below_threshold": bool(abs(error) < goals.threshold_ms),
                        "residual_display_periods": int(round(error / row["display_period_ms"]))})
    return result, records


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


def analyze_directories(directories: list[str | Path], *, goals: Goals | None = None) -> dict:
    """Compute in memory. Writing reports is a separate, explicit operation."""
    goals = goals or Goals()
    if not (all(math.isfinite(value) for value in (goals.median_ms, goals.threshold_ms, goals.coverage_pct))
            and goals.median_ms > 0 and goals.threshold_ms > 0 and 0 < goals.coverage_pct <= 100):
        raise ValueError("Goals require positive milliseconds and coverage in (0, 100]")
    sessions = [load_session(directory) for directory in directories]
    if not sessions:
        raise ValueError("Supply at least one recording or analysis directory")
    if len({s.name for s in sessions}) != len(sessions):
        raise ValueError("Input analysis directory names must be distinct")
    if len({str(s.recording.resolve()) for s in sessions}) != len(sessions):
        raise ValueError("Multiple analyses of the same source recording are not independent inputs")
    if len({s.audit["camera_journal_sha256"] for s in sessions}) != len(sessions):
        raise ValueError("Duplicate camera journals detected; supply each recording only once")
    previous_frames = set()
    for session in sessions:
        identities = {(row["stream_key"], row["pts_ns"]) for row in session.frames
                      if row["stream_key"] is not None and row["pts_ns"] is not None}
        if previous_frames & identities:
            raise ValueError("Recordings overlap in stream/PTS frames; cross-recording checks would leak observations")
        previous_frames.update(identities)
    experiments, predictions, fitted_models = [], [], []

    def experiment(kind, sources, target, training_groups, evaluation_rows):
        label = f"{kind}:{'+'.join(sources)}->{target.name}"
        try:
            models, selection = _select_models(training_groups, goals)
        except ValueError as error:
            experiments.append({"id": label, "kind": kind, "sources": sources, "target": target.name,
                                "status": "insufficient_training_evidence", "reason": str(error)})
            return
        fitted_models.append({"experiment": label, "selection": selection, "models": models})
        for family, model in models.items():
            metrics, rows = _evaluate(model, evaluation_rows, goals)
            experiments.append({"id": label, "kind": kind, "sources": sources, "target": target.name,
                                "strategy": family, "selected_family": selection["selected_family"],
                                "metrics": metrics, "status": metrics["status"]})
            predictions.extend(dict(row, experiment=label, strategy=family) for row in rows)

    for session in sessions:
        cut = int(len(session.frames) * 0.7)
        experiment("chronological_holdout", [session.name], session,
                   [session.frames[:max(0, cut - HISTORY)]], session.frames[cut:])
    for source in sessions:
        # Fit once on source; prediction/evaluation uses every target unchanged.
        try:
            models, selection = _select_models([source.frames], goals)
        except ValueError as error:
            experiments.append({"kind": "frozen_source", "sources": [source.name],
                                "status": "insufficient_training_evidence", "reason": str(error)})
            continue
        fitted_models.append({"experiment": f"frozen_source:{source.name}",
                              "selection": selection, "models": models})
        for target in sessions:
            if source is target:
                continue
            relation = ("shared_stream" if source.stream_keys & target.stream_keys else
                        "distinct_stream" if source.identity_complete and target.identity_complete else
                        "unknown_stream_relation")
            label = f"frozen_source:{source.name}->{target.name}"
            for family, model in models.items():
                metrics, rows = _evaluate(model, target.frames, goals)
                experiments.append({"id": label, "kind": "frozen_source", "sources": [source.name],
                                    "target": target.name, "strategy": family, "stream_relation": relation,
                                    "selected_family": selection["selected_family"],
                                    "metrics": metrics, "status": metrics["status"]})
                predictions.extend(dict(row, experiment=label, strategy=family) for row in rows)
    components = _stream_components(sessions)
    if len(components) >= 2:
        for held_out in components:
            held_names = {session.name for session in held_out}
            sources = [session for group in components for session in group if session.name not in held_names]
            for target in held_out:
                experiment("leave_stream_out", [s.name for s in sources], target,
                           [s.frames for s in sources], target.frames)
    primary = [row for row in experiments if row.get("strategy") == "preselected"
               and row["kind"] == "leave_stream_out"]
    independent_pass = bool(primary) and all(
        row.get("metrics", {}).get("conditional_goal_passed", False)
        and row["metrics"].get("enough_evaluation_frames", False)
        and row["metrics"].get("below_threshold_all_camera_lower_bound_pct", 0) >= goals.coverage_pct
        for row in primary)
    expected_targets = sum(len(group) for group in components) if len(components) >= 2 else 0
    independent_pass = independent_pass and len(primary) == expected_targets and expected_targets == len(sessions)
    return {
        "schema_version": VERSION, "generated_at": datetime.now().astimezone().isoformat(),
        "goals": {"median_absolute_below_ms": goals.median_ms, "absolute_below_ms": goals.threshold_ms,
                  "minimum_coverage_pct": goals.coverage_pct, "strict_threshold_comparison": True},
        "feature_order": FEATURE_NAMES,
        "evidence_policy": {
            "primary": "Newest decoded generation to next presentation; conditional on no missed newer QR",
            "sensitivity": "Newest-marker lifetime and common lifetime of all markers; never used for selection",
            "contradiction": "Empty common visibility is unexplained; do not attribute it automatically to rolling shutter",
            "saved_input_limit": "QR values were decoded/remapped upstream; original pixels and omitted detections are not reassessed",
            "features": "Current/past raw PTS gaps; receipt model also uses past/current receipt gaps and changes of receipt-minus-PTS age",
            "forbidden_predictors": "QR values, cell IDs, display indices, absolute recording time, old fitted corrections",
            "reference": "Raw epoch pipeline-zero monotonic time plus raw camera PTS; no saved corrected timestamps",
            "selection": "Two forward inner folds; rank worst recording/fold by goal violations then P95, median and MAE; frozen before outer scoring",
            "purge_frames": HISTORY, "primary_holdout_fraction": 0.3,
            "fit_target": "Interval midpoint for robust regression/neighbors; training-only coverage shift for regression",
            "missing_features": "Training-only median imputation plus explicit missing indicators; reset history at gaps/epochs/pauses",
            "uncertainty": "Report midpoint error, worst-endpoint error, interval width and block resampling alongside interval distance",
        },
        "sessions": [{"name": s.name, "recording_directory": str(s.recording),
                      "analysis_directory": str(s.directory), "audit": s.audit,
                      "source_provenance_sha256": s.provenance, "stream_keys": sorted(s.stream_keys),
                      "stream_identity_complete": s.identity_complete} for s in sessions],
        "stream_groups": [[s.name for s in group] for group in components],
        "experiments": experiments, "models": fitted_models, "predictions": predictions,
        "frames": [row for s in sessions for row in s.frames],
        "verdict": {
            "conditional_independent_goals_passed": bool(independent_pass),
            "physical_accuracy_established": False,
            "status": ("conditional_goals_met_exposure_unverified" if independent_pass else
                       "needs_independent_stream_evidence" if len(components) < 2 else "conditional_goals_not_met"),
            "explanation": "A low interval error cannot establish actual exposure timing when QR generation, visibility, or physical presentation is unresolved",
        },
    }


def _markdown(report: dict) -> str:
    goals = report["goals"]
    lines = ["# Final calibration analysis", "", f"Generated: {report['generated_at']}", "",
             f"Status: **{report['verdict']['status']}**", "",
             f"Goals: median < {goals['median_absolute_below_ms']:g} ms; "
             f"at least {goals['minimum_coverage_pct']:g}% of errors < {goals['absolute_below_ms']:g} ms.", "",
             "All scores are conditional software-marker scores. Physical exposure accuracy is unverified.", "",
             "## Evidence and assumptions", ""]
    lines.extend(f"- **{key}:** {value}" for key, value in report["evidence_policy"].items())
    lines.extend(["", "## Input audit", "",
                  "| Recording | Camera frames | Scored | More QRs than configured visible | Visibility contradictions |",
                  "|---|---:|---:|---:|---:|"])
    for session in report["sessions"]:
        counts = session["audit"]["counts"]
        lines.append(f"| {session['name']} | {counts['camera_frames']} | {counts['primary_scored_frames']} | "
                     f"{counts.get('more_codes_than_configured_visible', 0)} | {counts.get('visibility_contradiction', 0)} |")
    lines.extend(["", "## Models selected before evaluation", "",
                  "These rows use the model family and settings selected inside the training data. "
                  "All competing families are exported separately; the best outer result is not rebranded as a validated winner.", "",
                  "| Check | Source → target | Selected family | MAE | Median | P95 | Below threshold | Goal pass |",
                  "|---|---|---|---:|---:|---:|---:|---|"])
    for row in report["experiments"]:
        if row.get("strategy") != "preselected" or row["status"] != "evaluated":
            continue
        m = row["metrics"]
        lines.append(f"| {row['kind']} | {' + '.join(row['sources'])} → {row['target']} | "
                     f"{row['selected_family']} | {m['mae_ms']:.3f} | {m['median_absolute_ms']:.3f} | "
                     f"{m['p95_absolute_ms']:.3f} | {m['below_threshold_pct']:.1f}% | {m['conditional_goal_passed']} |")
    skipped = [row for row in report["experiments"] if row["status"] != "evaluated"]
    lines.extend(["", "## Stream groups", "", *[f"- {', '.join(group)}" for group in report["stream_groups"]], "",
                  "Unknown stream identities cannot pass independent validation. All recordings sharing any stream epoch "
                  "stay together when an entire stream is held out.", "",
                  "## Skipped checks", "", *[f"- {row.get('id', row['kind'])}: {row.get('reason', row['status'])}"
                                              for row in skipped], "",
                  "## Graph guide", "",
                  "- Residual CDF: each full curve uses held-out predictions; dashed lines show the requested thresholds.",
                  "- Overview: coverage and P95 per evaluated target; each point retains its source/target label.",
                  "- Evidence: decoded QR count versus configured visibility; contradictions are not automatically explained away.",
                  "", "## Reproducibility", "",
                  "The JSON includes source hashes, selected settings, training models, feature preprocessing, "
                  "inner-fold scores, all hypotheses, uncertainty ranges and cohort scores. The CSVs retain every camera "
                  "frame audit and each evaluated prediction. Existing analyses and live corrections are unchanged.", ""])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _plot(report: dict, output: Path, svg: bool) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    files = []
    goals = report["goals"]

    def save(figure, stem):
        for extension in (["png", "svg"] if svg else ["png"]):
            name = f"{stem}.{extension}"
            figure.savefig(output / name, dpi=160, bbox_inches="tight")
            files.append(name)
        plt.close(figure)

    sessions = report["sessions"]
    figure, axes = plt.subplots(len(sessions), 1, figsize=(11, 4 * len(sessions)), squeeze=False,
                                layout="constrained")
    for session, axis in zip(sessions, axes[:, 0]):
        for family in (*_recipes(), "preselected"):
            values = [abs(row["interval_residual_ms"]) for row in report["predictions"]
                      if row["experiment"].startswith("chronological_holdout:")
                      and row["recording"] == session["name"] and row["strategy"] == family]
            if values:
                x = np.sort(values)
                axis.step(np.r_[0, x], np.r_[0, 100 * np.arange(1, len(x) + 1) / len(x)],
                          where="post", label=family, linewidth=2.3 if family == "preselected" else 1)
        axis.axvline(goals["absolute_below_ms"], color="black", linestyle="--")
        axis.axvline(goals["median_absolute_below_ms"], color="gray", linestyle=":")
        axis.axhline(goals["minimum_coverage_pct"], color="black", linestyle="--")
        axis.axhline(50, color="gray", linestyle=":")
        axis.set(title=f"{session['name']}: chronological holdout (conditional QR interval)",
                 xlabel="Absolute distance outside interval (ms)", ylabel="Frames at or below error (%)",
                 xlim=(0, None), ylim=(0, 101))
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    save(figure, "final_analysis_residual_cdf")
    results = [row for row in report["experiments"] if row.get("strategy") == "preselected"
               and row["status"] == "evaluated"]
    figure, axes = plt.subplots(1, 2, figsize=(15, max(5, len(results) * 0.35)), layout="constrained")
    labels = [f"{row['kind']}: {'+'.join(row['sources'])} → {row['target']}" for row in results]
    for axis, key, goal, title in zip(axes, ("below_threshold_pct", "p95_absolute_ms"),
                                   (goals["minimum_coverage_pct"], goals["absolute_below_ms"]),
                                   ("Coverage below threshold (%)", "P95 interval error (ms)")):
        axis.barh(np.arange(len(results)), [row["metrics"][key] for row in results])
        axis.set_yticks(np.arange(len(results)), labels, fontsize=7)
        axis.axvline(goal, linestyle="--", color="black")
        axis.set_title(title)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    save(figure, "final_analysis_overview")
    figure, axes = plt.subplots(len(sessions), 1, figsize=(11, 3 * len(sessions)), squeeze=False,
                                layout="constrained")
    for session, axis in zip(sessions, axes[:, 0]):
        rows = [row for row in report["frames"] if row["recording"] == session["name"]]
        axis.scatter([row["frame_number"] for row in rows], [row["decoded_unique_qrs"] for row in rows],
                     c=["#bd3b39" if row["flags"]["visibility_contradiction"] else "#287e9b" for row in rows], s=8)
        axis.axhline(session["audit"]["visible_qrs"], color="black", linestyle="--", label="Configured visible count")
        axis.set(title=f"{session['name']} — red: incompatible marker lifetimes",
                 xlabel="Camera frame", ylabel="Unique decoded QRs")
        axis.legend(fontsize=8)
    save(figure, "final_analysis_evidence")
    return files


def write_report(report: dict, output_directory: str | Path, *, plots: bool = True, svg: bool = False) -> dict:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = ["final_analysis.json", "final_analysis.md", "final_analysis_frames.csv",
             "final_analysis_predictions.csv", "final_analysis_results.csv"]
    if plots:
        files.extend(_plot(report, output, svg))
    report = dict(report, output_directory=str(output), output_files=files)
    _write_csv(output / "final_analysis_frames.csv", report["frames"])
    _write_csv(output / "final_analysis_predictions.csv", report["predictions"])
    _write_csv(output / "final_analysis_results.csv", [
        {**{key: value for key, value in row.items() if key != "metrics"}, **row.get("metrics", {})}
        for row in report["experiments"]])
    (output / "final_analysis.md").write_text(_markdown(report), encoding="utf-8")
    (output / "final_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _self_test() -> None:
    from unittest.mock import patch

    goals = Goals()
    bounds = np.asarray([[80., 90.], [80., 90.], [80., 90.]])
    np.testing.assert_array_equal(_residual(bounds, np.asarray([75., 85., 95.])), [5., 0., -5.])
    assert _bounds(10**18 + 90_000_000, 10**18, 10**18 + 10_000_000) == [80., 90.]
    assert _integer(10**18 + 1) == _integer(str(10**18 + 1)) == 10**18 + 1
    assert not _score(bounds[:1], np.asarray([70.]), goals)["conditional_goal_passed"]
    rows = []
    for index in range(240):
        step = 20. if index % 3 == 0 else 40.
        midpoint = 50 + step
        rows.append({"features": [step] * HISTORY + [step] * 8,
                     "targets": {"newest_generation": [midpoint - 2, midpoint + 2]}})
    models, selection = _select_models([rows], goals)
    assert selection["inner_folds"] == 2
    for model in models.values():
        prediction = _predict(model, rows)
        assert np.all(np.isfinite(prediction))
    np.testing.assert_allclose(_predict(models["robust_pts_history"], rows),
                               [np.mean(row["targets"]["newest_generation"]) for row in rows], atol=0.2)
    original = _predict(models["preselected"], rows)
    changed = [dict(row, targets={"newest_generation": [1000., 2000.]}) for row in rows]
    np.testing.assert_array_equal(original, _predict(models["preselected"], changed))
    x, parameters = _preprocess(np.asarray([[1., np.nan], [3., np.nan]]))
    assert np.isfinite(x).all()
    before = json.dumps(parameters, sort_keys=True)
    _preprocess(np.asarray([[1e9, -1e9]]), parameters)
    assert json.dumps(parameters, sort_keys=True) == before
    def session(name, keys, complete=True):
        return Session(name, Path(name), Path(name), [], {}, {}, set(keys), complete)
    groups = _stream_components([session("a", ["1"]), session("b", ["2"]),
                                 session("bridge", ["1", "2"]), session("c", ["3"]),
                                 session("unknown", [], False)])
    assert sorted(sorted(s.name for s in group) for group in groups) == [["a", "b", "bridge"], ["c"]]

    # Synthetic raw journals verify that offsets/acceptance labels from an old
    # report cannot alter the new bounds; all data stays in memory via mocks.
    root = Path("/synthetic/recording")
    analysis = root.with_name("recording_analysis")
    epoch_zero = 10**18 + 1
    display = [{"kind": "session", "grid_qrs": 2, "visible_qrs": 1}]
    for index in range(500):
        stamp = epoch_zero + index * 10_000_000
        display.append({"kind": "frame", "index": index, "cell": index % 2,
                        "marker_ns": stamp, "presentation_return_ns": stamp})
    cameras, decoded = [], []
    for index in range(240):
        latest = 2 * index + 10
        pts = latest * 10_000_000 + 85_000_000
        filename = f"images/{index}.jpg"
        cameras.append({"frame": filename, "stream_epoch": 1 if index < 120 else 2,
                        "pts_ns": pts, "received_monotonic_ns": epoch_zero + pts + 150_000_000})
        payload = f"{(epoch_zero + latest * 10_000_000) // 1_000_000 % 1_000_000_000_000:012d}"
        decoded.append({"filename": filename, "qr_values_ms": [payload, None],
                        "validation": "rejected_by_old_assumption", "offset_interval_lower_ms": -9999,
                        "offset_interval_upper_ms": -9998})
    files = {analysis / "calibration_analysis.json": {"recording_directory": str(root), "frames": decoded},
             root / "camera_timing_session.json": {"epochs": [
                 {"stream_epoch": epoch, "pipeline_zero_monotonic_ns": epoch_zero} for epoch in (1, 2)]},
             root / "camera_timestamps.jsonl": cameras, root / "display_timestamps.jsonl": display}
    with patch.object(Path, "is_file", lambda path: path in files), \
            patch(__name__ + "._json", side_effect=lambda path: files[path]), \
            patch(__name__ + "._journal", side_effect=lambda path: files[path]), \
            patch(__name__ + "._hash", return_value="synthetic"):
        loaded = load_session(analysis)
        assert len(loaded.frames) == 240
        assert loaded.frames[0]["targets"]["newest_generation"] == [75.0, 85.0]
        assert loaded.frames[0]["targets"]["common_visibility"] == [75.0, 85.0]
        assert all(value is None for value in loaded.frames[120]["features"])
        assert loaded.frames[121]["features"][0] == 20.0
        assert loaded.frames[0]["targets"] == loaded.frames[-1]["targets"]
        # A nonunique marker cannot be resolved by an assumed correction.
        display[12]["marker_ns"] = display[11]["marker_ns"]
        ambiguous = load_session(analysis)
        assert ambiguous.frames[0]["targets"]["newest_generation"] is None
        assert "ambiguous_unmapped_or_conflicting_qr_identity" in ambiguous.frames[0]["exclusion_reasons"]
    print("Self-tests passed: precise interval arithmetic, strict goals, frozen predictions, "
          "training-only preprocessing, stream grouping, raw-journal reconstruction, epoch resets and ambiguous QR rejection.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--median-goal-ms", type=float, default=5.0)
    parser.add_argument("--threshold-ms", type=float, default=10.0)
    parser.add_argument("--coverage-pct", type=float, default=95.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--svg", action="store_true", help="Also save SVG graphs; PNG is the default")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic in-memory checks without writing reports")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.directories:
        parser.error("Supply explicit recording/analysis directories, or use --self-test")
    if not args.no_plots:
        # Match the existing launcher behavior without importing old analysis code.
        # Some hosts have user-site NumPy paired with an incompatible system Matplotlib.
        environments = [os.environ.copy(), dict(os.environ, PYTHONNOUSERSITE="1")]
        selected = None
        for environment in environments:
            environment.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "segcom-final-analysis-matplotlib"))
            probe = subprocess.run([sys.executable, "-c", "import numpy, matplotlib"],
                                   env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if probe.returncode == 0:
                selected = environment
                break
        if selected is None:
            parser.exit(1, "No compatible NumPy/Matplotlib pair found. Use --no-plots or repair those dependencies.\n")
        if selected.get("PYTHONNOUSERSITE") != os.environ.get("PYTHONNOUSERSITE"):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=selected)
            raise SystemExit(result.returncode)
        os.environ.setdefault("MPLCONFIGDIR", selected["MPLCONFIGDIR"])
    output = args.output_directory or args.directories[0].resolve().parent / "final_analysis"
    try:
        report = analyze_directories(args.directories, goals=Goals(args.median_goal_ms, args.threshold_ms, args.coverage_pct))
        report = write_report(report, output, plots=not args.no_plots, svg=args.svg)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Final analysis failed: {error}\n")
    print(f"Saved final_analysis reports to {report['output_directory']}")
    print(report["verdict"]["status"])


if __name__ == "__main__":
    main()
