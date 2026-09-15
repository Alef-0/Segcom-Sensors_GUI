#!/usr/bin/env python3
"""Create a quantitative camera-offset verdict from saved QR analysis files."""

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

from sensors.camera.timing_defaults import DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS

CURRENT_CORRECTION_MS = DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS
NOMINAL_PTS_STEP_MS = 1000.0 / 30.0
MAX_CONTINUOUS_PTS_STEP_MS = 100.0
TRAIN_FRACTION = 0.70
MINIMUM_CLEAN_FRAMES = 30
STEP_BUCKET_MS = 5.0
HISTORY_RIDGE = 1e-3
MINIMUM_CADENCE_STATE_FRAMES = 10
GENERATION_ALTERNATIVES = 2
FIXED_REPEATABILITY_LIMIT_MS = 2.0
DYNAMIC_MINIMUM_GAIN_MS = 0.5
DYNAMIC_P95_TOLERANCE_MS = 0.5

ANALYSIS_JSON = "calibration_analysis.json"
FRAMES_CSV = "calibration_frames.csv"
VERDICT_JSON = "calibration_verdict.json"
VERDICT_MARKDOWN = "calibration_verdict.md"
PREDICTIONS_CSV = "calibration_strategy_predictions.csv"
TIMELINE_GRAPH = "calibration_offset_timeline.png"
RESIDUAL_GRAPH = "calibration_residual_cdf.png"
OFFSET_HISTOGRAM = "calibration_offset_histogram.png"
FIXED_RESIDUAL_HISTOGRAM = "calibration_fixed_residual_histogram.png"
PTS_RESIDUAL_HISTOGRAM = "calibration_pts_residual_histogram.png"
READABILITY_GRAPH = "calibration_readability_diagnostics.png"
_PLOTTING = None


def _plotting():
    """Load Matplotlib only when creating graphs, after the viewer has closed."""
    global _PLOTTING
    if _PLOTTING is None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot
        from matplotlib.ticker import PercentFormatter
        _PLOTTING = pyplot, PercentFormatter
    return _PLOTTING


def matplotlib_environment() -> dict[str, str]:
    """Select an installed NumPy/Matplotlib pair that imports together."""
    candidates = [os.environ.copy()]
    system_only = os.environ.copy()
    system_only["PYTHONNOUSERSITE"] = "1"
    candidates.append(system_only)
    probe = [sys.executable, "-c", "import matplotlib, numpy"]
    for environment in candidates:
        environment.setdefault("MPLCONFIGDIR", "/tmp/segcom-calibration-matplotlib")
        checked = subprocess.run(
            probe,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if checked.returncode == 0:
            return environment
    raise RuntimeError(
        "No compatible NumPy/Matplotlib installation was found. Install the "
        "versions listed in requirements.txt before creating the verdict graphs."
    )


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    frame_number: int
    filename: str
    validation: str
    timing_status: str | None
    offset_ms: float | None
    pts_step_ms: float
    pts_history_ms: tuple[float, ...]
    interval_lower_ms: float | None
    interval_upper_ms: float | None
    matched_readable_qrs: int | None
    grid_qrs: int | None
    latest_cell: int | None
    latest_cell_name: str | None
    readable_cells: tuple[bool, ...]

    @property
    def clean(self) -> bool:
        return (
            self.validation == "accepted_clean"
            and self.timing_status == "Clean"
            and self.offset_ms is not None
        )

    @property
    def has_interval(self) -> bool:
        return (
            self.interval_lower_ms is not None
            and self.interval_upper_ms is not None
            and self.interval_lower_ms <= self.interval_upper_ms
        )

    @property
    def usable(self) -> bool:
        if self.has_interval:
            return self.validation.startswith("accepted_")
        return self.clean

    @property
    def representative_offset_ms(self) -> float:
        if self.has_interval:
            return (self.interval_lower_ms + self.interval_upper_ms) / 2
        if self.offset_ms is None:
            raise ValueError("Evidence has neither an interval nor an exact offset")
        return self.offset_ms

    @property
    def fully_readable(self) -> bool:
        return (
            self.matched_readable_qrs is not None
            and self.grid_qrs is not None
            and self.grid_qrs > 0
            and self.matched_readable_qrs >= self.grid_qrs
        )

    @property
    def missing_qrs(self) -> int | None:
        if self.matched_readable_qrs is None or self.grid_qrs is None:
            return None
        return max(0, self.grid_qrs - self.matched_readable_qrs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_saved_analysis(output: Path) -> tuple[dict, list[dict], dict]:
    json_path = output / ANALYSIS_JSON
    csv_path = output / FRAMES_CSV
    if not json_path.is_file() or not csv_path.is_file():
        raise ValueError(
            f"{output} must contain both {ANALYSIS_JSON} and {FRAMES_CSV}"
        )
    try:
        report = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {json_path}") from error
    frames = report.get("frames") if isinstance(report, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{ANALYSIS_JSON} has no frame evidence")

    with csv_path.open(encoding="utf-8", newline="") as source:
        csv_rows = list(csv.DictReader(source))
    if len(csv_rows) != len(frames):
        raise ValueError(
            f"Frame count differs between {ANALYSIS_JSON} ({len(frames)}) and "
            f"{FRAMES_CSV} ({len(csv_rows)})"
        )
    for index, (json_row, csv_row) in enumerate(zip(frames, csv_rows), 1):
        if str(json_row.get("filename", "")) != str(csv_row.get("filename", "")):
            raise ValueError(f"Frame order differs between saved files at row {index}")
    provenance = {
        ANALYSIS_JSON: _sha256(json_path),
        FRAMES_CSV: _sha256(csv_path),
    }
    return report, frames, provenance


def _prepare_evidence(
    frames: list[dict], *, default_grid_qrs: int | None = None
) -> list[FrameEvidence]:
    evidence = []
    previous_pts: float | None = None
    history: list[float] = []
    for position, row in enumerate(frames, 1):
        pts = _finite_number(row.get("pts_ns"))
        step = NOMINAL_PTS_STEP_MS
        if pts is not None and previous_pts is not None:
            candidate = (pts - previous_pts) / 1e6
            if 0 < candidate <= MAX_CONTINUOUS_PTS_STEP_MS:
                step = candidate
            else:
                history.clear()
        elif previous_pts is not None:
            history.clear()

        recent = [step, *reversed(history[-5:])]
        recent.extend([NOMINAL_PTS_STEP_MS] * (6 - len(recent)))
        offset = _finite_number(row.get("pts_minus_latest_qr_ms"))
        interval_lower = _finite_number(row.get("offset_interval_lower_ms"))
        interval_upper = _finite_number(row.get("offset_interval_upper_ms"))
        if (
            interval_lower is None
            or interval_upper is None
            or interval_lower > interval_upper
        ):
            interval_lower = None
            interval_upper = None
        evidence.append(FrameEvidence(
            frame_number=int(row.get("frame_number") or position),
            filename=str(row.get("filename") or f"row-{position}"),
            validation=str(row.get("validation") or "unknown"),
            timing_status=row.get("timing_status"),
            offset_ms=offset,
            pts_step_ms=step,
            pts_history_ms=tuple(recent[:6]),
            interval_lower_ms=interval_lower,
            interval_upper_ms=interval_upper,
            matched_readable_qrs=(
                int(row["matched_readable_qrs"])
                if _finite_number(row.get("matched_readable_qrs")) is not None
                else None
            ),
            grid_qrs=(
                int(row["grid_qrs"])
                if _finite_number(row.get("grid_qrs")) is not None
                else default_grid_qrs
            ),
            latest_cell=(
                int(row["latest_cell"])
                if _finite_number(row.get("latest_cell")) is not None
                else None
            ),
            latest_cell_name=(
                str(row["latest_cell_name"])
                if row.get("latest_cell_name") is not None
                else None
            ),
            readable_cells=(
                tuple(value is not None and str(value).strip() != "" for value in row["qr_values_ms"])
                if isinstance(row.get("qr_values_ms"), list)
                else ()
            ),
        ))
        history.append(step)
        previous_pts = pts
    return evidence


def _interval_arrays(rows: Iterable[FrameEvidence]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(rows)
    representative = np.asarray(
        [row.representative_offset_ms for row in rows], dtype=float
    )
    lower = np.asarray([
        row.interval_lower_ms if row.has_interval else row.representative_offset_ms
        for row in rows
    ], dtype=float)
    upper = np.asarray([
        row.interval_upper_ms if row.has_interval else row.representative_offset_ms
        for row in rows
    ], dtype=float)
    return representative, lower, upper


def _interval_residual(
    lower: np.ndarray, upper: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    """Return signed distance to an interval, with zero for predictions inside it."""
    return np.where(
        prediction < lower,
        lower - prediction,
        np.where(prediction > upper, upper - prediction, 0.0),
    )


def _metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    residual = (
        target - prediction
        if lower is None or upper is None
        else _interval_residual(lower, upper, prediction)
    )
    absolute = np.abs(residual)
    return {
        "n": int(len(target)),
        "mae_ms": float(np.mean(absolute)),
        "median_absolute_ms": float(np.median(absolute)),
        "p95_absolute_ms": float(np.percentile(absolute, 95)),
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "bias_ms": float(np.mean(residual)),
        "within_interval_pct": float(100 * np.mean(absolute == 0)),
        "within_5ms_pct": float(100 * np.mean(absolute <= 5)),
        "within_10ms_pct": float(100 * np.mean(absolute <= 10)),
    }


def _describe(values: np.ndarray) -> dict:
    median = float(np.median(values))
    return {
        "n": int(len(values)),
        "mean_ms": float(np.mean(values)),
        "median_ms": median,
        "standard_deviation_ms": float(np.std(values)),
        "mad_ms": float(np.median(np.abs(values - median))),
        "p05_ms": float(np.percentile(values, 5)),
        "p25_ms": float(np.percentile(values, 25)),
        "p75_ms": float(np.percentile(values, 75)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def _maximum_interval_consensus(rows: list[FrameEvidence]) -> dict | None:
    intervals = [
        (row.interval_lower_ms, row.interval_upper_ms)
        for row in rows
        if row.has_interval
    ]
    if not intervals:
        return None
    boundaries = sorted({value for interval in intervals for value in interval})
    candidates: list[tuple[float, float, float, int]] = []
    for index, boundary in enumerate(boundaries):
        coverage = sum(lower <= boundary <= upper for lower, upper in intervals)
        candidates.append((boundary, boundary, boundary, coverage))
        if index + 1 < len(boundaries):
            following = boundaries[index + 1]
            midpoint = (boundary + following) / 2
            coverage = sum(lower <= midpoint <= upper for lower, upper in intervals)
            candidates.append((boundary, following, midpoint, coverage))
    maximum = max(candidate[3] for candidate in candidates)
    preferred = float(np.median([
        (lower + upper) / 2 for lower, upper in intervals
    ]))
    winners = [candidate for candidate in candidates if candidate[3] == maximum]
    regions: list[list[float]] = []
    for start, end, _midpoint, _coverage in winners:
        if regions and start <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], end)
        else:
            regions.append([start, end])
    chosen = min(
        regions,
        key=lambda region: (
            0 if region[0] <= preferred <= region[1]
            else min(abs(preferred - region[0]), abs(preferred - region[1])),
            -(region[1] - region[0]),
        ),
    )
    estimate = min(max(preferred, chosen[0]), chosen[1])
    return {
        "method": "maximum overlap across all interval-usable frames",
        "offset_range_lower_ms": chosen[0],
        "offset_range_upper_ms": chosen[1],
        "estimated_offset_ms": estimate,
        "contributing_frames": len(intervals),
        "maximum_consistent_frames": maximum,
        "maximum_consistent_pct": 100 * maximum / len(intervals),
    }


def _generation_step_analysis(
    rows: list[FrameEvidence], consensus: dict | None
) -> dict | None:
    interval_rows = [row for row in rows if row.has_interval]
    if not interval_rows or consensus is None:
        return None
    widths = np.asarray([
        row.interval_upper_ms - row.interval_lower_ms for row in interval_rows
    ])
    period_ms = float(np.median(widths))
    if not math.isfinite(period_ms) or period_ms <= 0:
        return None
    anchor_ms = float(consensus["estimated_offset_ms"])
    groups: dict[int, list[FrameEvidence]] = {}
    for row in interval_rows:
        generation = round((row.representative_offset_ms - anchor_ms) / period_ms)
        groups.setdefault(generation, []).append(row)
    bands = []
    for generation, members in sorted(groups.items()):
        exact = [row.offset_ms for row in members if row.offset_ms is not None]
        bands.append({
            "generation_steps_from_anchor": generation,
            "frames": len(members),
            "interval_midpoint_median_ms": float(np.median([
                row.representative_offset_ms for row in members
            ])),
            "exact_marker_offset_median_ms": (
                float(np.median(exact)) if exact else None
            ),
        })
    return {
        "estimated_display_period_ms": period_ms,
        "anchor_ms": anchor_ms,
        "classification": (
            "Diagnostic only: integer presentation-generation bands are reported "
            "but are not supplied to live PTS predictors."
        ),
        "bands": bands,
    }


def _readability_analysis(
    rows: list[FrameEvidence], display_period_ms: float | None
) -> dict:
    known = [row for row in rows if row.matched_readable_qrs is not None]
    fully_readable = [row for row in known if row.fully_readable]
    partial = [row for row in known if not row.fully_readable]

    def subset_metrics(members: list[FrameEvidence]) -> dict | None:
        if not members:
            return None
        target, lower, upper = _interval_arrays(members)
        return {
            "frames": len(members),
            "midpoint_distribution": _describe(target),
            "current_fixed_metrics": _metrics(
                target,
                np.full(len(target), CURRENT_CORRECTION_MS),
                lower,
                upper,
            ),
        }

    by_cell = []
    cells = sorted({row.latest_cell for row in known if row.latest_cell is not None})
    for cell in cells:
        members = [row for row in known if row.latest_cell == cell]
        missing = [row.missing_qrs or 0 for row in members]
        by_cell.append({
            "cell": cell,
            "cell_name": next(
                (row.latest_cell_name for row in members if row.latest_cell_name),
                f"Cell {cell}",
            ),
            "selected_as_newest_frames": len(members),
            "fully_readable_frames": sum(row.fully_readable for row in members),
            "fully_readable_pct": 100 * sum(row.fully_readable for row in members) / len(members),
            "mean_matched_readable_qrs": float(np.mean([
                row.matched_readable_qrs for row in members
            ])),
            "mean_missing_qrs": float(np.mean(missing)),
            "interval_midpoint_median_ms": float(np.median([
                row.representative_offset_ms for row in members
            ])),
        })

    cell_map_rows = [row for row in known if row.readable_cells]
    cell_count = max((len(row.readable_cells) for row in cell_map_rows), default=0)
    per_cell = []
    selected_counts = {
        row["cell"]: row["selected_as_newest_frames"] for row in by_cell
    }
    for cell in range(cell_count):
        observed = [row for row in cell_map_rows if cell < len(row.readable_cells)]
        readable_frames = sum(row.readable_cells[cell] for row in observed)
        per_cell.append({
            "cell": cell,
            "observed_frames": len(observed),
            "readable_frames": readable_frames,
            "readable_pct": 100 * readable_frames / len(observed),
            "selected_as_newest_frames": selected_counts.get(cell, 0),
        })

    ambiguity = None
    if display_period_ms is not None and math.isfinite(display_period_ms) and display_period_ms > 0:
        deltas = [
            round((row.representative_offset_ms - CURRENT_CORRECTION_MS) / display_period_ms)
            for row in known
        ]
        ambiguity = {
            "reference_correction_ms": CURRENT_CORRECTION_MS,
            "classification": (
                "Diagnostic only. Positive steps mean the selected marker interval is one or more "
                "display periods older than the provisional fixed correction. No alternative is "
                "chosen for fitting or live prediction."
            ),
            "steps_from_current_counts": {
                str(step): deltas.count(step) for step in sorted(set(deltas))
            },
            "one_or_more_older_generation_frames": sum(step >= 1 for step in deltas),
            "two_or_more_older_generation_frames": sum(step >= 2 for step in deltas),
        }

    return {
        "known_readability_frames": len(known),
        "fully_readable_frames": len(fully_readable),
        "partial_readability_frames": len(partial),
        "fully_readable_pct": 100 * len(fully_readable) / len(known) if known else None,
        "fully_readable_metrics": subset_metrics(fully_readable),
        "partial_readability_metrics": subset_metrics(partial),
        "selected_newest_by_cell": by_cell,
        "per_cell_readability": per_cell,
        "generation_ambiguity": ambiguity,
        "alternative_interval_policy": {
            "alternatives_per_frame": GENERATION_ALTERNATIVES,
            "method": (
                "Report the selected interval shifted earlier by one and two measured display "
                "periods. Alternatives remain diagnostic and are never selected by residual size."
            ),
        },
    }


def _step_bucket(step_ms: float) -> float:
    return round(step_ms / STEP_BUCKET_MS) * STEP_BUCKET_MS


def _fit_interval_constant(rows: list[FrameEvidence], *, squared: bool) -> float:
    representative, lower, upper = _interval_arrays(rows)
    if squared:
        low = float(np.min(lower))
        high = float(np.max(upper))
        for _ in range(100):
            candidate = (low + high) / 2
            residual_sum = float(np.sum(_interval_residual(
                lower, upper, np.full(len(rows), candidate)
            )))
            if residual_sum > 0:
                low = candidate
            else:
                high = candidate
        return (low + high) / 2

    preferred = float(np.median(representative))
    candidates = np.unique(np.concatenate((lower, upper, [preferred])))
    losses = np.asarray([
        np.mean(np.abs(_interval_residual(
            lower, upper, np.full(len(rows), candidate)
        )))
        for candidate in candidates
    ])
    best = float(np.min(losses))
    winners = candidates[np.isclose(losses, best, rtol=1e-12, atol=1e-12)]
    return float(min(winners, key=lambda value: abs(float(value) - preferred)))


def _fit_step_medians(train: list[FrameEvidence], fallback: float) -> dict[float, float]:
    groups: dict[float, list[FrameEvidence]] = {}
    for row in train:
        groups.setdefault(_step_bucket(row.pts_step_ms), []).append(row)
    return {
        bucket: _fit_interval_constant(values, squared=False)
        for bucket, values in groups.items()
        if len(values) >= 10
    } or {_step_bucket(NOMINAL_PTS_STEP_MS): fallback}


def _fit_history_model(
    train: list[FrameEvidence], *, history_length: int = 6
) -> dict:
    if not 1 <= history_length <= 6:
        raise ValueError("PTS history length must be between 1 and 6")
    features = np.asarray([
        row.pts_history_ms[:history_length] for row in train
    ], dtype=float)
    target, lower, upper = _interval_arrays(train)
    center = np.median(features, axis=0)
    design = np.column_stack((np.ones(len(features)), features - center))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    ridge = HISTORY_RIDGE
    ridge_matrix = np.diag([0.0, *([ridge] * (design.shape[1] - 1))])
    converged = False
    gradient_norm = math.inf
    iterations = 0
    for iterations in range(1, 201):
        prediction = design @ coefficients
        residual = _interval_residual(lower, upper, prediction)
        gradient = -(design.T @ residual) / len(design)
        gradient[1:] += ridge * coefficients[1:]
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= 1e-9:
            converged = True
            break
        active = residual != 0
        hessian = design[active].T @ design[active] / len(design) + ridge_matrix
        hessian += np.eye(design.shape[1]) * 1e-12
        try:
            direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            direction = -gradient
        directional_derivative = float(gradient @ direction)
        if directional_derivative >= 0:
            direction = -gradient
            directional_derivative = -float(gradient @ gradient)

        objective = (
            0.5 * float(np.mean(residual**2))
            + 0.5 * ridge * float(coefficients[1:] @ coefficients[1:])
        )
        step_size = 1.0
        while step_size >= 1e-10:
            updated = coefficients + step_size * direction
            updated_residual = _interval_residual(lower, upper, design @ updated)
            updated_objective = (
                0.5 * float(np.mean(updated_residual**2))
                + 0.5 * ridge * float(updated[1:] @ updated[1:])
            )
            if updated_objective <= objective + 1e-4 * step_size * directional_derivative:
                coefficients = updated
                break
            step_size *= 0.5
        else:
            break
    slopes = coefficients[1:]
    return {
        "center_ms": center.tolist(),
        "centered_intercept_ms": float(coefficients[0]),
        "intercept_ms": float(coefficients[0] - center @ slopes),
        "slopes": slopes.tolist(),
        "history_length": history_length,
        "ridge": ridge,
        "rank": int(rank),
        "fit_objective": "squared distance outside presentation interval",
        "optimizer_iterations": iterations,
        "optimizer_converged": converged,
        "optimizer_final_gradient_max_abs": gradient_norm,
        "features": [
            "pts_step_ms",
            *[f"pts_step_lag{i}_ms" for i in range(1, history_length)],
        ],
    }


def _predict_history(model: dict, rows: Iterable[FrameEvidence]) -> np.ndarray:
    history_length = int(model.get("history_length", len(model["slopes"])))
    features = np.asarray([
        row.pts_history_ms[:history_length] for row in rows
    ], dtype=float)
    center = np.asarray(model["center_ms"], dtype=float)
    slopes = np.asarray(model["slopes"], dtype=float)
    return model["centered_intercept_ms"] + (features - center) @ slopes


def _cadence_key(row: FrameEvidence, depth: int) -> tuple[float, ...]:
    return tuple(_step_bucket(value) for value in row.pts_history_ms[:depth])


def _fit_cadence_model(
    train: list[FrameEvidence], *, depth: int, fallback: float
) -> dict:
    step_fallbacks = _fit_step_medians(train, fallback)
    groups: dict[tuple[float, ...], list[FrameEvidence]] = {}
    for row in train:
        groups.setdefault(_cadence_key(row, depth), []).append(row)
    states = {
        key: _fit_interval_constant(members, squared=False)
        for key, members in groups.items()
        if len(members) >= MINIMUM_CADENCE_STATE_FRAMES
    }
    return {
        "depth": depth,
        "fallback_ms": fallback,
        "step_fallbacks": step_fallbacks,
        "states": states,
    }


def _predict_cadence(model: dict, rows: Iterable[FrameEvidence]) -> np.ndarray:
    depth = int(model["depth"])
    return np.asarray([
        model["states"].get(
            _cadence_key(row, depth),
            model["step_fallbacks"].get(
                _step_bucket(row.pts_step_ms), model["fallback_ms"]
            ),
        )
        for row in rows
    ])


def _select_cadence_depth(train: list[FrameEvidence]) -> tuple[dict, list[dict]]:
    inner_count = max(1, min(len(train) - 1, int(len(train) * TRAIN_FRACTION)))
    inner_train = train[:inner_count]
    inner_holdout = train[inner_count:]
    target, lower, upper = _interval_arrays(inner_holdout)
    trials = []
    for depth in range(1, 4):
        fallback = _fit_interval_constant(inner_train, squared=False)
        model = _fit_cadence_model(inner_train, depth=depth, fallback=fallback)
        metrics = _metrics(target, _predict_cadence(model, inner_holdout), lower, upper)
        trials.append({"depth": depth, "metrics": metrics})
    selected = min(
        trials,
        key=lambda trial: (
            trial["metrics"]["mae_ms"],
            trial["metrics"]["p95_absolute_ms"],
            trial["depth"],
        ),
    )["depth"]
    fallback = _fit_interval_constant(train, squared=False)
    return _fit_cadence_model(train, depth=selected, fallback=fallback), trials


def _select_history_length(train: list[FrameEvidence]) -> tuple[dict, list[dict]]:
    inner_count = max(1, min(len(train) - 1, int(len(train) * TRAIN_FRACTION)))
    inner_train = train[:inner_count]
    inner_holdout = train[inner_count:]
    target, lower, upper = _interval_arrays(inner_holdout)
    trials = []
    for history_length in range(1, 7):
        model = _fit_history_model(inner_train, history_length=history_length)
        metrics = _metrics(target, _predict_history(model, inner_holdout), lower, upper)
        trials.append({"history_length": history_length, "metrics": metrics})
    selected = min(
        trials,
        key=lambda trial: (
            trial["metrics"]["mae_ms"],
            trial["metrics"]["p95_absolute_ms"],
            trial["history_length"],
        ),
    )["history_length"]
    return _fit_history_model(train, history_length=selected), trials


def _strategy_predictions(
    usable: list[FrameEvidence], train_count: int
) -> tuple[dict, dict[str, np.ndarray]]:
    train = usable[:train_count]
    median = _fit_interval_constant(train, squared=False)
    mean = _fit_interval_constant(train, squared=True)
    step_medians = _fit_step_medians(train, median)
    cadence_model, cadence_trials = _select_cadence_depth(train)
    selected_history_model, history_trials = _select_history_length(train)
    history_model = _fit_history_model(train, history_length=6)

    count = len(usable)
    predictions = {
        "fixed_current_ms": np.full(count, CURRENT_CORRECTION_MS),
        "fixed_train_median": np.full(count, median),
        "fixed_train_mean": np.full(count, mean),
        "pts_step_median": np.asarray([
            step_medians.get(_step_bucket(row.pts_step_ms), median) for row in usable
        ]),
        "pts_cadence_state": _predict_cadence(cadence_model, usable),
        "pts_history_selected_linear": _predict_history(selected_history_model, usable),
        "pts_history6_linear": _predict_history(history_model, usable),
    }
    definitions = {
        "fixed_current_ms": {
            "label": f"Current fixed {CURRENT_CORRECTION_MS:.3f} ms",
            "fit": "No fitting; current configured correction",
            "parameters": {"correction_ms": CURRENT_CORRECTION_MS},
        },
        "fixed_train_median": {
            "label": "Interval-median fixed",
            "fit": "Minimizes training absolute distance outside presentation intervals",
            "parameters": {"correction_ms": median},
        },
        "fixed_train_mean": {
            "label": "Interval least-squares fixed",
            "fit": "Minimizes training squared distance outside presentation intervals",
            "parameters": {"correction_ms": mean},
        },
        "pts_step_median": {
            "label": "PTS-step interval median",
            "fit": f"Training interval medians in {STEP_BUCKET_MS:g} ms PTS-step buckets",
            "parameters": {
                "fallback_ms": median,
                "bucket_corrections_ms": {f"{key:g}": value for key, value in sorted(step_medians.items())},
            },
        },
        "pts_cadence_state": {
            "label": "PTS cadence-state interval median",
            "fit": (
                "Chronologically selected categorical state using the current and up to two "
                "previous 5 ms PTS-step buckets"
            ),
            "parameters": {
                "selected_depth": cadence_model["depth"],
                "minimum_state_frames": MINIMUM_CADENCE_STATE_FRAMES,
                "fallback_ms": cadence_model["fallback_ms"],
                "step_fallbacks_ms": {
                    f"{key:g}": value
                    for key, value in sorted(cadence_model["step_fallbacks"].items())
                },
                "state_corrections_ms": {
                    "/".join(f"{value:g}" for value in key): correction
                    for key, correction in sorted(cadence_model["states"].items())
                },
                "inner_validation_trials": cadence_trials,
            },
        },
        "pts_history_selected_linear": {
            "label": "Selected-length regularized PTS history",
            "fit": (
                "Chronologically selects 1-6 current/lagged PTS intervals inside the training "
                "portion, then refits that length with ridge regularization"
            ),
            "parameters": {
                **selected_history_model,
                "inner_validation_trials": history_trials,
            },
        },
        "pts_history6_linear": {
            "label": "Regularized six-step PTS interval model",
            "fit": (
                "Squared interval-distance fit using current and five previous PTS intervals "
                "with ridge regularization"
            ),
            "parameters": history_model,
        },
    }
    return definitions, predictions


def _write_timeline_graph(
    path: Path,
    evidence: list[FrameEvidence],
    session_correction: float,
    *,
    save_svg: bool = False,
) -> None:
    plt, _ = _plotting()
    usable = [row for row in evidence if row.usable]
    original_clean = [row for row in usable if row.clean]
    retained_flagged = [row for row in usable if not row.clean]
    unusable = [row for row in evidence if not row.usable and row.offset_ms is not None]
    figure, axis = plt.subplots(figsize=(12, 5.2), layout="constrained")
    if unusable:
        axis.scatter(
            [row.frame_number for row in unusable],
            [row.offset_ms for row in unusable],
            s=9,
            alpha=0.28,
            color="#7a858f",
            label="Unusable exact-marker evidence",
        )
    axis.scatter(
        [row.frame_number for row in original_clean],
        [row.representative_offset_ms for row in original_clean],
        s=11,
        alpha=0.62,
        color="#167f8c",
        label="Originally clean interval midpoint",
    )
    if retained_flagged:
        axis.scatter(
            [row.frame_number for row in retained_flagged],
            [row.representative_offset_ms for row in retained_flagged],
            s=10,
            alpha=0.45,
            color="#d38b3d",
            label="Retained timing-flagged interval midpoint",
        )
    axis.axhline(
        CURRENT_CORRECTION_MS,
        color="#b4465a",
        linestyle="--",
        label=f"Current {CURRENT_CORRECTION_MS:.3f} ms",
    )
    axis.axhline(
        session_correction,
        color="#147a88",
        linestyle="--",
        label=f"Interval-aware fixed {session_correction:.3f} ms",
    )
    axis.set(
        title="Presentation-interval midpoint across the recording",
        xlabel="Camera frame number",
        ylabel="PTS minus presentation-interval midpoint (ms)",
    )
    axis.grid(alpha=0.22)
    axis.legend(ncols=2, fontsize=9)
    _save_figure(figure, path, save_svg=save_svg)


def _write_readability_graph(
    path: Path,
    usable: list[FrameEvidence],
    readability: dict,
    *,
    save_svg: bool = False,
) -> None:
    """Show missing-QR timing bands and selected-newest reliability per cell."""
    plt, PercentFormatter = _plotting()
    figure, (timeline, cells) = plt.subplots(
        1, 2, figsize=(14, 5.2), layout="constrained"
    )
    known = [row for row in usable if row.missing_qrs is not None]
    unknown = [row for row in usable if row.missing_qrs is None]
    if known:
        points = timeline.scatter(
            [row.frame_number for row in known],
            [row.representative_offset_ms for row in known],
            c=[row.missing_qrs for row in known],
            cmap="viridis",
            vmin=0,
            vmax=max(1, max(row.missing_qrs or 0 for row in known)),
            s=12,
            alpha=0.7,
        )
        colorbar = figure.colorbar(points, ax=timeline, pad=0.02)
        colorbar.set_label("Unreadable or unmatched QR cells")
    if unknown:
        timeline.scatter(
            [row.frame_number for row in unknown],
            [row.representative_offset_ms for row in unknown],
            color="#7a858f",
            s=10,
            alpha=0.45,
            label="Readability unavailable",
        )
    timeline.axhline(
        CURRENT_CORRECTION_MS,
        color="#b4465a",
        linestyle="--",
        linewidth=1.4,
        label=f"Current {CURRENT_CORRECTION_MS:.3f} ms",
    )
    timeline.set(
        title="Timing bands versus QR readability",
        xlabel="Camera frame number",
        ylabel="Presentation-interval midpoint (ms)",
    )
    timeline.grid(alpha=0.22)
    timeline.legend(fontsize=8)

    cell_rows = readability.get("per_cell_readability", [])
    if cell_rows:
        positions = np.arange(len(cell_rows))
        bars = cells.bar(
            positions,
            [row["readable_pct"] for row in cell_rows],
            color="#3f8f68",
        )
        cells.set_xticks(
            positions,
            [str(row["cell"]) for row in cell_rows],
        )
        for bar, row in zip(bars, cell_rows):
            cells.text(
                bar.get_x() + bar.get_width() / 2,
                min(98, bar.get_height() + 2),
                f"newest={row['selected_as_newest_frames']}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
        cells.set_ylim(0, 112)
        cells.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    else:
        cells.text(
            0.5,
            0.5,
            "Selected-cell metadata unavailable",
            transform=cells.transAxes,
            ha="center",
            va="center",
        )
    cells.set(
        title="Per-cell QR readability and newest-selection count",
        xlabel="Detected grid cell (zero-based)",
        ylabel="Frames where cell decoded (%)",
    )
    cells.grid(axis="y", alpha=0.22)
    _save_figure(figure, path, save_svg=save_svg)


def _write_residual_graph(
    path: Path,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    predictions: dict[str, np.ndarray],
    definitions: dict,
    holdout_slice: slice,
    best_key: str,
    *,
    save_svg: bool = False,
) -> None:
    plt, _ = _plotting()
    keys = ["fixed_current_ms", "fixed_train_median"]
    if best_key not in keys:
        keys.append(best_key)
    colors = ("#b4465a", "#8b7a2f", "#147a88")
    residuals = {
        key: np.sort(np.abs(_interval_residual(
            lower[holdout_slice],
            upper[holdout_slice],
            predictions[key][holdout_slice],
        )))
        for key in keys
    }
    figure, axis = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for key, color in zip(keys, colors):
        values = residuals[key]
        cumulative = np.arange(1, len(values) + 1) / len(values) * 100
        axis.plot(values, cumulative, color=color, linewidth=2.2, label=definitions[key]["label"])
    x_max = max(10.0, max(float(np.percentile(values, 99)) for values in residuals.values()))
    axis.set(
        title="Chronological holdout residual comparison",
        xlabel="Absolute distance outside presentation interval (ms)",
        ylabel="Holdout frames within residual (%)",
        xlim=(0, x_max * 1.04),
        ylim=(0, 100),
    )
    axis.grid(alpha=0.22)
    axis.legend(fontsize=9)
    _save_figure(figure, path, save_svg=save_svg)


def _save_figure(figure, png_path: Path, *, save_svg: bool = False) -> None:
    """Save a display-friendly PNG and, when requested, a vector SVG."""
    plt, _ = _plotting()
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    if save_svg:
        figure.savefig(png_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _succinct_histogram_edges(values: np.ndarray) -> np.ndarray:
    """Use Freedman-Diaconis spacing while keeping the display to 8-18 bins."""
    values = np.asarray(values, dtype=float)
    automatic = np.histogram_bin_edges(values, bins="fd")
    bin_count = max(8, min(18, len(automatic) - 1))
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        low, high = low - 0.5, high + 0.5
    return np.linspace(low, high, bin_count + 1)


def _write_histogram(
    path: Path,
    title: str,
    x_label: str,
    series: list[tuple[str, np.ndarray, str]],
    *,
    symmetric_about_zero: bool = False,
    references: tuple[tuple[float, str, str], ...] = (),
    save_svg: bool = False,
) -> None:
    """Render one strategy per panel using compact bins and an individual range."""
    plt, PercentFormatter = _plotting()
    panel_count = len(series)
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(5.1 * panel_count, 4.9),
        layout="constrained",
        squeeze=False,
    )
    figure.suptitle(title, fontsize=14, fontweight="bold")
    for axis, (label, raw_values, color) in zip(axes[0], series):
        values = np.asarray(raw_values, dtype=float)
        edges = _succinct_histogram_edges(values)
        weights = np.full(len(values), 100.0 / len(values))
        axis.hist(values, bins=edges, weights=weights, color=color, edgecolor="white", linewidth=0.8)
        data_low, data_high = float(np.min(values)), float(np.max(values))
        if symmetric_about_zero:
            extent = max(abs(data_low), abs(data_high), 1.0) * 1.06
            axis.set_xlim(-extent, extent)
            axis.axvline(0, color="#25313c", linewidth=1.5, linestyle="--")
            mae = float(np.mean(np.abs(values)))
            summary = f"MAE {mae:.3f} ms\nBias {np.mean(values):+.3f} ms\n{len(edges) - 1} bins"
        else:
            span = max(data_high - data_low, 1.0)
            axis.set_xlim(data_low - 0.05 * span, data_high + 0.05 * span)
            summary = f"Median {np.median(values):.3f} ms\n{len(edges) - 1} bins"
        axis.text(
            0.98,
            0.96,
            summary,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#c9d0d6", "alpha": 0.9},
        )
        for value, reference_label, reference_color in references:
            axis.axvline(value, color=reference_color, linewidth=1.8, linestyle=":", label=reference_label)
        axis.set(title=label, xlabel=x_label, ylabel="Frames in bin (%)")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        axis.grid(axis="y", alpha=0.22)
        if references:
            axis.legend(fontsize=8, loc="upper left")
    _save_figure(figure, path, save_svg=save_svg)


def _write_predictions(
    path: Path,
    usable: list[FrameEvidence],
    train_count: int,
    predictions: dict[str, np.ndarray],
    *,
    display_period_ms: float | None,
) -> None:
    fields = [
        "frame_number", "filename", "split", "source_validation",
        "source_timing_status", "evidence_policy", "observed_offset_ms",
        "interval_lower_ms", "interval_upper_ms", "representative_offset_ms",
        "pts_step_ms", "matched_readable_qrs", "grid_qrs", "missing_qrs",
        "readability_class", "latest_cell", "latest_cell_name",
        "diagnostic_generation_steps_from_current",
        *[
            field
            for generation in range(1, GENERATION_ALTERNATIVES + 1)
            for field in (
                f"newer_generation_{generation}_interval_lower_ms",
                f"newer_generation_{generation}_interval_upper_ms",
            )
        ],
        *[f"correction_{key}" for key in predictions],
        *[f"interval_residual_{key}" for key in predictions],
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(usable):
            lower = (
                row.interval_lower_ms if row.has_interval
                else row.representative_offset_ms
            )
            upper = (
                row.interval_upper_ms if row.has_interval
                else row.representative_offset_ms
            )
            record = {
                "frame_number": row.frame_number,
                "filename": row.filename,
                "split": "train" if index < train_count else "holdout",
                "source_validation": row.validation,
                "source_timing_status": row.timing_status,
                "evidence_policy": (
                    "presentation_interval" if row.has_interval
                    else "legacy_exact_marker_fallback"
                ),
                "observed_offset_ms": row.offset_ms,
                "interval_lower_ms": lower,
                "interval_upper_ms": upper,
                "representative_offset_ms": row.representative_offset_ms,
                "pts_step_ms": row.pts_step_ms,
                "matched_readable_qrs": row.matched_readable_qrs,
                "grid_qrs": row.grid_qrs,
                "missing_qrs": row.missing_qrs,
                "readability_class": (
                    "full" if row.fully_readable
                    else "partial" if row.matched_readable_qrs is not None
                    else "unknown"
                ),
                "latest_cell": row.latest_cell,
                "latest_cell_name": row.latest_cell_name,
                "diagnostic_generation_steps_from_current": (
                    round(
                        (row.representative_offset_ms - CURRENT_CORRECTION_MS)
                        / display_period_ms
                    )
                    if display_period_ms is not None and display_period_ms > 0
                    else None
                ),
            }
            if display_period_ms is not None and display_period_ms > 0:
                for generation in range(1, GENERATION_ALTERNATIVES + 1):
                    shift = generation * display_period_ms
                    record[f"newer_generation_{generation}_interval_lower_ms"] = lower - shift
                    record[f"newer_generation_{generation}_interval_upper_ms"] = upper - shift
            record.update({
                f"correction_{key}": float(values[index])
                for key, values in predictions.items()
            })
            record.update({
                f"interval_residual_{key}": float(_interval_residual(
                    np.asarray([lower]),
                    np.asarray([upper]),
                    np.asarray([values[index]]),
                )[0])
                for key, values in predictions.items()
            })
            writer.writerow(record)


def _markdown(report: dict) -> str:
    sample = report["interval_midpoint_distribution"]
    verdict = report["verdict"]
    quality = report["data_quality"]
    graph_suffixes = " / ".join(f"`.{suffix}`" for suffix in report["graph_formats"])
    lines = [
        "# Calibration timing verdict",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        verdict["operational_recommendation"],
        "",
        verdict["current_correction_assessment"],
        "",
        verdict["dynamic_strategy_assessment"],
        "",
        "This correction is scored against the software presentation interval associated with each "
        "journal-matched QR. It is not a measured physical exposure time or a completed camera/radar alignment proof.",
        "",
        "## Evidence quality",
        "",
        f"- Processed frames: {quality['processed_frames']}",
        f"- Interval-usable frames used: {quality['usable_frames']} ({quality['usable_pct']:.1f}%)",
        f"- Originally clean frames used: {quality['source_clean_frames']}",
        f"- Timing-suspect frames retained: {quality['timing_suspect_frames_retained']}",
        f"- Unknown-replacement frames retained: {quality['unknown_timing_frames_retained']}",
        f"- Incomplete frames excluded: {quality['incomplete_frames']}",
        f"- Other unusable frames excluded: {quality['other_excluded_frames']}",
        f"- Display journal: {quality['display_late_submissions']} late submissions, "
        f"{quality['display_irregular_intervals']} irregular intervals, and "
        f"{quality['display_missed_period_candidates']} missed-period candidates. These remain diagnostics "
        "and do not invalidate an otherwise complete observed interval.",
    ]
    interval_analysis = report.get("presentation_interval_analysis")
    primary_interval = (
        interval_analysis.get("primary_pts")
        if isinstance(interval_analysis, dict) else None
    )
    if isinstance(primary_interval, dict):
        lines.extend([
            "",
            "## Presentation-interval reconstruction",
            "",
            "Each decoded newest QR constrains exposure to the software presentation interval "
            "between that display event and the following one. This avoids treating the QR marker "
            "as an exact physical exposure timestamp.",
            "",
            f"- Maximum-overlap offset range: "
            f"{primary_interval['offset_range_lower_ms']:.3f} to "
            f"{primary_interval['offset_range_upper_ms']:.3f} ms",
            f"- Representative estimate inside that range: "
            f"{primary_interval['estimated_offset_ms']:.3f} ms",
            f"- Consistent interval-usable frames: {primary_interval['maximum_consistent_frames']} / "
            f"{primary_interval['contributing_frames']} "
            f"({primary_interval['maximum_consistent_pct']:.1f}%)",
            "- Late submission, irregular cadence, and later QR replacement remain visible as quality "
            "flags, but the recorded monotonic boundaries are retained.",
        ])
    generation = report.get("generation_step_analysis")
    if isinstance(generation, dict):
        lines.extend([
            "",
            "## Presentation-generation steps",
            "",
            f"The inferred display period is {generation['estimated_display_period_ms']:.3f} ms. "
            "Offsets are grouped by integer presentation periods from the maximum-overlap anchor; "
            "these groups are diagnostics and are not live predictor inputs.",
            "",
            "| Steps from anchor | Frames | Interval-midpoint median | Exact-marker median |",
            "|---:|---:|---:|---:|",
        ])
        for band in generation["bands"]:
            exact = band["exact_marker_offset_median_ms"]
            exact_text = f"{exact:.3f} ms" if exact is not None else "unavailable"
            lines.append(
                f"| {band['generation_steps_from_anchor']:+d} | {band['frames']} | "
                f"{band['interval_midpoint_median_ms']:.3f} ms | {exact_text} |"
            )
    readability = report.get("readability_analysis")
    if isinstance(readability, dict) and readability["known_readability_frames"]:
        lines.extend([
            "",
            "## QR readability and generation ambiguity",
            "",
            f"- Fully readable grids: {readability['fully_readable_frames']} / "
            f"{readability['known_readability_frames']} "
            f"({readability['fully_readable_pct']:.1f}%)",
            f"- Partial grids retained: {readability['partial_readability_frames']}",
            "- Each frame now reports one- and two-generation newer interval alternatives. "
            "They are diagnostics only and are not chosen by residual size or used by a live model.",
        ])
        ambiguity = readability.get("generation_ambiguity")
        if isinstance(ambiguity, dict):
            lines.extend([
                f"- Default-relative diagnostic: "
                f"{ambiguity['one_or_more_older_generation_frames']} frames are at least one "
                f"display period older than {ambiguity['reference_correction_ms']:.3f} ms; "
                f"{ambiguity['two_or_more_older_generation_frames']} are at least two periods older.",
            ])
            if readability["per_cell_readability"]:
                lines.extend([
                    "",
                    "| Detected grid cell | Readable frames | Readability | Selected newest |",
                    "|---:|---:|---:|---:|",
                ])
                for cell in readability["per_cell_readability"]:
                    lines.append(
                        f"| {cell['cell']} | {cell['readable_frames']} / "
                        f"{cell['observed_frames']} | {cell['readable_pct']:.1f}% | "
                        f"{cell['selected_as_newest_frames']} |"
                    )
            lines.extend([
                "",
                "| Selected newest cell | Frames | Fully readable | Mean matched QRs | Midpoint median |",
                "|---:|---:|---:|---:|---:|",
            ])
            for cell in readability["selected_newest_by_cell"]:
                lines.append(
                    f"| {cell['cell']} ({cell['cell_name']}) | "
                    f"{cell['selected_as_newest_frames']} | {cell['fully_readable_pct']:.1f}% | "
                    f"{cell['mean_matched_readable_qrs']:.2f} | "
                    f"{cell['interval_midpoint_median_ms']:.3f} ms |"
                )
    lines.extend([
        "",
        "## Presentation-interval midpoint distribution",
        "",
        "| Statistic | Milliseconds |",
        "|---|---:|",
        f"| Median | {sample['median_ms']:.3f} |",
        f"| Mean | {sample['mean_ms']:.3f} |",
        f"| MAD | {sample['mad_ms']:.3f} |",
        f"| Standard deviation | {sample['standard_deviation_ms']:.3f} |",
        f"| 5th–95th percentile | {sample['p05_ms']:.3f}–{sample['p95_ms']:.3f} |",
        "",
        "## Strategy comparison",
        "",
        "Models were fitted on the first 70% of interval-usable frames and evaluated on the later 30%. "
        "The later portion was not used to fit coefficients.",
        "",
        "| Strategy | Holdout MAE | Median absolute | P95 absolute | RMSE | Bias | Inside interval | Within 10 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in report["strategy_order"]:
        strategy = report["strategies"][key]
        metrics = strategy["holdout_metrics"]
        lines.append(
            f"| {strategy['label']} | {metrics['mae_ms']:.3f} | "
            f"{metrics['median_absolute_ms']:.3f} | {metrics['p95_absolute_ms']:.3f} | "
            f"{metrics['rmse_ms']:.3f} | {metrics['bias_ms']:+.3f} | "
            f"{metrics['within_interval_pct']:.1f}% | "
            f"{metrics['within_10ms_pct']:.1f}% |"
        )
    lines.extend([
        "",
        "## What absolute residual means",
        "",
        "Each frame supplies a lower and upper correction bound. Residual is zero when the prediction "
        "falls inside that presentation interval. Below it, residual is `lower bound - prediction`; "
        "above it, residual is `upper bound - prediction`. This prevents one uncertain display refresh "
        "from being treated as an exact continuous error.",
        "",
        "MAE is the average absolute residual across the evaluated frames. P95 absolute residual is the value "
        "that 95% of those frames meet or beat; it shows the less-common large errors that an average can hide.",
        "",
        "The PTS models use only current and previous camera-message intervals. They do not use QR values, "
        "display indices, generation-band labels, future frames, or elapsed recording time as predictors. "
        "The dynamic ranking remains exploratory "
        "because it comes from one recording; confirm it on another independently recorded calibration before enabling it live.",
        "The cadence-state depth and regularized history length are selected only inside the training "
        "portion before the final chronological holdout is scored.",
        "",
        "## Acceptance criteria",
        "",
        f"- Fixed candidates must stay within {report['acceptance_criteria']['fixed_repeatability_limit_ms']:.1f} ms "
        "across independently restarted sessions.",
        f"- A dynamic strategy must improve MAE by at least "
        f"{report['acceptance_criteria']['dynamic_minimum_mae_gain_ms']:.1f} ms.",
        f"- Its P95 may not exceed the fixed model by more than "
        f"{report['acceptance_criteria']['dynamic_p95_tolerance_ms']:.1f} ms.",
        "- Cross-session no-regression is evaluated separately; this single recording cannot pass it.",
        "",
        "## Files",
        "",
        f"- `{Path(TIMELINE_GRAPH).stem}` ({graph_suffixes}): retained interval midpoints over camera-frame order",
        f"- `{Path(RESIDUAL_GRAPH).stem}` ({graph_suffixes}): holdout interval-distance distributions",
        f"- `{Path(OFFSET_HISTOGRAM).stem}` ({graph_suffixes}): distribution of retained interval midpoints",
        f"- `{Path(FIXED_RESIDUAL_HISTOGRAM).stem}` ({graph_suffixes}): one panel per fixed strategy",
        f"- `{Path(PTS_RESIDUAL_HISTOGRAM).stem}` ({graph_suffixes}): one panel per PTS strategy",
        f"- `{Path(READABILITY_GRAPH).stem}` ({graph_suffixes}): missing-QR timing bands and per-cell readability",
        f"- `{PREDICTIONS_CSV}`: per-usable-frame interval bounds, predictions, residuals, and split labels",
        f"- `{VERDICT_JSON}`: complete metrics, model coefficients, provenance, and machine-readable verdict",
        "",
    ])
    return "\n".join(lines)


def analyze_output_directory(
    output_directory: str | Path, *, save_svg: bool = False
) -> dict:
    """Analyze the two files created by the recording window and write a verdict."""
    output = Path(output_directory).expanduser().resolve()
    if not output.is_dir():
        raise ValueError(f"Analysis output directory does not exist: {output}")
    source, frames, provenance = _load_saved_analysis(output)
    grid = source.get("grid") if isinstance(source.get("grid"), dict) else {}
    default_grid_qrs = (
        int(grid["qr_count"])
        if _finite_number(grid.get("qr_count")) is not None
        else None
    )
    evidence = _prepare_evidence(frames, default_grid_qrs=default_grid_qrs)
    usable = [row for row in evidence if row.usable]
    if len(usable) < MINIMUM_CLEAN_FRAMES:
        raise ValueError(
            f"Need at least {MINIMUM_CLEAN_FRAMES} usable offsets or presentation intervals "
            f"for a quantitative verdict; found {len(usable)}"
        )

    train_count = max(1, min(len(usable) - 1, int(len(usable) * TRAIN_FRACTION)))
    holdout_slice = slice(train_count, None)
    target, lower, upper = _interval_arrays(usable)
    definitions, predictions = _strategy_predictions(usable, train_count)
    strategies = {}
    for key, definition in definitions.items():
        strategies[key] = {
            **definition,
            "holdout_median_correction_ms": float(np.median(predictions[key][holdout_slice])),
            "holdout_metrics": _metrics(
                target[holdout_slice],
                predictions[key][holdout_slice],
                lower[holdout_slice],
                upper[holdout_slice],
            ),
        }

    candidate_keys = (
        "fixed_train_median",
        "pts_step_median",
        "pts_cadence_state",
        "pts_history_selected_linear",
        "pts_history6_linear",
    )
    best_key = min(candidate_keys, key=lambda key: strategies[key]["holdout_metrics"]["mae_ms"])
    calibrated_fixed_mae = strategies["fixed_train_median"]["holdout_metrics"]["mae_ms"]
    best_mae = strategies[best_key]["holdout_metrics"]["mae_ms"]
    gain_ms = calibrated_fixed_mae - best_mae
    meaningful_dynamic_gain = (
        best_key != "fixed_train_median"
        and gain_ms >= max(DYNAMIC_MINIMUM_GAIN_MS, 0.05 * calibrated_fixed_mae)
        and strategies[best_key]["holdout_metrics"]["p95_absolute_ms"]
        <= strategies["fixed_train_median"]["holdout_metrics"]["p95_absolute_ms"]
        + DYNAMIC_P95_TOLERANCE_MS
    )

    distribution = _describe(target)
    source_clean = [row for row in evidence if row.clean]
    source_clean_target = np.asarray(
        [row.offset_ms for row in source_clean], dtype=float
    )
    source_clean_distribution = (
        _describe(source_clean_target) if len(source_clean_target) else None
    )
    session_correction = _fit_interval_constant(usable, squared=False)
    current_difference = CURRENT_CORRECTION_MS - session_correction
    replaces_current = not math.isclose(
        CURRENT_CORRECTION_MS,
        session_correction,
        abs_tol=0.0005,
    )
    consensus = _maximum_interval_consensus(usable)
    generation_steps = _generation_step_analysis(usable, consensus)
    display_period_ms = (
        generation_steps["estimated_display_period_ms"]
        if isinstance(generation_steps, dict)
        else None
    )
    readability = _readability_analysis(usable, display_period_ms)
    if meaningful_dynamic_gain:
        dynamic_assessment = (
            f"The lowest observed chronological-holdout interval error came from "
            f"{strategies[best_key]['label']} at {best_mae:.3f} ms MAE, "
            f"{gain_ms:.3f} ms better than the trained interval-median fixed correction. "
            "Treat it as a candidate only: "
            "one later independent calibration is required before enabling a dynamic live correction."
        )
    else:
        dynamic_assessment = (
            f"No tested dynamic PTS strategy produced a sufficiently reliable improvement over the calibrated "
            f"interval-median fixed correction on the chronological holdout. The lowest observed candidate was "
            f"{strategies[best_key]['label']} at {best_mae:.3f} ms MAE."
        )

    validation_counts: dict[str, int] = {}
    for row in evidence:
        validation_counts[row.validation] = validation_counts.get(row.validation, 0) + 1
    display = source.get("display") if isinstance(source.get("display"), dict) else {}
    incomplete_frames = (
        validation_counts.get("skipped_incomplete", 0)
        + validation_counts.get("skipped_no_readable_qr", 0)
    )
    source_clean_frames = len(source_clean)
    timing_suspect_retained = sum(
        row.validation == "accepted_timing_suspect" for row in usable
    )
    unknown_retained = sum(row.validation == "accepted_unknown" for row in usable)
    quality = {
        "processed_frames": len(evidence),
        "usable_frames": len(usable),
        "usable_pct": 100 * len(usable) / len(evidence),
        "source_clean_frames": source_clean_frames,
        "timing_suspect_frames_retained": timing_suspect_retained,
        "unknown_timing_frames_retained": unknown_retained,
        "clean_frames": source_clean_frames,
        "clean_pct": 100 * source_clean_frames / len(evidence),
        "timing_suspect_frames": validation_counts.get("accepted_timing_suspect", 0),
        "unknown_timing_frames": validation_counts.get("accepted_unknown", 0),
        "incomplete_frames": incomplete_frames,
        "other_excluded_frames": max(0, len(evidence) - len(usable) - incomplete_frames),
        "validation_counts": validation_counts,
        "display_late_submissions": int(display.get("late_submissions", 0)),
        "display_irregular_intervals": int(display.get("irregular_intervals", 0)),
        "display_missed_period_candidates": int(display.get("missed_period_candidates", 0)),
    }

    current_all = _metrics(
        target, np.full(len(target), CURRENT_CORRECTION_MS), lower, upper
    )
    session_all = _metrics(
        target, np.full(len(target), session_correction), lower, upper
    )
    if len(source_clean_target):
        source_clean_current = _metrics(
            source_clean_target,
            np.full(len(source_clean_target), CURRENT_CORRECTION_MS),
        )
        source_clean_median = float(np.median(source_clean_target))
        source_clean_session = _metrics(
            source_clean_target,
            np.full(len(source_clean_target), source_clean_median),
        )
    else:
        source_clean_current = None
        source_clean_session = None
    graph_files = [
        TIMELINE_GRAPH,
        RESIDUAL_GRAPH,
        OFFSET_HISTOGRAM,
        FIXED_RESIDUAL_HISTOGRAM,
        PTS_RESIDUAL_HISTOGRAM,
        READABILITY_GRAPH,
    ]
    if save_svg:
        graph_files.extend(str(Path(path).with_suffix(".svg")) for path in graph_files.copy())
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "output_directory": str(output),
        "target": (
            "Host-anchored camera PTS constrained by consecutive software presentation "
            "boundaries for each journal-matched decoded QR; not physical exposure truth"
        ),
        "source_recording_directory": source.get("recording_directory"),
        "source_provenance_sha256": provenance,
        "evidence_policy": {
            "primary": "journal-matched accepted frames with finite ordered presentation bounds",
            "legacy_fallback": "exact marker offsets only when source status is accepted_clean",
            "timing_flags": "retained as diagnostics; they do not discard complete intervals",
            "residual": "zero inside interval; signed distance to nearest bound outside interval",
        },
        "presentation_interval_analysis": {
            "method": "interval consensus recomputed from every interval-usable frame",
            "primary_pts": consensus,
            "source_recording_analysis": source.get("presentation_interval_analysis"),
        },
        "generation_step_analysis": generation_steps,
        "readability_analysis": readability,
        "data_quality": quality,
        "interval_midpoint_distribution": distribution,
        "clean_offset_distribution": source_clean_distribution,
        "chronological_split": {
            "train_fraction": TRAIN_FRACTION,
            "train_usable_frames": train_count,
            "holdout_usable_frames": len(usable) - train_count,
            "train_clean_frames": sum(row.clean for row in usable[:train_count]),
            "holdout_clean_frames": sum(row.clean for row in usable[train_count:]),
            "train_last_camera_frame": usable[train_count - 1].frame_number,
            "holdout_first_camera_frame": usable[train_count].frame_number,
        },
        "current_and_session_fixed_metrics_all_usable": {
            "current_fixed": current_all,
            "session_full_interval_median": session_all,
        },
        "current_and_session_fixed_metrics_all_clean": {
            "current_fixed": source_clean_current,
            "session_full_median": source_clean_session,
        },
        "strategy_order": list(definitions),
        "strategies": strategies,
        "verdict": {
            "status": "provisional_interval_session_correction",
            "recommended_fixed_correction_ms": session_correction,
            "replaces_current_correction": replaces_current,
            "current_correction_ms": CURRENT_CORRECTION_MS,
            "current_minus_recommended_ms": current_difference,
            "best_observed_holdout_strategy": best_key,
            "dynamic_candidate_is_meaningful": meaningful_dynamic_gain,
            "operational_recommendation": (
                f"For this recording, the interval-aware low-complexity candidate is "
                f"{session_correction:.3f} ms. It minimizes absolute distance outside the retained "
                "presentation intervals. "
                + (
                    f"If adopted provisionally, it replaces the {CURRENT_CORRECTION_MS:.3f} ms "
                    "subtraction; it is not added to it."
                    if replaces_current else
                    "The configured provisional default already matches this result at 0.001 ms precision."
                )
            ),
            "current_correction_assessment": (
                f"The current {CURRENT_CORRECTION_MS:.3f} ms correction falls inside "
                f"{current_all['within_interval_pct']:.1f}% of retained presentation intervals "
                f"and has {current_all['mae_ms']:.3f} ms interval-distance MAE."
            ),
            "dynamic_strategy_assessment": dynamic_assessment,
            "deployment_boundary": (
                "Do not enable a learned dynamic correction from this one recording. Confirm the same "
                "preselected strategy on a later independent recording. Preserve late, irregular, and "
                "replacement events as diagnostics without discarding complete presentation intervals."
            ),
        },
        "output_files": [VERDICT_JSON, VERDICT_MARKDOWN, PREDICTIONS_CSV, *graph_files],
        "graph_formats": ["png", *(("svg",) if save_svg else ())],
        "histogram_policy": {
            "renderer": "Matplotlib",
            "layout": "One strategy per histogram panel; no overlapping distributions",
            "bin_width": "Freedman-Diaconis, constrained to 8-18 bins per panel",
            "offset_range": "Each panel's full interval-midpoint range plus 5% padding",
            "residual_range": "Each panel's full residual range, symmetric around zero plus 6% padding",
            "height": "Percentage of evaluated frames in each bin",
        },
        "acceptance_criteria": {
            "fixed_repeatability_limit_ms": FIXED_REPEATABILITY_LIMIT_MS,
            "dynamic_minimum_mae_gain_ms": DYNAMIC_MINIMUM_GAIN_MS,
            "dynamic_p95_tolerance_ms": DYNAMIC_P95_TOLERANCE_MS,
            "single_recording_status": (
                "Cross-recording repeatability and no-regression criteria require separately "
                "recorded sessions and are evaluated by calibration.cross_recording_analysis."
            ),
            "dynamic_gain_passed_this_holdout": meaningful_dynamic_gain,
        },
    }

    _write_predictions(
        output / PREDICTIONS_CSV,
        usable,
        train_count,
        predictions,
        display_period_ms=display_period_ms,
    )
    _write_timeline_graph(
        output / TIMELINE_GRAPH, evidence, session_correction, save_svg=save_svg
    )
    _write_residual_graph(
        output / RESIDUAL_GRAPH,
        target,
        lower,
        upper,
        predictions,
        definitions,
        holdout_slice,
        best_key,
        save_svg=save_svg,
    )
    holdout_lower = lower[holdout_slice]
    holdout_upper = upper[holdout_slice]
    _write_histogram(
        output / OFFSET_HISTOGRAM,
        "Distribution of retained presentation-interval midpoints",
        "PTS minus presentation-interval midpoint (ms)",
        [("Retained interval midpoints", target, "#147a88")],
        references=(
            (
                CURRENT_CORRECTION_MS,
                f"Current correction: {CURRENT_CORRECTION_MS:.3f} ms",
                "#b4465a",
            ),
            (session_correction, f"Interval-aware fixed: {session_correction:.3f} ms", "#8b7a2f"),
        ),
        save_svg=save_svg,
    )
    _write_histogram(
        output / FIXED_RESIDUAL_HISTOGRAM,
        "Holdout interval-distance distribution — fixed corrections",
        "Signed distance outside presentation interval (ms)",
        [
            (
                f"Current {CURRENT_CORRECTION_MS:.3f} ms",
                _interval_residual(
                    holdout_lower,
                    holdout_upper,
                    predictions["fixed_current_ms"][holdout_slice],
                ),
                "#b4465a",
            ),
            ("Interval median", _interval_residual(holdout_lower, holdout_upper, predictions["fixed_train_median"][holdout_slice]), "#8b7a2f"),
            ("Interval least-squares", _interval_residual(holdout_lower, holdout_upper, predictions["fixed_train_mean"][holdout_slice]), "#6d7d8a"),
        ],
        symmetric_about_zero=True,
        save_svg=save_svg,
    )
    _write_readability_graph(
        output / READABILITY_GRAPH,
        usable,
        readability,
        save_svg=save_svg,
    )
    _write_histogram(
        output / PTS_RESIDUAL_HISTOGRAM,
        "Holdout interval-distance distribution — PTS strategies",
        "Signed distance outside presentation interval (ms)",
        [
            ("PTS-step interval median", _interval_residual(holdout_lower, holdout_upper, predictions["pts_step_median"][holdout_slice]), "#3f8f68"),
            ("PTS cadence state", _interval_residual(holdout_lower, holdout_upper, predictions["pts_cadence_state"][holdout_slice]), "#577a3f"),
            ("Selected-length PTS history", _interval_residual(holdout_lower, holdout_upper, predictions["pts_history_selected_linear"][holdout_slice]), "#5f6fb2"),
            ("Regularized six-step PTS", _interval_residual(holdout_lower, holdout_upper, predictions["pts_history6_linear"][holdout_slice]), "#147a88"),
        ],
        symmetric_about_zero=True,
        save_svg=save_svg,
    )
    (output / VERDICT_MARKDOWN).write_text(_markdown(report), encoding="utf-8")
    (output / VERDICT_JSON).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis_directory",
        type=Path,
        help=f"Folder containing {ANALYSIS_JSON} and {FRAMES_CSV}",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also save vector SVG copies of every graph (PNG is always saved)",
    )
    arguments = parser.parse_args()
    plotting_environment = matplotlib_environment()
    current_system_only = os.environ.get("PYTHONNOUSERSITE") == "1"
    selected_system_only = plotting_environment.get("PYTHONNOUSERSITE") == "1"
    if selected_system_only != current_system_only:
        command = [
            sys.executable,
            "-m",
            "calibration.quantitative_analysis",
            str(arguments.analysis_directory),
        ]
        if arguments.svg:
            command.append("--svg")
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=plotting_environment,
        )
        raise SystemExit(completed.returncode)
    os.environ.setdefault("MPLCONFIGDIR", plotting_environment["MPLCONFIGDIR"])
    report = analyze_output_directory(arguments.analysis_directory, save_svg=arguments.svg)
    verdict = report["verdict"]
    print(verdict["operational_recommendation"])
    print(verdict["dynamic_strategy_assessment"])
    print(f"Saved {VERDICT_MARKDOWN} and supporting files in {report['output_directory']}")


if __name__ == "__main__":
    main()
