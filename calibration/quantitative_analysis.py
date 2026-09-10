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


CURRENT_CORRECTION_MS = 109.0
NOMINAL_PTS_STEP_MS = 1000.0 / 30.0
MAX_CONTINUOUS_PTS_STEP_MS = 100.0
TRAIN_FRACTION = 0.70
MINIMUM_CLEAN_FRAMES = 30
STEP_BUCKET_MS = 5.0

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

    @property
    def clean(self) -> bool:
        return (
            self.validation == "accepted_clean"
            and self.timing_status == "Clean"
            and self.offset_ms is not None
        )


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


def _prepare_evidence(frames: list[dict]) -> list[FrameEvidence]:
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
        evidence.append(FrameEvidence(
            frame_number=int(row.get("frame_number") or position),
            filename=str(row.get("filename") or f"row-{position}"),
            validation=str(row.get("validation") or "unknown"),
            timing_status=row.get("timing_status"),
            offset_ms=offset,
            pts_step_ms=step,
            pts_history_ms=tuple(recent[:6]),
        ))
        history.append(step)
        previous_pts = pts
    return evidence


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    residual = target - prediction
    absolute = np.abs(residual)
    return {
        "n": int(len(target)),
        "mae_ms": float(np.mean(absolute)),
        "median_absolute_ms": float(np.median(absolute)),
        "p95_absolute_ms": float(np.percentile(absolute, 95)),
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "bias_ms": float(np.mean(residual)),
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


def _step_bucket(step_ms: float) -> float:
    return round(step_ms / STEP_BUCKET_MS) * STEP_BUCKET_MS


def _fit_step_medians(train: list[FrameEvidence], fallback: float) -> dict[float, float]:
    groups: dict[float, list[float]] = {}
    for row in train:
        groups.setdefault(_step_bucket(row.pts_step_ms), []).append(float(row.offset_ms))
    return {
        bucket: float(np.median(values))
        for bucket, values in groups.items()
        if len(values) >= 10
    } or {_step_bucket(NOMINAL_PTS_STEP_MS): fallback}


def _fit_history_model(train: list[FrameEvidence]) -> dict:
    features = np.asarray([row.pts_history_ms for row in train], dtype=float)
    target = np.asarray([row.offset_ms for row in train], dtype=float)
    center = np.median(features, axis=0)
    design = np.column_stack((np.ones(len(features)), features - center))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    slopes = coefficients[1:]
    return {
        "center_ms": center.tolist(),
        "centered_intercept_ms": float(coefficients[0]),
        "intercept_ms": float(coefficients[0] - center @ slopes),
        "slopes": slopes.tolist(),
        "rank": int(rank),
        "features": ["pts_step_ms", *[f"pts_step_lag{i}_ms" for i in range(1, 6)]],
    }


def _predict_history(model: dict, rows: Iterable[FrameEvidence]) -> np.ndarray:
    features = np.asarray([row.pts_history_ms for row in rows], dtype=float)
    center = np.asarray(model["center_ms"], dtype=float)
    slopes = np.asarray(model["slopes"], dtype=float)
    return model["centered_intercept_ms"] + (features - center) @ slopes


def _strategy_predictions(
    clean: list[FrameEvidence], train_count: int
) -> tuple[dict, dict[str, np.ndarray]]:
    train = clean[:train_count]
    target_train = np.asarray([row.offset_ms for row in train], dtype=float)
    median = float(np.median(target_train))
    mean = float(np.mean(target_train))
    step_medians = _fit_step_medians(train, median)
    history_model = _fit_history_model(train)

    count = len(clean)
    predictions = {
        "fixed_109_ms": np.full(count, CURRENT_CORRECTION_MS),
        "fixed_train_median": np.full(count, median),
        "fixed_train_mean": np.full(count, mean),
        "pts_step_median": np.asarray([
            step_medians.get(_step_bucket(row.pts_step_ms), median) for row in clean
        ]),
        "pts_history6_linear": _predict_history(history_model, clean),
    }
    definitions = {
        "fixed_109_ms": {
            "label": "Current fixed 109 ms",
            "fit": "No fitting; current configured correction",
            "parameters": {"correction_ms": CURRENT_CORRECTION_MS},
        },
        "fixed_train_median": {
            "label": "Calibrated fixed median",
            "fit": "Training median; minimizes training absolute error",
            "parameters": {"correction_ms": median},
        },
        "fixed_train_mean": {
            "label": "Calibrated fixed mean",
            "fit": "Training mean; minimizes training squared error",
            "parameters": {"correction_ms": mean},
        },
        "pts_step_median": {
            "label": "PTS-step median",
            "fit": f"Training medians in {STEP_BUCKET_MS:g} ms PTS-step buckets",
            "parameters": {
                "fallback_ms": median,
                "bucket_corrections_ms": {f"{key:g}": value for key, value in sorted(step_medians.items())},
            },
        },
        "pts_history6_linear": {
            "label": "Six-step PTS history",
            "fit": "Least-squares fit using current and five previous PTS intervals",
            "parameters": history_model,
        },
    }
    return definitions, predictions


def _write_timeline_graph(
    path: Path,
    evidence: list[FrameEvidence],
    clean_median: float,
    *,
    save_svg: bool = False,
) -> None:
    plt, _ = _plotting()
    plotted = [row for row in evidence if row.offset_ms is not None]
    clean = [row for row in plotted if row.clean]
    excluded = [row for row in plotted if not row.clean]
    figure, axis = plt.subplots(figsize=(12, 5.2), layout="constrained")
    axis.scatter(
        [row.frame_number for row in excluded],
        [row.offset_ms for row in excluded],
        s=10,
        alpha=0.38,
        color="#d38b3d",
        label="Excluded suspect/unknown evidence",
    )
    axis.scatter(
        [row.frame_number for row in clean],
        [row.offset_ms for row in clean],
        s=11,
        alpha=0.62,
        color="#167f8c",
        label="Clean evidence",
    )
    axis.axhline(CURRENT_CORRECTION_MS, color="#b4465a", linestyle="--", label="Current 109 ms")
    axis.axhline(clean_median, color="#147a88", linestyle="--", label=f"Clean median {clean_median:.3f} ms")
    axis.set(
        title="QR-derived offset across the recording",
        xlabel="Camera frame number",
        ylabel="PTS minus newest valid displayed QR (ms)",
    )
    axis.grid(alpha=0.22)
    axis.legend(ncols=2, fontsize=9)
    _save_figure(figure, path, save_svg=save_svg)


def _write_residual_graph(
    path: Path,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    definitions: dict,
    holdout_slice: slice,
    best_key: str,
    *,
    save_svg: bool = False,
) -> None:
    plt, _ = _plotting()
    keys = ["fixed_109_ms", "fixed_train_median"]
    if best_key not in keys:
        keys.append(best_key)
    colors = ("#b4465a", "#8b7a2f", "#147a88")
    residuals = {
        key: np.sort(np.abs(target[holdout_slice] - predictions[key][holdout_slice]))
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
        xlabel="Absolute residual against decoded QR marker (ms)",
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
    clean: list[FrameEvidence],
    train_count: int,
    predictions: dict[str, np.ndarray],
) -> None:
    fields = [
        "frame_number", "filename", "split", "observed_offset_ms", "pts_step_ms",
        *[f"correction_{key}" for key in predictions],
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(clean):
            record = {
                "frame_number": row.frame_number,
                "filename": row.filename,
                "split": "train" if index < train_count else "holdout",
                "observed_offset_ms": row.offset_ms,
                "pts_step_ms": row.pts_step_ms,
            }
            record.update({
                f"correction_{key}": float(values[index])
                for key, values in predictions.items()
            })
            writer.writerow(record)


def _markdown(report: dict) -> str:
    sample = report["clean_offset_distribution"]
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
        "This correction targets the host-anchored camera PTS minus the newest journal-matched decoded QR marker. "
        "It is a software-marker reference, not a measured physical exposure time or a completed camera/radar alignment proof.",
        "",
        "## Evidence quality",
        "",
        f"- Processed frames: {quality['processed_frames']}",
        f"- Clean offsets used: {quality['clean_frames']} ({quality['clean_pct']:.1f}%)",
        f"- Accepted but timing-suspect frames excluded: {quality['timing_suspect_frames']}",
        f"- Accepted with unknown replacement timing excluded: {quality['unknown_timing_frames']}",
        f"- Incomplete frames excluded: {quality['incomplete_frames']}",
        f"- Display journal: {quality['display_late_submissions']} late submissions, "
        f"{quality['display_irregular_intervals']} irregular intervals, and "
        f"{quality['display_missed_period_candidates']} missed-period candidates",
        "",
        "## Clean offset distribution",
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
        "Models were fitted on the first 70% of clean frames and evaluated on the later 30%. "
        "The later portion was not used to fit coefficients.",
        "",
        "| Strategy | Holdout MAE | Median absolute | P95 absolute | RMSE | Bias | Within 10 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in report["strategy_order"]:
        strategy = report["strategies"][key]
        metrics = strategy["holdout_metrics"]
        lines.append(
            f"| {strategy['label']} | {metrics['mae_ms']:.3f} | "
            f"{metrics['median_absolute_ms']:.3f} | {metrics['p95_absolute_ms']:.3f} | "
            f"{metrics['rmse_ms']:.3f} | {metrics['bias_ms']:+.3f} | "
            f"{metrics['within_10ms_pct']:.1f}% |"
        )
    lines.extend([
        "",
        "## What absolute residual means",
        "",
        "For each frame, the signed residual is `observed QR-derived offset - predicted correction`. "
        "The absolute residual removes the sign: `abs(observed offset - predicted correction)`. "
        "For example, if the observed offset is 104 ms and a strategy subtracts 98 ms, the residual is +6 ms "
        "and the absolute residual is 6 ms. A negative 6 ms residual also has a 6 ms absolute residual.",
        "",
        "MAE is the average absolute residual across the evaluated frames. P95 absolute residual is the value "
        "that 95% of those frames meet or beat; it shows the less-common large errors that an average can hide.",
        "",
        "The PTS-step models use only current and previous camera-message intervals. They do not use QR values, "
        "display indices, future frames, or elapsed recording time as predictors. The dynamic ranking remains exploratory "
        "because it comes from one recording; confirm it on another independently recorded calibration before enabling it live.",
        "",
        "## Files",
        "",
        f"- `{Path(TIMELINE_GRAPH).stem}` ({graph_suffixes}): clean and excluded offsets over camera-frame order",
        f"- `{Path(RESIDUAL_GRAPH).stem}` ({graph_suffixes}): holdout absolute-residual distributions",
        f"- `{Path(OFFSET_HISTOGRAM).stem}` ({graph_suffixes}): distribution of the clean observed offsets",
        f"- `{Path(FIXED_RESIDUAL_HISTOGRAM).stem}` ({graph_suffixes}): one panel per fixed strategy",
        f"- `{Path(PTS_RESIDUAL_HISTOGRAM).stem}` ({graph_suffixes}): one panel per PTS strategy",
        f"- `{PREDICTIONS_CSV}`: per-clean-frame predictions and train/holdout labels",
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
    evidence = _prepare_evidence(frames)
    clean = [row for row in evidence if row.clean]
    if len(clean) < MINIMUM_CLEAN_FRAMES:
        raise ValueError(
            f"Need at least {MINIMUM_CLEAN_FRAMES} clean offsets for a quantitative verdict; "
            f"found {len(clean)}"
        )

    train_count = max(1, min(len(clean) - 1, int(len(clean) * TRAIN_FRACTION)))
    holdout_slice = slice(train_count, None)
    target = np.asarray([row.offset_ms for row in clean], dtype=float)
    definitions, predictions = _strategy_predictions(clean, train_count)
    strategies = {}
    for key, definition in definitions.items():
        strategies[key] = {
            **definition,
            "holdout_median_correction_ms": float(np.median(predictions[key][holdout_slice])),
            "holdout_metrics": _metrics(target[holdout_slice], predictions[key][holdout_slice]),
        }

    candidate_keys = ("fixed_train_median", "pts_step_median", "pts_history6_linear")
    best_key = min(candidate_keys, key=lambda key: strategies[key]["holdout_metrics"]["mae_ms"])
    calibrated_fixed_mae = strategies["fixed_train_median"]["holdout_metrics"]["mae_ms"]
    best_mae = strategies[best_key]["holdout_metrics"]["mae_ms"]
    gain_ms = calibrated_fixed_mae - best_mae
    meaningful_dynamic_gain = (
        best_key in ("pts_step_median", "pts_history6_linear")
        and gain_ms >= max(0.5, 0.05 * calibrated_fixed_mae)
        and strategies[best_key]["holdout_metrics"]["p95_absolute_ms"]
        <= strategies["fixed_train_median"]["holdout_metrics"]["p95_absolute_ms"] + 0.5
    )

    distribution = _describe(target)
    session_median = distribution["median_ms"]
    current_difference = CURRENT_CORRECTION_MS - session_median
    direction = "over-corrects" if current_difference > 0 else "under-corrects"
    if meaningful_dynamic_gain:
        dynamic_assessment = (
            f"The lowest observed chronological-holdout error came from "
            f"{strategies[best_key]['label']} at {best_mae:.3f} ms MAE, "
            f"{gain_ms:.3f} ms better than the trained fixed median. Treat it as a candidate only: "
            "one later independent calibration is required before enabling a dynamic live correction."
        )
    else:
        dynamic_assessment = (
            f"No tested dynamic PTS strategy produced a sufficiently reliable improvement over the calibrated "
            f"fixed median on the chronological holdout. The lowest observed candidate was "
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
    quality = {
        "processed_frames": len(evidence),
        "clean_frames": len(clean),
        "clean_pct": 100 * len(clean) / len(evidence),
        "timing_suspect_frames": validation_counts.get("accepted_timing_suspect", 0),
        "unknown_timing_frames": validation_counts.get("accepted_unknown", 0),
        "incomplete_frames": incomplete_frames,
        "other_excluded_frames": len(evidence) - len(clean)
        - validation_counts.get("accepted_timing_suspect", 0)
        - validation_counts.get("accepted_unknown", 0)
        - incomplete_frames,
        "validation_counts": validation_counts,
        "display_late_submissions": int(display.get("late_submissions", 0)),
        "display_irregular_intervals": int(display.get("irregular_intervals", 0)),
        "display_missed_period_candidates": int(display.get("missed_period_candidates", 0)),
    }

    current_all = _metrics(target, np.full(len(target), CURRENT_CORRECTION_MS))
    median_all = _metrics(target, np.full(len(target), session_median))
    graph_files = [
        TIMELINE_GRAPH,
        RESIDUAL_GRAPH,
        OFFSET_HISTOGRAM,
        FIXED_RESIDUAL_HISTOGRAM,
        PTS_RESIDUAL_HISTOGRAM,
    ]
    if save_svg:
        graph_files.extend(str(Path(path).with_suffix(".svg")) for path in graph_files.copy())
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "output_directory": str(output),
        "target": (
            "Host-anchored camera PTS minus newest journal-matched decoded QR marker; "
            "software reference, not physical exposure truth"
        ),
        "source_recording_directory": source.get("recording_directory"),
        "source_provenance_sha256": provenance,
        "data_quality": quality,
        "clean_offset_distribution": distribution,
        "chronological_split": {
            "train_fraction": TRAIN_FRACTION,
            "train_clean_frames": train_count,
            "holdout_clean_frames": len(clean) - train_count,
            "train_last_camera_frame": clean[train_count - 1].frame_number,
            "holdout_first_camera_frame": clean[train_count].frame_number,
        },
        "current_and_session_fixed_metrics_all_clean": {
            "fixed_109_ms": current_all,
            "session_full_median": median_all,
        },
        "strategy_order": list(definitions),
        "strategies": strategies,
        "verdict": {
            "status": "provisional_session_correction",
            "recommended_fixed_correction_ms": session_median,
            "replaces_current_correction": True,
            "current_correction_ms": CURRENT_CORRECTION_MS,
            "current_minus_recommended_ms": current_difference,
            "best_observed_holdout_strategy": best_key,
            "dynamic_candidate_is_meaningful": meaningful_dynamic_gain,
            "operational_recommendation": (
                f"For this recording, the defensible low-complexity correction is the clean median "
                f"of {session_median:.3f} ms. If adopted for this session, it replaces the 109.000 ms "
                "subtraction; it is not added to it."
            ),
            "current_correction_assessment": (
                f"Relative to the decoded QR marker, 109.000 ms {direction} by "
                f"{abs(current_difference):.3f} ms at the clean-sample median."
            ),
            "dynamic_strategy_assessment": dynamic_assessment,
            "deployment_boundary": (
                "Do not enable a learned dynamic correction from this one recording. Confirm the same "
                "preselected strategy on a later independent recording and retain replacement/display timing exclusions."
            ),
        },
        "output_files": [VERDICT_JSON, VERDICT_MARKDOWN, PREDICTIONS_CSV, *graph_files],
        "graph_formats": ["png", *(("svg",) if save_svg else ())],
        "histogram_policy": {
            "renderer": "Matplotlib",
            "layout": "One strategy per histogram panel; no overlapping distributions",
            "bin_width": "Freedman-Diaconis, constrained to 8-18 bins per panel",
            "offset_range": "Each panel's full observed range plus 5% padding",
            "residual_range": "Each panel's full residual range, symmetric around zero plus 6% padding",
            "height": "Percentage of evaluated frames in each bin",
        },
    }

    _write_predictions(output / PREDICTIONS_CSV, clean, train_count, predictions)
    _write_timeline_graph(
        output / TIMELINE_GRAPH, evidence, session_median, save_svg=save_svg
    )
    _write_residual_graph(
        output / RESIDUAL_GRAPH,
        target,
        predictions,
        definitions,
        holdout_slice,
        best_key,
        save_svg=save_svg,
    )
    holdout_target = target[holdout_slice]
    _write_histogram(
        output / OFFSET_HISTOGRAM,
        "Distribution of clean QR-derived offsets",
        "Observed PTS minus newest valid displayed QR (ms)",
        [("Clean observed offsets", target, "#147a88")],
        references=(
            (CURRENT_CORRECTION_MS, "Current correction: 109.000 ms", "#b4465a"),
            (session_median, f"Clean median: {session_median:.3f} ms", "#8b7a2f"),
        ),
        save_svg=save_svg,
    )
    _write_histogram(
        output / FIXED_RESIDUAL_HISTOGRAM,
        "Holdout residual distribution — fixed corrections",
        "Observed offset minus predicted correction (ms)",
        [
            ("Current 109 ms", holdout_target - predictions["fixed_109_ms"][holdout_slice], "#b4465a"),
            ("Calibrated median", holdout_target - predictions["fixed_train_median"][holdout_slice], "#8b7a2f"),
            ("Calibrated mean", holdout_target - predictions["fixed_train_mean"][holdout_slice], "#6d7d8a"),
        ],
        symmetric_about_zero=True,
        save_svg=save_svg,
    )
    _write_histogram(
        output / PTS_RESIDUAL_HISTOGRAM,
        "Holdout residual distribution — PTS strategies",
        "Observed offset minus predicted correction (ms)",
        [
            ("PTS-step median", holdout_target - predictions["pts_step_median"][holdout_slice], "#3f8f68"),
            ("Six-step PTS", holdout_target - predictions["pts_history6_linear"][holdout_slice], "#147a88"),
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
