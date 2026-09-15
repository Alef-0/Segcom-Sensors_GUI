#!/usr/bin/env python3
"""Evaluate calibration strategies unchanged across complete recordings."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np

from calibration.quantitative_analysis import (
    ANALYSIS_JSON,
    CURRENT_CORRECTION_MS,
    DYNAMIC_MINIMUM_GAIN_MS,
    DYNAMIC_P95_TOLERANCE_MS,
    FIXED_REPEATABILITY_LIMIT_MS,
    FRAMES_CSV,
    MINIMUM_CLEAN_FRAMES,
    _fit_history_model,
    _fit_interval_constant,
    _fit_step_medians,
    _interval_arrays,
    _load_saved_analysis,
    _metrics,
    matplotlib_environment,
    _predict_cadence,
    _predict_history,
    _prepare_evidence,
    _select_cadence_depth,
    _select_history_length,
    _step_bucket,
    _plotting,
)

REPORT_JSON = "calibration_cross_recording.json"
REPORT_MARKDOWN = "calibration_cross_recording.md"
RESULTS_CSV = "calibration_cross_recording_results.csv"
COMPARISON_GRAPH = "calibration_cross_recording_comparison.png"


def discover_analysis_directories(parent: str | Path) -> list[Path]:
    """Find usable sibling analyses while avoiding duplicate source recordings."""
    parent = Path(parent).expanduser().resolve()
    selected: dict[str, Path] = {}
    for directory in sorted(path for path in parent.iterdir() if path.is_dir()):
        if not (directory / ANALYSIS_JSON).is_file() or not (directory / FRAMES_CSV).is_file():
            continue
        try:
            source = json.loads((directory / ANALYSIS_JSON).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        identity = str(source.get("recording_directory") or directory)
        selected.setdefault(identity, directory)
    return list(selected.values())


def _stream_identity(source_recording: str | None) -> dict:
    if not source_recording:
        return {"status": "unknown"}
    path = Path(source_recording).expanduser() / "camera_timing_session.json"
    if not path.is_file():
        return {"status": "unknown", "session_file": str(path)}
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unknown", "session_file": str(path)}
    epoch = session.get("stream_epoch_at_start")
    matching = next(
        (
            item for item in session.get("epochs", [])
            if item.get("stream_epoch") == epoch
        ),
        {},
    )
    pipeline_zero = matching.get("pipeline_zero_monotonic_ns")
    signature = None
    if epoch is not None and pipeline_zero is not None:
        signature = f"{pipeline_zero}:{epoch}"
    return {
        "status": "identified" if signature else "unknown",
        "signature": signature,
        "stream_epoch": epoch,
        "pipeline_zero_monotonic_ns": pipeline_zero,
        "recording_started_at": session.get("recording_started_at"),
        "session_file": str(path),
    }


def _load_session(directory: Path) -> dict:
    source, frames, provenance = _load_saved_analysis(directory)
    grid = source.get("grid") if isinstance(source.get("grid"), dict) else {}
    grid_qrs = grid.get("qr_count")
    evidence = _prepare_evidence(
        frames,
        default_grid_qrs=int(grid_qrs) if grid_qrs is not None else None,
    )
    usable = [row for row in evidence if row.usable]
    if len(usable) < MINIMUM_CLEAN_FRAMES:
        raise ValueError(
            f"{directory} has {len(usable)} usable frames; "
            f"at least {MINIMUM_CLEAN_FRAMES} are required"
        )
    source_recording = source.get("recording_directory")
    return {
        "name": directory.name,
        "directory": directory,
        "source_recording_directory": source_recording,
        "stream_identity": _stream_identity(source_recording),
        "provenance": provenance,
        "usable": usable,
        "fixed_candidate_ms": _fit_interval_constant(usable, squared=False),
    }


def _fit_frozen_models(source_rows: list) -> dict:
    fixed = _fit_interval_constant(source_rows, squared=False)
    step = _fit_step_medians(source_rows, fixed)
    cadence, cadence_trials = _select_cadence_depth(source_rows)
    selected_history, history_trials = _select_history_length(source_rows)
    history6 = _fit_history_model(source_rows, history_length=6)
    return {
        "fixed_source_median": {"correction_ms": fixed},
        "pts_step_median": {"fallback_ms": fixed, "bucket_corrections_ms": step},
        "pts_cadence_state": {"model": cadence, "selection_trials": cadence_trials},
        "pts_history_selected_linear": {
            "model": selected_history,
            "selection_trials": history_trials,
        },
        "pts_history6_linear": {"model": history6},
    }


def _predict_frozen(models: dict, target_rows: list) -> dict[str, np.ndarray]:
    count = len(target_rows)
    step = models["pts_step_median"]
    return {
        "current_fixed": np.full(count, CURRENT_CORRECTION_MS),
        "fixed_source_median": np.full(
            count, models["fixed_source_median"]["correction_ms"]
        ),
        "pts_step_median": np.asarray([
            step["bucket_corrections_ms"].get(
                _step_bucket(row.pts_step_ms), step["fallback_ms"]
            )
            for row in target_rows
        ]),
        "pts_cadence_state": _predict_cadence(
            models["pts_cadence_state"]["model"], target_rows
        ),
        "pts_history_selected_linear": _predict_history(
            models["pts_history_selected_linear"]["model"], target_rows
        ),
        "pts_history6_linear": _predict_history(
            models["pts_history6_linear"]["model"], target_rows
        ),
    }


def _strategy_label(key: str) -> str:
    return {
        "current_fixed": f"Current fixed {CURRENT_CORRECTION_MS:.3f} ms",
        "fixed_source_median": "Source-recording fixed median",
        "pts_step_median": "PTS-step interval median",
        "pts_cadence_state": "PTS cadence-state median",
        "pts_history_selected_linear": "Selected-length regularized history",
        "pts_history6_linear": "Regularized six-step history",
    }[key]


def _write_graph(path: Path, summary: dict) -> None:
    plt, _ = _plotting()
    rows = summary["strategy_summary"]
    positions = np.arange(len(rows))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5.5), layout="constrained")
    axis.bar(
        positions - width / 2,
        [row["weighted_mae_ms"] for row in rows],
        width,
        label="Weighted cross-recording MAE",
        color="#147a88",
    )
    axis.bar(
        positions + width / 2,
        [row["worst_pair_p95_ms"] for row in rows],
        width,
        label="Worst source→target P95",
        color="#d38b3d",
    )
    axis.set_xticks(positions, [row["short_label"] for row in rows], rotation=20, ha="right")
    axis.set(
        title="Frozen-model cross-recording performance",
        ylabel="Distance outside presentation interval (ms)",
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=9)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown(report: dict) -> str:
    criteria = report["acceptance_criteria"]
    lines = [
        "# Cross-recording calibration validation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        f"- Fixed-candidate range: {criteria['fixed_candidate_range_ms']:.3f} ms "
        f"(limit {criteria['fixed_repeatability_limit_ms']:.3f} ms)",
        f"- Distinct identified camera stream sessions: {criteria['distinct_stream_sessions']}",
        f"- Independently restarted-session requirement: "
        f"{'passed' if criteria['independent_sessions_passed'] else 'not passed'}",
        "",
        "## Sessions",
        "",
        "| Analysis | Usable frames | Fixed candidate | Camera stream signature |",
        "|---|---:|---:|---|",
    ]
    for session in report["sessions"]:
        signature = session["stream_identity"].get("signature") or "unknown"
        lines.append(
            f"| {session['name']} | {session['usable_frames']} | "
            f"{session['fixed_candidate_ms']:.3f} ms | `{signature}` |"
        )
    lines.extend([
        "",
        "## Frozen model parameters",
        "",
        "| Source analysis | Cadence depth | Selected history | Six-step slopes |",
        "|---|---:|---:|---|",
    ])
    for model in report["frozen_model_parameters"]:
        slopes = ", ".join(f"{value:.3f}" for value in model["six_step_slopes"])
        lines.append(
            f"| {model['source']} | {model['cadence_depth']} | "
            f"{model['selected_history_length']} | `{slopes}` |"
        )
    ranges = ", ".join(
        f"lag {index}: {value:.3f}"
        for index, value in enumerate(report["six_step_slope_ranges"])
    )
    lines.extend([
        "",
        f"Six-step coefficient ranges across fitted source recordings: {ranges}.",
        "",
        "## Frozen strategy summary",
        "",
        "Each source recording fits a model once. That model is evaluated unchanged on every other recording.",
        "",
        "| Strategy | Weighted MAE | Worst P95 | Worst MAE change vs source-fixed | Acceptance |",
        "|---|---:|---:|---:|---|",
    ])
    for strategy in report["strategy_summary"]:
        lines.append(
            f"| {strategy['label']} | {strategy['weighted_mae_ms']:.3f} | "
            f"{strategy['worst_pair_p95_ms']:.3f} | "
            f"{strategy['worst_pair_mae_change_vs_fixed_ms']:+.3f} | "
            f"{'passed' if strategy['acceptance_passed'] else 'not passed'} |"
        )
    lines.extend([
        "",
        "Passing model accuracy is not sufficient for deployment unless the recordings contain at "
        "least two independently identified camera stream sessions.",
        "",
        "## Files",
        "",
        f"- `{RESULTS_CSV}`: every frozen source-to-target strategy score",
        f"- `{COMPARISON_GRAPH}`: aggregate MAE and worst P95 comparison",
        f"- `{REPORT_JSON}`: machine-readable report and acceptance checks",
        "",
    ])
    return "\n".join(lines)


def analyze_directories(
    analysis_directories: Iterable[str | Path], output_directory: str | Path
) -> dict:
    directories = [Path(path).expanduser().resolve() for path in analysis_directories]
    if len(directories) < 2:
        raise ValueError("Cross-recording analysis requires at least two analysis directories")
    sessions = [_load_session(directory) for directory in directories]
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    pair_results = []
    frozen_parameters = []
    for source in sessions:
        models = _fit_frozen_models(source["usable"])
        selected_history = models["pts_history_selected_linear"]["model"]
        history6 = models["pts_history6_linear"]["model"]
        frozen_parameters.append({
            "source": source["name"],
            "fixed_correction_ms": models["fixed_source_median"]["correction_ms"],
            "step_corrections_ms": {
                f"{bucket:g}": correction
                for bucket, correction in sorted(
                    models["pts_step_median"]["bucket_corrections_ms"].items()
                )
            },
            "cadence_depth": models["pts_cadence_state"]["model"]["depth"],
            "selected_history_length": selected_history["history_length"],
            "selected_history_slopes": selected_history["slopes"],
            "six_step_slopes": history6["slopes"],
            "history_ridge": history6["ridge"],
        })
        for target in sessions:
            if target is source:
                continue
            target_values, lower, upper = _interval_arrays(target["usable"])
            predictions = _predict_frozen(models, target["usable"])
            for key, values in predictions.items():
                metrics = _metrics(target_values, values, lower, upper)
                pair_results.append({
                    "source": source["name"],
                    "target": target["name"],
                    "strategy": key,
                    "label": _strategy_label(key),
                    **metrics,
                })

    keys = list(dict.fromkeys(row["strategy"] for row in pair_results))
    summary = []
    for key in keys:
        rows = [row for row in pair_results if row["strategy"] == key]
        total = sum(row["n"] for row in rows)
        weighted_mae = sum(row["mae_ms"] * row["n"] for row in rows) / total
        changes = []
        p95_changes = []
        for row in rows:
            fixed = next(
                candidate for candidate in pair_results
                if candidate["source"] == row["source"]
                and candidate["target"] == row["target"]
                and candidate["strategy"] == "fixed_source_median"
            )
            changes.append(row["mae_ms"] - fixed["mae_ms"])
            p95_changes.append(row["p95_absolute_ms"] - fixed["p95_absolute_ms"])
        dynamic = key not in ("current_fixed", "fixed_source_median")
        mean_gain = -sum(change * row["n"] for change, row in zip(changes, rows)) / total
        acceptance = (
            dynamic
            and mean_gain >= DYNAMIC_MINIMUM_GAIN_MS
            and max(p95_changes) <= DYNAMIC_P95_TOLERANCE_MS
            and max(changes) <= DYNAMIC_P95_TOLERANCE_MS
        )
        summary.append({
            "strategy": key,
            "label": _strategy_label(key),
            "short_label": {
                "current_fixed": "Current",
                "fixed_source_median": "Source fixed",
                "pts_step_median": "PTS step",
                "pts_cadence_state": "Cadence",
                "pts_history_selected_linear": "Selected history",
                "pts_history6_linear": "Six-step",
            }[key],
            "source_target_pairs": len(rows),
            "weighted_mae_ms": weighted_mae,
            "weighted_mae_gain_vs_fixed_ms": mean_gain,
            "worst_pair_p95_ms": max(row["p95_absolute_ms"] for row in rows),
            "worst_pair_p95_change_vs_fixed_ms": max(p95_changes),
            "worst_pair_mae_change_vs_fixed_ms": max(changes),
            "acceptance_passed": acceptance,
        })

    fixed_candidates = [session["fixed_candidate_ms"] for session in sessions]
    six_step_slope_ranges = [
        max(model["six_step_slopes"][index] for model in frozen_parameters)
        - min(model["six_step_slopes"][index] for model in frozen_parameters)
        for index in range(6)
    ]
    signatures = {
        session["stream_identity"].get("signature")
        for session in sessions
        if session["stream_identity"].get("signature")
    }
    independent_sessions = len(signatures) >= 2
    fixed_range = max(fixed_candidates) - min(fixed_candidates)
    passing = [row for row in summary if row["acceptance_passed"]]
    best = min(passing, key=lambda row: row["weighted_mae_ms"]) if passing else None
    deployment_ready = (
        independent_sessions
        and fixed_range <= FIXED_REPEATABILITY_LIMIT_MS
        and best is not None
    )
    verdict = (
        f"{best['label']} passes the frozen-model accuracy checks across these recordings. "
        if best else
        "No dynamic strategy passes every frozen-model accuracy check across these recordings. "
    )
    if not independent_sessions:
        verdict += (
            "Deployment remains blocked because fewer than two distinct restarted camera stream "
            "sessions were identified."
        )
    elif deployment_ready:
        verdict += "The configured acceptance criteria are satisfied for a provisional live trial."
    else:
        verdict += "The configured acceptance criteria are not yet satisfied for a live trial."

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "output_directory": str(output),
        "sessions": [{
            "name": session["name"],
            "analysis_directory": str(session["directory"]),
            "source_recording_directory": session["source_recording_directory"],
            "usable_frames": len(session["usable"]),
            "fixed_candidate_ms": session["fixed_candidate_ms"],
            "stream_identity": session["stream_identity"],
            "source_provenance_sha256": session["provenance"],
        } for session in sessions],
        "pair_results": pair_results,
        "frozen_model_parameters": frozen_parameters,
        "six_step_slope_ranges": six_step_slope_ranges,
        "strategy_summary": summary,
        "acceptance_criteria": {
            "fixed_repeatability_limit_ms": FIXED_REPEATABILITY_LIMIT_MS,
            "fixed_candidate_range_ms": fixed_range,
            "fixed_repeatability_passed": fixed_range <= FIXED_REPEATABILITY_LIMIT_MS,
            "dynamic_minimum_mae_gain_ms": DYNAMIC_MINIMUM_GAIN_MS,
            "dynamic_p95_tolerance_ms": DYNAMIC_P95_TOLERANCE_MS,
            "distinct_stream_sessions": len(signatures),
            "independent_sessions_passed": independent_sessions,
            "best_passing_dynamic_strategy": best["strategy"] if best else None,
            "deployment_ready": deployment_ready,
        },
        "verdict": verdict,
        "output_files": [REPORT_JSON, REPORT_MARKDOWN, RESULTS_CSV, COMPARISON_GRAPH],
    }

    fields = [
        "source", "target", "strategy", "label", "n", "mae_ms",
        "median_absolute_ms", "p95_absolute_ms", "rmse_ms", "bias_ms",
        "within_interval_pct", "within_5ms_pct", "within_10ms_pct",
    ]
    with (output / RESULTS_CSV).open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_results)
    _write_graph(output / COMPARISON_GRAPH, report)
    (output / REPORT_MARKDOWN).write_text(_markdown(report), encoding="utf-8")
    (output / REPORT_JSON).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_directories", nargs="+", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    arguments = parser.parse_args()
    plotting_environment = matplotlib_environment()
    current_system_only = os.environ.get("PYTHONNOUSERSITE") == "1"
    selected_system_only = plotting_environment.get("PYTHONNOUSERSITE") == "1"
    if selected_system_only != current_system_only:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "calibration.cross_recording_analysis",
                *sys.argv[1:],
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=plotting_environment,
        )
        raise SystemExit(completed.returncode)
    os.environ.setdefault("MPLCONFIGDIR", plotting_environment["MPLCONFIGDIR"])
    report = analyze_directories(
        arguments.analysis_directories, arguments.output_directory
    )
    print(report["verdict"])
    print(f"Saved cross-recording report in {report['output_directory']}")


if __name__ == "__main__":
    main()
