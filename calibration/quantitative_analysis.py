"""Create a descriptive report from saved calibration QR results."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from sensors.camera.timing_defaults import DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS

CURRENT_CORRECTION_MS = DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS
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
POOLED_REPORT_JSON = "quantitative_analysis.json"
POOLED_REPORT_MARKDOWN = "quantitative_analysis.md"
POOLED_PREDICTIONS_CSV = "quantitative_predictions.csv"
POOLED_RESIDUAL_GRAPH = "quantitative_residual_cdf.png"
POOLED_FIXED_RESIDUAL_GRAPH = "quantitative_current_correction_residual_cdf.png"
POOLED_OFFSET_GRAPH = "quantitative_offsets_by_recording.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _load_saved_analysis(output: Path) -> tuple[dict, list[dict], dict]:
    path = output / ANALYSIS_JSON
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("frames"), list):
        raise ValueError(f"{ANALYSIS_JSON} must contain a frames list")
    frames = source["frames"]
    csv_path = output / FRAMES_CSV
    provenance = {path.name: _sha256(path)}
    if csv_path.is_file():
        provenance[csv_path.name] = _sha256(csv_path)
    return source, frames, provenance


def _number(value) -> float | None:
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _metrics(values: np.ndarray) -> dict:
    return {"n": len(values), "mae_ms": float(np.mean(np.abs(values))),
            "median_absolute_ms": float(np.median(np.abs(values))),
            "p95_absolute_ms": float(np.percentile(np.abs(values), 95)),
            "maximum_absolute_ms": float(np.max(np.abs(values)))}


def _plot(path: Path, title: str, x_label: str, series: list[tuple[str, np.ndarray]], *, save_svg=False, histogram=False, legend_below=False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axis = plt.subplots(figsize=(10, 6.5 if legend_below else 5))
    cdf_values = []
    for label, values in series:
        if not len(values):
            continue
        if histogram:
            axis.hist(values, bins="auto", alpha=.55, label=label)
        else:
            ordered = np.sort(values)
            cdf_values.append(ordered)
            axis.plot(ordered, 100 * np.arange(1, len(ordered) + 1) / len(ordered), label=label)
    if cdf_values:
        p95 = float(np.percentile(np.concatenate(cdf_values), 95))
        axis.set_xlim(0, max(p95, np.finfo(float).eps))
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 10))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.axvline(p95, color="#555555", linestyle=":", linewidth=1.2,
                     label=f"P95 cutoff · {p95:.2f} ms")
        axis.axhline(95, color="#777777", linestyle=":", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Frames (%)" if not histogram else "Frames")
    axis.grid(True, alpha=.25)
    if series:
        if legend_below:
            handles, labels = axis.get_legend_handles_labels()
            figure.subplots_adjust(left=.10, right=.98, top=.90, bottom=.25)
            figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015),
                          ncol=3, fontsize=8)
        else:
            axis.legend()
            figure.tight_layout()
    else:
        figure.tight_layout()
    figure.savefig(path, dpi=150)
    if save_svg:
        figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def _write_timeline_graph(path: Path, frames: list[dict], median: float, *, save_svg=False) -> None:
    values = np.asarray([_number(row.get("pts_minus_latest_qr_ms")) for row in frames if _number(row.get("pts_minus_latest_qr_ms")) is not None])
    _plot(path, "QR matched camera timing offsets", "Observed offset (ms)", [(f"Frame offsets · median {median:.3f} ms", values)], save_svg=save_svg, histogram=True)


def _write_residual_graph(path: Path, values: np.ndarray, median: float, *, save_svg=False) -> None:
    _plot(path, "Residual CDF through P95", "Absolute residual (ms)", [("Session median", np.abs(values - median))], save_svg=save_svg)


def _write_histogram(path: Path, title: str, label: str, values: np.ndarray, *, save_svg=False) -> None:
    _plot(path, title, label, [(title, values)], save_svg=save_svg, histogram=True)


def analyze_output_directory(output_directory: str | Path, *, save_svg: bool = False) -> dict:
    """Summarize saved clean offsets; this report is descriptive, not deployment validation."""
    output = Path(output_directory).expanduser().resolve()
    if not output.is_dir():
        raise ValueError(f"Analysis output directory does not exist: {output}")
    source, frames, provenance = _load_saved_analysis(output)
    by_validation = Counter()
    for row in frames:
        by_validation[row.get("validation", "unknown")] += 1
    clean = [(int(row.get("frame_number", index + 1)), _number(row.get("pts_minus_latest_qr_ms")))
             for index, row in enumerate(frames)
             if row.get("validation") == "accepted_clean" and _number(row.get("pts_minus_latest_qr_ms")) is not None]
    offsets = np.asarray([value for _, value in clean], dtype=float)
    if not len(offsets):
        raise ValueError("No clean QR matched offsets are available for a descriptive report")
    median = float(np.median(offsets))
    current_residual = offsets - CURRENT_CORRECTION_MS
    median_residual = offsets - median
    display = source.get("display") if isinstance(source.get("display"), dict) else {}
    quality = {
        "processed_frames": len(frames), "clean_frames": len(clean),
        "clean_pct": 100 * len(clean) / len(frames) if frames else 0.0,
        "timing_suspect_frames": by_validation["accepted_timing_suspect"],
        "unknown_timing_frames": by_validation["accepted_unknown"],
        "incomplete_frames": by_validation["skipped_incomplete"] + by_validation["skipped_no_readable_qr"],
        "other_excluded_frames": len(frames) - len(clean) - by_validation["accepted_timing_suspect"] - by_validation["accepted_unknown"],
        "validation_counts": dict(by_validation),
        "display_late_submissions": int(display.get("late_submissions", 0)),
        "display_irregular_intervals": int(display.get("irregular_intervals", 0)),
        "display_missed_period_candidates": int(display.get("missed_period_candidates", 0)),
    }
    graph_names = [TIMELINE_GRAPH, RESIDUAL_GRAPH, OFFSET_HISTOGRAM,
                   FIXED_RESIDUAL_HISTOGRAM, PTS_RESIDUAL_HISTOGRAM]
    extensions = ["png", "svg"] if save_svg else ["png"]
    output_files = [VERDICT_JSON, VERDICT_MARKDOWN, PREDICTIONS_CSV,
                    *[str(Path(name).with_suffix("." + ext)) for name in graph_names for ext in extensions]]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "output_directory": str(output),
        "status": "descriptive_only",
        "target": "Saved software presentation interval evidence; this report does not establish physical exposure time.",
        "source_recording_directory": source.get("recording_directory"),
        "source_provenance_sha256": provenance,
        "data_quality": quality,
        "clean_offset_distribution": {
            "n": len(offsets), "minimum_ms": float(np.min(offsets)),
            "median_ms": median, "mean_ms": float(np.mean(offsets)),
            "p95_ms": float(np.percentile(offsets, 95)), "maximum_ms": float(np.max(offsets)),
            "standard_deviation_ms": float(np.std(offsets)),
        },
        "current_and_session_fixed_metrics_all_clean": {
            "runtime_default_correction": _metrics(current_residual),
            "session_full_median": _metrics(median_residual),
        },
        "verdict": {
            "status": "provisional_session_correction",
            "recommended_fixed_correction_ms": median,
            "replaces_current_correction": True,
            "current_correction_ms": CURRENT_CORRECTION_MS,
            "current_minus_recommended_ms": CURRENT_CORRECTION_MS - median,
            "deployment_validated": False,
            "physical_accuracy_established": False,
            "operational_recommendation": f"Descriptive session median: {median:.3f} ms. Validate on independent streams before use.",
            "deployment_boundary": "One recording and software QR presentation returns do not establish independent validation or physical exposure accuracy.",
        },
        "output_files": output_files,
        "graph_formats": extensions,
    }
    _write_timeline_graph(output / TIMELINE_GRAPH, frames, median, save_svg=save_svg)
    _write_residual_graph(output / RESIDUAL_GRAPH, offsets, median, save_svg=save_svg)
    _write_histogram(output / OFFSET_HISTOGRAM, "Observed clean offsets", "PTS minus QR (ms)", offsets, save_svg=save_svg)
    _write_histogram(output / FIXED_RESIDUAL_HISTOGRAM, "Residual with current correction", "Residual (ms)", current_residual, save_svg=save_svg)
    _write_histogram(output / PTS_RESIDUAL_HISTOGRAM, "Residual with session median", "Residual (ms)", median_residual, save_svg=save_svg)
    with (output / PREDICTIONS_CSV).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_number", "observed_offset_ms", "current_residual_ms", "session_median_residual_ms"])
        writer.writeheader()
        writer.writerows({"frame_number": frame, "observed_offset_ms": value,
                          "current_residual_ms": value - CURRENT_CORRECTION_MS,
                          "session_median_residual_ms": value - median} for frame, value in clean)
    (output / VERDICT_MARKDOWN).write_text(
        "# Descriptive QR timing report\n\n"
        f"Clean frames: {len(clean)} / {len(frames)}.\n\n"
        f"Observed median offset: {median:.3f} ms.\n\n"
        "This report describes software timestamp intervals. It does not establish camera exposure timing, physical monitor scanout, independent-stream validation, or deployment readiness.\n",
        encoding="utf-8")
    (output / VERDICT_JSON).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _write_pooled_offset_graph(path: Path, recordings: list[dict], *, save_svg=False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 5))
    values = [recording["offsets"] for recording in recordings]
    axis.boxplot(values, labels=[recording["name"] for recording in recordings], showfliers=True)
    axis.axhline(CURRENT_CORRECTION_MS, color="#555555", linestyle="--", linewidth=1.2,
                 label=f"Current correction · {CURRENT_CORRECTION_MS:.1f} ms")
    axis.set_title("Clean QR matched offsets by recording")
    axis.set_xlabel("Recording")
    axis.set_ylabel("PTS minus QR (ms)")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(True, axis="y", alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    if save_svg:
        figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def analyze_output_directories(
    output_directories: list[str | Path],
    pooled_output_directory: str | Path,
    *,
    save_svg: bool = False,
) -> dict:
    """Compile descriptive reports from multiple saved recordings into one folder."""
    pooled_output = Path(pooled_output_directory).expanduser().resolve()
    recordings = []
    used_names = Counter()
    for raw_directory in output_directories:
        output = Path(raw_directory).expanduser().resolve()
        if output == pooled_output:
            raise ValueError("The pooled report directory cannot also be an input recording directory")
        source, frames, provenance = _load_saved_analysis(output)
        clean = [(int(row.get("frame_number", index + 1)), _number(row.get("pts_minus_latest_qr_ms")))
                 for index, row in enumerate(frames)
                 if row.get("validation") == "accepted_clean" and _number(row.get("pts_minus_latest_qr_ms")) is not None]
        if not clean:
            raise ValueError(f"No clean QR matched offsets are available in {output}")
        name = output.name
        used_names[name] += 1
        if used_names[name] > 1:
            name = f"{name} ({used_names[name]})"
        offsets = np.asarray([value for _, value in clean], dtype=float)
        median = float(np.median(offsets))
        recordings.append({
            "name": name,
            "directory": str(output),
            "source_recording_directory": source.get("recording_directory"),
            "source_provenance_sha256": provenance,
            "processed_frames": len(frames),
            "clean_frames": len(clean),
            "offsets": offsets,
            "frames": clean,
            "median_ms": median,
        })

    if not recordings:
        raise ValueError("Provide at least one recording analysis directory")
    pooled_offsets = np.concatenate([recording["offsets"] for recording in recordings])
    pooled_fixed_residuals = pooled_offsets - CURRENT_CORRECTION_MS
    pooled_session_residuals = np.concatenate([
        recording["offsets"] - recording["median_ms"] for recording in recordings
    ])
    processed_frames = sum(recording["processed_frames"] for recording in recordings)
    clean_frames = len(pooled_offsets)
    extensions = ["png", "svg"] if save_svg else ["png"]
    graph_names = [POOLED_RESIDUAL_GRAPH, POOLED_FIXED_RESIDUAL_GRAPH, POOLED_OFFSET_GRAPH]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "output_directory": str(pooled_output),
        "status": "descriptive_only",
        "analysis_basis": "Saved rows marked accepted_clean with a finite pts_minus_latest_qr_ms value.",
        "recording_count": len(recordings),
        "data_quality": {
            "processed_frames": processed_frames,
            "clean_frames": clean_frames,
            "clean_pct": 100 * clean_frames / processed_frames if processed_frames else 0.0,
        },
        "pooled_clean_offset_distribution": {
            "n": clean_frames,
            "minimum_ms": float(np.min(pooled_offsets)),
            "median_ms": float(np.median(pooled_offsets)),
            "mean_ms": float(np.mean(pooled_offsets)),
            "p95_ms": float(np.percentile(pooled_offsets, 95)),
            "maximum_ms": float(np.max(pooled_offsets)),
            "standard_deviation_ms": float(np.std(pooled_offsets)),
        },
        "pooled_residual_metrics_all_clean": {
            "runtime_default_correction": _metrics(pooled_fixed_residuals),
            "within_recording_median": _metrics(pooled_session_residuals),
        },
        "recordings": [{
            "name": recording["name"],
            "analysis_directory": recording["directory"],
            "source_recording_directory": recording["source_recording_directory"],
            "source_provenance_sha256": recording["source_provenance_sha256"],
            "processed_frames": recording["processed_frames"],
            "clean_frames": recording["clean_frames"],
            "clean_pct": 100 * recording["clean_frames"] / recording["processed_frames"] if recording["processed_frames"] else 0.0,
            "median_offset_ms": recording["median_ms"],
            "runtime_default_correction": _metrics(recording["offsets"] - CURRENT_CORRECTION_MS),
            "within_recording_median": _metrics(recording["offsets"] - recording["median_ms"]),
        } for recording in recordings],
        "verdict": {
            "status": "descriptive_only",
            "current_correction_ms": CURRENT_CORRECTION_MS,
            "deployment_validated": False,
            "physical_accuracy_established": False,
            "operational_recommendation": "Use this pooled report to compare recordings; it does not validate a correction for deployment.",
            "deployment_boundary": "The input recordings share a camera and software presentation evidence; pooled counts do not establish independent-stream validation or physical exposure accuracy.",
        },
        "output_files": [POOLED_REPORT_JSON, POOLED_REPORT_MARKDOWN, POOLED_PREDICTIONS_CSV,
                         *[str(Path(name).with_suffix("." + ext)) for name in graph_names for ext in extensions]],
        "graph_formats": extensions,
    }

    pooled_output.mkdir(parents=True, exist_ok=True)
    cdf_series = [(recording["name"], np.abs(recording["offsets"] - recording["median_ms"]))
                  for recording in recordings]
    cdf_series.append(("All recordings", np.abs(pooled_session_residuals)))
    _plot(pooled_output / POOLED_RESIDUAL_GRAPH,
          "Within-recording absolute residuals (P95 display cutoff)",
          "Absolute residual (ms)", cdf_series, save_svg=save_svg, legend_below=True)
    fixed_cdf_series = [(recording["name"], np.abs(recording["offsets"] - CURRENT_CORRECTION_MS))
                        for recording in recordings]
    fixed_cdf_series.append(("All recordings", np.abs(pooled_fixed_residuals)))
    _plot(pooled_output / POOLED_FIXED_RESIDUAL_GRAPH,
          "Absolute residuals from current correction (P95 display cutoff)",
          "Absolute residual (ms)", fixed_cdf_series, save_svg=save_svg, legend_below=True)
    _write_pooled_offset_graph(pooled_output / POOLED_OFFSET_GRAPH, recordings, save_svg=save_svg)

    with (pooled_output / POOLED_PREDICTIONS_CSV).open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["recording", "frame_number", "observed_offset_ms", "current_correction_residual_ms",
                      "within_recording_median_residual_ms"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for recording in recordings:
            for frame_number, offset in recording["frames"]:
                writer.writerow({
                    "recording": recording["name"],
                    "frame_number": frame_number,
                    "observed_offset_ms": offset,
                    "current_correction_residual_ms": offset - CURRENT_CORRECTION_MS,
                    "within_recording_median_residual_ms": offset - recording["median_ms"],
                })

    table = "\n".join(
        f"| {recording['name']} | {recording['clean_frames']} / {recording['processed_frames']} | "
        f"{recording['median_offset_ms']:.3f} | {recording['runtime_default_correction']['p95_absolute_ms']:.3f} | "
        f"{recording['runtime_default_correction']['maximum_absolute_ms']:.3f} |"
        for recording in report["recordings"]
    )
    (pooled_output / POOLED_REPORT_MARKDOWN).write_text(
        "# Pooled quantitative QR timing report\n\n"
        f"Recordings: {len(recordings)}. Clean frames: {clean_frames} / {processed_frames} "
        f"({report['data_quality']['clean_pct']:.2f}%).\n\n"
        f"Pooled observed median offset: {report['pooled_clean_offset_distribution']['median_ms']:.3f} ms. "
        f"Runtime default correction: {CURRENT_CORRECTION_MS:.3f} ms. Pooled residual P95: {report['pooled_residual_metrics_all_clean']['runtime_default_correction']['p95_absolute_ms']:.3f} ms; "
        f"maximum: {report['pooled_residual_metrics_all_clean']['runtime_default_correction']['maximum_absolute_ms']:.3f} ms.\n\n"
        "| Recording | Clean / processed | Median offset (ms) | Runtime default P95 absolute residual (ms) | Maximum absolute residual (ms) |\n"
        "|---|---:|---:|---:|---:|\n" + table + "\n\n"
        "This report pools saved software timestamp intervals from the listed recordings. It is descriptive and does not establish independent-stream validation, camera exposure timing, physical monitor scanout, or deployment readiness. Both residual CDFs end their x-axes at the corresponding pooled P95 and show frame percentages on the y-axes. The standard residual CDF is centered on each recording median; the separate residual CDF compares offsets with the runtime default correction.\n",
        encoding="utf-8",
    )
    (pooled_output / POOLED_REPORT_JSON).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_directories", type=Path, nargs="+")
    parser.add_argument("--output-directory", type=Path,
                        help="Pooled report folder (defaults to a quantitative_analysis folder beside the first input)")
    parser.add_argument("--svg", action="store_true", help="Also write SVG copies of the charts")
    args = parser.parse_args()
    if len(args.analysis_directories) == 1:
        if args.output_directory is not None:
            parser.error("--output-directory is only used when pooling multiple recordings")
        report = analyze_output_directory(args.analysis_directories[0], save_svg=args.svg)
        print(report["verdict"]["operational_recommendation"])
        print(f"Saved descriptive report in {report['output_directory']}")
        return

    per_recording = [analyze_output_directory(directory, save_svg=args.svg)
                     for directory in args.analysis_directories]
    pooled_directory = args.output_directory or (args.analysis_directories[0].expanduser().resolve().parent / "quantitative_analysis")
    pooled = analyze_output_directories(args.analysis_directories, pooled_directory, save_svg=args.svg)
    print(f"Updated {len(per_recording)} per-recording reports.")
    print(f"Saved pooled report in {pooled['output_directory']}")


if __name__ == "__main__":
    main()
