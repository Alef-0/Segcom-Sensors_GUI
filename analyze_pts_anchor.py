#!/usr/bin/env python3
"""Calibrate one fixed offset, then score it on independent stream recordings.

python3 analyze_pts_anchor.py calibrate LAB_ANALYSIS --output offset.json
python3 analyze_pts_anchor.py evaluate TEST_ANALYSIS --offset-file offset.json \
    --output-directory anchor_report

QR evidence determines the laboratory offset and scores held-out predictions.
The predictor itself reads only camera clocks, segment running time, and c0.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from calibration.final_analysis import Goals, _journal, _score, load_session, _stream_components


def epoch_key(row):
    return tuple(row.get(k) for k in ("stream_epoch", "mapping_revision", "segment_epoch"))


def predict_timestamps(camera_rows, epochs):
    """Use only clocks; preserve integer nanoseconds and reset at epoch changes.

    A clock pair is (pipeline_running_time_observed_ns,
    pipeline_clock_mapping_monotonic_ns), not (frame PTS, arrival time).
    Pairing PTS with arrival would incorrectly absorb delivery latency.
    """
    anchors = {}
    for epoch in epochs:
        key = epoch_key(epoch)
        zero = epoch.get("pipeline_zero_monotonic_ns")
        if key in anchors and anchors[key] != zero:
            raise ValueError(f"Conflicting clock anchors for {key}")
        anchors[key] = zero
    result = []
    for row in camera_rows:
        key = epoch_key(row)
        zero = anchors.get(key)
        running = row.get("running_time_ns")
        xa = row.get("pipeline_running_time_observed_ns")
        xb = row.get("pipeline_clock_mapping_monotonic_ns")
        sampled_zero = xb - xa if xa is not None and xb is not None else None
        values = {
            "epoch_running_time": zero + running if zero is not None and running is not None else None,
        }
        values.update(
            sampled_zero_minus_epoch_ms=(sampled_zero - zero) / 1e6
            if sampled_zero is not None and zero is not None else None,
        )
        result.append(values)
    return result


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return dict(n=len(values), minimum=float(min(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(max(values)),
                maximum_absolute=float(max(abs(v) for v in values)))


def score_predictions(frames, predictions, method="estimated_monotonic_ns"):
    eligible = [(f, p[method]) for f, p in zip(frames, predictions)
                if f["targets"]["newest_generation"] is not None
                and p[method] is not None]
    metrics = {}
    if eligible:
        bounds = np.asarray([f["targets"]["newest_generation"] for f, _ in eligible])
        corrections = np.asarray([(f["media_reference_monotonic_ns"] - t) / 1e6
                                  for f, t in eligible])
        metrics = _score(bounds, corrections, Goals())
    complete = len(eligible) == len(frames) and bool(frames)
    return dict(metrics, every_saved_frame_scored=complete,
                every_saved_frame_meets_goals=complete and metrics.get("conditional_goal_passed", False),
                camera_frames=len(frames), scored_frames=len(eligible),
                no_qr_interval=sum(f["targets"]["newest_generation"] is None for f in frames),
                missing_prediction=sum(p[method] is None for p in predictions),
                unscored_frames=len(frames) - len(eligible))


def camera_predictions(recording, offset_ns):
    """No display journal, QR evidence, arrival fit, or saved correction input."""
    camera_path = next(recording / name for name in
                       ("camera_timestamps.jsonl", "camera_timestamps.json")
                       if (recording / name).is_file())
    rows = _journal(camera_path)
    epochs = json.loads((recording / "camera_timing_session.json").read_text())["epochs"]
    predictions = predict_timestamps(rows, epochs)
    for prediction in predictions:
        media = prediction["epoch_running_time"]
        prediction["estimated_monotonic_ns"] = media - offset_ns if media is not None else None
    return rows, predictions


def aligned_predictions(session, offset_ns):
    rows, predictions = camera_predictions(session.recording, offset_ns)
    if [r.get("frame") or r.get("camera_frame") for r in rows] != [f["filename"] for f in session.frames]:
        raise ValueError("Camera journal and evidence rows do not align")
    return rows, predictions


def calibrate(directories, minimum_stream_age_seconds=30.0):
    """Median interval midpoints per independent stream, then equal-weight median.

    The age cutoff is declared before looking at QR results. It is a test
    condition, not a claim that 30 seconds establishes camera stabilization.
    """
    if not math.isfinite(minimum_stream_age_seconds) or minimum_stream_age_seconds < 0:
        raise ValueError("Minimum stream age must be finite and nonnegative")
    sessions = [load_session(path) for path in directories]
    if not sessions or any(not s.identity_complete or not s.stream_keys
                           or any(not key.startswith("pipeline-base:") for key in s.stream_keys)
                           for s in sessions):
        raise ValueError("Calibration requires recorded pipeline-base stream identities")
    hashes = [s.audit["camera_journal_sha256"] for s in sessions]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate calibration camera journals")
    groups = []
    for group in _stream_components(sessions):
        midpoints, recordings = [], []
        for session in group:
            cameras, predictions = aligned_predictions(session, 0)
            values = []
            for frame, camera, prediction in zip(session.frames, cameras, predictions):
                age = camera.get("pipeline_running_time_observed_ns")
                bounds = frame["targets"]["newest_generation"]
                media = prediction["epoch_running_time"]
                if (age is None or age < minimum_stream_age_seconds * 1e9
                        or bounds is None or media is None):
                    continue
                correction = media - frame["media_reference_monotonic_ns"]
                values.append(correction + round(sum(bounds) * 0.5 * 1e6))
            if not values:
                raise ValueError(f"No usable post-cutoff anchored QR evidence: {session.name}")
            midpoints.extend(values)
            recordings.append(dict(name=session.name, usable_frames=len(values),
                                   midpoint_offset_ms=distribution(v / 1e6 for v in values)))
        groups.append(dict(stream_keys=sorted(set().union(*(s.stream_keys for s in group))),
                           offset_ns=round(float(np.median(midpoints))), recordings=recordings))
    return dict(schema_version=1, kind="frozen_pts_anchor_offset",
                created_at=datetime.now().astimezone().isoformat(),
                formula="pipeline_zero_monotonic_ns + running_time_ns - offset_ns",
                offset_ns=round(float(np.median([g["offset_ns"] for g in groups]))),
                estimator="median_of_independent_stream_medians_of_interval_midpoints",
                minimum_stream_age_seconds=minimum_stream_age_seconds,
                calibration_stream_keys=sorted(set().union(*(s.stream_keys for s in sessions))),
                calibration_camera_hashes=hashes,
                source_provenance_sha256={k: v for s in sessions for k, v in s.provenance.items()},
                groups=groups, deployment_validated=False, physical_accuracy_established=False)


def read_offset(path):
    artifact = json.loads(Path(path).read_text())
    if (artifact.get("schema_version") != 1 or artifact.get("kind") != "frozen_pts_anchor_offset"
            or type(artifact.get("offset_ns")) is not int
            or not artifact.get("calibration_stream_keys") or not artifact.get("calibration_camera_hashes")):
        raise ValueError("Expected a frozen offset artifact produced by the calibrate command")
    return artifact


def evaluate(directories, artifact):
    results, exported = [], []
    for directory in directories:
        session = load_session(directory)
        if (not session.identity_complete or not session.stream_keys
                or any(not key.startswith("pipeline-base:") for key in session.stream_keys)):
            raise ValueError(f"Cannot establish independent stream identity: {session.name}")
        if (session.stream_keys & set(artifact["calibration_stream_keys"])
                or session.audit["camera_journal_sha256"] in artifact["calibration_camera_hashes"]):
            raise ValueError(f"Calibration and evaluation share a stream or camera journal: {session.name}")
        cameras, predictions = aligned_predictions(session, artifact["offset_ns"])
        scores = score_predictions(session.frames, predictions)
        for frame, camera, prediction in zip(session.frames, cameras, predictions):
            stamp = prediction["estimated_monotonic_ns"]
            bounds = frame["targets"]["newest_generation"]
            start = end = residual = None
            if bounds is not None:
                reference = frame["media_reference_monotonic_ns"]
                start, end = reference - round(bounds[1] * 1e6), reference - round(bounds[0] * 1e6)
                if stamp is not None:
                    residual = (max(stamp - end, 0) - max(start - stamp, 0)) / 1e6
            exported.append(dict(recording=session.name, frame_number=frame["frame_number"],
                                 filename=frame["filename"], stream_key=frame["stream_key"],
                                 estimated_monotonic_ns=stamp, offset_ns=artifact["offset_ns"],
                                 stream_age_ns=camera.get("pipeline_running_time_observed_ns"),
                                 qr_start_monotonic_ns=start, qr_end_monotonic_ns=end,
                                 signed_interval_error_ms=residual,
                                 diagnostic_offset_interval_ms=frame["diagnostic_newest_decoded_interval_ms"],
                                 evidence_status=frame["evidence_assessment"]["status"],
                                 evidence_warnings=";".join(frame["evidence_assessment"].get("warnings", [])),
                                 exclusion_reasons=";".join(frame["exclusion_reasons"]),
                                 sampled_anchor_difference_ms=prediction["sampled_zero_minus_epoch_ms"]))
        results.append(dict(name=session.name, scores=scores, stream_keys=sorted(session.stream_keys),
                            audit=session.audit, source_provenance_sha256=session.provenance,
                            clock_mapping_difference_ms=distribution(
                                p["sampled_zero_minus_epoch_ms"] for p in predictions)))
    return dict(offset=artifact, sessions=results, physical_accuracy_established=False,
                evaluation_qr_used_for_prediction=False), exported


def write_report(report, rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report["output_directory"] = str(output.resolve())
    report["implementation_sha256"] = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).parent / "calibration/final_analysis.py",
                  Path(__file__).parent / "calibration/evidence.py")}
    (output / "pts_anchor_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "pts_anchor_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Frozen clock-anchor evaluation", "",
             f"Fixed laboratory offset: {report['offset']['offset_ns'] / 1e6:.6f} ms.",
             "Prediction: pipeline anchor + segment running time - fixed offset.",
             "Test QR evidence scores predictions; it never recalibrates the offset.", "",
             "| Recording | Scored / saved | Median ms | P95 ms | Maximum ms | Conditional goals |",
             "|---|---:|---:|---:|---:|---|"]
    for session in report["sessions"]:
        score = session["scores"]
        values = [f"{score[k]:.3f}" if k in score else "unavailable"
                  for k in ("median_absolute_ms", "p95_absolute_ms", "maximum_absolute_ms")]
        lines.append(f"| {session['name']} | {score['scored_frames']}/{score['camera_frames']} | "
                     + " | ".join(values) + f" | {score.get('conditional_goal_passed', False)} |")
    lines += ["", "Goals: maximum interval error <10 ms and median <5 ms; equality fails.",
              "Errors are distances outside conditional newest-readable QR intervals, not physical exposure error.",
              "Unscorable saved frames remain in the denominator. Unsaved-frame counts are in the JSON audit.",
              "Passing scored frames cannot verify missing/unscorable frames or establish deployment readiness."]
    (output / "pts_anchor_analysis.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("calibrate", help="Use laboratory recordings only; save a candidate c0")
    fit.add_argument("directories", nargs="+", type=Path)
    fit.add_argument("--minimum-stream-age-seconds", type=float, default=30.0)
    fit.add_argument("--output", required=True, type=Path)
    test = commands.add_parser("evaluate", help="Apply saved c0 to independent stream recordings")
    test.add_argument("directories", nargs="+", type=Path)
    test.add_argument("--offset-file", required=True, type=Path)
    test.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "calibrate":
            if args.output.exists():
                raise ValueError("Offset output already exists; use a new versioned filename")
            artifact = calibrate(args.directories, args.minimum_stream_age_seconds)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(json.dumps(artifact, indent=2) + "\n")
            print(f"Candidate offset: {artifact['offset_ns'] / 1e6:.6f} ms; saved {args.output}")
        else:
            artifact = read_offset(args.offset_file)
            report, rows = evaluate(args.directories, artifact)
            report["offset_file_sha256"] = hashlib.sha256(args.offset_file.read_bytes()).hexdigest()
            write_report(report, rows, args.output_directory)
            print(f"Evaluation saved in {args.output_directory}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
