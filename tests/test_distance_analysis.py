"""Focused tests for interval-aware calibration distance analysis."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from calibration.distance_analysis import (
    DisplayEvent,
    OVERVIEW_GRAPH,
    _finite_int,
    _full_marker_interval,
    _payload,
    analyze_recording,
)


class DistanceAnalysisTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        recording = root / "sample"
        analysis = root / "sample_analysis"
        recording.mkdir()
        analysis.mkdir()

        base_ns = 10_000_000_000
        period_ns = 10_000_000
        display_frames = []
        for index in range(45):
            presentation_ns = base_ns + index * period_ns
            display_frames.append({
                "kind": "frame",
                "index": index,
                "cell": index % 4,
                "marker_ns": presentation_ns - 500_000,
                "predicted_flip_ns": presentation_ns - 500_000,
                "presentation_return_ns": presentation_ns,
                "presentation_event_kind": "test_flip_return",
                "physical_presentation_measured": False,
                "interval_ns": period_ns if index else None,
                "late_submit": False,
                "irregular_interval": False,
                "skipped_periods": 0,
            })
        display_rows = [{
            "kind": "session",
            "grid_qrs": 4,
            "visible_qrs": 3,
            "presentation_semantics": "synthetic software flip boundary",
        }, *display_frames]
        (recording / "display_timestamps.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in display_rows),
            encoding="utf-8",
        )

        newest_indices = (5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38)
        qr_gap_frames = {5, 9}
        camera_rows = []
        analysis_frames = []
        for frame_number, newest in enumerate(newest_indices, 1):
            observation_ns = (
                display_frames[newest]["presentation_return_ns"] + period_ns // 2
            )
            filename = f"images/camera_{frame_number:06d}.jpg"
            camera_rows.append({
                "frame": filename,
                "stream_epoch": 1,
                "pts_ns": observation_ns + 100_000_000,
                "received_monotonic_ns": observation_ns + 200_000_000,
            })
            values = [None] * 4
            indices = []
            if frame_number not in qr_gap_frames:
                indices = list(range(newest - 2, newest + 1))
                for index in indices:
                    values[index % 4] = _payload(display_frames[index]["marker_ns"])
            analysis_frames.append({
                "frame_number": frame_number,
                "filename": filename,
                "validation": (
                    "accepted_clean" if indices else "skipped_no_readable_qr"
                ),
                "timing_status": "Clean" if indices else None,
                "qr_values_ms": values,
                "display_indices": indices,
                "latest_display_index": newest if indices else None,
                "camera_reference_monotonic_ns": observation_ns + 100_000_000,
                "received_monotonic_ns": observation_ns + 200_000_000,
            })
        (recording / "camera_timestamps.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in camera_rows),
            encoding="utf-8",
        )
        (recording / "camera_timing_session.json").write_text(
            json.dumps({
                "epochs": [{
                    "stream_epoch": 1,
                    "pipeline_zero_monotonic_ns": 0,
                }],
            }),
            encoding="utf-8",
        )
        (analysis / "calibration_analysis.json").write_text(
            json.dumps({
                "processed": len(analysis_frames),
                "total": len(analysis_frames),
                "frames": analysis_frames,
            }),
            encoding="utf-8",
        )
        return recording, analysis

    def test_analysis_maps_markers_and_fills_qr_gaps_from_arrival_time(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording, analysis = self._fixture(root)
            output = root / "distance_output"

            def graph_file(_report, graph_output):
                (graph_output / OVERVIEW_GRAPH).write_bytes(b"PNG")

            with patch(
                "calibration.distance_analysis._write_overview_graph_compatibly",
                side_effect=graph_file,
            ):
                report = analyze_recording(
                    recording,
                    analysis_directory=analysis,
                    output_directory=output,
                    minimum_anchors=4,
                )

            self.assertEqual(report["counts"]["camera_frames"], 12)
            self.assertEqual(report["counts"]["marker_observations"], 30)
            self.assertEqual(
                [marker["display_index"] for marker in report["markers"][:3]],
                [3, 4, 5],
            )
            self.assertFalse(report["markers"][0]["is_latest_observed"])
            self.assertTrue(report["markers"][2]["is_latest_observed"])
            self.assertEqual(
                report["counts"]["all_markers_state_status"],
                {"compatible_global_state": 10, "no_mapped_markers": 2},
            )
            self.assertEqual(report["segments"][0]["status"], "chronologically_validated")
            model = report["segments"][0]["arrival_model"]
            self.assertAlmostEqual(model["delay_intercept_ms"], 200.0, places=6)
            self.assertAlmostEqual(model["delay_drift_ms_per_second"], 0.0, places=6)
            self.assertAlmostEqual(
                report["segments"][0]["pts_constant_baseline"]["correction_ms"],
                100.0,
                places=6,
            )

            gap = report["frames"][4]
            expected = 10_000_000_000 + 17 * 10_000_000 + 5_000_000
            self.assertEqual(gap["anchor_status"], "missing_qr_anchor")
            self.assertEqual(gap["estimation_kind"], "interpolated_qr_gap")
            self.assertEqual(gap["estimated_observation_monotonic_ns"], expected)

            first = report["frames"][0]
            self.assertEqual(
                first["anchor_interval_start_ns"],
                10_000_000_000 + 5 * 10_000_000,
            )
            self.assertTrue(first["inside_qr_interval"])
            self.assertTrue((output / "distance_analysis.json").is_file())
            self.assertTrue((output / "distance_frames.csv").is_file())
            self.assertTrue((output / "distance_markers.csv").is_file())
            self.assertTrue((output / OVERVIEW_GRAPH).is_file())
            self.assertIn(OVERVIEW_GRAPH, report["output_files"])
            self.assertEqual(report["graph_formats"], ["png"])

    def test_full_marker_intersection_reports_rolling_or_mixed_generations(self):
        events = [
            DisplayEvent(
                index=index,
                cell=index % 4,
                marker_ns=index * 10_000_000,
                predicted_flip_ns=index * 10_000_000,
                presentation_return_ns=index * 10_000_000 + 1_000_000,
                presentation_event_kind="test",
                physical_presentation_measured=False,
                timing_issues=(),
            )
            for index in range(12)
        ]

        lower, upper, status = _full_marker_interval([1, 6], events, 3)

        self.assertGreaterEqual(lower, upper)
        self.assertEqual(status, "mixed_or_rolling_generations")

    def test_integer_timestamp_conversion_preserves_nanosecond_precision(self):
        timestamp = 1_789_406_435_539_860_935

        self.assertEqual(_finite_int(timestamp), timestamp)
        self.assertEqual(_finite_int(str(timestamp)), timestamp)


if __name__ == "__main__":
    unittest.main()
