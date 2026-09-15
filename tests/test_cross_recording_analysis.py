"""Headless tests for multi-recording calibration validation."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from calibration.cross_recording_analysis import (
    analyze_directories,
    discover_analysis_directories,
)


class CrossRecordingAnalysisTests(unittest.TestCase):
    def make_analysis(
        self,
        root: Path,
        name: str,
        *,
        pipeline_zero: int,
        offset_shift_ms: float = 0.0,
    ) -> Path:
        recording = root / name
        recording.mkdir()
        session = {
            "recording_started_at": f"2026-01-01T00:00:0{name[-1]}+00:00",
            "stream_epoch_at_start": 1,
            "epochs": [{
                "stream_epoch": 1,
                "pipeline_zero_monotonic_ns": pipeline_zero,
            }],
        }
        (recording / "camera_timing_session.json").write_text(
            json.dumps(session), encoding="utf-8"
        )

        analysis = root / f"{name}_analysis"
        analysis.mkdir()
        frames = []
        pts_ns = 0
        for index in range(72):
            step_ms = 20.0 if index % 3 == 1 else 40.0
            pts_ns += int(step_ms * 1e6)
            midpoint = (77.0 if step_ms == 20.0 else 87.0) + offset_shift_ms
            frames.append({
                "frame_number": index + 1,
                "filename": f"camera_{index + 1:06d}.jpg",
                "validation": "accepted_clean",
                "timing_status": "Clean",
                "pts_ns": pts_ns,
                "pts_minus_latest_qr_ms": midpoint,
                "offset_interval_lower_ms": midpoint - 4.0,
                "offset_interval_upper_ms": midpoint + 4.0,
                "matched_readable_qrs": 10 if index % 5 else 8,
                "grid_qrs": 10,
                "latest_cell": index % 10,
                "latest_cell_name": f"Cell {index % 10}",
            })
        report = {
            "recording_directory": str(recording),
            "grid": {"qr_count": 10},
            "frames": frames,
        }
        (analysis / "calibration_analysis.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        with (analysis / "calibration_frames.csv").open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=frames[0].keys())
            writer.writeheader()
            writer.writerows(frames)
        return analysis

    def test_frozen_models_are_scored_across_distinct_stream_sessions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_analysis(root, "01_calibration", pipeline_zero=100)
            second = self.make_analysis(
                root,
                "02_calibration",
                pipeline_zero=200,
                offset_shift_ms=0.3,
            )
            output = root / "cross"
            with patch(
                "calibration.cross_recording_analysis._write_graph",
                side_effect=lambda path, _report: path.write_bytes(b"PNG"),
            ):
                report = analyze_directories([first, second], output)

            self.assertEqual(
                report["acceptance_criteria"]["distinct_stream_sessions"], 2
            )
            self.assertTrue(
                report["acceptance_criteria"]["independent_sessions_passed"]
            )
            self.assertLessEqual(
                report["acceptance_criteria"]["fixed_candidate_range_ms"], 2.0
            )
            self.assertEqual(len(report["pair_results"]), 12)
            self.assertEqual(len(report["frozen_model_parameters"]), 2)
            self.assertEqual(len(report["six_step_slope_ranges"]), 6)
            self.assertTrue((output / "calibration_cross_recording.md").is_file())
            self.assertTrue((output / "calibration_cross_recording_results.csv").is_file())
            self.assertTrue(any(
                row["acceptance_passed"]
                for row in report["strategy_summary"]
            ))

    def test_discovery_deduplicates_the_same_source_recording(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.make_analysis(root, "01_calibration", pipeline_zero=100)
            duplicate = root / "latest_calibration_analysis"
            duplicate.mkdir()
            for filename in ("calibration_analysis.json", "calibration_frames.csv"):
                duplicate.joinpath(filename).write_bytes(original.joinpath(filename).read_bytes())

            self.assertEqual(discover_analysis_directories(root), [original])

    def test_same_stream_epoch_does_not_count_as_independent_validation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_analysis(root, "01_calibration", pipeline_zero=100)
            second = self.make_analysis(root, "02_calibration", pipeline_zero=100)
            with patch(
                "calibration.cross_recording_analysis._write_graph",
                side_effect=lambda path, _report: path.write_bytes(b"PNG"),
            ):
                report = analyze_directories([first, second], root / "cross")

            criteria = report["acceptance_criteria"]
            self.assertEqual(criteria["distinct_stream_sessions"], 1)
            self.assertFalse(criteria["independent_sessions_passed"])
            self.assertFalse(criteria["deployment_ready"])


if __name__ == "__main__":
    unittest.main()
