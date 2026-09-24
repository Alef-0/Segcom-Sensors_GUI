"""Command-line routing tests for calibration recording analysis."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import json

import analyze_calibration_recording as recording_launcher


class TestRecordingLauncher(unittest.TestCase):
    def test_root_launcher_runs_only_pts_anchor_analysis_with_explicit_output(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            destination = output / "pts_anchor_analysis"
            destination.mkdir()
            (destination / "pts_anchor_analysis.json").write_text(
                json.dumps({"output_directory": str(destination)}),
                encoding="utf-8",
            )
            with patch.object(recording_launcher.subprocess, "run") as run:
                report = recording_launcher._run_anchor_analysis(output, Path("/offset.json"))

            command = run.call_args.args[0]
            self.assertEqual(
                command[:3],
                [recording_launcher.sys.executable, str(recording_launcher.PROJECT_ROOT / "analyze_pts_anchor.py"), "evaluate"],
            )
            self.assertIn(str(output), command)
            self.assertEqual(report["output_directory"], str(destination))

    def test_root_launcher_runs_pts_anchor_analysis_after_the_window(self):
        recording = Path("/recordings/sample")
        output = Path("/recordings/sample_analysis")
        with (
            patch.object(recording_launcher, "run_recording_display", return_value=output) as display,
            patch.object(recording_launcher, "_run_anchor_analysis", return_value={
                "output_directory": str(output / "pts_anchor_analysis")
            }) as analyze,
            patch("sys.argv", ["analyze_calibration_recording.py", str(recording), "--offset-file", "/offset.json"]),
        ):
            recording_launcher.main()
        display.assert_called_once()
        analyze.assert_called_once_with(output, Path("/offset.json"))

    def test_closing_saved_review_does_not_rerun_pts_anchor_analysis(self):
        with (
            patch.object(recording_launcher, "run_recording_display", return_value=None),
            patch.object(recording_launcher, "_run_anchor_analysis") as analyze,
            patch("sys.argv", ["analyze_calibration_recording.py", "/recordings/sample"]),
        ):
            recording_launcher.main()
        analyze.assert_not_called()

    def test_fresh_decode_without_offset_only_exports_evidence(self):
        with (
            patch.object(recording_launcher, "run_recording_display", return_value=Path("/recordings/lab_analysis")),
            patch.object(recording_launcher, "_run_anchor_analysis") as analyze,
            patch("sys.argv", ["analyze_calibration_recording.py", "/recordings/lab"]),
        ):
            recording_launcher.main()
        analyze.assert_not_called()
