"""Focused tests for calibration report generation."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from calibration.quantitative_analysis import CURRENT_CORRECTION_MS, analyze_output_directory
from sensors.camera.timing_defaults import DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS


class TestQuantitativeAnalysis(unittest.TestCase):
    def test_saved_analysis_defaults_to_png_and_optionally_saves_svg_graphs(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            frames = []
            for index in range(45):
                clean = index != 8
                frames.append({
                    "frame_number": index + 1,
                    "filename": f"camera_{index + 1:06d}.jpg",
                    "validation": "accepted_clean" if clean else "accepted_timing_suspect",
                    "timing_status": "Clean" if clean else "Timing suspect",
                    "pts_ns": index * (20_000_000 if index % 2 else 40_000_000),
                    "pts_minus_latest_qr_ms": 96.0 + index % 4,
                })
            source = {
                "recording_directory": "/recordings/sample",
                "processed": len(frames),
                "display": {
                    "late_submissions": 2,
                    "irregular_intervals": 1,
                    "missed_period_candidates": 0,
                },
                "frames": frames,
            }
            (output / "calibration_analysis.json").write_text(
                json.dumps(source), encoding="utf-8"
            )
            with (output / "calibration_frames.csv").open(
                "w", encoding="utf-8", newline=""
            ) as destination:
                writer = csv.DictWriter(destination, fieldnames=frames[0].keys())
                writer.writeheader()
                writer.writerows(frames)

            def graph_file(path, *_args, save_svg=False, **_kwargs):
                path.write_bytes(b"PNG")
                if save_svg:
                    path.with_suffix(".svg").write_text("<svg/>", encoding="utf-8")

            with (
                patch("calibration.quantitative_analysis._write_timeline_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_residual_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_histogram", side_effect=graph_file),
            ):
                report = analyze_output_directory(output)

            self.assertEqual(report["data_quality"]["clean_frames"], 44)
            self.assertEqual(report["data_quality"]["timing_suspect_frames"], 1)
            self.assertEqual(CURRENT_CORRECTION_MS, DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS)
            self.assertEqual(report["verdict"]["current_correction_ms"], CURRENT_CORRECTION_MS)
            self.assertTrue((output / "calibration_verdict.md").is_file())
            self.assertEqual(report["graph_formats"], ["png"])
            graph_stems = (
                "calibration_offset_timeline",
                "calibration_residual_cdf",
                "calibration_offset_histogram",
                "calibration_fixed_residual_histogram",
                "calibration_pts_residual_histogram",
            )
            for stem in graph_stems:
                self.assertTrue((output / f"{stem}.png").is_file())
                self.assertFalse((output / f"{stem}.svg").exists())
                self.assertNotIn(f"{stem}.svg", report["output_files"])

            with (
                patch("calibration.quantitative_analysis._write_timeline_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_residual_graph", side_effect=graph_file),
                patch("calibration.quantitative_analysis._write_histogram", side_effect=graph_file),
            ):
                svg_report = analyze_output_directory(output, save_svg=True)

            self.assertEqual(svg_report["graph_formats"], ["png", "svg"])
            for stem in graph_stems:
                filename = f"{stem}.svg"
                self.assertIn("<svg", (output / filename).read_text())
                self.assertIn(filename, svg_report["output_files"])
