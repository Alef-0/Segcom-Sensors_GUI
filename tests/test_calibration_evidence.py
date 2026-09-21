import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from calibration.evidence import assess_evidence, detection_evidence, screen_cell, screen_geometry
from calibration.final_analysis import Goals, _evaluate, _select_models, analyze_directories, load_session, write_report


def observation(index=10, box=(10, 10, 40, 40)):
    return {"raw": None if index is None else str(index), "bbox": box, "cell": 0,
            "confidence": 0.9, "marker": None if index is None else {"cell": index % 4},
            "display_index": index, "original_points": None}


def evidence(index):
    return detection_evidence([observation(index)], (100, 100), {"status": "unavailable"})


def recording(root, *, legacy=False, resets=False, count=180):
    folder = root / "recording"
    folder.mkdir()
    analysis = root / "recording_analysis"
    analysis.mkdir()
    zero = 10_000_000_000
    display = [{"kind": "session", "grid_qrs": 4, "visible_qrs": 1}]
    for i in range(count * 2 + 20):
        stamp = zero + i * 10_000_000
        display.append({"kind": "frame", "index": i, "cell": i % 4,
                        "marker_ns": stamp, "presentation_return_ns": stamp})
    cameras, decoded = [], []
    for i in range(count):
        latest = 2 * i + 10
        pts = latest * 10_000_000 + 85_000_000 + (i % 2) * 1_000_000
        filename = f"images/camera_{i+1:06d}.jpg"
        cameras.append({"frame": filename, "stream_epoch": 1, "segment_epoch": 1,
                        "mapping_revision": i if resets else 0,
                        "running_time_ns": pts, "pts_ns": pts,
                        "media_monotonic_ns": zero + pts,
                        "application_arrival_monotonic_ns": zero + pts + 50_000_000})
        values = [None] * 4
        values[latest % 4] = f"{(zero + latest * 10_000_000)//1_000_000:012d}"
        row = {"filename": filename, "qr_values_ms": values}
        if not legacy:
            row["qr_evidence"] = evidence(latest)
        decoded.append(row)
    for name, rows in (("display_timestamps.jsonl", display), ("camera_timestamps.jsonl", cameras)):
        (folder / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (folder / "camera_timing_session.json").write_text(json.dumps({"epochs": [
        {"stream_epoch": 1, "mapping_revision": 0, "segment_epoch": 1,
         "pipeline_zero_monotonic_ns": zero}]}))
    (analysis / "calibration_analysis.json").write_text(json.dumps({
        "recording_directory": str(folder), "frames": decoded}))
    return analysis


class EvidenceTests(unittest.TestCase):
    def test_unreadable_retry_of_readable_region_is_not_missing_code(self):
        saved = detection_evidence([observation(), observation(None)], (100, 100), {})
        self.assertEqual(saved["physical_detection_groups"], 1)
        self.assertEqual(saved["unreadable_groups"], [])
        self.assertEqual(len(saved["detections"]), 2)
        self.assertTrue(assess_evidence({"qr_evidence": saved}, [10])["primary_usable"])

    def test_missing_newer_clipped_region_prevents_definite_timing_score(self):
        saved = detection_evidence([observation(944), observation(None, (60, 70, 95, 100))],
                                   (100, 100), {})
        result = assess_evidence({"qr_evidence": saved}, [944])
        self.assertFalse(result["primary_usable"])
        self.assertIn("clipped_qr_regions", result["reasons"])
        self.assertIn("unreadable_regions_may_contain_newer_qr", result["reasons"])

    def test_payload_conflicts_and_transition_are_not_fitted(self):
        saved = detection_evidence([observation(10), observation(11)], (100, 100), {})
        result = assess_evidence({"qr_evidence": saved}, [10, 11], transition=True)
        self.assertEqual(result["status"], "multiple_generation_transition")
        self.assertIn("conflicting_payloads_in_one_region", result["reasons"])
        self.assertFalse(result["primary_usable"])

    def test_screen_mapping_uses_screen_instead_of_camera_grid(self):
        config = {"default": {"image_size": [1000, 800], "undistortion_alpha": 0.25,
                              "corners": [[600, 100], [900, 150], [880, 500], [620, 480]]}}
        geometry = screen_geometry(config, "a.jpg", (1000, 800), 0.25)
        positions = [(0, 0), (0, 1), (1, 1), (1, 0)]
        self.assertEqual(geometry["status"], "valid")
        self.assertEqual(screen_cell((650, 180), geometry, positions), 0)
        self.assertIsNone(screen_cell((100, 100), geometry, positions))
        self.assertEqual(screen_geometry(config, "a.jpg", (1000, 800), 0.5)["status"], "invalid")
        self.assertEqual(screen_geometry(config, "a.jpg", (500, 400), 0.25)["status"], "invalid")
        config["frames"] = {"a.jpg": {**config["default"], "corners": [[-5, 100], [900, 150], [880, 500], [0, 480]]}}
        self.assertEqual(screen_geometry(config, "a.jpg", (1000, 800), 0.25)["status"], "clipped")

    def test_stale_geometry_mismatch_is_flagged(self):
        saved = evidence(10)
        saved["geometry"] = {"status": "valid"}
        saved["detections"][0]["screen_cell"] = 0  # Journal cell is 2.
        self.assertIn("screen_geometry_disagrees_with_journal",
                      assess_evidence({"qr_evidence": saved}, [10])["reasons"])

    def test_legacy_intervals_are_diagnostic_only_and_report_does_not_pass(self):
        with TemporaryDirectory() as temp:
            analysis = recording(Path(temp), legacy=True)
            report = analyze_directories([analysis])
            self.assertEqual(report["verdict"]["status"], "insufficient_trustworthy_evidence")
            self.assertFalse(report["verdict"]["independent_every_frame_verified"])
            self.assertEqual(report["sessions"][0]["audit"]["counts"]["primary_scored_frames"], 0)
            self.assertIsNotNone(report["frames"][0]["diagnostic_newest_decoded_interval_ms"])
            self.assertIsNone(report["frames"][0]["targets"]["newest_generation"])
            saved = write_report(report, Path(temp) / "output", plots=False)
            self.assertIn("final_analysis_diagnostics.csv", saved["output_files"])
            self.assertIn("unknown_pixel_evidence", (Path(temp)/"output/final_analysis.md").read_text())

    def test_resets_disable_history_models_but_allow_constant_baseline(self):
        with TemporaryDirectory() as temp:
            session = load_session(recording(Path(temp), resets=True))
            models, selection = _select_models([session.frames], Goals())
            self.assertEqual(selection["selected_family"], "model_a_constant")
            self.assertNotIn("model_c_interval_history", models)
            self.assertNotIn("model_b_cadence_state", models)
            self.assertTrue(selection["unsupported_recipes"]["model_c_interval_history"])
            self.assertEqual(session.frames[1]["history_reset_reasons"], ["stream_mapping_or_segment_changed"])
            self.assertEqual(session.frames[1]["history_length"], 0)

    def test_ambiguous_frames_stay_in_denominator_and_cannot_verify_every_frame(self):
        with TemporaryDirectory() as temp:
            session = load_session(recording(Path(temp)))
            rows = session.frames[:2]
            rows[1]["targets"]["newest_generation"] = None
            rows[1]["evidence_assessment"] = {"status": "potentially_missing_newer_generation", "reasons": ["clipped_qr_regions"]}
            model = {"recipe": {"kind": "fixed_interval"}, "correction_ms": 80.0}
            metrics, predictions = _evaluate(model, rows, Goals())
            self.assertEqual(metrics["n"], 1)
            self.assertEqual(metrics["camera_frames"], 2)
            self.assertEqual(metrics["scored_camera_pct"], 50)
            self.assertEqual(metrics["below_threshold_all_camera_lower_bound_pct"], 50)
            self.assertFalse(metrics["every_camera_frame_verified"])
            self.assertIsNone(predictions[1]["interval_residual_ms"])

    def test_complete_variable_history_is_supported_and_fallback_is_explicit(self):
        with TemporaryDirectory() as temp:
            session = load_session(recording(Path(temp)))
            models, _ = _select_models([session.frames], Goals())
            self.assertIn("model_c_interval_history", models)
            _, predictions = _evaluate(models["model_c_interval_history"], session.frames, Goals())
            self.assertEqual(predictions[0]["estimate_status"], "constant_fallback_missing_history")
            self.assertEqual(predictions[-1]["estimate_status"], "valid")

    def test_export_preserves_pixel_evidence_and_final_analysis_gates_it(self):
        from collections import OrderedDict
        from unittest.mock import Mock
        from calibration.recording_display import DisplayTimeline, RecordingAnalyzer
        with TemporaryDirectory() as temp:
            analysis = recording(Path(temp), count=1)
            folder = analysis.with_name("recording")
            model = RecordingAnalyzer.__new__(RecordingAnalyzer)
            model.grid_qrs = 4
            model.cell_positions = [(0, 0), (0, 1), (1, 1), (1, 0)]
            model.screen_geometry_config = None
            model.timeline = DisplayTimeline(folder)
            model.epochs = {epoch["stream_epoch"]: epoch for epoch in json.loads(
                (folder / "camera_timing_session.json").read_text())["epochs"]}
            model.undistorter = Mock()
            model.undistorter.to_original.side_effect = lambda points, *_: points
            model.cache = OrderedDict()
            model.cache_limit = 8
            row = json.loads((folder / "camera_timestamps.jsonl").read_text())
            row["filename"] = row["frame"]
            source = json.loads((analysis / "calibration_analysis.json").read_text())
            payload = source["frames"][0]["qr_values_ms"][2]
            pixels = np.zeros((100, 100, 3), dtype=np.uint8)
            decoded = [{"raw": payload, "bbox": np.array([10., 10., 40., 40.]),
                        "center": (25., 25.), "confidence": 0.9},
                       {"raw": None, "bbox": np.array([60., 70., 99., 100.]),
                        "center": (79., 85.), "confidence": 0.8}]
            result = model._finish_analysis(0, row, pixels, pixels, decoded, 0.25)
            source["frames"][0]["qr_evidence"] = result["qr_evidence"]
            (analysis / "calibration_analysis.json").write_text(json.dumps(source))
            frame = load_session(analysis).frames[0]
            self.assertEqual(len(frame["qr_evidence"]["detections"]), 2)
            self.assertIsNone(frame["targets"]["newest_generation"])
            self.assertIsNotNone(frame["diagnostic_newest_decoded_interval_ms"])
            self.assertIn("clipped_qr_regions", frame["exclusion_reasons"])

    def test_positive_report_exports_diagnostics_without_changing_thresholds(self):
        with TemporaryDirectory() as temp:
            report = analyze_directories([recording(Path(temp))])
            self.assertTrue(report["predictions"])
            saved = write_report(report, Path(temp) / "output", plots=False)
            text = (Path(temp) / "output/final_analysis.md").read_text()
            self.assertIn("prediction range", text)
            self.assertIn("Largest scored errors", text)
            self.assertEqual(saved["goals"]["absolute_below_ms"], 10)
            self.assertIn("missing_independent_stream_validation", saved["verdict"]["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
