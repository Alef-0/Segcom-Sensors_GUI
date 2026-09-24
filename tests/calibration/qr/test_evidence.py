import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from calibration.evidence import assess_evidence, detection_evidence, repeated_state_checks, screen_cell, screen_geometry
from calibration.anchor_analysis import _stream_components, load_session
from calibration.qr import timestamp_payload
from calibration.recording_display import RecordingAnalyzer
from analyze_pts_anchor import score_predictions
from tests.calibration.support import build_calibration_analysis_recording


def observation(index=10, box=(10, 10, 40, 40)):
    return {"raw": None if index is None else str(index), "bbox": box, "cell": 0,
            "confidence": 0.9, "marker": None if index is None else {"cell": index % 4},
            "display_index": index, "original_points": None}


def evidence(index):
    return detection_evidence([observation(index)], (100, 100), {"status": "unavailable"})


def temporal_confirmation_pair():
    timeline_frames = [
        {"index": index, "cell": index % 4,
         "marker_ns": 10_000_000_000_000 + index * 16_000_000}
        for index in range(8)
    ]

    def detection(group, display_index, cell, box, *, raw=None, confidence=0.9):
        if raw is None and display_index is not None:
            raw = timestamp_payload(timeline_frames[display_index]["marker_ns"])
        return {
            "group": group, "display_index": display_index, "raw": raw,
            "detected_cell": cell, "screen_cell": None,
            "position_basis": "camera_image_grid",
            "journal_cell": timeline_frames[display_index]["cell"] if display_index is not None else None,
            "bbox": list(box), "original_points": [[box[0], box[1]], [box[2], box[1]],
                                                     [box[2], box[3]], [box[0], box[3]]],
            "confidence": confidence, "clipped": False,
        }

    previous_detections = [
        detection(0, 4, 0, (5, 5, 25, 25)),
        detection(1, 5, 1, (10, 10, 30, 30)),
        detection(2, None, 2, (40, 10, 60, 30), raw=None),
    ]
    following_detections = [
        detection(0, 5, 1, (11, 11, 31, 31)),
        detection(1, 6, 2, (41, 11, 61, 31)),
    ]

    def result(index, detections, unreadable_groups=()):
        observations = [{"undistorted_points": np.asarray(row["original_points"], dtype=float)}
                        for row in detections]
        return {
            "index": index,
            "row": {"stream_epoch": 1, "mapping_revision": 0, "segment_epoch": 1},
            "observations": observations,
            "qr_evidence": {"image_size": [100, 100], "detections": detections,
                            "unreadable_groups": list(unreadable_groups), "conflicting_groups": []},
        }

    model = RecordingAnalyzer.__new__(RecordingAnalyzer)
    model.manual_values = {}
    model.timeline = SimpleNamespace(frames=timeline_frames)
    return model, result(18, previous_detections, (2,)), result(19, following_detections)


class EvidenceTests(unittest.TestCase):
    def test_temporal_successor_is_inferred_after_sequence_and_position_confirmation(self):
        model, previous, following = temporal_confirmation_pair()

        inferred = model._confirm_temporal_successor(previous, following)

        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0]["display_index"], 6)
        self.assertEqual(inferred[0]["raw"], timestamp_payload(model.timeline.frames[6]["marker_ns"]))
        self.assertEqual(inferred[0]["cell"], 2)
        self.assertEqual(inferred[0]["supporting_readable_display_indices"], [4, 5])
        self.assertEqual(inferred[0]["confirmation_frame_number"], 20)
        self.assertGreaterEqual(inferred[0]["candidate_bbox_iou"], 0.35)
        self.assertGreaterEqual(inferred[0]["predecessor_bbox_iou"], 0.35)

    def test_temporal_successor_requires_two_consecutive_readable_predecessors(self):
        model, previous, following = temporal_confirmation_pair()
        previous["qr_evidence"]["detections"] = previous["qr_evidence"]["detections"][1:]
        previous["qr_evidence"]["unreadable_groups"] = [2]
        previous["observations"] = previous["observations"][1:]

        self.assertEqual(model._confirm_temporal_successor(previous, following), [])

    def test_temporal_successor_requires_same_candidate_and_predecessor_cells(self):
        for target, row_index, changed_cell in (("previous", 2, 3), ("following", 0, 0)):
            with self.subTest(target=target, cell=changed_cell):
                model, previous, following = temporal_confirmation_pair()
                frame = previous if target == "previous" else following
                frame["qr_evidence"]["detections"][row_index]["detected_cell"] = changed_cell

                self.assertEqual(model._confirm_temporal_successor(previous, following), [])

    def test_temporal_successor_requires_bbox_overlap_in_both_tracks(self):
        model, previous, following = temporal_confirmation_pair()
        following["qr_evidence"]["detections"][1]["bbox"] = [70, 10, 90, 30]

        self.assertEqual(model._confirm_temporal_successor(previous, following), [])

    def test_unreadable_retry_of_readable_region_is_not_missing_code(self):
        saved = detection_evidence([observation(), observation(None)], (100, 100), {})
        self.assertEqual(saved["physical_detection_groups"], 1)
        self.assertEqual(saved["unreadable_groups"], [])
        self.assertEqual(len(saved["detections"]), 2)
        self.assertTrue(assess_evidence({"qr_evidence": saved}, [10])["primary_usable"])

    def test_unreadable_clipped_region_allows_conditional_newest_readable_score(self):
        saved = detection_evidence([observation(944), observation(None, (60, 70, 95, 100))],
                                   (100, 100), {})
        result = assess_evidence({"qr_evidence": saved}, [944])
        self.assertTrue(result["primary_usable"])
        self.assertIn("clipped_qr_regions", result["warnings"])
        self.assertIn("unreadable_regions_may_contain_newer_qr", result["warnings"])

    def test_older_readable_artifacts_do_not_veto_newest(self):
        saved = detection_evidence([observation(10), observation(11, (60, 60, 90, 90))],
                                   (100, 100), {})
        result = assess_evidence({"qr_evidence": saved}, [10, 11], transition=True)
        self.assertTrue(result["primary_usable"])
        self.assertEqual(result["status"], "usable_newest_readable_with_artifacts")
        self.assertIn("multiple_generations_without_common_visibility", result["warnings"])

    def test_payload_conflicts_and_transition_are_not_fitted(self):
        saved = detection_evidence([observation(10), observation(11)], (100, 100), {})
        result = assess_evidence({"qr_evidence": saved}, [10, 11], transition=True)
        self.assertIn("multiple_generations_without_common_visibility", result["warnings"])
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

    def test_repeated_state_flags_entire_run_without_fitted_offset(self):
        times = [i * 16_666_667 for i in range(170)]
        rows = [{"latest_display_index": latest,
                 "media_reference_monotonic_ns": 5_000_000_000 + elapsed,
                 "segment": 1}
                for latest, elapsed in ((146, 0), (146, 40_000_000),
                                        (146, 59_975_294), (152, 99_942_150),
                                        (155, 139_899_708), (158, 159_853_021))]
        checks = repeated_state_checks(rows, times)
        self.assertEqual([item is not None for item in checks], [True, True, True, False, False, False])
        self.assertAlmostEqual(checks[0]["camera_span_ms"], 59.975294)
        result = assess_evidence({"qr_evidence": evidence(146), "temporal_evidence": checks[0]}, [146])
        self.assertFalse(result["primary_usable"])
        self.assertEqual(result["status"], "suspected_stale_visual_state")

    def test_normal_oversampling_long_journal_holds_and_pauses_are_not_stale(self):
        rows = [{"latest_display_index": 2, "media_reference_monotonic_ns": n,
                 "segment": 1} for n in (0, 10_000_000, 20_000_000)]
        times = [i * 16_666_667 for i in range(6)]
        self.assertEqual(repeated_state_checks(rows, times), [None] * 3)
        rows[-1]["media_reference_monotonic_ns"] = 60_000_000
        self.assertEqual(repeated_state_checks(rows, times, {2}), [None] * 3)
        times[3:] = [150_000_000, 166_666_667, 183_333_334]
        self.assertEqual(repeated_state_checks(rows, times), [None] * 3)

    def test_repeated_state_does_not_bridge_missing_reads_or_segment_resets(self):
        times = [i * 10_000_000 for i in range(6)]
        for middle in ({"latest_display_index": None}, {"media_reference_monotonic_ns": None},
                       {"segment": 2}, {"media_reference_monotonic_ns": -1}):
            with self.subTest(middle=middle):
                rows = [{"latest_display_index": 2, "media_reference_monotonic_ns": n,
                         "segment": 1} for n in (0, 20_000_000, 40_000_000)]
                rows[1].update(middle)
                self.assertEqual(repeated_state_checks(rows, times), [None] * 3)

    def test_anchor_analysis_recomputes_stale_runs_and_retains_diagnostics(self):
        with TemporaryDirectory() as temp:
            analysis = build_calibration_analysis_recording(Path(temp), count=8)
            path = analysis / "calibration_analysis.json"
            source = json.loads(path.read_text())
            for row in source["frames"][1:3]:
                row["qr_values_ms"] = source["frames"][0]["qr_values_ms"]
                row["qr_evidence"] = evidence(10)
            path.write_text(json.dumps(source))
            session = load_session(analysis)
            self.assertEqual(session.audit["counts"]["primary_scored_frames"], 5)
            self.assertEqual(session.audit["counts"]["evidence_suspected_stale_visual_state"], 3)
            self.assertEqual(len(session.frames), 8)
            for row in session.frames[:3]:
                self.assertIsNotNone(row["diagnostic_newest_decoded_interval_ms"])
                self.assertTrue(all(target is None for target in row["targets"].values()))
                self.assertFalse(row["evidence_assessment"]["primary_usable"])

    def test_stale_geometry_mismatch_is_flagged(self):
        saved = evidence(10)
        saved["geometry"] = {"status": "valid"}
        saved["detections"][0]["screen_cell"] = 0  # Journal cell is 2.
        self.assertIn("screen_geometry_disagrees_with_journal",
                      assess_evidence({"qr_evidence": saved}, [10])["reasons"])

    def test_legacy_intervals_are_diagnostic_only_and_report_does_not_pass(self):
        with TemporaryDirectory() as temp:
            analysis = build_calibration_analysis_recording(Path(temp), legacy=True)
            session = load_session(analysis)
            self.assertEqual(session.audit["counts"]["primary_scored_frames"], 0)
            self.assertIsNotNone(session.frames[0]["diagnostic_newest_decoded_interval_ms"])
            self.assertIsNone(session.frames[0]["targets"]["newest_generation"])

    def test_resets_preserve_continuity_diagnostics(self):
        with TemporaryDirectory() as temp:
            session = load_session(build_calibration_analysis_recording(Path(temp), resets=True))
            self.assertEqual(session.frames[1]["continuity_reset_reasons"],
                             ["stream_mapping_or_segment_changed"])

    def test_remapped_anchors_do_not_make_same_pipeline_independent(self):
        with TemporaryDirectory() as temp:
            sessions = []
            for i, base_time in enumerate((123_000_000, 123_000_000, 456_000_000)):
                root = Path(temp) / str(i)
                root.mkdir()
                analysis = build_calibration_analysis_recording(root)
                path = root / "recording" / "camera_timing_session.json"
                saved = json.loads(path.read_text())
                saved["epochs"][0]["pipeline_zero_monotonic_ns"] += i * 10_000
                saved["epochs"][0]["pipeline_base_time_ns"] = base_time
                path.write_text(json.dumps(saved))
                sessions.append(load_session(analysis))
            self.assertEqual(sessions[0].stream_keys, sessions[1].stream_keys)
            self.assertNotEqual(sessions[0].stream_keys, sessions[2].stream_keys)
            self.assertEqual(sorted(map(len, _stream_components(sessions))), [1, 2])

    def test_ambiguous_frames_stay_in_denominator_and_cannot_verify_every_frame(self):
        with TemporaryDirectory() as temp:
            session = load_session(build_calibration_analysis_recording(Path(temp)))
            rows = session.frames[:2]
            rows[1]["targets"]["newest_generation"] = None
            rows[1]["evidence_assessment"] = {"status": "potentially_missing_newer_generation", "reasons": ["clipped_qr_regions"]}
            predictions = [{"estimated_monotonic_ns": r["media_reference_monotonic_ns"] - 80_000_000}
                           for r in rows]
            metrics = score_predictions(rows, predictions)
            self.assertEqual(metrics["scored_frames"], 1)
            self.assertEqual(metrics["camera_frames"], 2)
            self.assertFalse(metrics["every_saved_frame_meets_goals"])

    def test_export_preserves_artifacts_without_discarding_newest_interval(self):
        from collections import OrderedDict
        from unittest.mock import Mock
        from calibration.recording_display import DisplayTimeline, RecordingAnalyzer
        with TemporaryDirectory() as temp:
            analysis = build_calibration_analysis_recording(Path(temp), count=1)
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
            self.assertIsNotNone(frame["targets"]["newest_generation"])
            self.assertIsNotNone(frame["diagnostic_newest_decoded_interval_ms"])
            self.assertIn("clipped_qr_regions", frame["evidence_assessment"]["warnings"])

    def test_existing_saved_decode_can_be_reanalyzed_with_older_generations(self):
        with TemporaryDirectory() as temp:
            analysis = build_calibration_analysis_recording(Path(temp))
            path = analysis / "calibration_analysis.json"
            source = json.loads(path.read_text())
            for i, row in enumerate(source["frames"]):
                latest = 2 * i + 10
                older = latest - 1
                row["qr_values_ms"][older % 4] = f"{(10_000_000_000 + older * 10_000_000)//1_000_000:012d}"
                row["qr_evidence"] = detection_evidence(
                    [observation(older), observation(latest, (60, 60, 90, 90))], (100, 100), {})
            path.write_text(json.dumps(source))
            session = load_session(analysis)
            self.assertEqual(session.audit["counts"]["primary_scored_frames"], 180)
            for frame in session.frames:
                self.assertEqual(frame["targets"]["newest_generation"],
                                 frame["diagnostic_newest_decoded_interval_ms"])
                self.assertIsNone(frame["targets"]["common_visibility"])


if __name__ == "__main__":
    unittest.main()
