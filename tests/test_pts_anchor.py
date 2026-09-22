"""Clock mapping, epoch reset, and independence from QR/arrival evidence."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analyze_pts_anchor import calibrate, evaluate, predict_timestamps, read_offset, score_predictions, write_report
from tests.test_calibration_evidence import recording


class PtsAnchorTests(unittest.TestCase):
    def make_recording(self, root, name, base):
        location = root / name
        location.mkdir()
        analysis = recording(location, count=60)
        journal = location / "recording/camera_timestamps.jsonl"
        cameras = [json.loads(line) for line in journal.read_text().splitlines()]
        for camera in cameras:
            camera["pipeline_running_time_observed_ns"] = 40_000_000_000 + camera["running_time_ns"]
            camera["pipeline_base_time_ns"] = base
        journal.write_text("".join(json.dumps(row) + "\n" for row in cameras))
        epoch = location / "recording/camera_timing_session.json"
        data = json.loads(epoch.read_text())
        data["epochs"][0]["pipeline_base_time_ns"] = base
        epoch.write_text(json.dumps(data))
        return analysis

    def test_frozen_lab_offset_transfers_without_test_qr_or_arrival_feedback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_recording(root, "lab", 111)
            target = self.make_recording(root, "test", 222)
            artifact = calibrate([source])
            self.assertEqual(artifact["offset_ns"], 80_500_000)
            report, rows = evaluate([target], artifact)
            self.assertEqual(report["sessions"][0]["scores"]["maximum_absolute_ms"], 0)
            before = deepcopy(artifact)
            path = target / "calibration_analysis.json"
            data = json.loads(path.read_text())
            for frame in data["frames"]:
                frame["qr_values_ms"] = [None] * 4
            path.write_text(json.dumps(data))
            journal = target.with_name("recording") / "camera_timestamps.jsonl"
            cameras = [json.loads(line) for line in journal.read_text().splitlines()]
            for row in cameras:
                row["application_arrival_monotonic_ns"] += 999_000_000
            journal.write_text("".join(json.dumps(row) + "\n" for row in cameras))
            changed, other = evaluate([target], artifact)
            self.assertEqual(artifact, before)
            self.assertEqual([r["estimated_monotonic_ns"] for r in rows],
                             [r["estimated_monotonic_ns"] for r in other])
            self.assertEqual(changed["sessions"][0]["scores"]["scored_frames"], 0)
            self.assertFalse(changed["sessions"][0]["scores"]["every_saved_frame_meets_goals"])
            write_report(changed, other, root / "report")
            self.assertTrue((root / "report/pts_anchor_predictions.csv").is_file())

    def test_same_pipeline_is_rejected_even_in_a_separate_recording(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_recording(root, "lab", 111)
            target = self.make_recording(root, "same_stream", 111)
            artifact = calibrate([source])
            with self.assertRaisesRegex(ValueError, "share a stream"):
                evaluate([target], artifact)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                calibrate([source, source])

    def test_predeclared_stream_age_cutoff_does_not_silently_expand(self):
        with TemporaryDirectory() as directory:
            source = self.make_recording(Path(directory), "lab", 111)
            with self.assertRaisesRegex(ValueError, "No usable post-cutoff"):
                calibrate([source], 100)
            with self.assertRaisesRegex(ValueError, "finite"):
                calibrate([source], float("nan"))

    def test_offset_roundtrip_and_malformed_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = calibrate([self.make_recording(root, "lab", 111)])
            path = root / "offset.json"
            path.write_text(json.dumps(artifact))
            self.assertEqual(read_offset(path), artifact)
            artifact["offset_ns"] = True
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "frozen offset"):
                read_offset(path)

    def test_clock_anchor_without_pipeline_identity_cannot_prove_independence(self):
        with TemporaryDirectory() as directory:
            source = self.make_recording(Path(directory), "lab", 111)
            path = source.with_name("recording") / "camera_timing_session.json"
            metadata = json.loads(path.read_text())
            del metadata["epochs"][0]["pipeline_base_time_ns"]
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "pipeline-base"):
                calibrate([source])

    def test_independent_streams_have_equal_weight(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            low = self.make_recording(root, "low", 111)
            high = self.make_recording(root, "high", 222)
            # Shift anchored media time in the high stream by 20 ms relative
            # to the unchanged display journal. Keep only ten frames there.
            epoch = high.with_name("recording") / "camera_timing_session.json"
            data = json.loads(epoch.read_text())
            data["epochs"][0]["pipeline_zero_monotonic_ns"] += 20_000_000
            epoch.write_text(json.dumps(data))
            journal = high.with_name("recording") / "camera_timestamps.jsonl"
            journal.write_text("\n".join(journal.read_text().splitlines()[:10]) + "\n")
            artifact = calibrate([low, high])
            self.assertEqual([g["offset_ns"] for g in artifact["groups"]], [80_500_000, 100_500_000])
            self.assertEqual(artifact["offset_ns"], 90_500_000)

    def test_clock_mapping_uses_running_time_not_raw_pts_or_arrival(self):
        epoch = dict(stream_epoch=1, mapping_revision=0, segment_epoch=1,
                     pipeline_zero_monotonic_ns=10_000_000_000)
        row = dict(stream_epoch=1, mapping_revision=0, segment_epoch=1,
                   pts_ns=900_000_000, running_time_ns=100_000_000,
                   pipeline_running_time_observed_ns=200_000_000,
                   pipeline_clock_mapping_monotonic_ns=10_200_000_000,
                   application_arrival_monotonic_ns=10_180_000_000)
        result = predict_timestamps([row], [epoch])[0]
        self.assertEqual(result['epoch_running_time'], 10_100_000_000)
        self.assertEqual(result['sampled_zero_minus_epoch_ms'], 0)
        altered = dict(row, application_arrival_monotonic_ns=99_000_000_000,
                       targets={'newest_generation': [-1000, 1000]},
                       capture_estimator_correction_ns=900_000_000)
        self.assertEqual(result, predict_timestamps([altered], [epoch])[0])

    def test_anchor_is_frozen_until_new_recorded_epoch_and_never_uses_clock_fallback(self):
        rows = [dict(stream_epoch=1, mapping_revision=m, segment_epoch=1,
                     running_time_ns=r, pipeline_running_time_observed_ns=r + 10,
                     pipeline_clock_mapping_monotonic_ns=r + 10 + zero)
                for m, r, zero in [(0, 100, 1000), (0, 200, 1005), (1, 300, 2000)]]
        epochs = [dict(stream_epoch=1, mapping_revision=m, segment_epoch=1,
                       pipeline_zero_monotonic_ns=zero) for m, zero in ((0, 1000), (1, 2000))]
        result = predict_timestamps(rows, epochs)
        self.assertEqual([r['epoch_running_time'] for r in result], [1100, 1200, 2300])
        self.assertEqual(result[1]['sampled_zero_minus_epoch_ms'], 5 / 1e6)
        self.assertTrue(all(r['epoch_running_time'] is None for r in predict_timestamps(rows, [])))
        changed = deepcopy(rows)
        changed[-1]['pipeline_clock_mapping_monotonic_ns'] += 999
        self.assertEqual([p['epoch_running_time'] for p in predict_timestamps(changed, epochs)],
                         [p['epoch_running_time'] for p in result])

    def test_scoring_keeps_unscorable_frames_and_interval_boundaries(self):
        frames = [dict(media_reference_monotonic_ns=1_100_000_000,
                       targets={'newest_generation': b})
                  for b in ([90, 100], [90, 100], [90, 100], None)]
        predictions = [{'epoch_running_time': t} for t in
                       (1_000_000_000, 1_010_000_000, 1_020_000_000, 1_000_000_000)]
        score = score_predictions(frames, predictions, 'epoch_running_time')
        self.assertEqual(score['camera_frames'], 4)
        self.assertEqual(score['scored_frames'], 3)
        self.assertEqual(score['unscored_frames'], 1)
        self.assertEqual(score['maximum_absolute_ms'], 10)
        self.assertEqual(score['errors_at_or_above_threshold'], 1)
        self.assertFalse(score['strict_maximum_goal_passed'])



if __name__ == '__main__':
    unittest.main()
