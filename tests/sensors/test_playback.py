from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2 as cv

from menu_layout import build_graph_tab, build_record_tab, build_video_tab
from processing.visualization.graph_draw import Graph_radar
import sensors.playback as playback_module


class PlaybackControllerTests(unittest.TestCase):
    @staticmethod
    def _elements(layout):
        for row in layout:
            for element in row:
                yield element
                rows = getattr(element, "Rows", None)
                if rows:
                    yield from PlaybackControllerTests._elements(rows)

    def test_synced_only_checkbox_defaults_to_checked(self):
        elements = {
            element.Key: element
            for element in self._elements(build_record_tab())
            if getattr(element, "Key", None)
        }
        self.assertTrue(elements["playback_synced_only"].InitialState)

    def test_synced_only_loader_omits_unpaired_entries(self):
        synced = SimpleNamespace(point_cloud=Path("frame.pcd"), camera_frame=Path("camera.jpg"))
        pcd_only = SimpleNamespace(point_cloud=Path("only.pcd"), camera_frame=None)
        image_only = SimpleNamespace(point_cloud=None, camera_frame=Path("only.jpg"))

        with patch.object(
            playback_module,
            "load_recording_entries",
            return_value=(pcd_only, synced, image_only),
        ):
            _, entries = playback_module.load_snapshot_entries(".", synced_only=True)
            _, all_entries = playback_module.load_snapshot_entries(".", synced_only=False)

        self.assertEqual(entries, [synced])
        self.assertEqual(all_entries, [pcd_only, synced, image_only])

    def test_synced_only_loader_warns_when_folder_has_one_modality(self):
        pcd_only = SimpleNamespace(point_cloud=Path("only.pcd"), camera_frame=None)

        with patch.object(
            playback_module,
            "load_recording_entries",
            return_value=(pcd_only,),
        ):
            with self.assertRaisesRegex(ValueError, "No synced image \\+ PCD pairs"):
                playback_module.load_snapshot_entries(".", synced_only=True)

    def test_structure_detection_distinguishes_snapshots_and_recordings(self):
        synced = SimpleNamespace(point_cloud=Path("frame.pcd"), camera_frame=Path("camera.jpg"))
        pcd_only = SimpleNamespace(point_cloud=Path("only.pcd"), camera_frame=None)

        self.assertEqual(playback_module.detect_playback_structure("snapshot_001", (synced,)), "snapshot")
        self.assertEqual(playback_module.detect_playback_structure("custom_folder", (synced,)), "snapshot")
        self.assertEqual(playback_module.detect_playback_structure("recording_A", (pcd_only,)), "recording")

    def test_graph_resolution_and_range_share_one_apply_button(self):
        layout = build_graph_tab()
        elements = {
            element.Key: element
            for element in self._elements(layout)
            if getattr(element, "Key", None)
        }
        self.assertIn("graph_settings_apply", elements)
        self.assertIn("point_cutoff_apply", elements)
        self.assertNotIn("graph_resolution_apply", elements)
        self.assertNotIn("graph_range_apply", elements)

    def test_video_tab_contains_resolution_rate_and_latency_controls(self):
        layout = build_video_tab()
        elements = {
            element.Key: element
            for element in self._elements(layout)
            if getattr(element, "Key", None)
        }
        self.assertIn("playback_resolution_apply", elements)
        self.assertIn("recording_rate_apply", elements)
        self.assertIn("camera_latency_apply", elements)
        self.assertIn("camera_pipeline_latency", elements)
        self.assertIn("camera_latency_adjustment", elements)
        self.assertIn("camera_latency_status", elements)

    def test_clicked_point_is_printed_on_one_line(self):
        graph = Graph_radar.__new__(Graph_radar)
        graph.displayed_points = [{
            "pixel": (10, 20),
            "x": 1.25,
            "y": 3.5,
            "point": SimpleNamespace(dynamic_property=0, rcs=-7.0),
        }]
        output = StringIO()

        with redirect_stdout(output):
            graph._on_mouse(cv.EVENT_LBUTTONDOWN, 10, 20, None, None)

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("[RADAR POINT] x=1.25 m | y=3.50 m", lines[0])

    def test_single_playback_controller_exports(self):
        self.assertTrue(hasattr(playback_module, "PlaybackController"))
        self.assertTrue(hasattr(playback_module, "load_recording_entries"))
        self.assertTrue(hasattr(playback_module, "load_snapshot_entries"))
        self.assertTrue(hasattr(playback_module, "playback_main"))
        self.assertTrue(hasattr(playback_module, "snapshot_playback_main"))

    def test_playback_controller_handles_step_and_pause(self):
        controller = playback_module.PlaybackController(
            connection=SimpleNamespace(poll=lambda *_: False),
            pool=SimpleNamespace(put=lambda *_: None),
            shutdown_event=SimpleNamespace(is_set=lambda: False, set=lambda: None),
            initial_values={},
        )
        entry1 = playback_module.PlaybackEntry(point_cloud=None, recorded_at=playback_module.datetime.now())
        entry2 = playback_module.PlaybackEntry(point_cloud=None, recorded_at=playback_module.datetime.now())
        controller.entries = [entry1, entry2]
        controller.index = 0
        controller.paused = False

        with patch.object(controller, "_render"):
            controller._handle("playback_pause", None)
            self.assertTrue(controller.paused)

            controller._handle("playback_next", None)
            self.assertEqual(controller.index, 1)
            self.assertTrue(controller.paused)

            controller._handle("playback_previous", None)
            self.assertEqual(controller.index, 0)

            controller._handle("playback_stop", None)
            self.assertTrue(controller.stop_requested)

    def test_load_recording_entries_from_subdirectories(self):
        from tempfile import TemporaryDirectory
        import json

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session = root / "recording_B_2026-09-27"
            pcd_dir = session / "pcd"
            pcd_dir.mkdir(parents=True)
            pcd_file = pcd_dir / "frame_000001.pcd"
            pcd_file.write_text("# .PCD v.7 - Point Cloud Data\n", encoding="utf-8")

            rec_json = session / "recording.json"
            rec_json.write_text(json.dumps([{
                "point_cloud": "pcd/frame_000001.pcd",
                "recorded_at": "2026-09-27T12:00:00",
                "camera_frame": None,
            }]), encoding="utf-8")

            entries = playback_module.load_recording_entries(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].point_cloud.resolve(), pcd_file.resolve())


if __name__ == "__main__":
    unittest.main()
