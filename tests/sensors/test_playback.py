from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2 as cv
import numpy as np

from menu_layout import build_graph_tab, build_record_tab, build_video_tab
from sensors.filter import Graph_radar
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
            self.assertEqual(entries[0].camera, "B")

            # With explicit Camera field
            rec_json.write_text(json.dumps([{
                "point_cloud": "pcd/frame_000001.pcd",
                "recorded_at": "2026-09-27T12:00:00",
                "camera_frame": None,
                "Camera": "A",
            }]), encoding="utf-8")
            entries = playback_module.load_recording_entries(root)
            self.assertEqual(entries[0].camera, "A")

    def test_record_tab_folder_defaults_to_relative_recordings(self):
        elements = {
            element.Key: element
            for element in self._elements(build_record_tab())
            if getattr(element, "Key", None)
        }
        self.assertEqual(elements["record_folder"].DefaultText, "./recordings")
        self.assertEqual(elements["playback_folder"].DefaultText, "./recordings")

    def test_format_relative_path(self):
        from menu_functions import format_relative_path
        cwd = Path.cwd()
        self.assertEqual(format_relative_path(str(cwd)), "./")
        self.assertEqual(format_relative_path(str(cwd / "recordings")), "./recordings")
        self.assertEqual(format_relative_path(str(cwd / "recordings" / "session_1")), "./recordings/session_1")
        self.assertEqual(format_relative_path("./recordings"), "./recordings")
        self.assertEqual(format_relative_path("recordings"), "./recordings")
        self.assertEqual(format_relative_path("/non/existent/path/outside/workspace"), "/non/existent/path/outside/workspace")
        self.assertEqual(format_relative_path(""), "")

    def test_folder_gui_events_convert_to_relative(self):
        from menu_functions import handle_gui_event, RuntimeState
        cwd = Path.cwd()
        abs_folder = str(cwd / "recordings" / "test_session")
        mock_elem = SimpleNamespace(update=unittest.mock.MagicMock())
        mock_win = unittest.mock.MagicMock()
        mock_win.key_dict = {"playback_folder": True, "record_folder": True}
        mock_win.__getitem__.return_value = mock_elem
        controls = SimpleNamespace(window=mock_win)
        values = {"playback_folder": abs_folder}
        handle_gui_event("playback_folder", values, controls, RuntimeState(), None, None)
        mock_elem.update.assert_called_with("./recordings/test_session")
        self.assertEqual(values["playback_folder"], "./recordings/test_session")

    def test_transposition_persists_on_playback_start(self):
        import tempfile
        from menu_functions import handle_gui_event, RuntimeState
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pcd_dir = tmp_path / "pcd"
            pcd_dir.mkdir()
            (pcd_dir / "frame.pcd").write_text("dummy")
            (tmp_path / "recording.json").write_text("[]")
            mock_win = unittest.mock.MagicMock()
            mock_win.key_dict = {}
            controls = SimpleNamespace(
                playback=False,
                recording=False,
                recording_pending=False,
                connected_radar=False,
                connected_cam=False,
                transposition=True,
                show_playback_error=unittest.mock.MagicMock(),
                validate_playback_resolution=lambda values: (1280, 720),
                set_playback_pending=unittest.mock.MagicMock(),
                window=mock_win,
            )
            runtime = RuntimeState()
            pipes = SimpleNamespace(
                cam=SimpleNamespace(send=unittest.mock.MagicMock()),
                radar=SimpleNamespace(send=unittest.mock.MagicMock()),
                playback=SimpleNamespace(send=unittest.mock.MagicMock()),
            )
            shutdown = SimpleNamespace(set=unittest.mock.MagicMock())
            values = {"playback_folder": str(tmp_path), "record_folder": str(tmp_path), "playback_synced_only": False}
            handle_gui_event("playback_toggle", values, controls, runtime, pipes, shutdown)
            self.assertTrue(controls.transposition)
            pipes.playback.send.assert_called_once()
            ev, payload = pipes.playback.send.call_args[0][0]
            self.assertEqual(ev, "playback_start")
            self.assertTrue(payload["transposition"])

    def test_load_recording_entries_from_file_path(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pcd_dir = root / "pcd"
            pcd_dir.mkdir()
            (pcd_dir / "frame_000001.pcd").write_text("test", encoding="utf-8")
            rec_json = root / "recording.json"
            rec_json.write_text(json.dumps([{
                "point_cloud": "pcd/frame_000001.pcd",
                "recorded_at": "2026-09-27T12:00:00",
                "camera_frame": None,
                "Camera": "B",
            }]), encoding="utf-8")
            entries = playback_module.load_recording_entries(rec_json)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].camera, "B")

    def test_graph_tab_contains_transposition_and_centered_inputs(self):
        layout = build_graph_tab()
        elements = {
            element.Key: element
            for element in self._elements(layout)
            if getattr(element, "Key", None)
        }
        self.assertIn("transposition_toggle", elements)
        self.assertIn("transposition_status", elements)
        for key in ("point_cutoff", "graph_width", "graph_height", "graph_x_range", "graph_y_range"):
            self.assertEqual(elements[key].Justification, "center")

    def test_video_tab_centered_inputs(self):
        layout = build_video_tab()
        elements = {
            element.Key: element
            for element in self._elements(layout)
            if getattr(element, "Key", None)
        }
        for key in ("playback_width", "playback_height", "camera_pipeline_latency", "camera_latency_adjustment"):
            self.assertEqual(elements[key].Justification, "center")


    def test_playback_transposition_maps_camera_b_points_to_image(self):
        controller = playback_module.PlaybackController(
            connection=SimpleNamespace(poll=lambda *_: False),
            pool=SimpleNamespace(put=lambda *_: None),
            shutdown_event=SimpleNamespace(is_set=lambda: False, set=lambda: None),
            initial_values={"transposition": True},
        )
        fake_overlay = SimpleNamespace(draw=unittest.mock.MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8)))
        controller.transposition_overlay = fake_overlay
        controller.transposition_active = True

        entry_b = playback_module.PlaybackEntry(
            point_cloud=Path("frame.pcd"),
            recorded_at=playback_module.datetime.now(),
            camera_frame=Path("camera.jpg"),
            camera="B",
        )
        controller.entries = [entry_b]
        controller.index = 0

        dummy_reader = SimpleNamespace(
            frame_type="cluster",
            clusters=(SimpleNamespace(dist_long=5.0, dist_latitude=2.0, rcs=-5.0, dynamic_property=0, pdh=2, ambiguity_state=3, invalid_flag=0),),
        )

        with patch("sensors.playback.PointCloudReader", return_value=dummy_reader), \
             patch("sensors.playback.cv.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch("sensors.playback.cv.imshow"), \
             patch("sensors.playback.cv.waitKey"):
            controller._render()

        fake_overlay.draw.assert_called_once()
        _, kwargs = fake_overlay.draw.call_args
        self.assertEqual(kwargs["source_size"], (100, 100))

    def test_playback_transposition_does_not_map_camera_a(self):
        controller = playback_module.PlaybackController(
            connection=SimpleNamespace(poll=lambda *_: False),
            pool=SimpleNamespace(put=lambda *_: None),
            shutdown_event=SimpleNamespace(is_set=lambda: False, set=lambda: None),
            initial_values={"transposition": True},
        )
        fake_overlay = SimpleNamespace(draw=unittest.mock.MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8)))
        controller.transposition_overlay = fake_overlay
        controller.transposition_active = True

        entry_a = playback_module.PlaybackEntry(
            point_cloud=Path("frame.pcd"),
            recorded_at=playback_module.datetime.now(),
            camera_frame=Path("camera.jpg"),
            camera="A",
        )
        controller.entries = [entry_a]
        controller.index = 0

        dummy_reader = SimpleNamespace(
            frame_type="cluster",
            clusters=(SimpleNamespace(dist_long=5.0, dist_latitude=2.0, rcs=-5.0, dynamic_property=0, pdh=2, ambiguity_state=3, invalid_flag=0),),
        )

        with patch("sensors.playback.PointCloudReader", return_value=dummy_reader), \
             patch("sensors.playback.cv.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch("sensors.playback.cv.imshow"), \
             patch("sensors.playback.cv.waitKey"):
            controller._render()

        fake_overlay.draw.assert_not_called()


if __name__ == "__main__":
    unittest.main()
