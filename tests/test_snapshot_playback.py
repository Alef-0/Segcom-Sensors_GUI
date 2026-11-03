from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2 as cv

import tests  # Ensure GUI mocks are loaded in headless test environments
import processing.playback.snapshot_playback as playback_module
from processing.visualization.graph_draw import Graph_radar
from menu_configurations import Configurations


class FakeConnection:
    def __init__(self, events):
        self.events = list(events)

    def poll(self, _timeout=None):
        return bool(self.events)

    def recv(self):
        return self.events.pop(0)


class FakePool:
    def __init__(self):
        self.items = []

    def put(self, item, timeout=None):
        self.items.append(item)


class FakeShutdownEvent:
    def __init__(self):
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True


class SnapshotPlaybackTests(unittest.TestCase):
    @staticmethod
    def _elements(layout):
        for row in layout:
            for element in row:
                yield element
                rows = getattr(element, "Rows", None)
                if rows:
                    yield from SnapshotPlaybackTests._elements(rows)

    def test_synced_only_checkbox_defaults_to_checked(self):
        elements = {
            element.Key: element
            for element in self._elements(Configurations._create_snapshot_layout())
            if getattr(element, "Key", None)
        }

        self.assertTrue(elements["snapshot_playback_synced_only"].InitialState)

    def test_synced_only_loader_omits_unpaired_entries(self):
        synced = SimpleNamespace(point_cloud=Path("frame.pcd"), camera_frame=Path("camera.jpg"))
        pcd_only = SimpleNamespace(point_cloud=Path("only.pcd"), camera_frame=None)
        image_only = SimpleNamespace(point_cloud=None, camera_frame=Path("only.jpg"))

        with patch.object(
            playback_module,
            "load_recording_entries",
            return_value=(pcd_only, synced, image_only),
        ):
            _, entries = playback_module._load_entries(".", synced_only=True)
            _, all_entries = playback_module._load_entries(".", synced_only=False)

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
                playback_module._load_entries(".", synced_only=True)

    def test_synced_only_warning_does_not_render_any_entry(self):
        pool = FakePool()
        controller = playback_module.SnapshotPlaybackController(
            FakeConnection([]),
            pool,
            FakeShutdownEvent(),
            {},
        )

        with (
            patch.object(
                playback_module,
                "_load_entries",
                side_effect=ValueError("No synced image + PCD pairs were found"),
            ),
            patch.object(controller, "_render") as render,
            patch.object(controller, "_close_windows"),
        ):
            controller._play({"folder": ".", "synced_only": True})

        render.assert_not_called()
        self.assertEqual(pool.items[0][0], "snapshot_playback_error")

    def test_graph_resolution_and_range_share_one_apply_button(self):
        layout = Configurations._create_general_configurations_layout()
        elements = {
            element.Key: element
            for element in self._elements(layout)
            if getattr(element, "Key", None)
        }

        self.assertIn("graph_settings_apply", elements)
        self.assertNotIn("graph_resolution_apply", elements)
        self.assertNotIn("graph_range_apply", elements)

    def test_playback_starts_playing_and_processes_clicks_while_paused(self):
        connection = FakeConnection([
            ("snapshot_playback_pause", None),
            ("snapshot_playback_stop", None),
        ])
        pool = FakePool()
        controller = playback_module.SnapshotPlaybackController(
            connection,
            pool,
            FakeShutdownEvent(),
            {},
        )
        entry = SimpleNamespace(recorded_at=datetime.now(timezone.utc))

        with (
            patch.object(playback_module, "_load_entries", return_value=(Path("."), [entry])),
            patch.object(controller, "_render"),
            patch.object(controller, "_close_windows"),
            patch.object(controller, "_process_window_events") as process_events,
        ):
            controller._play({"folder": "."})

        first_message, first_state = pool.items[0]
        self.assertEqual(first_message, "snapshot_playback_state")
        self.assertTrue(first_state["active"])
        self.assertFalse(first_state["paused"])
        self.assertGreaterEqual(process_events.call_count, 2)

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

    def test_close_windows_uses_process_local_cleanup(self):
        with (
            patch.object(playback_module.cv, "destroyAllWindows") as destroy_all,
            patch.object(playback_module.cv, "waitKey") as wait_key,
        ):
            playback_module.SnapshotPlaybackController._close_windows()

        destroy_all.assert_called_once_with()
        wait_key.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
