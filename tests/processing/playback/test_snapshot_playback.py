from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2 as cv

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

if __name__ == "__main__":
    unittest.main()
