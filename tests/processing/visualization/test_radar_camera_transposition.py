from datetime import datetime, timezone
from pathlib import Path
import queue
from types import SimpleNamespace
import unittest

import numpy as np

from processing.visualization.radar_camera_transformer import (
    RadarCameraTransformer,
)
from processing.visualization.transposition import (
    RADAR_GROUP_B,
    RadarCameraOverlay,
    get_latest,
    put_latest,
    transposition_payload,
)


def transformer_with_depth(depth):
    intrinsic = [
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ]
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[2, 3] = depth
    return RadarCameraTransformer(intrinsic, [0, 0, 0, 0, 0], extrinsic)


class RadarCameraTranspositionTests(unittest.TestCase):
    def test_newest_payload_replaces_unread_payload(self):
        channel = queue.Queue(maxsize=1)
        put_latest(channel, {"sequence": 1})
        put_latest(channel, {"sequence": 2})

        self.assertEqual(get_latest(channel), {"sequence": 2})

    def test_payload_uses_forward_lateral_order_and_distance_cutoff(self):
        points = (
            SimpleNamespace(dist_long=3.0, dist_latitude=-1.0),
            SimpleNamespace(dist_long=20.0, dist_latitude=0.0),
        )
        payload = transposition_payload(
            points,
            ((1, 2, 3), (4, 5, 6)),
            frame_type="object",
            recorded_at=datetime.now(timezone.utc),
            distance_cutoff=15.0,
        )

        self.assertEqual(payload["group"], RADAR_GROUP_B)
        self.assertEqual(payload["points"], ({
            "radar_xyz": (3.0, -1.0, 0.0),
            "color": (1, 2, 3),
        },))

    def test_overlay_projects_and_scales_a_visible_point(self):
        overlay = RadarCameraOverlay(transformer_with_depth(10.0))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        payload = {
            "group": RADAR_GROUP_B,
            "published_monotonic": 10.0,
            "points": ({
                "radar_xyz": (1.0, 2.0, 0.0),
                "color": (10, 20, 30),
            },),
        }

        result = overlay.draw(
            frame,
            payload,
            source_size=(200, 200),
            now_monotonic=10.0,
        )

        self.assertEqual(tuple(result[35, 30]), (10, 20, 30))
        self.assertFalse(np.any(frame))

    def test_overlay_rejects_behind_camera_and_stale_points(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        payload = {
            "group": RADAR_GROUP_B,
            "published_monotonic": 10.0,
            "points": ({
                "radar_xyz": (1.0, 2.0, 0.0),
                "color": (10, 20, 30),
            },),
        }

        behind = RadarCameraOverlay(transformer_with_depth(-10.0)).draw(
            frame,
            payload,
            source_size=(100, 100),
            now_monotonic=10.0,
        )
        stale = RadarCameraOverlay(transformer_with_depth(10.0)).draw(
            frame,
            payload,
            source_size=(100, 100),
            now_monotonic=11.0,
        )

        self.assertIs(behind, frame)
        self.assertIs(stale, frame)

    def test_repository_camera_matrix_loads(self):
        matrix_path = Path(__file__).resolve().parents[3] / "camera_matrixes.json"
        transformer = RadarCameraTransformer.from_json(matrix_path)

        self.assertEqual(transformer.intrinsic.shape, (3, 3))
        self.assertEqual(transformer.extrinsic.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
