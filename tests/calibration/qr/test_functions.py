"""Focused unit tests for QR payload, grid, decode, and contrast helpers."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration.qr import (
    DETECTION_BATCH_SIZE,
    GRID_LAYOUTS,
    QUIET_ZONE_MODULES,
    cell_index_for,
    decode_qrs_batch,
    decode_qrs_with_quadrant_retries,
    detect_contrast_cells,
    grid_bounds,
    grid_cell_names,
    grid_positions,
    grid_shape,
    order_by_quadrant,
    qr_matrix,
    timestamp_payload,
)
from tests.calibration.support import (
    FakeReader,
    image_fixture,
    recording_frame_fixture,
)


class TestQRFunctions(unittest.TestCase):
    def test_contrast_detection_finds_bright_content_in_an_otherwise_dark_cell(self):
        image = image_fixture("bright_qr_cell_0.png")

        detections = detect_contrast_cells(image, 4)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["cell"], 0)
        self.assertIsNone(detections[0]["raw"])
        self.assertTrue(np.allclose(detections[0]["bbox"], [12, 12, 37, 37]))

    def test_contrast_detection_ignores_empty_and_low_contrast_cells(self):
        for filename in ("empty_screen.png", "low_contrast_cell.png"):
            with self.subTest(filename=filename):
                image = image_fixture(filename)
                self.assertEqual(detect_contrast_cells(image, 4), [])

    def test_contrast_detection_finds_content_in_saved_recording_frames(self):
        # Restrict the measurement to the laptop panel; the camera also sees a
        # bright room around the display in these unmodified recordings.
        panel_bounds = (28, 1075, 175, 1832)
        top, bottom, left, right = panel_bounds
        for filename in ("camera_000019.jpg", "camera_000020.jpg"):
            with self.subTest(filename=filename):
                frame = recording_frame_fixture(filename)
                panel = frame[top:bottom, left:right]
                self.assertGreater(len(detect_contrast_cells(panel, 4)), 0)

    def test_payload_and_reduced_quiet_zone(self):
        self.assertEqual(timestamp_payload(12_345_678_900_000), "000012345678")
        matrix = qr_matrix("000012345678")
        self.assertEqual(QUIET_ZONE_MODULES, 2)
        self.assertFalse(matrix[:QUIET_ZONE_MODULES].any())
        self.assertFalse(matrix[:, :QUIET_ZONE_MODULES].any())

    def test_fixed_qr_masks_are_supported_and_validated(self):
        automatic = qr_matrix("000012345678")
        fixed = qr_matrix("000012345678", mask_pattern=0)

        self.assertEqual(fixed.shape, automatic.shape)
        self.assertFalse(fixed[:QUIET_ZONE_MODULES].any())
        with self.assertRaisesRegex(ValueError, "mask pattern"):
            qr_matrix("000012345678", mask_pattern=8)

    def test_every_fixed_qr_mask_decodes_at_module_aligned_scale(self):
        detector = cv2.QRCodeDetector()
        for mask_pattern in range(8):
            with self.subTest(mask_pattern=mask_pattern):
                matrix = qr_matrix("000012345678", mask_pattern=mask_pattern)
                image = np.repeat(
                    np.repeat(255 - matrix * 255, 12, axis=0),
                    12,
                    axis=1,
                )
                decoded, points, _ = detector.detectAndDecode(image)
                self.assertEqual(decoded, "000012345678")
                self.assertIsNotNone(points)

    def test_bounding_boxes_are_ordered_clockwise_by_sector(self):
        detections = [
            {"raw": "3", "bbox": np.array([10, 60, 30, 80]), "center": (20, 70), "confidence": 1},
            {"raw": "1", "bbox": np.array([10, 10, 30, 30]), "center": (20, 20), "confidence": 1},
            {"raw": "4", "bbox": np.array([60, 60, 80, 80]), "center": (70, 70), "confidence": 1},
            {"raw": "2", "bbox": np.array([60, 10, 80, 30]), "center": (70, 20), "confidence": 1},
        ]
        ordered = order_by_quadrant(detections, (100, 100))
        self.assertEqual([item["raw"] for item in ordered], ["1", "2", "4", "3"])
        self.assertEqual([item["quadrant"] for item in ordered], [0, 1, 2, 3])

    def test_missing_quadrant_is_retried_with_qreader(self):
        class RetryReader:
            def __init__(self):
                self.calls = []

            def detect_and_decode(self, image, return_detections=False, is_bgr=False):
                self.calls.append(image.shape[:2])
                if image.shape[:2] == (100, 100):
                    decoded = ("1", "2", "3")
                    boxes = ([10, 10, 30, 30], [60, 10, 80, 30], [60, 60, 80, 80])
                else:
                    decoded = ("4",) if len(self.calls) == 2 else ()
                    boxes = ([10, 10, 30, 30],) if decoded else ()
                detections = tuple({
                    "bbox_xyxy": np.asarray(box, dtype=np.float32),
                    "confidence": 0.9,
                } for box in boxes)
                return (decoded, detections) if return_detections else decoded

        reader = RetryReader()
        detections = decode_qrs_with_quadrant_retries(
            reader, np.zeros((100, 100, 3), np.uint8)
        )
        ordered = order_by_quadrant(detections, (100, 100))
        self.assertEqual([item["raw"] for item in ordered], ["1", "2", "3", "4"])
        self.assertEqual(reader.calls, [(100, 100), (50, 50)])

    def test_supported_grid_shapes_and_snake_order(self):
        self.assertEqual(tuple(GRID_LAYOUTS), (4, 6, 8, 9, 10, 12))
        self.assertEqual(grid_shape(6), (2, 3))
        self.assertEqual(
            grid_positions(6),
            ((0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)),
        )
        self.assertEqual(cell_index_for((90, 75), (120, 100), 6), 3)
        self.assertEqual(grid_cell_names(4)[2], "Bottom-right")

    def test_batch_detection_uses_one_model_prediction_and_keeps_image_order(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                return [f"prediction-{index}" for index in range(len(kwargs["source"]))]

        class BatchReader:
            def __init__(self):
                self.detector = type("Detector", (), {
                    "model": FakeModel(),
                    "_conf_th": 0.3,
                    "_nms_iou": 0.3,
                })()

            @staticmethod
            def decode(image, detection_result):
                return detection_result["raw"]

        reader = BatchReader()
        images = [np.full((20, 20, 3), index, np.uint8) for index in range(3)]

        with (
            patch("qrdet._prepare_input", side_effect=lambda source, is_bgr: source),
            patch(
                "qrdet._yolo_v8_results_to_dict",
                side_effect=lambda results, image: [{
                    "raw": results,
                    "bbox_xyxy": np.asarray((1, 2, 11, 12), dtype=float),
                    "confidence": 0.9,
                }],
            ),
        ):
            batches = decode_qrs_batch(reader, images)

        self.assertEqual(len(reader.detector.model.calls), 1)
        self.assertEqual(
            [[item["raw"] for item in batch] for batch in batches],
            [["prediction-0"], ["prediction-1"], ["prediction-2"]],
        )
