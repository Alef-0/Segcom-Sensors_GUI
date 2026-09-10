from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RadarCameraTransformer:
    """Project radar coordinates through calibrated camera coefficients."""

    def __init__(
        self,
        intrinsic: Any,
        distortion: Any,
        extrinsic: Any,
    ) -> None:
        self.set_intrinsic(intrinsic, distortion)
        self.set_extrinsic_matrix(extrinsic)

    @classmethod
    def from_json(cls, path: str | Path) -> "RadarCameraTransformer":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{source}: expected a JSON object")

        intrinsic = cls._first(
            data,
            ("intrinsic", "intrinsic_matrix", "camera_matrix"),
        )
        distortion = cls._first(
            data,
            (
                "distortion",
                "distortion_coefficients",
                "dist_coefficients",
                "dist_coeffs",
            ),
        )
        extrinsic = cls._first(
            data,
            (
                "extrinsic",
                "transform_radar_to_camera_4x4",
                "radar_to_camera_matrix",
            ),
        )
        if intrinsic is None or distortion is None or extrinsic is None:
            raise KeyError(
                f"{source}: intrinsic, distortion, and extrinsic coefficients are required"
            )
        return cls(intrinsic, distortion, extrinsic)

    @staticmethod
    def _first(mapping: dict[str, Any], names: tuple[str, ...]) -> Any | None:
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
        return None

    @staticmethod
    def _points(points: Any, columns: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] not in columns:
            expected = " or ".join(str(value) for value in columns)
            raise ValueError(f"points must have {expected} columns, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("points must contain finite values")
        return array

    def set_intrinsic(self, intrinsic: Any, distortion: Any) -> None:
        matrix = np.asarray(intrinsic, dtype=np.float64)
        coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3):
            raise ValueError(f"intrinsic must be 3x3, got {matrix.shape}")
        if coefficients.size < 4:
            raise ValueError("distortion must contain at least four coefficients")
        if not np.isfinite(matrix).all() or not np.isfinite(coefficients).all():
            raise ValueError("intrinsic and distortion must contain finite values")
        self.intrinsic = matrix.copy()
        self.distortion = coefficients.copy()

    def set_extrinsic_matrix(self, matrix: Any) -> None:
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape == (3, 4):
            transform = np.vstack([transform, [0.0, 0.0, 0.0, 1.0]])
        if transform.shape != (4, 4):
            raise ValueError(
                f"extrinsic_matrix must be 4x4 or 3x4, got {transform.shape}"
            )
        if not np.isfinite(transform).all():
            raise ValueError("extrinsic_matrix must contain finite values")

        # Keep the calibrated translation while normalizing small numerical
        # errors in the rotation block, as in the supplied reference module.
        left, _, right = np.linalg.svd(transform[:3, :3])
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right

        self.extrinsic = np.eye(4, dtype=np.float64)
        self.extrinsic[:3, :3] = rotation
        self.extrinsic[:3, 3] = transform[:3, 3]
        self.rotation_matrix = rotation
        self.rotation_vector, _ = cv2.Rodrigues(rotation)
        self.translation_vector = transform[:3, 3].reshape(3, 1)

    def radar_to_camera(self, radar_points: Any) -> np.ndarray:
        points = self._points(radar_points, (3,))
        homogeneous = np.column_stack([points, np.ones(len(points))])
        return (self.extrinsic @ homogeneous.T).T[:, :3]

    def radar_to_image(
        self,
        radar_points: Any,
        *,
        distorted: bool = True,
        z: float = 0.0,
    ) -> np.ndarray:
        points = self._points(radar_points, (2, 3))
        if points.shape[1] == 2:
            points = np.column_stack([points, np.full(len(points), float(z))])
        distortion = self.distortion if distorted else None
        result, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3),
            self.rotation_vector,
            self.translation_vector,
            self.intrinsic,
            distortion,
        )
        return result.reshape(-1, 2).astype(np.float64)
