"""Validated radar-to-camera coordinate transformation and projection."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RadarCameraTransformer:
    def __init__(self, intrinsic: Any, distortion: Any, extrinsic: Any):
        self.set_intrinsic(intrinsic, distortion)
        self.set_extrinsic_matrix(extrinsic)

    @classmethod
    def from_json(cls, path: str | Path):
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{source}: expected a JSON object")
        intrinsic = cls._first(data, ("intrinsic", "intrinsic_matrix", "camera_matrix"))
        distortion = cls._first(data, (
            "distortion", "distortion_coefficients", "dist_coefficients", "dist_coeffs",
        ))
        extrinsic = cls._first(data, (
            "extrinsic", "transform_radar_to_camera_4x4", "radar_to_camera_matrix",
        ))
        if intrinsic is None or distortion is None or extrinsic is None:
            raise ValueError(f"{source}: intrinsic, distortion, and extrinsic are required")
        return cls(intrinsic, distortion, extrinsic)

    @staticmethod
    def _first(data: dict, names: tuple[str, ...]):
        return next((data[name] for name in names if data.get(name) is not None), None)

    @staticmethod
    def _points(value, widths: tuple[int, ...]) -> np.ndarray:
        points = np.asarray(value, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] not in widths or not np.isfinite(points).all():
            raise ValueError(f"points must be finite rows with {widths} columns")
        return points

    def set_intrinsic(self, intrinsic: Any, distortion: Any) -> None:
        matrix = np.asarray(intrinsic, dtype=np.float64)
        coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3) or coefficients.size < 4:
            raise ValueError("intrinsic must be 3x3 and distortion needs at least four values")
        if not np.isfinite(matrix).all() or not np.isfinite(coefficients).all():
            raise ValueError("camera coefficients must be finite")
        self.intrinsic = matrix.copy()
        self.distortion = coefficients.copy()

    def set_extrinsic_matrix(self, matrix: Any) -> None:
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape == (3, 4):
            transform = np.vstack((transform, (0.0, 0.0, 0.0, 1.0)))
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("extrinsic must be a finite 4x4 or 3x4 matrix")
        left, _, right = np.linalg.svd(transform[:3, :3])
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        self.extrinsic = np.eye(4)
        self.extrinsic[:3, :3] = rotation
        self.extrinsic[:3, 3] = transform[:3, 3]
        self.rotation_matrix = rotation
        self.rotation_vector, _ = cv2.Rodrigues(rotation)
        self.translation_vector = transform[:3, 3].reshape(3, 1)

    def radar_to_camera(self, radar_points: Any) -> np.ndarray:
        points = self._points(radar_points, (3,))
        return (self.extrinsic @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]

    def radar_to_image(self, radar_points: Any, *, distorted: bool = True, z: float = 0.0) -> np.ndarray:
        points = self._points(radar_points, (2, 3))
        if points.shape[1] == 2:
            points = np.column_stack((points, np.full(len(points), float(z))))
        pixels, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3), self.rotation_vector, self.translation_vector,
            self.intrinsic, self.distortion if distorted else None,
        )
        return pixels.reshape(-1, 2)
