"""Radar-to-camera coordinate transformation, projection, and real-time image overlay."""

import json
import math
from pathlib import Path
import queue
import time
from typing import Any

import cv2 as cv
import numpy as np

DEFAULT_MATRIX_PATH = Path(__file__).resolve().parents[2] / "camera_matrixes.json"
RADAR_GROUP_B = 2
OVERLAY_MAX_AGE_SECONDS = 0.5


def clear_latest(channel) -> None:
    """Drain all queued items from an IPC channel."""
    if channel is None:
        return
    while True:
        try:
            channel.get_nowait()
        except queue.Empty:
            return


def put_latest(channel, payload: dict[str, Any]) -> None:
    """Enqueue payload replacing older unread payload when channel is full."""
    if channel is None:
        return
    try:
        channel.put_nowait(payload)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(payload)
        except queue.Full:
            pass


def get_latest(channel, previous=None):
    """Retrieve the newest payload available from an IPC channel."""
    latest = previous
    if channel is not None:
        while True:
            try:
                latest = channel.get_nowait()
            except queue.Empty:
                break
    return latest


def transposition_payload(points, colors, *, frame_type: str, recorded_at,
                          distance_cutoff: float) -> dict[str, Any]:
    """Serialize filtered radar points within distance cutoff for camera overlay."""
    entries = []
    for point, color in zip(points, colors):
        forward, lateral = point.dist_long, point.dist_latitude
        try:
            if not math.isfinite(forward) or not math.isfinite(lateral):
                continue
        except TypeError:
            continue
        if math.hypot(forward, lateral) <= float(distance_cutoff):
            entries.append({
                "radar_xyz": (float(forward), float(lateral), 0.0),
                "color": tuple(int(component) for component in color),
            })
    return {"group": RADAR_GROUP_B, "frame_type": str(frame_type),
            "recorded_at": recorded_at.isoformat(), "published_monotonic": time.monotonic(),
            "points": tuple(entries)}


class RadarCameraTransformer:
    """Transforms 3D radar coordinates to camera space and 2D pixel coordinates."""

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
        self.rotation_vector, _ = cv.Rodrigues(rotation)
        self.translation_vector = transform[:3, 3].reshape(3, 1)

    def radar_to_camera(self, radar_points: Any) -> np.ndarray:
        points = self._points(radar_points, (3,))
        return (self.extrinsic @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]

    def radar_to_image(self, radar_points: Any, *, distorted: bool = True, z: float = 0.0) -> np.ndarray:
        points = self._points(radar_points, (2, 3))
        if points.shape[1] == 2:
            points = np.column_stack((points, np.full(len(points), float(z))))
        pixels, _ = cv.projectPoints(
            points.reshape(-1, 1, 3), self.rotation_vector, self.translation_vector,
            self.intrinsic, self.distortion if distorted else None,
        )
        return pixels.reshape(-1, 2)


class RadarCameraOverlay:
    """Renders projected radar detections as visual overlays on camera frames."""

    def __init__(self, transformer: RadarCameraTransformer,
                 *, max_age_seconds: float = OVERLAY_MAX_AGE_SECONDS):
        self.transformer = transformer
        self.max_age_seconds = float(max_age_seconds)

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_MATRIX_PATH):
        return cls(RadarCameraTransformer.from_json(path))

    def draw(self, frame: np.ndarray, payload: dict | None, *,
             source_size: tuple[int, int], now_monotonic: float | None = None) -> np.ndarray:
        if payload is None or payload.get("group") != RADAR_GROUP_B:
            return frame
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if now - float(payload.get("published_monotonic", float("-inf"))) > self.max_age_seconds:
            return frame
        entries = tuple(payload.get("points", ()))
        if not entries:
            return frame
        xyz = np.asarray([entry["radar_xyz"] for entry in entries], dtype=np.float64)
        camera = self.transformer.radar_to_camera(xyz)
        visible = camera[:, 2] > 0
        if not visible.any():
            return frame
        source_width, source_height = source_size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Camera source dimensions must be positive")
        pixels = self.transformer.radar_to_image(xyz[visible])
        chosen = [entry for entry, keep in zip(entries, visible) if keep]
        scale_x, scale_y = frame.shape[1] / source_width, frame.shape[0] / source_height
        result = frame.copy()
        for pixel, entry in zip(pixels, chosen):
            x, y = int(round(pixel[0] * scale_x)), int(round(pixel[1] * scale_y))
            if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                color = tuple(int(channel) for channel in entry["color"])
                cv.circle(result, (x, y), 6, (255, 255, 255), 2)
                cv.circle(result, (x, y), 4, color, -1)
        return result
