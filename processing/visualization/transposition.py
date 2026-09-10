from __future__ import annotations

import math
from pathlib import Path
import queue
import time
from typing import Any

import cv2 as cv
import numpy as np

from processing.visualization.radar_camera_transformer import (
    RadarCameraTransformer,
)


DEFAULT_MATRIX_PATH = Path(__file__).resolve().parents[2] / "camera_matrixes.json"
RADAR_GROUP_B = 2
OVERLAY_MAX_AGE_SECONDS = 0.5


def clear_latest(channel) -> None:
    if channel is None:
        return
    while True:
        try:
            channel.get_nowait()
        except queue.Empty:
            return


def put_latest(channel, payload: dict[str, Any]) -> None:
    """Replace an unread payload instead of adding display latency."""
    if channel is None:
        return
    try:
        channel.put_nowait(payload)
        return
    except queue.Full:
        pass
    try:
        channel.get_nowait()
    except queue.Empty:
        pass
    try:
        channel.put_nowait(payload)
    except queue.Full:
        pass


def get_latest(channel, previous=None):
    latest = previous
    if channel is None:
        return latest
    while True:
        try:
            latest = channel.get_nowait()
        except queue.Empty:
            return latest


def transposition_payload(
    points,
    colors,
    *,
    frame_type: str,
    recorded_at,
    distance_cutoff: float,
) -> dict[str, Any]:
    projected_points = []
    for point, color in zip(points, colors):
        forward = point.dist_long
        lateral = point.dist_latitude
        try:
            coordinates_are_finite = math.isfinite(forward) and math.isfinite(lateral)
        except TypeError:
            coordinates_are_finite = False
        if not coordinates_are_finite:
            continue
        if math.hypot(forward, lateral) > float(distance_cutoff):
            continue
        projected_points.append({
            "radar_xyz": (float(forward), float(lateral), 0.0),
            "color": tuple(int(component) for component in color),
        })
    return {
        "group": RADAR_GROUP_B,
        "frame_type": str(frame_type),
        "recorded_at": recorded_at.isoformat(),
        "published_monotonic": time.monotonic(),
        "points": tuple(projected_points),
    }


class RadarCameraOverlay:
    def __init__(
        self,
        transformer: RadarCameraTransformer,
        *,
        max_age_seconds: float = OVERLAY_MAX_AGE_SECONDS,
    ) -> None:
        self.transformer = transformer
        self.max_age_seconds = float(max_age_seconds)

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_MATRIX_PATH):
        return cls(RadarCameraTransformer.from_json(path))

    def draw(
        self,
        frame: np.ndarray,
        payload: dict[str, Any] | None,
        *,
        source_size: tuple[int, int],
        now_monotonic: float | None = None,
    ) -> np.ndarray:
        if payload is None or payload.get("group") != RADAR_GROUP_B:
            return frame
        published = float(payload.get("published_monotonic", float("-inf")))
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if now - published > self.max_age_seconds:
            return frame

        entries = tuple(payload.get("points", ()))
        if not entries:
            return frame
        radar_points = np.asarray(
            [entry["radar_xyz"] for entry in entries],
            dtype=np.float64,
        )
        camera_points = self.transformer.radar_to_camera(radar_points)
        visible = camera_points[:, 2] > 0.0
        if not np.any(visible):
            return frame

        pixels = self.transformer.radar_to_image(
            radar_points[visible],
            distorted=True,
        )
        visible_entries = [
            entry for entry, keep in zip(entries, visible) if keep
        ]
        source_width, source_height = source_size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("The camera source dimensions must be positive")
        height, width = frame.shape[:2]
        scale_x = width / float(source_width)
        scale_y = height / float(source_height)
        result = frame.copy()
        for pixel, entry in zip(pixels, visible_entries):
            x = int(round(float(pixel[0]) * scale_x))
            y = int(round(float(pixel[1]) * scale_y))
            if not (0 <= x < width and 0 <= y < height):
                continue
            color = tuple(int(component) for component in entry["color"])
            cv.circle(result, (x, y), 6, (255, 255, 255), 2)
            cv.circle(result, (x, y), 4, color, -1)
        return result
