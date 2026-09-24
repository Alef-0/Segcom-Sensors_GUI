"""Low-latency radar group B overlay on camera frames."""

import math
from pathlib import Path
import queue
import time
from typing import Any

import cv2 as cv
import numpy as np

from processing.visualization.radar_camera_transformer import RadarCameraTransformer

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


class RadarCameraOverlay:
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
