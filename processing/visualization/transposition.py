from __future__ import annotations

import math
from pathlib import Path
import queue
import time
from typing import Any

import cv2 as cv
import numpy as np

from processing.visualization.multi_radar_fusion import (
    DEFAULT_CAM_B_MATRIX_PATH,
    DEFAULT_EXTRINSICS_PATH,
    MultiSensorRegistry,
)
from processing.visualization.radar_camera_transformer import (
    RadarCameraTransformer,
)


DEFAULT_MATRIX_PATH = DEFAULT_CAM_B_MATRIX_PATH
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
    group: int = RADAR_GROUP_B,
) -> dict[str, Any]:
    """Build local-pod radar projection payload in [dist_long, dist_latitude, 0.0] order."""
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
        "group": int(group),
        "radar_source": "local",
        "frame_type": str(frame_type),
        "recorded_at": recorded_at.isoformat(),
        "published_monotonic": time.monotonic(),
        "points": tuple(projected_points),
    }


def global_transposition_payload(
    snapshot,
    *,
    recorded_at,
    frame_type: str = "global",
) -> dict[str, Any]:
    """Build unified-frame projection payload from a GlobalRadarSnapshot in Radar B 3D order."""
    projected_points = []
    for idx in range(len(snapshot.unified_xy)):
        lat_b = float(snapshot.unified_xy[idx, 0])
        long_b = float(snapshot.unified_xy[idx, 1])
        ch = snapshot.channels[idx]
        pose = snapshot.poses.get(ch)
        z_b = float(pose.offset_z_m) if pose is not None else 0.0
        color = snapshot.colors[idx]
        projected_points.append({
            "radar_xyz": (long_b, lat_b, z_b),
            "color": tuple(int(component) for component in color),
        })
    return {
        "group": 0,  # 0 denotes global fused map usable by any camera A/B/C
        "radar_source": "global",
        "frame_type": str(frame_type),
        "recorded_at": recorded_at.isoformat(),
        "published_monotonic": time.monotonic(),
        "points": tuple(projected_points),
    }


class RadarCameraOverlay:
    """Projects a radar payload onto a camera frame using a single RadarCameraTransformer."""

    def __init__(
        self,
        transformer: RadarCameraTransformer,
        *,
        max_age_seconds: float = OVERLAY_MAX_AGE_SECONDS,
        expected_group: int | None = RADAR_GROUP_B,
    ) -> None:
        self.transformer = transformer
        self.max_age_seconds = float(max_age_seconds)
        self.expected_group = expected_group

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
        if payload is None:
            return frame
        payload_group = payload.get("group")
        if (
            self.expected_group is not None
            and payload_group != 0
            and payload_group != self.expected_group
        ):
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


class MultiGroupCameraOverlay:
    """Manages per-camera (A, B, C) local and global radar overlays via MultiSensorRegistry."""

    def __init__(
        self,
        extrinsics_path: str | Path = DEFAULT_EXTRINSICS_PATH,
        cam_b_path: str | Path = DEFAULT_MATRIX_PATH,
        *,
        max_age_seconds: float = OVERLAY_MAX_AGE_SECONDS,
    ) -> None:
        self.registry = MultiSensorRegistry(extrinsics_path, cam_b_path)
        self.max_age_seconds = float(max_age_seconds)
        self._cache: dict[tuple[int, str], tuple[RadarCameraOverlay, dict[str, Any]]] = {}
        self.reload()

    def reload(self) -> None:
        """Reload JSON extrinsics and rebuild cached per-channel transformers."""
        self.registry.reload()
        self._cache.clear()
        for ch in (1, 2, 3):
            for src in ("local", "global"):
                transformer, info = self.registry.get_camera_transformer(ch, radar_source=src)
                expected = None if src == "global" else ch
                self._cache[(ch, src)] = (
                    RadarCameraOverlay(
                        transformer,
                        max_age_seconds=self.max_age_seconds,
                        expected_group=expected,
                    ),
                    info,
                )

    def status_for_channel(self, channel: int, radar_source: str = "local") -> str:
        _, info = self._cache.get((int(channel), radar_source), (None, {}))
        pod = {1: "A", 2: "B", 3: "C"}.get(int(channel), str(channel))
        src_lbl = "GLOBAL MAP" if radar_source == "global" else f"RADAR {pod}"
        if int(channel) == 2:
            return f"ON · CAM B + {src_lbl} · calibrated reference"
        if info.get("calibrated"):
            rmse = info.get("rmse_px")
            rmse_txt = f"RMSE {rmse:.2f} px" if isinstance(rmse, (int, float)) else "calibrated"
            return f"ON · CAM {pod} + {src_lbl} · {rmse_txt}"
        return f"ON · CAM {pod} + {src_lbl} · using shift prior (uncalibrated)"

    def draw(
        self,
        channel: int,
        frame: np.ndarray,
        payload: dict[str, Any] | None,
        *,
        source_size: tuple[int, int],
        now_monotonic: float | None = None,
    ) -> np.ndarray:
        if payload is None:
            return frame
        src = str(payload.get("radar_source", "local"))
        entry = self._cache.get((int(channel), src))
        if entry is None:
            return frame
        overlay, _ = entry
        return overlay.draw(
            frame,
            payload,
            source_size=source_size,
            now_monotonic=now_monotonic,
        )
