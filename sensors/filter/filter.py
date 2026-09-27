"""Radar point and object filtering, schema options, and dynamic color assignment."""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from sensors.radar.connection_packages import (
    Clusters_messages,
    MISSING_QUALITY,
    Objects_messages,
    RadarObject,
    RadarPoint,
)

FILTER_PREFIX = "filter."
PDH_KEY = "filter.pdh.max"
RCS_KEY = "filter.rcs.min"


@dataclass(frozen=True)
class FilterOption:
    """Descriptor for UI filter toggle options."""

    field: str
    value: int
    label: str
    default: bool = False
    enabled: bool = True
    color: str | None = None

    @property
    def key(self) -> str:
        return f"{FILTER_PREFIX}{self.field}.{self.value}"


DYNAMIC_PROPERTY_OPTIONS = (
    FilterOption("dynamic_property", 0, "Moving", True, color="#FF0000"),
    FilterOption("dynamic_property", 1, "Stationary", True, color="#FF7B00"),
    FilterOption("dynamic_property", 2, "Oncoming", True, color="#FFE600"),
    FilterOption("dynamic_property", 3, "Stationary Candidate", color="#00FF00"),
    FilterOption("dynamic_property", 4, "Unknown", True, color="#0000FF"),
    FilterOption("dynamic_property", 5, "Crossing Stationary", True, color="#00FFFF"),
    FilterOption("dynamic_property", 6, "Crossing Moving", True, color="#8400FF"),
    FilterOption("dynamic_property", 7, "Stopped", True, color="#000000"),
)
AMBIGUITY_STATE_OPTIONS = (
    FilterOption("ambiguity_state", 1, "Ambiguous"),
    FilterOption("ambiguity_state", 2, "Staggered Ramp"),
    FilterOption("ambiguity_state", 3, "Unambiguous", True),
    FilterOption("ambiguity_state", 4, "Stationary Candidates", True),
)
_DEFAULT_INVALID = {0x00, 0x04, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0F, 0x10, 0x11}
_DISABLED_INVALID = {0x05, 0x0D}
INVALID_STATE_OPTIONS = tuple(
    FilterOption("invalid_state", value, f"0x{value:X}",
                 value in _DEFAULT_INVALID, value not in _DISABLED_INVALID)
    for value in range(0x12)
)
DYNAMIC_COLORS_BGR = tuple(
    tuple(int(option.color[index:index + 2], 16) for index in (5, 3, 1))
    for option in DYNAMIC_PROPERTY_OPTIONS
)
UNKNOWN_DYNAMIC_COLOR_BGR = (128, 128, 128)


def parse_filter_key(key: object) -> tuple[str, int | str] | None:
    """Parse GUI event keys into filter field and option value."""
    if not isinstance(key, str) or not key.startswith(FILTER_PREFIX):
        return None
    parts = key.split(".")
    if len(parts) != 3:
        return None
    field, raw = parts[1:]
    if (field, raw) in (("pdh", "max"), ("rcs", "min")):
        return field, raw
    try:
        return field, int(raw, 0)
    except ValueError:
        return None


def filter_radar_points(
    points: Iterable[RadarPoint | RadarObject],
    *,
    cluster: bool = True,
    rcs_min: float | None = None,
    pdh_max: int | None = None,
    allowed_dynamic: Iterable[int] | None = None,
    allowed_ambiguity: Iterable[int] | None = None,
    allowed_invalid: Iterable[int] | None = None,
    max_distance: float | None = None,
    distance_cutoff: float | None = None,
) -> tuple[list[float], list[float], list[tuple[int, int, int]], tuple]:
    """NumPy vectorized filter and color mapper for radar points."""
    pts = tuple(points)
    if not pts:
        return [], [], [], ()

    x = np.array([getattr(p, "dist_latitude", np.nan) for p in pts], dtype=float)
    y = np.array([getattr(p, "dist_long", np.nan) for p in pts], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if max_distance is not None:
        mask &= (y <= float(max_distance))
    if distance_cutoff is not None:
        mask &= (np.hypot(x, y) <= float(distance_cutoff))
    if rcs_min is not None:
        rcs = np.array([getattr(p, "rcs", np.nan) for p in pts], dtype=float)
        mask &= np.isnan(rcs) | (rcs == MISSING_QUALITY) | (rcs >= float(rcs_min))

    dyn = np.array([v if (v := getattr(p, "dynamic_property", -1)) is not None else -1 for p in pts], dtype=int)
    if allowed_dynamic:
        mask &= (dyn < 0) | (dyn == MISSING_QUALITY) | np.isin(dyn, list(allowed_dynamic))

    if cluster:
        if pdh_max is not None:
            pdh = np.array([v if (v := getattr(p, "pdh", -1)) is not None else -1 for p in pts], dtype=int)
            mask &= (pdh <= 0) | (pdh == MISSING_QUALITY) | (pdh <= int(pdh_max))
        if allowed_ambiguity:
            amb = np.array([v if (v := getattr(p, "ambiguity_state", -1)) is not None else -1 for p in pts], dtype=int)
            mask &= (amb < 0) | (amb == MISSING_QUALITY) | np.isin(amb, list(allowed_ambiguity))
        if allowed_invalid:
            inv = np.array([v if (v := getattr(p, "invalid_flag", -1)) is not None else -1 for p in pts], dtype=int)
            mask &= (inv < 0) | (inv == MISSING_QUALITY) | np.isin(inv, list(allowed_invalid))

    indices = np.flatnonzero(mask)
    colors = [
        DYNAMIC_COLORS_BGR[d] if 0 <= d < len(DYNAMIC_COLORS_BGR) else UNKNOWN_DYNAMIC_COLOR_BGR
        for d in dyn[indices]
    ]
    return x[indices].tolist(), y[indices].tolist(), colors, tuple(pts[i] for i in indices)


def filter_point_cutoff(points: Iterable[RadarPoint | RadarObject], max_distance: float):
    """Backward-compatible point cutoff filter using vectorized evaluation."""
    _, _, _, selected = filter_radar_points(points, max_distance=max_distance)
    return list(selected)


def filter_rcs(points: Iterable[RadarPoint | RadarObject], min_rcs: float):
    """Backward-compatible RCS filter using vectorized evaluation."""
    _, _, _, selected = filter_radar_points(points, rcs_min=min_rcs)
    return list(selected)


class Filter_graph:
    """Manages active filter state dictionary and filters radar detections."""

    def __init__(self, values: dict):
        self.enabled_values = {name: set() for name in (
            "dynamic_property", "ambiguity_state", "invalid_state",
        )}
        self.pdh_max = int(values.get(PDH_KEY, 3))
        self.rcs_min = float(values.get(RCS_KEY, -20.0))
        self.last_points = ()
        for key, enabled in values.items():
            self._update(key, enabled)

    def update_values(self, event: str, values: dict) -> None:
        if event in values:
            self._update(event, values[event])

    def _update(self, key: str, value) -> None:
        parsed = parse_filter_key(key)
        if parsed is None:
            return
        field, choice = parsed
        if field == "pdh":
            self.pdh_max = int(value)
        elif field == "rcs":
            self.rcs_min = float(value)
        elif field in self.enabled_values and isinstance(choice, int):
            (self.enabled_values[field].add if value else self.enabled_values[field].discard)(choice)

    def _filter(self, points: Iterable, *, cluster: bool):
        x, y, colors, selected = filter_radar_points(
            points,
            cluster=cluster,
            rcs_min=self.rcs_min,
            pdh_max=self.pdh_max,
            allowed_dynamic=self.enabled_values["dynamic_property"],
            allowed_ambiguity=self.enabled_values["ambiguity_state"],
            allowed_invalid=self.enabled_values["invalid_state"],
        )
        self.last_points = selected
        return x, y, colors

    def filter_point_sequence(self, points: Iterable[RadarPoint]):
        return self._filter(points, cluster=True)

    def filter_object_sequence(self, objects: Iterable[RadarObject]):
        return self._filter(objects, cluster=False)

    def filter_points(self, messages: Clusters_messages):
        return self.filter_point_sequence(messages.snapshot())

    def filter_objects(self, messages: Objects_messages):
        return self.filter_object_sequence(messages.snapshot())
