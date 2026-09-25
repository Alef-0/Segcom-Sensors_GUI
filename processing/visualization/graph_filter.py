"""Apply user-selected radar filters consistently to points and objects."""

import math
from typing import Iterable

from processing.visualization.filter_schema import (
    DYNAMIC_COLORS_BGR, PDH_KEY, RCS_KEY, parse_filter_key,
)
from sensors.radar.connection_packages import (
    Clusters_messages, MISSING_QUALITY, Objects_messages, RadarObject, RadarPoint,
)

UNKNOWN_DYNAMIC_COLOR_BGR = (128, 128, 128)


class Filter_graph:
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

    @staticmethod
    def _missing(value) -> bool:
        if value is None:
            return True
        try:
            return value == MISSING_QUALITY or math.isnan(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    @classmethod
    def _selected(cls, value, allowed: set[int]) -> bool:
        return cls._missing(value) or not allowed or value in allowed

    @staticmethod
    def _finite_coordinates(point) -> bool:
        try:
            return math.isfinite(point.dist_long) and math.isfinite(point.dist_latitude)
        except (AttributeError, TypeError, ValueError):
            return False

    def _allowed(self, point, *, cluster: bool) -> bool:
        quality_ok = True
        pdh = getattr(point, "pdh", None)
        rcs = getattr(point, "rcs", None)
        if cluster and not self._missing(pdh):
            quality_ok = 0 < pdh <= self.pdh_max
        rcs_ok = self._missing(rcs) or rcs >= self.rcs_min
        return (
            self._finite_coordinates(point)
            and self._selected(getattr(point, "dynamic_property", None), self.enabled_values["dynamic_property"])
            and self._selected(getattr(point, "ambiguity_state", None), self.enabled_values["ambiguity_state"])
            and self._selected(getattr(point, "invalid_flag", None), self.enabled_values["invalid_state"])
            and rcs_ok and quality_ok
        )

    @staticmethod
    def _color(value):
        if Filter_graph._missing(value):
            return UNKNOWN_DYNAMIC_COLOR_BGR
        try:
            return DYNAMIC_COLORS_BGR[int(value)]
        except (IndexError, TypeError, ValueError, OverflowError):
            return UNKNOWN_DYNAMIC_COLOR_BGR

    def _filter(self, points: Iterable, *, cluster: bool):
        selected = tuple(point for point in points if self._allowed(point, cluster=cluster))
        self.last_points = selected
        return (
            [point.dist_latitude for point in selected],
            [point.dist_long for point in selected],
            [self._color(getattr(point, "dynamic_property", None)) for point in selected],
        )

    def filter_point_sequence(self, points: Iterable[RadarPoint]):
        return self._filter(points, cluster=True)

    def filter_object_sequence(self, objects: Iterable[RadarObject]):
        return self._filter(objects, cluster=False)

    def filter_points(self, messages: Clusters_messages):
        return self.filter_point_sequence(messages.snapshot())

    def filter_objects(self, messages: Objects_messages):
        return self.filter_object_sequence(messages.snapshot())
