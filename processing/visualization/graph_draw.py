"""OpenCV top-down radar plot with range scale and point inspection."""

import math

import cv2 as cv
import numpy as np

from processing.visualization.filter_schema import DYNAMIC_PROPERTY_OPTIONS

WIDTH, HEIGHT, MAX_VALUES, MARGIN = 800, 600, 15.0, 50
DYNAMIC_PROPERTY_LABELS = {option.value: option.label for option in DYNAMIC_PROPERTY_OPTIONS}


class Graph_radar:
    def __init__(self, distance_cutoff=MAX_VALUES, width=WIDTH, height=HEIGHT,
                 x_range=MAX_VALUES, y_range=MAX_VALUES):
        self.margin = MARGIN
        self.displayed_points = []
        self.width, self.height = self._resolution(width, height)
        self.x_range, self.y_range = self._ranges(x_range, y_range)
        self.distance_cutoff = self._cutoff(distance_cutoff)
        self._geometry()
        self._refresh()

    @staticmethod
    def _resolution(width, height):
        width, height = int(width), int(height)
        if width <= 2 * MARGIN or height <= 2 * MARGIN:
            raise ValueError("Graph dimensions must exceed 100 pixels")
        return width, height

    @staticmethod
    def _ranges(x_range, y_range):
        values = float(x_range), float(y_range)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Graph ranges must be positive finite numbers")
        return values

    @staticmethod
    def _cutoff(value):
        cutoff = float(value)
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("Point distance cutoff must be positive")
        return cutoff

    def _geometry(self):
        self.graph_width = self.width - 2 * self.margin
        self.graph_height = self.height - 2 * self.margin
        self.origin_x, self.origin_y = self.width // 2, self.height - self.margin

    def _refresh(self):
        self.base_image = self._draw_grid()

    def set_distance_cutoff(self, value):
        self.distance_cutoff = self._cutoff(value)
        self._refresh()

    def set_resolution(self, width, height):
        self.width, self.height = self._resolution(width, height)
        self._geometry()
        self._refresh()

    def set_range(self, x_range, y_range):
        self.x_range, self.y_range = self._ranges(x_range, y_range)
        self._refresh()

    def graph_to_pixel(self, x, y):
        px = round(self.margin + (x + self.x_range) / (2 * self.x_range) * self.graph_width)
        py = round(self.origin_y - y / self.y_range * self.graph_height)
        return int(px), int(py)

    @staticmethod
    def _step(limit, divisions=5):
        raw = limit / divisions
        scale = 10 ** math.floor(math.log10(raw))
        return next(factor for factor in (1, 2, 5, 10) if raw / scale <= factor) * scale

    def _draw_grid(self):
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        cv.rectangle(image, (self.margin // 2, self.margin // 2),
                     (self.width - self.margin // 2, self.height - self.margin // 2), (0, 0, 0), 2)
        cv.line(image, (self.margin, self.origin_y), (self.width - self.margin, self.origin_y), (0, 0, 0), 2)
        cv.line(image, (self.origin_x, self.margin), (self.origin_x, self.origin_y), (0, 0, 0), 2)
        for limit, axis in ((self.x_range, "x"), (self.y_range, "y")):
            step = self._step(limit, 15)
            for i in range(1, int(limit / step) + 1):
                value = i * step
                if axis == "x":
                    for signed in (-value, value):
                        px, _ = self.graph_to_pixel(signed, 0)
                        cv.line(image, (px, self.margin), (px, self.origin_y), (210, 210, 210), 1)
                else:
                    _, py = self.graph_to_pixel(0, value)
                    cv.line(image, (self.margin, py), (self.width - self.margin, py), (210, 210, 210), 1)
        self._draw_arcs(image)
        return image

    def _draw_arcs(self, image):
        center = self.graph_to_pixel(0, 0)
        step = self._step(self.y_range, 15)
        for distance in np.arange(step, self.y_range + step * 0.25, step):
            edge = self.graph_to_pixel(distance, distance)
            axes = (edge[0] - center[0], center[1] - edge[1])
            if min(axes) > 0:
                cv.ellipse(image, center, axes, 0, 210, 330, (70, 70, 70), 1)
        if self.distance_cutoff <= self.y_range:
            edge = self.graph_to_pixel(self.distance_cutoff, self.distance_cutoff)
            axes = (edge[0] - center[0], center[1] - edge[1])
            if min(axes) > 0:
                cv.ellipse(image, center, axes, 0, 210, 330, (0, 0, 255), 2)

    @staticmethod
    def _point_class(point):
        value = getattr(point, "dynamic_property", None)
        return DYNAMIC_PROPERTY_LABELS.get(value, "N/A" if value is None else f"UNKNOWN_{value}")

    def _on_mouse(self, event, pixel_x, pixel_y, _flags, _param):
        if event != cv.EVENT_LBUTTONDOWN or not self.displayed_points:
            return
        item = min(self.displayed_points, key=lambda point: (
            (point["pixel"][0] - pixel_x) ** 2 + (point["pixel"][1] - pixel_y) ** 2
        ))
        point = item["point"]
        rcs = getattr(point, "rcs", None)
        print(
            f"[RADAR POINT] x={item['x']:.2f} m | y={item['y']:.2f} m | "
            f"class={self._point_class(point)} | rcs={rcs if rcs is not None else 'N/A'}"
        )

    def show_points(self, x_group, y_group, colors, points=None):
        image = self.base_image.copy()
        point_rows = tuple(points or ())
        self.displayed_points = []
        for index, (x, y, color) in enumerate(zip(x_group, y_group, colors)):
            if math.hypot(x, y) > self.distance_cutoff:
                continue
            pixel = self.graph_to_pixel(x, y)
            cv.circle(image, pixel, 4, color, -1)
            self.displayed_points.append({
                "pixel": pixel, "x": x, "y": y,
                "point": point_rows[index] if index < len(point_rows) else None,
            })
        cv.namedWindow("RADAR")
        cv.setMouseCallback("RADAR", self._on_mouse)
        cv.imshow("RADAR", image)
        cv.waitKey(1)

    def close(self):
        cv.destroyAllWindows()
        cv.waitKey(1)
