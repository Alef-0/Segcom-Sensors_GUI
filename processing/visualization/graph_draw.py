import math

import cv2 as cv
import numpy as np

from processing.visualization.filter_schema import DYNAMIC_PROPERTY_OPTIONS
from processing.visualization.multi_radar_fusion import CHANNEL_TO_POD, POD_COLORS_BGR

# Frame dimensions
WIDTH = 800
HEIGHT = 600

MAX_VALUES = 15
DYNAMIC_PROPERTY_LABELS = {
    option.value: option.label for option in DYNAMIC_PROPERTY_OPTIONS
}

# Parameters
# argv[1] = Camera Topic to subscribe


class Graph_radar():
    def __init__(
        self,
        distance_cutoff=MAX_VALUES,
        width=WIDTH,
        height=HEIGHT,
        x_range=MAX_VALUES,
        y_range=MAX_VALUES,
    ):
        self.margin = 50
        self.width, self.height = self._validated_resolution(width, height)
        self.x_range, self.y_range = self._validated_range(x_range, y_range)
        self._update_geometry()
        self.distance_cutoff = self._validated_cutoff(distance_cutoff)
        self.displayed_points = []
        self._refresh_base_image()

    @staticmethod
    def _validated_resolution(width, height):
        width, height = int(width), int(height)
        if width <= 2 * 50 or height <= 2 * 50:
            raise ValueError("Graph width and height must be greater than 100 pixels")
        return width, height

    @staticmethod
    def _validated_range(x_range, y_range):
        x_range, y_range = float(x_range), float(y_range)
        if (
            not math.isfinite(x_range)
            or not math.isfinite(y_range)
            or x_range <= 0
            or y_range <= 0
        ):
            raise ValueError("Graph X and Y ranges must be positive numbers")
        return x_range, y_range

    def _update_geometry(self):
        self.graph_width = self.width - 2 * self.margin
        self.graph_height = self.height - 2 * self.margin
        self.origin_x = self.width // 2
        self.origin_y = self.height - self.margin
        self.x_min, self.x_max = -self.x_range, self.x_range
        self.y_min, self.y_max = 0.0, self.y_range

    def _refresh_base_image(self):
        self.base_image = self.create_base_image()
        self.base_image = self.create_details()

    @staticmethod
    def _validated_cutoff(value):
        cutoff = float(value)
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("Point distance cutoff must be a positive number")
        return cutoff

    def set_distance_cutoff(self, value):
        self.distance_cutoff = self._validated_cutoff(value)
        self._refresh_base_image()

    def set_resolution(self, width, height):
        self.width, self.height = self._validated_resolution(width, height)
        self._update_geometry()
        self._refresh_base_image()

    def set_range(self, x_range, y_range):
        self.x_range, self.y_range = self._validated_range(x_range, y_range)
        self._update_geometry()
        self._refresh_base_image()

    def graph_to_pixel(self, x, y):
        pixel_x = int(self.origin_x + (x * self.graph_width) / (self.x_max - self.x_min))
        pixel_y = int(self.origin_y - (y * self.graph_height) / (self.y_max - self.y_min))
        return (pixel_x, pixel_y)

    @staticmethod
    def _tick_step(limit, target_count):
        raw_step = float(limit) / target_count
        magnitude = 10 ** math.floor(math.log10(raw_step))
        normalized = raw_step / magnitude
        factor = next(value for value in (1, 2, 5, 10) if normalized <= value)
        return factor * magnitude

    @staticmethod
    def _positive_ticks(limit, target_count):
        step = Graph_radar._tick_step(limit, target_count)
        count = int(math.floor(limit / step + 1e-9))
        return [step * index for index in range(1, count + 1)]

    @staticmethod
    def _tick_label(value):
        return f"{value:g}"

    def create_base_image(self):
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Draw border
        cv.rectangle(frame, (self.margin // 2, self.margin // 2),
                    (self.width - self.margin // 2, self.height - self.margin // 2), (0, 0, 0), 2)

        # Draw X-axis
        cv.line(frame, (self.margin, self.origin_y), (self.width - self.margin, self.origin_y), (0, 0, 0), 2)
        # Draw Y-axis
        cv.line(frame, (self.origin_x, self.height - self.margin), (self.origin_x, self.margin), (0, 0, 0), 2)

        # Draw grid lines (lighter)
        x_grid = self._positive_ticks(self.x_range, 15)
        for x in [*[-value for value in reversed(x_grid)], *x_grid]:
            pixel_x, _ = self.graph_to_pixel(x, 0)
            cv.line(frame, (pixel_x, self.margin), (pixel_x, self.height - self.margin), (200, 200, 200), 1)

        for y in self._positive_ticks(self.y_range, 15):
            _, pixel_y = self.graph_to_pixel(0, y)
            cv.line(frame, (self.margin, pixel_y), (self.width - self.margin, pixel_y), (200, 200, 200), 1)

        # Draw arrows
        # X-axis arrow
        cv.arrowedLine(frame, (self.width - self.margin - 10, self.origin_y),
                        (self.width - self.margin, self.origin_y), (0, 0, 0), 2, tipLength=0.02)
        # Y-axis arrow
        cv.arrowedLine(frame, (self.origin_x, self.margin + 10),
                        (self.origin_x, self.margin), (0, 0, 0), 2, tipLength=0.02)

        # Draw tick marks and labels for X-axis
        x_ticks = self._positive_ticks(self.x_range, 4)
        for x in [*[-value for value in reversed(x_ticks)], *x_ticks]:
            pixel_x, pixel_y = self.graph_to_pixel(x, 0)
            cv.line(frame, (pixel_x, pixel_y - 5), (pixel_x, pixel_y + 5), (0, 0, 0), 1)
            cv.putText(frame, self._tick_label(x), (pixel_x - 10, pixel_y + 20),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw tick marks and labels for Y-axis
        for y in self._positive_ticks(self.y_range, 4):
            pixel_x, pixel_y = self.graph_to_pixel(0, y)
            cv.line(frame, (pixel_x - 5, pixel_y), (pixel_x + 5, pixel_y), (0, 0, 0), 1)
            cv.putText(frame, self._tick_label(y), (pixel_x - 25, pixel_y + 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw origin label
        cv.putText(frame, "0", (self.origin_x - 15, self.origin_y + 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw axis labels
        cv.putText(frame, "X", (self.width - self.margin + 5, self.origin_y + 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv.putText(frame, "Y", (self.origin_x - 5, self.margin - 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display the frame
        return frame

    def _draw_range_arc(self, image, distance, color, thickness):
        x, y = self.graph_to_pixel(distance, distance)
        center = self.graph_to_pixel(0, 0)
        axes = (x - center[0], center[1] - y)
        if axes[0] > 0 and axes[1] > 0:
            cv.ellipse(image, center, axes, 0, 270 - 60, 270 + 60, color, thickness)

    def create_details(self):
        new_img = self.base_image.copy()
        center = self.graph_to_pixel(0, 0)

        angle_rad = math.radians(30)
        slope = math.tan(angle_rad)

        # Compute endpoints for ±60° diagonals
        end_right = self.graph_to_pixel(self.x_max, self.x_max * slope)
        end_left = self.graph_to_pixel(self.x_min, -self.x_min * slope)
        cv.line(new_img, center, end_left, (0, 0, 0), 1)
        cv.line(new_img, center, end_right, (0, 0, 0), 1)

        # Draw the semicircles
        for distance in self._positive_ticks(self.y_range, 15):
            self._draw_range_arc(new_img, distance, (0, 0, 0), 1)

        if self.distance_cutoff <= self.y_max:
            self._draw_range_arc(new_img, self.distance_cutoff, (0, 0, 255), 2)

        return new_img

    @staticmethod
    def _dynamic_class(point):
        dynamic_property = getattr(point, "dynamic_property", None)
        if dynamic_property is None:
            return "N/A"
        return DYNAMIC_PROPERTY_LABELS.get(
            dynamic_property,
            f"UNKNOWN_{dynamic_property}",
        )

    @staticmethod
    def _object_class(point):
        class_name = getattr(point, "object_class_name", None)
        return class_name or None

    @staticmethod
    def _format_rcs(point):
        rcs = getattr(point, "rcs", None)
        if rcs is None:
            return "N/A"
        try:
            return f"{float(rcs):.1f} dBm²"
        except (TypeError, ValueError):
            return str(rcs)

    def _on_mouse(self, event, pixel_x, pixel_y, _flags, _param):
        if event != cv.EVENT_LBUTTONDOWN:
            return
        if not self.displayed_points:
            print()
            return

        closest = min(
            self.displayed_points,
            key=lambda item: (
                (item["pixel"][0] - pixel_x) ** 2
                + (item["pixel"][1] - pixel_y) ** 2
            ),
        )
        point = closest["point"]
        pod = closest.get("pod")
        prefix = f"pod={pod} | " if pod else ""
        details = (
            f"{prefix}x={closest['x']:.2f} m | y={closest['y']:.2f} m | "
            f"class={self._dynamic_class(point)} | rcs={self._format_rcs(point)}"
        )
        if "local_x" in closest and "local_y" in closest:
            details += f" | local=({closest['local_x']:.2f}, {closest['local_y']:.2f}) m"
        object_class = self._object_class(point)
        if object_class:
            details += f" | object_class={object_class}"

        print(f"[RADAR POINT] {details}")

    def show_points(self, x_group, y_group, colors, points=None):
        new_img = self.base_image.copy()
        point_group = tuple(points) if points is not None else ()
        self.displayed_points = []

        for index, (x, y, color) in enumerate(zip(x_group, y_group, colors)):
            if math.hypot(x, y) > self.distance_cutoff:
                continue
            pixel = self.graph_to_pixel(x, y)
            cv.circle(new_img, pixel, 4, color, -1)
            point = point_group[index] if index < len(point_group) else None
            self.displayed_points.append({
                "pixel": pixel,
                "x": x,
                "y": y,
                "point": point,
            })

        cv.namedWindow("RADAR")
        cv.setMouseCallback("RADAR", self._on_mouse)
        cv.imshow("RADAR", new_img)
        cv.waitKey(1)

    def _draw_sensor_cone(self, image, pose, color, *, online=True, fov_half_deg=60.0):
        """Render a radar pod origin marker and rotated FoV cone on the global canvas."""
        origin_px = self.graph_to_pixel(pose.offset_x_m, pose.offset_y_m)
        if not online:
            cv.circle(image, origin_px, 5, (150, 150, 150), 2)
            cv.putText(
                image,
                f"{pose.pod}:OFF",
                (origin_px[0] - 18, origin_px[1] + 16),
                cv.FONT_HERSHEY_SIMPLEX,
                0.38,
                (130, 130, 130),
                1,
            )
            return

        cone_range = min(float(self.distance_cutoff), math.hypot(self.x_range, self.y_range))
        angles = np.linspace(-fov_half_deg, fov_half_deg, 25)
        arc_local = np.column_stack([
            cone_range * np.sin(np.radians(angles)),
            cone_range * np.cos(np.radians(angles)),
        ])
        arc_uni = pose.transform_xy(arc_local)
        poly_pts = [origin_px, *[self.graph_to_pixel(float(px), float(py)) for px, py in arc_uni]]
        poly_arr = np.asarray(poly_pts, dtype=np.int32).reshape(-1, 1, 2)

        overlay = image.copy()
        cv.fillPoly(overlay, [poly_arr], color)
        cv.addWeighted(overlay, 0.08, image, 0.92, 0, image)
        cv.line(image, origin_px, poly_pts[1], color, 1)
        cv.line(image, origin_px, poly_pts[-1], color, 1)
        cv.polylines(image, [np.asarray(poly_pts[1:], dtype=np.int32).reshape(-1, 1, 2)], False, color, 1)
        cv.circle(image, origin_px, 6, color, -1)
        cv.circle(image, origin_px, 6, (0, 0, 0), 1)
        cv.putText(
            image,
            pose.pod,
            (origin_px[0] - 5, origin_px[1] - 9),
            cv.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )

    def render_global_image(self, snapshot, *, show_cones=True, show_co_detections=True):
        """Render a GlobalRadarSnapshot onto a fresh BGR canvas without calling imshow."""
        canvas = self.create_base_image()
        for distance in self._positive_ticks(self.y_range, 15):
            self._draw_range_arc(canvas, distance, (220, 220, 220), 1)

        if show_cones:
            for ch in (1, 2, 3):
                pose = snapshot.poses.get(ch)
                if pose is None:
                    continue
                online = bool(snapshot.channel_status.get(ch, {}).get("online", False))
                self._draw_sensor_cone(canvas, pose, POD_COLORS_BGR[ch], online=online)

        self.displayed_points = []
        for idx in range(len(snapshot.unified_xy)):
            ux, uy = float(snapshot.unified_xy[idx, 0]), float(snapshot.unified_xy[idx, 1])
            lx, ly = float(snapshot.local_xy[idx, 0]), float(snapshot.local_xy[idx, 1])
            color = snapshot.colors[idx]
            ch = snapshot.channels[idx]
            pixel = self.graph_to_pixel(ux, uy)
            if self.margin <= pixel[0] <= self.width - self.margin and self.margin <= pixel[1] <= self.height - self.margin:
                cv.circle(canvas, pixel, 5, (255, 255, 255), 1)
                cv.circle(canvas, pixel, 4, color, -1)
            self.displayed_points.append({
                "pixel": pixel,
                "x": ux,
                "y": uy,
                "local_x": lx,
                "local_y": ly,
                "pod": CHANNEL_TO_POD.get(ch, str(ch)),
                "channel": ch,
                "point": snapshot.points[idx] if idx < len(snapshot.points) else None,
            })

        if show_co_detections:
            for matches in snapshot.co_detections.values():
                for match in matches:
                    px_ref = self.graph_to_pixel(*match["ref_xy"])
                    px_tgt = self.graph_to_pixel(*match["tgt_xy"])
                    cv.circle(canvas, px_ref, 8, (0, 180, 0), 1)
                    cv.circle(canvas, px_tgt, 8, (0, 180, 0), 1)
                    cv.line(canvas, px_ref, px_tgt, (0, 180, 0), 2)
                    mid = ((px_ref[0] + px_tgt[0]) // 2 + 4, (px_ref[1] + px_tgt[1]) // 2 - 4)
                    cv.putText(
                        canvas,
                        f"{match['distance_m'] * 100.0:.0f}cm",
                        mid,
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (0, 130, 0),
                        1,
                    )

        hud_parts = ["GLOBAL"]
        for ch in (1, 2, 3):
            st = snapshot.channel_status.get(ch, {})
            pod = CHANNEL_TO_POD[ch]
            hud_parts.append(f"{pod}:{'ON(' + str(st.get('count', 0)) + ')' if st.get('online') else 'OFF'}")
        n_ab = len(snapshot.co_detections.get("AB", ()))
        n_cb = len(snapshot.co_detections.get("CB", ()))
        if show_co_detections and (n_ab or n_cb):
            hud_parts.append(f"CoDet AB:{n_ab} CB:{n_cb}")
        cv.putText(
            canvas,
            " | ".join(hud_parts),
            (self.margin // 2 + 6, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.44,
            (20, 20, 20),
            1,
        )
        return canvas

    def show_global_snapshot(self, snapshot, *, show_cones=True, show_co_detections=True):
        """Render and display fused 3-radar global map on the OpenCV RADAR window."""
        canvas = self.render_global_image(
            snapshot,
            show_cones=show_cones,
            show_co_detections=show_co_detections,
        )
        cv.namedWindow("RADAR")
        cv.setMouseCallback("RADAR", self._on_mouse)
        cv.imshow("RADAR", canvas)
        cv.waitKey(1)

    def close(self):
        cv.destroyAllWindows()
        cv.waitKey(1)
