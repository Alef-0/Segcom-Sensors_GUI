from types import SimpleNamespace
import unittest

from processing.visualization.graph_filter import Filter_graph
from processing.visualization.filter_schema import (
    AMBIGUITY_STATE_OPTIONS,
    DYNAMIC_PROPERTY_OPTIONS,
    INVALID_STATE_OPTIONS,
    PDH_KEY,
    RCS_KEY,
)


def initial_values():
    values = {
        PDH_KEY: 3,
        RCS_KEY: -10.0,
    }
    for option in DYNAMIC_PROPERTY_OPTIONS:
        values[option.key] = option.default
    for option in AMBIGUITY_STATE_OPTIONS:
        values[option.key] = option.default
    for option in INVALID_STATE_OPTIONS:
        values[option.key] = option.default
    return values


class Messages:
    def __init__(self, *points):
        self.points = points

    def snapshot(self):
        return self.points


class GraphFilterTests(unittest.TestCase):
    def test_cluster_rcs_below_threshold_is_filtered(self):
        graph_filter = Filter_graph(initial_values())
        accepted = SimpleNamespace(
            dynamic_property=0,
            pdh=2,
            ambiguity_state=3,
            invalid_flag=0,
            rcs=-10.0,
            dist_latitude=1.0,
            dist_long=2.0,
        )
        rejected = SimpleNamespace(**{**vars(accepted), "rcs": -10.5})

        x, y, _ = graph_filter.filter_points(Messages(accepted, rejected))

        self.assertEqual(x, [1.0])
        self.assertEqual(y, [2.0])

    def test_object_rcs_filter_uses_same_threshold(self):
        graph_filter = Filter_graph(initial_values())
        accepted = SimpleNamespace(
            dynamic_property=1,
            rcs=-9.5,
            dist_latitude=-1.0,
            dist_long=4.0,
        )
        rejected = SimpleNamespace(**{**vars(accepted), "rcs": -10.5})

        x, y, _ = graph_filter.filter_objects(Messages(accepted, rejected))

        self.assertEqual(x, [-1.0])
        self.assertEqual(y, [4.0])

    def test_default_rcs_is_minus_twenty(self):
        values = initial_values()
        values.pop(RCS_KEY)

        self.assertEqual(Filter_graph(values).rcs_min, -20.0)

    def test_rcs_slider_update_changes_minimum(self):
        values = initial_values()
        graph_filter = Filter_graph(values)
        values[RCS_KEY] = 12.5

        graph_filter.update_values(RCS_KEY, values)

        self.assertEqual(graph_filter.rcs_min, 12.5)

    def test_filter_radar_points_vectorized_all_options(self):
        from sensors.auxiliary import filter_radar_points, filter_point_cutoff, filter_rcs
        p1 = SimpleNamespace(dist_long=5.0, dist_latitude=2.0, rcs=-5.0, dynamic_property=0, pdh=2, ambiguity_state=3, invalid_flag=0)
        p2 = SimpleNamespace(dist_long=25.0, dist_latitude=2.0, rcs=-5.0, dynamic_property=0, pdh=2, ambiguity_state=3, invalid_flag=0)
        p3 = SimpleNamespace(dist_long=5.0, dist_latitude=2.0, rcs=-25.0, dynamic_property=0, pdh=2, ambiguity_state=3, invalid_flag=0)
        p4 = SimpleNamespace(dist_long=float("nan"), dist_latitude=2.0, rcs=-5.0, dynamic_property=0)

        # Cutoff filter
        self.assertEqual(len(filter_point_cutoff((p1, p2), 10.0)), 1)
        # RCS filter
        self.assertEqual(len(filter_rcs((p1, p3), -10.0)), 1)

        # Full vectorized filter
        x, y, colors, selected = filter_radar_points(
            (p1, p2, p3, p4), cluster=True, rcs_min=-10.0, max_distance=10.0,
            allowed_dynamic={0}, allowed_ambiguity={3}, allowed_invalid={0},
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0], p1)
        self.assertEqual((x, y), ([2.0], [5.0]))
        self.assertEqual(colors, [(0, 0, 255)])


if __name__ == "__main__":
    unittest.main()
