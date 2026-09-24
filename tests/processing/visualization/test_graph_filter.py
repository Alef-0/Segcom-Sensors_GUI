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


if __name__ == "__main__":
    unittest.main()
