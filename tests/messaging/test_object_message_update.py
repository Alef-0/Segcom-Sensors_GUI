from types import SimpleNamespace
import math
import unittest

from sensors.radar.connection_packages import (
    MISSING_QUALITY,
    ObjectStatus,
    Objects_messages,
    create_200_radar_configuration,
    read_201_radar_state,
    read_201_radar_state_extended,
    read_60d_object_extended,
    read_60e_object_warning,
)
from processing.visualization.graph_filter import Filter_graph, UNKNOWN_DYNAMIC_COLOR_BGR
from processing.visualization.filter_schema import (
    AMBIGUITY_STATE_OPTIONS,
    DYNAMIC_PROPERTY_OPTIONS,
    INVALID_STATE_OPTIONS,
    PDH_KEY,
    RCS_KEY,
)


def put_motorola(payload, start, length, value):
    bit = start
    for value_bit in range(length):
        byte_index, bit_index = divmod(bit, 8)
        mask = 1 << bit_index
        if value & (1 << value_bit):
            payload[byte_index] |= mask
        else:
            payload[byte_index] &= ~mask
        bit = bit - 15 if bit_index == 7 else bit + 1


def initial_filter_values():
    values = {PDH_KEY: 3, RCS_KEY: -10.0}
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


class ObjectMessageUpdateTests(unittest.TestCase):
    def test_60d_extended_fields(self):
        payload = bytearray(8)
        payload[0] = 12
        put_motorola(payload, 21, 11, 1500)
        put_motorola(payload, 24, 3, 5)
        put_motorola(payload, 28, 9, 300)
        put_motorola(payload, 46, 10, 500)
        put_motorola(payload, 48, 8, 20)
        put_motorola(payload, 56, 8, 30)

        decoded = read_60d_object_extended(payload)

        self.assertEqual(decoded[0], 12)
        self.assertAlmostEqual(decoded[1], 5.0)
        self.assertEqual(decoded[2], 5)
        self.assertAlmostEqual(decoded[3], 0.5)
        self.assertAlmostEqual(decoded[4], 20.0)
        self.assertAlmostEqual(decoded[5], 4.0)
        self.assertAlmostEqual(decoded[6], 6.0)

    def test_60e_warning_fields(self):
        payload = bytes((9, 0b10100101, 0, 0, 0, 0, 0, 0))
        self.assertEqual(read_60e_object_warning(payload), (9, 0b10100101))

    def test_object_messages_merge_all_object_packets_by_id(self):
        messages = Objects_messages()
        messages.fill_60a(ObjectStatus(1, 42, 1))
        messages.fill_60d((5, 1.25, 4, -0.25, 12.0, 4.2, 1.8))
        messages.fill_60e((5, 0x81))
        messages.fill_60b((5, 12.0, -3.0, 2.5, 0.0, 1, -8.0))

        obj, = messages.snapshot()

        self.assertEqual(obj.object_id, 5)
        self.assertEqual(obj.object_class, 4)
        self.assertAlmostEqual(obj.acceleration_longitude, 1.25)
        self.assertAlmostEqual(obj.acceleration_latitude, -0.25)
        self.assertAlmostEqual(obj.orientation_angle, 12.0)
        self.assertAlmostEqual(obj.length, 4.2)
        self.assertAlmostEqual(obj.width, 1.8)
        self.assertEqual(obj.collision_detection_regions, 0x81)

    def test_configuration_adds_extended_and_relay_flags(self):
        raw = create_200_radar_configuration(
            1, 511, 1, 3, 1, 2, 1, 1,
            1, 1, 1,
            ok_ext=True, ext_info=1,
            ok_relay=True, ctrl_relay=1,
        )
        payload = raw.to_bytes(8, "big")

        self.assertEqual(payload[0], 0b10111101)
        self.assertEqual(payload[5], 0b10001111)

    def test_existing_configuration_call_remains_compatible(self):
        raw = create_200_radar_configuration(1, 511, 1, 3, 1, 2, 1, 1, 1, 1, 1)
        payload = raw.to_bytes(8, "big")
        self.assertEqual(payload[0], 0b10011101)
        self.assertEqual(payload[5], 0b10000100)

    def test_extended_201_state_exposes_new_flags(self):
        payload = bytearray(8)
        payload[5] = 0b00110110

        extended = read_201_radar_state_extended(payload)
        legacy = read_201_radar_state(payload)

        self.assertEqual(extended[2], 1)
        self.assertEqual(extended[4:7], (1, 1, 1))
        self.assertEqual(len(legacy), 6)
        self.assertEqual(legacy[4], 1)

    def test_graph_accepts_missing_quality(self):
        graph_filter = Filter_graph(initial_filter_values())
        point = SimpleNamespace(
            dynamic_property=0,
            pdh=MISSING_QUALITY,
            ambiguity_state=MISSING_QUALITY,
            invalid_flag=MISSING_QUALITY,
            rcs=-9.0,
            dist_latitude=1.0,
            dist_long=2.0,
        )

        x, y, _ = graph_filter.filter_points(Messages(point))

        self.assertEqual((x, y), ([1.0], [2.0]))

    def test_graph_accepts_missing_dynamic_and_rcs(self):
        graph_filter = Filter_graph(initial_filter_values())
        obj = SimpleNamespace(
            dynamic_property=None,
            rcs=math.nan,
            dist_latitude=-1.0,
            dist_long=4.0,
        )

        x, y, colors = graph_filter.filter_objects(Messages(obj))

        self.assertEqual((x, y), ([-1.0], [4.0]))
        self.assertEqual(colors, [UNKNOWN_DYNAMIC_COLOR_BGR])

    def test_empty_dynamic_selection_means_show_all(self):
        values = initial_filter_values()
        for option in DYNAMIC_PROPERTY_OPTIONS:
            values[option.key] = False
        graph_filter = Filter_graph(values)
        obj = SimpleNamespace(
            dynamic_property=7,
            rcs=-9.0,
            dist_latitude=3.0,
            dist_long=5.0,
        )

        x, y, _ = graph_filter.filter_objects(Messages(obj))

        self.assertEqual((x, y), ([3.0], [5.0]))

    def test_default_rcs_preserves_latest_minus_twenty_setting(self):
        values = initial_filter_values()
        values.pop(RCS_KEY)
        self.assertEqual(Filter_graph(values).rcs_min, -20.0)

    def test_playback_sequence_filter_uses_optional_field_rules(self):
        graph_filter = Filter_graph(initial_filter_values())
        obj = SimpleNamespace(
            dynamic_property=None,
            rcs=None,
            dist_latitude=2.0,
            dist_long=6.0,
        )
        x, y, colors = graph_filter.filter_object_sequence((obj,))
        self.assertEqual((x, y), ([2.0], [6.0]))
        self.assertEqual(colors, [UNKNOWN_DYNAMIC_COLOR_BGR])

    def test_missing_coordinates_are_not_drawn(self):
        graph_filter = Filter_graph(initial_filter_values())
        obj = SimpleNamespace(
            dynamic_property=0,
            rcs=-9.0,
            dist_latitude=math.nan,
            dist_long=5.0,
        )

        self.assertEqual(graph_filter.filter_objects(Messages(obj)), ([], [], []))


if __name__ == "__main__":
    unittest.main()
