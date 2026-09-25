import unittest

from sensors.radar import connection_packages
from sensors.radar.cluster_messages import read_701_cluster_list
from sensors.radar.object_messages import (
    OBJECT_CLASSES,
    Objects_messages,
    RadarObject,
    read_60d_object_extended,
)


class MessageModuleSplitTests(unittest.TestCase):
    def test_cluster_decoder_is_defined_in_cluster_module(self):
        self.assertEqual(read_701_cluster_list.__module__, "sensors.radar.cluster_messages")
        self.assertIs(connection_packages.read_701_cluster_list, read_701_cluster_list)

    def test_object_decoder_is_defined_in_object_module(self):
        self.assertEqual(read_60d_object_extended.__module__, "sensors.radar.object_messages")
        self.assertIs(connection_packages.read_60d_object_extended, read_60d_object_extended)

    def test_object_class_dictionary_matches_60d_values(self):
        self.assertEqual(OBJECT_CLASSES, {
            0: "POINT",
            1: "CAR",
            2: "TRUCK",
            3: "RESERVED_01",
            4: "MOTORCYCLE",
            5: "BICYCLE",
            6: "WIDE",
            7: "RESERVED_02",
        })
        self.assertIs(connection_packages.OBJECT_CLASSES, OBJECT_CLASSES)

    def test_object_class_name_is_available_on_decoded_objects(self):
        messages = Objects_messages()
        messages.fill_60d((4, 0.0, 6, 0.0, 0.0, 4.0, 2.0))
        messages.fill_60b((4, 12.0, 1.0, 0.0, 0.0, 0, -10.0))

        obj, = messages.snapshot()

        self.assertEqual(obj.object_class, 6)
        self.assertEqual(obj.object_class_name, "WIDE")
        self.assertIsNone(RadarObject(1).object_class_name)


if __name__ == "__main__":
    unittest.main()
