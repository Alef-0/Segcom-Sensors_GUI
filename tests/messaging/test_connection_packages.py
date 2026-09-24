import unittest

from sensors.radar.connection_packages import (
    Clusters_messages,
    KINEMATIC_RMS_VALUES,
    MISSING_QUALITY,
    ORIENTATION_RMS_VALUES,
    ObjectStatus,
    Objects_messages,
    create_200_radar_configuration,
    read_201_radar_state,
    read_60a_object_status,
    read_60b_object_general,
    read_60c_object_quality,
    read_701_cluster_list,
    read_702_quality_info,
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


class RadarPackageTests(unittest.TestCase):
    def test_701_fields_cross_byte_boundaries(self):
        dist_long = 0x123
        dist_latitude = 0x5AB
        velocity_longitude = 0x2AB
        velocity_latitude = 0x155
        payload = bytearray(8)
        payload[0] = 7
        payload[1] = dist_long >> 5
        payload[2] = ((dist_long & 0x1F) << 3) | ((dist_latitude >> 8) & 0x07)
        payload[3] = dist_latitude & 0xFF
        payload[4] = velocity_longitude >> 2
        payload[5] = ((velocity_longitude & 0x03) << 6) | ((velocity_latitude >> 3) & 0x3F)
        payload[6] = ((velocity_latitude & 0x07) << 5) | 6
        payload[7] = 180

        decoded = read_701_cluster_list(payload)

        self.assertEqual(decoded[0], 7)
        self.assertAlmostEqual(decoded[1], dist_long * 0.2 - 500.0)
        self.assertAlmostEqual(decoded[2], dist_latitude * 0.2 - 102.3)
        self.assertAlmostEqual(decoded[3], velocity_longitude * 0.25 - 128.0)
        self.assertAlmostEqual(decoded[4], velocity_latitude * 0.25 - 64.0)
        self.assertEqual(decoded[5], 6)
        self.assertAlmostEqual(decoded[6], 26.0)

    def test_702_quality_fields(self):
        payload = bytearray(8)
        payload[0] = 11
        payload[3] = 5
        payload[4] = (0x12 << 3) | 3
        self.assertEqual(read_702_quality_info(payload), (11, 5, 3, 0x12))

    def test_snapshot_keeps_general_points_without_quality(self):
        messages = Clusters_messages()
        messages.fill_701((9, 10.0, -2.0, 3.0, 0.0, 4, -12.5))

        point, = messages.snapshot()

        self.assertEqual(point.cluster_id, 9)
        self.assertEqual(point.velocity_latitude, 0.0)
        self.assertEqual(point.pdh, MISSING_QUALITY)
        self.assertEqual(point.ambiguity_state, MISSING_QUALITY)
        self.assertEqual(point.invalid_flag, MISSING_QUALITY)

    def test_quality_can_arrive_before_general_data(self):
        messages = Clusters_messages()
        messages.fill_702((4, 6, 2, 11))
        messages.fill_701((4, 5.0, 1.0, -2.0, 0.5, 3, 8.0))

        point, = messages.snapshot()

        self.assertEqual((point.pdh, point.ambiguity_state, point.invalid_flag), (6, 2, 11))

    def test_60a_object_status(self):
        payload = bytes((17, 0x12, 0x34, 0x10, 0, 0, 0, 0))
        self.assertEqual(
            read_60a_object_status(payload),
            ObjectStatus(17, 0x1234, 1),
        )

    def test_60b_object_general_fields(self):
        payload = bytearray(8)
        payload[0] = 7
        put_motorola(payload, 19, 13, 2600)
        put_motorola(payload, 24, 11, 1023)
        put_motorola(payload, 46, 10, 512)
        put_motorola(payload, 48, 3, 6)
        put_motorola(payload, 53, 9, 256)
        payload[7] = 128

        decoded = read_60b_object_general(payload)

        self.assertEqual(decoded[0], 7)
        self.assertAlmostEqual(decoded[1], 20.0)
        self.assertAlmostEqual(decoded[2], 0.0)
        self.assertAlmostEqual(decoded[3], 0.0)
        self.assertAlmostEqual(decoded[4], 0.0)
        self.assertEqual(decoded[5], 6)
        self.assertAlmostEqual(decoded[6], 0.0)

    def test_60c_object_quality_fields_and_lookup_tables(self):
        payload = bytearray(8)
        payload[0] = 9
        indices = (4, 10, 12, 16, 20, 24, 30)
        for start, value in zip((11, 17, 22, 28, 34, 39, 45), indices):
            put_motorola(payload, start, 5, value)
        put_motorola(payload, 50, 3, 3)
        put_motorola(payload, 53, 3, 6)

        decoded = read_60c_object_quality(payload)

        self.assertEqual(decoded[0], 9)
        self.assertEqual(decoded[1:7], tuple(KINEMATIC_RMS_VALUES[index] for index in indices[:6]))
        self.assertEqual(decoded[7], ORIENTATION_RMS_VALUES[30])
        self.assertEqual(decoded[8:], (3, 6))

    def test_60c_invalid_rms_values_are_none(self):
        payload = bytearray(8)
        for start in (11, 17, 22, 28, 34, 39, 45):
            put_motorola(payload, start, 5, 31)
        decoded = read_60c_object_quality(payload)
        self.assertEqual(decoded[1:8], (None,) * 7)

    def test_object_messages_merge_quality_and_general_by_id(self):
        messages = Objects_messages()
        status = ObjectStatus(1, 42, 1)
        messages.fill_60a(status)
        messages.fill_60c((5, 0.014, 0.063, 0.105, 0.288, 0.794, 2.187, 180.0, 2, 7))
        messages.fill_60b((5, 12.0, -3.0, 2.5, 0.0, 1, -8.0))

        obj, = messages.snapshot()

        self.assertEqual(messages.status, status)
        self.assertEqual(obj.object_id, 5)
        self.assertEqual(obj.measurement_state, 2)
        self.assertEqual(obj.probability_of_existence, 7)
        self.assertEqual(obj.dist_long, 12.0)

    def test_201_masks_radar_power_and_rcs(self):
        payload = bytearray(8)
        payload[3] = 0xFE
        payload[4] = 0x80
        payload[7] = 0xFC
        _, radar_power, _, rcs, _, _ = read_201_radar_state(payload)
        self.assertEqual(radar_power, 5)
        self.assertEqual(rcs, 7)

    def test_200_round_trip_layout(self):
        raw = create_200_radar_configuration(1, 511, 1, 3, 1, 2, 1, 1, 1, 1, 1)
        payload = raw.to_bytes(8, "big")
        self.assertEqual(payload[0], 0b10011101)
        self.assertEqual((payload[1] << 2) | (payload[2] >> 6), 511)
        self.assertEqual((payload[4] >> 3) & 0x03, 2)
        self.assertEqual((payload[4] >> 5) & 0x07, 3)


if __name__ == "__main__":
    unittest.main()
