from dataclasses import dataclass

MISSING_QUALITY = 0xFFFFFFFF

KINEMATIC_RMS_VALUES = (
    0.005, 0.006, 0.008, 0.011, 0.014, 0.018, 0.023, 0.029,
    0.038, 0.049, 0.063, 0.081, 0.105, 0.135, 0.174, 0.224,
    0.288, 0.371, 0.478, 0.616, 0.794, 1.023, 1.317, 1.697,
    2.187, 2.817, 3.630, 4.676, 6.025, 7.762, 10.000, None,
)
ORIENTATION_RMS_VALUES = (
    0.005, 0.007, 0.010, 0.014, 0.020, 0.029, 0.041, 0.058,
    0.082, 0.116, 0.165, 0.234, 0.332, 0.471, 0.669, 0.949,
    1.346, 1.909, 2.709, 3.843, 5.451, 7.734, 10.971, 15.565,
    22.081, 31.325, 44.439, 63.044, 89.437, 126.881, 180.000, None,
)

OBJECT_CLASSES = {
    0: "POINT", 1: "CAR", 2: "TRUCK", 3: "RESERVED_01",
    4: "MOTORCYCLE", 5: "BICYCLE", 6: "WIDE", 7: "RESERVED_02",
}


def check_payload(payload: bytes) -> None:
    """Validate that payload is exactly 8 bytes."""
    if len(payload) != 8:
        raise ValueError(f"Expected an 8-byte CAN payload, got {len(payload)}")


_check_payload = check_payload


@dataclass(frozen=True)
class ObjectStatus:
    """Radar object interface status and counter."""
    number_of_objects: int
    measurement_counter: int
    interface_version: int


@dataclass
class RadarPoint:
    """Single radar detection point / cluster."""
    cluster_id: int
    dist_long: float | None = None
    dist_latitude: float | None = None
    velocity_longitude: float | None = None
    velocity_latitude: float | None = None
    dynamic_property: int | None = None
    rcs: float | None = None
    pdh: int = MISSING_QUALITY
    ambiguity_state: int = MISSING_QUALITY
    invalid_flag: int = MISSING_QUALITY

    @property
    def has_general_data(self) -> bool:
        return None not in (self.dist_long, self.dist_latitude)


@dataclass
class RadarObject:
    """Tracked radar obstacle object."""
    object_id: int
    dist_long: float | None = None
    dist_latitude: float | None = None
    velocity_longitude: float | None = None
    velocity_latitude: float | None = None
    dynamic_property: int | None = None
    rcs: float | None = None
    dist_long_rms: float | None = None
    velocity_longitude_rms: float | None = None
    dist_latitude_rms: float | None = None
    velocity_latitude_rms: float | None = None
    acceleration_latitude_rms: float | None = None
    acceleration_longitude_rms: float | None = None
    orientation_rms: float | None = None
    measurement_state: int | None = None
    probability_of_existence: int | None = None
    acceleration_longitude: float | None = None
    acceleration_latitude: float | None = None
    object_class: int | None = None
    orientation_angle: float | None = None
    length: float | None = None
    width: float | None = None
    collision_detection_regions: int | None = None

    @property
    def has_general_data(self) -> bool:
        return None not in (self.dist_long, self.dist_latitude)

    @property
    def object_class_name(self) -> str | None:
        if self.object_class is None:
            return None
        return OBJECT_CLASSES.get(self.object_class, f"UNKNOWN_{self.object_class}")


class Clusters_messages:
    """Per-channel accumulator for cluster messages."""
    def __init__(self):
        self.points: dict[int, RadarPoint] = {}

    def clear(self):
        self.points.clear()

    def fill_701(self, message: tuple):
        cluster_id, dist_long, dist_lat, vel_long, vel_lat, dyn, rcs = message
        point = self.points.setdefault(cluster_id, RadarPoint(cluster_id))
        point.dist_long, point.dist_latitude = dist_long, dist_lat
        point.velocity_longitude, point.velocity_latitude = vel_long, vel_lat
        point.dynamic_property, point.rcs = dyn, rcs

    def fill_702(self, message: tuple):
        cluster_id, pdh, ambiguity, invalid_flag = message
        point = self.points.setdefault(cluster_id, RadarPoint(cluster_id))
        point.pdh, point.ambiguity_state, point.invalid_flag = pdh, ambiguity, invalid_flag

    def snapshot(self) -> tuple[RadarPoint, ...]:
        return tuple(
            RadarPoint(**vars(point))
            for _, point in sorted(self.points.items())
            if point.has_general_data
        )


class Objects_messages:
    """Per-channel accumulator for object messages."""
    def __init__(self):
        self.status: ObjectStatus | None = None
        self.objects: dict[int, RadarObject] = {}

    def clear(self):
        self.status = None
        self.objects.clear()

    def fill_60a(self, status: ObjectStatus):
        self.status = status

    def fill_60b(self, message: tuple):
        obj_id, dist_long, dist_lat, vel_long, vel_lat, dyn, rcs = message
        obj = self.objects.setdefault(obj_id, RadarObject(obj_id))
        obj.dist_long, obj.dist_latitude = dist_long, dist_lat
        obj.velocity_longitude, obj.velocity_latitude = vel_long, vel_lat
        obj.dynamic_property, obj.rcs = dyn, rcs

    def fill_60c(self, message: tuple):
        (
            obj_id, dist_long_rms, vel_long_rms, dist_lat_rms, vel_lat_rms,
            acc_lat_rms, acc_long_rms, orient_rms, state, prob,
        ) = message
        obj = self.objects.setdefault(obj_id, RadarObject(obj_id))
        obj.dist_long_rms, obj.velocity_longitude_rms = dist_long_rms, vel_long_rms
        obj.dist_latitude_rms, obj.velocity_latitude_rms = dist_lat_rms, vel_lat_rms
        obj.acceleration_latitude_rms = acc_lat_rms
        obj.acceleration_longitude_rms = acc_long_rms
        obj.orientation_rms, obj.measurement_state = orient_rms, state
        obj.probability_of_existence = prob

    def fill_60d(self, message: tuple):
        obj_id, acc_long, obj_cls, acc_lat, orient_ang, length, width = message
        obj = self.objects.setdefault(obj_id, RadarObject(obj_id))
        obj.acceleration_longitude, obj.object_class = acc_long, obj_cls
        obj.acceleration_latitude, obj.orientation_angle = acc_lat, orient_ang
        obj.length, obj.width = length, width

    def fill_60e(self, message: tuple):
        obj_id, collision_regions = message
        obj = self.objects.setdefault(obj_id, RadarObject(obj_id))
        obj.collision_detection_regions = collision_regions

    def snapshot(self) -> tuple[RadarObject, ...]:
        return tuple(
            RadarObject(**vars(obj))
            for _, obj in sorted(self.objects.items())
            if obj.has_general_data
        )


def read_701_cluster_list(package: bytes):
    """Decode 0x701 cluster general list packet."""
    check_payload(package)
    cluster_id = package[0]
    dist_long = (package[1] << 5) | (package[2] >> 3)
    dist_latitude = ((package[2] & 0x07) << 8) | package[3]
    velocity_longitude = (package[4] << 2) | (package[5] >> 6)
    velocity_latitude = ((package[5] & 0x3F) << 3) | (package[6] >> 5)
    dynamic_property = package[6] & 0x07
    rcs = package[7]
    return (
        cluster_id,
        dist_long * 0.2 - 500.0,
        dist_latitude * 0.2 - 102.3,
        velocity_longitude * 0.25 - 128.0,
        velocity_latitude * 0.25 - 64.0,
        dynamic_property,
        rcs * 0.5 - 64.0,
    )


def read_702_quality_info(package: bytes):
    """Decode 0x702 cluster quality packet."""
    check_payload(package)
    return package[0], package[3] & 0x07, package[4] & 0x07, (package[4] >> 3) & 0x1F


def read_60a_object_status(package: bytes) -> ObjectStatus:
    """Decode 0x60A object status packet."""
    check_payload(package)
    return ObjectStatus(
        number_of_objects=package[0],
        measurement_counter=(package[1] << 8) | package[2],
        interface_version=(package[3] >> 4) & 0x0F,
    )


def read_60b_object_general(package: bytes):
    """Decode 0x60B object general position and velocity packet."""
    check_payload(package)
    obj_id = package[0]
    dist_long = (package[1] << 5) | (package[2] >> 3)
    dist_lat = ((package[2] & 0x07) << 8) | package[3]
    vel_long = (package[4] << 2) | (package[5] >> 6)
    vel_lat = ((package[5] & 0x3F) << 3) | (package[6] >> 5)
    dynamic_property = package[6] & 0x07
    rcs = package[7]
    return (
        obj_id,
        dist_long * 0.2 - 500.0,
        dist_lat * 0.2 - 204.6,
        vel_long * 0.25 - 128.0,
        vel_lat * 0.25 - 64.0,
        dynamic_property,
        rcs * 0.5 - 64.0,
    )


def read_60c_object_quality(package: bytes):
    """Decode 0x60C object RMS accuracy and existence probability packet."""
    check_payload(package)
    obj_id = package[0]
    dist_long_rms = package[1] >> 3
    vel_long_rms = (package[2] >> 1) & 0x1F
    dist_lat_rms = ((package[1] & 0x07) << 2) | (package[2] >> 6)
    vel_lat_rms = ((package[2] & 0x01) << 4) | (package[3] >> 4)
    acc_lat_rms = (package[4] >> 2) & 0x1F
    acc_long_rms = ((package[3] & 0x0F) << 1) | (package[4] >> 7)
    orient_rms = ((package[4] & 0x03) << 3) | (package[5] >> 5)
    measurement_state = (package[6] >> 2) & 0x07
    probability_of_existence = package[6] >> 5
    return (
        obj_id,
        KINEMATIC_RMS_VALUES[dist_long_rms],
        KINEMATIC_RMS_VALUES[vel_long_rms],
        KINEMATIC_RMS_VALUES[dist_lat_rms],
        KINEMATIC_RMS_VALUES[vel_lat_rms],
        KINEMATIC_RMS_VALUES[acc_lat_rms],
        KINEMATIC_RMS_VALUES[acc_long_rms],
        ORIENTATION_RMS_VALUES[orient_rms],
        measurement_state,
        probability_of_existence,
    )


def read_60d_object_extended(package: bytes):
    """Decode 0x60D object dimensions, orientation, and acceleration packet."""
    check_payload(package)
    obj_id = package[0]
    acc_long = (package[1] << 3) | (package[2] >> 5)
    obj_class = package[3] & 0x07
    acc_lat = ((package[2] & 0x1F) << 4) | (package[3] >> 4)
    orient_angle = (package[4] << 2) | (package[5] >> 6)
    length = package[6]
    width = package[7]
    return (
        obj_id,
        acc_long * 0.01 - 10.0,
        obj_class,
        acc_lat * 0.01 - 2.5,
        orient_angle * 0.4 - 180.0,
        length * 0.2,
        width * 0.2,
    )


def read_60e_object_warning(package: bytes):
    """Decode 0x60E object collision detection warning regions packet."""
    check_payload(package)
    return package[0], package[1]


def create_200_radar_configuration(
    ok_distance, distance, ok_radarpower, radarpower,
    ok_output, output, ok_rcs, rcs,
    ok_qual, quality, save_nvm,
    ok_ext=False, ext_info=0, ok_relay=False, ctrl_relay=0,
):
    """Encode 0x200 radar sensor configuration command."""
    payload = bytearray(8)
    payload[0] = (
        (int(bool(ok_distance)) << 0)
        | (int(bool(ok_radarpower)) << 2)
        | (int(bool(ok_output)) << 3)
        | (int(bool(ok_qual)) << 4)
        | (int(bool(ok_ext)) << 5)
        | (int(bool(save_nvm)) << 7)
    )
    payload[1] = (distance >> 2) & 0xFF
    payload[2] = (distance & 0x03) << 6
    payload[4] = ((output & 0x03) << 3) | ((radarpower & 0x07) << 5)
    payload[5] = (
        (int(bool(ok_relay)) << 0)
        | (int(bool(ctrl_relay)) << 1)
        | ((quality & 0x01) << 2)
        | ((ext_info & 0x01) << 3)
        | (int(bool(save_nvm)) << 7)
    )
    payload[6] = int(bool(ok_rcs)) | ((rcs & 0x07) << 1)
    return int.from_bytes(payload, byteorder="big", signed=False)


def read_201_radar_state_extended(package: bytes):
    """Decode 0x201 radar state with extended configuration flags."""
    check_payload(package)
    max_distance_cfg = (package[1] << 2) | (package[2] >> 6)
    radar_power_cfg = ((package[3] & 0x03) << 1) | ((package[4] >> 7) & 0x01)
    output_type_cfg = (package[5] >> 2) & 0x03
    ctrl_relay_cfg = (package[5] >> 1) & 0x01
    send_quality_cfg = (package[5] >> 4) & 0x01
    send_ext_info_cfg = (package[5] >> 5) & 0x01
    rcs_threshold = (package[7] >> 2) & 0x07
    raw_payload = hex(int.from_bytes(package, byteorder="big", signed=False))
    return (
        max_distance_cfg,
        radar_power_cfg,
        output_type_cfg,
        rcs_threshold,
        send_quality_cfg,
        send_ext_info_cfg,
        ctrl_relay_cfg,
        raw_payload,
    )


def read_201_radar_state(package: bytes):
    """Decode 0x201 radar state returning standard 6-tuple."""
    (
        max_distance_cfg, radar_power_cfg, output_type_cfg,
        rcs_threshold, send_quality_cfg, _, _, raw_payload,
    ) = read_201_radar_state_extended(package)
    return (
        max_distance_cfg,
        radar_power_cfg,
        output_type_cfg,
        rcs_threshold,
        send_quality_cfg,
        raw_payload,
    )
