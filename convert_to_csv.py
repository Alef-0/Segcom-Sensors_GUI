#!/usr/bin/env python3
"""Convert Segcom-Sensors_GUI recordings from PCD to CSV.

Usage:
    python3 convert_pointcloud_to_csv_segcom.py recordings

For:
    recordings/

the program creates a sibling directory:
    recordings - CSV/

The source tree is scanned recursively. The converted tree keeps the Segcom
recording data:
    - .pcd point clouds -> .csv
    - camera images -> copied unchanged
    - recording.json and timestamps.json -> copied with .pcd references changed
      to .csv

Unrelated files are ignored. Directory structure is preserved for every copied
or converted recording file.

Dependency:
    python -m pip install pypcd4 numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

try:
    import numpy as np
    from pypcd4 import PointCloud
except ImportError:
    PointCloud = None


DICTIONARY_FILENAME = "value_dictionaries.json"

DYNAMIC_PROPERTY = {
    0: "moving",
    1: "stationary",
    2: "oncoming",
    3: "stationary_candidate",
    4: "unknown",
    5: "crossing_stationary",
    6: "crossing_moving",
    7: "stopped",
}

PDH0_FALSE_ALARM_PROBABILITY = {
    0: "invalid",
    1: "probability_below_25_percent",
    2: "probability_below_50_percent",
    3: "probability_below_75_percent",
    4: "probability_below_90_percent",
    5: "probability_below_99_percent",
    6: "probability_below_99_9_percent",
    7: "probability_at_most_100_percent",
}

AMBIGUITY_STATE = {
    0: "invalid",
    1: "ambiguous",
    2: "staggered_ramp",
    3: "unambiguous",
    4: "stationary_candidates",
}

CLUSTER_INVALID_STATE = {
    0x00: "valid",
    0x01: "invalid_low_rcs",
    0x02: "invalid_near_field_artefact",
    0x03: "invalid_far_range_not_confirmed_in_near_range",
    0x04: "valid_low_rcs",
    0x05: "reserved",
    0x06: "invalid_high_mirror_probability",
    0x07: "invalid_outside_sensor_field_of_view",
    0x08: "valid_azimuth_correction_due_to_elevation",
    0x09: "valid_high_child_probability",
    0x0A: "valid_high_probability_of_50_degree_artefact",
    0x0B: "valid_without_local_maximum",
    0x0C: "valid_high_artefact_probability",
    0x0D: "reserved",
    0x0E: "invalid_harmonics",
    0x0F: "valid_above_95_m_in_near_range",
    0x10: "valid_high_multi_target_probability",
    0x11: "valid_suspicious_angle",
}

MEASUREMENT_STATE = {
    0: "deleted",
    1: "new",
    2: "measured",
    3: "predicted",
    4: "deleted_for_merge",
    5: "new_from_merge",
    6: "reserved",
    7: "reserved",
}

PROBABILITY_OF_EXISTENCE = {
    0: "invalid",
    1: "probability_below_25_percent",
    2: "probability_below_50_percent",
    3: "probability_below_75_percent",
    4: "probability_below_90_percent",
    5: "probability_below_99_percent",
    6: "probability_below_99_9_percent",
    7: "probability_at_most_100_percent",
}

OBJECT_CLASS = {
    0: "point",
    1: "car",
    2: "truck",
    3: "not_in_use",
    4: "motorcycle",
    5: "bicycle",
    6: "wide",
    7: "reserved",
}

COLLISION_DETECTION_REGIONS = {
    bit: f"collision_detection_region_{bit}" for bit in range(8)
}

ENUM_COLUMNS = {
    "dynamic_property": DYNAMIC_PROPERTY,
    "pdh": PDH0_FALSE_ALARM_PROBABILITY,
    "ambiguity_state": AMBIGUITY_STATE,
    "invalid_flag": CLUSTER_INVALID_STATE,
    "measurement_state": MEASUREMENT_STATE,
    "probability_of_existence": PROBABILITY_OF_EXISTENCE,
    "object_class": OBJECT_CLASS,
}

COLUMN_DOCUMENTATION = {
    "ID": {"description": "Cluster or object identifier.", "unit": None},
    "dist_long": {"description": "Longitudinal distance from the radar.", "unit": "m"},
    "dist_latitude": {"description": "Lateral distance from the radar.", "unit": "m"},
    "velocity_longitude": {"description": "Longitudinal relative velocity.", "unit": "m/s"},
    "velocity_latitude": {"description": "Lateral relative velocity.", "unit": "m/s"},
    "dynamic_property": {"description": "Encoded target motion state.", "unit": None},
    "dynamic_property_label": {"description": "Decoded dynamic_property value.", "unit": None},
    "rcs": {"description": "Radar cross section.", "unit": "dBm²"},
    "pdh": {"description": "False-alarm probability category (Pdh0).", "unit": None},
    "pdh_label": {"description": "Decoded pdh value.", "unit": None},
    "ambiguity_state": {"description": "Velocity ambiguity state.", "unit": None},
    "ambiguity_state_label": {"description": "Decoded ambiguity_state value.", "unit": None},
    "invalid_flag": {"description": "Cluster validity/quality state.", "unit": None},
    "invalid_flag_label": {"description": "Decoded invalid_flag value.", "unit": None},
    "dist_long_rms": {"description": "RMS error of longitudinal distance.", "unit": "m"},
    "velocity_longitude_rms": {"description": "RMS error of longitudinal velocity.", "unit": "m/s"},
    "dist_latitude_rms": {"description": "RMS error of lateral distance.", "unit": "m"},
    "velocity_latitude_rms": {"description": "RMS error of lateral velocity.", "unit": "m/s"},
    "acceleration_latitude_rms": {"description": "RMS error of lateral acceleration.", "unit": "m/s²"},
    "acceleration_longitude_rms": {"description": "RMS error of longitudinal acceleration.", "unit": "m/s²"},
    "orientation_rms": {"description": "RMS error of object orientation.", "unit": "degrees"},
    "measurement_state": {"description": "Object tracker measurement state.", "unit": None},
    "measurement_state_label": {"description": "Decoded measurement_state value.", "unit": None},
    "probability_of_existence": {"description": "Encoded probability-of-existence category.", "unit": None},
    "probability_of_existence_label": {"description": "Decoded probability_of_existence value.", "unit": None},
    "acceleration_longitude": {"description": "Longitudinal relative acceleration.", "unit": "m/s²"},
    "acceleration_latitude": {"description": "Lateral relative acceleration.", "unit": "m/s²"},
    "object_class": {"description": "Encoded object classification.", "unit": None},
    "object_class_label": {"description": "Decoded object_class value.", "unit": None},
    "orientation_angle": {
        "description": "Object orientation angle; positive values follow the radar protocol convention.",
        "unit": "degrees",
    },
    "length": {"description": "Estimated object length.", "unit": "m"},
    "width": {"description": "Estimated object width.", "unit": "m"},
    "collision_detection_regions": {
        "description": "Bit mask of active collision-detection regions.",
        "unit": None,
    },
    "collision_detection_regions_label": {
        "description": "Pipe-separated names of all active collision-detection-region bits.",
        "unit": None,
    },
}

VALUE_DICTIONARIES = {
    "source": {
        "protocol": "Continental ARS40X radar CAN interface",
        "notes": (
            "Labels describe coded values present in Segcom cluster/object recordings. "
            "CSV files retain the original numeric value and add a *_label column."
        ),
    },
    "dictionaries": {
        "dynamic_property": DYNAMIC_PROPERTY,
        "pdh": PDH0_FALSE_ALARM_PROBABILITY,
        "ambiguity_state": AMBIGUITY_STATE,
        "invalid_flag": CLUSTER_INVALID_STATE,
        "measurement_state": MEASUREMENT_STATE,
        "probability_of_existence": PROBABILITY_OF_EXISTENCE,
        "object_class": OBJECT_CLASS,
        "collision_detection_regions": COLLISION_DETECTION_REGIONS,
    },
    "columns": COLUMN_DOCUMENTATION,
}


def normalize_scalar(value: Any) -> Any:
    """Convert NumPy values into CSV/JSON-safe Python scalars."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def integer_code(value: Any) -> int | None:
    value = normalize_scalar(value)
    if value == "" or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def enum_label(column: str, value: Any) -> str:
    code = integer_code(value)
    if code is None:
        return ""
    return ENUM_COLUMNS[column].get(code, f"unknown_{code}")


def collision_region_label(value: Any) -> str:
    mask = integer_code(value)
    if mask is None:
        return ""
    if mask == 0:
        return "none"
    labels = [label for bit, label in COLLISION_DETECTION_REGIONS.items() if mask & (1 << bit)]
    return "|".join(labels) if labels else f"unknown_mask_{mask}"


def output_fields(source_fields: Iterable[str]) -> list[str]:
    fields: list[str] = []
    for field in source_fields:
        fields.append(field)
        if field in ENUM_COLUMNS:
            fields.append(f"{field}_label")
        elif field == "collision_detection_regions":
            fields.append("collision_detection_regions_label")
    return fields


def convert_pcd(source: Path, destination: Path) -> int:
    if PointCloud is None:
        raise RuntimeError("Missing dependency. Install it with: python -m pip install pypcd4 numpy")
    cloud = PointCloud.from_path(str(source))
    fields = list(cloud.fields)
    values = cloud.numpy(fields)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fields(fields))
        writer.writeheader()
        for raw_row in values:
            row: dict[str, Any] = {}
            for field, raw_value in zip(fields, raw_row):
                value = normalize_scalar(raw_value)
                row[field] = value
                if field in ENUM_COLUMNS:
                    row[f"{field}_label"] = enum_label(field, value)
                elif field == "collision_detection_regions":
                    row["collision_detection_regions_label"] = collision_region_label(value)
            writer.writerow(row)
    return len(values)


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SEGCOM_METADATA_FILENAMES = {
    "recording.json",
    "timestamps.json",
    "recording.csv",
    "timestamps.csv",
}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def is_segcom_metadata(path: Path) -> bool:
    return path.is_file() and path.name.lower() in SEGCOM_METADATA_FILENAMES


def replace_pcd_reference(value: Any) -> Any:
    """Recursively replace PCD filenames in Segcom metadata with CSV filenames."""
    if isinstance(value, dict):
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower().endswith(".pcd"):
                key = key[:-4] + ".csv"
            converted[key] = replace_pcd_reference(item)
        return converted

    if isinstance(value, list):
        return [replace_pcd_reference(item) for item in value]

    if isinstance(value, str) and value.lower().endswith(".pcd"):
        return value[:-4] + ".csv"

    return value


def copy_segcom_metadata(source: Path, destination: Path) -> None:
    """Copy Segcom metadata while updating references to converted point clouds."""
    if source.suffix.lower() == ".csv":
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("r", newline="", encoding="utf-8") as in_file:
            reader = csv.reader(in_file)
            rows = []
            for row in reader:
                converted_row = []
                for cell in row:
                    if cell.lower().endswith(".pcd"):
                        converted_row.append(cell[:-4] + ".csv")
                    else:
                        converted_row.append(cell)
                rows.append(converted_row)
        with destination.open("w", newline="", encoding="utf-8") as out_file:
            writer = csv.writer(out_file)
            writer.writerows(rows)
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
        converted = replace_pcd_reference(data)

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(converted, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def write_dictionary(directory: Path) -> None:
    """Write the radar value dictionary beside converted CSV files."""
    (directory / DICTIONARY_FILENAME).write_text(
        json.dumps(VALUE_DICTIONARIES, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def convert_tree(
    source_root: Path,
    output_root: Path,
) -> tuple[int, int, int, int, int]:
    """Convert/copy the useful Segcom recording files recursively.

    Returns:
        (
            pcd_directories,
            pcd_files,
            points,
            images,
            metadata_files,
        )
    """
    pcd_directories: set[Path] = set()
    frame_count = 0
    point_count = 0
    image_count = 0
    metadata_count = 0

    for source_path in sorted(
        (path for path in source_root.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(source_root)),
    ):
        relative_path = source_path.relative_to(source_root)
        destination_path = output_root / relative_path
        suffix = source_path.suffix.lower()

        if suffix == ".pcd":
            destination_path = destination_path.with_suffix(".csv")
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            point_count += convert_pcd(source_path, destination_path)
            frame_count += 1
            pcd_directories.add(destination_path.parent)

        elif is_image_file(source_path):
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            image_count += 1

        elif is_segcom_metadata(source_path):
            copy_segcom_metadata(source_path, destination_path)
            metadata_count += 1

    for directory in pcd_directories:
        write_dictionary(directory)

    return (
        len(pcd_directories),
        frame_count,
        point_count,
        image_count,
        metadata_count,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert Segcom-Sensors_GUI recordings from PCD to CSV. "
            "Camera images and Segcom recording metadata are preserved."
        )
    )
    parser.add_argument(
        "source_folder",
        type=Path,
        help="Folder containing one or more Segcom recording directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    source_root = args.source_folder.expanduser().resolve()

    if not source_root.exists():
        print(f"Source folder does not exist: {source_root}", file=sys.stderr)
        return 1

    if not source_root.is_dir():
        print(f"Source path is not a directory: {source_root}", file=sys.stderr)
        return 1

    output_root = source_root.with_name(f"{source_root.name} - CSV")

    if output_root.exists():
        print(
            f"Output folder already exists: {output_root}\n"
            "Remove or rename it before running the conversion again.",
            file=sys.stderr,
        )
        return 1

    try:
        (
            directories,
            frames,
            points,
            images,
            metadata_files,
        ) = convert_tree(source_root, output_root)
    except Exception as error:
        # Avoid leaving a partial conversion that looks complete.
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1

    if frames == 0 and images == 0 and metadata_files == 0:
        print(f"No Segcom recording files were found inside: {source_root}")
        return 0

    print(f"Output: {output_root}")
    print(f"PCD folders: {directories}")
    print(f"PCD files converted: {frames}")
    print(f"Radar points converted: {points}")
    print(f"Images copied: {images}")
    print(f"Metadata files copied: {metadata_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
