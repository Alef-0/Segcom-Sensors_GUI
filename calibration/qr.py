"""QR creation and QReader decoding for camera calibration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_L


PAYLOAD_DIGITS = 12
PAYLOAD_MODULUS_MS = 10**PAYLOAD_DIGITS
QUIET_ZONE_MODULES = 2
GRID_LAYOUTS = {
    4: (2, 2),
    6: (2, 3),
    8: (2, 4),
    9: (3, 3),
    10: (2, 5),
    12: (2, 6),
}


def timestamp_payload(timestamp_ns: int) -> str:
    """Encode monotonic milliseconds in the compact calibration payload."""
    return f"{timestamp_ns // 1_000_000 % PAYLOAD_MODULUS_MS:0{PAYLOAD_DIGITS}d}"


def qr_matrix(payload: str, border: int = QUIET_ZONE_MODULES) -> np.ndarray:
    """Return a black/white QR matrix, including the requested quiet zone."""
    code = qrcode.QRCode(
        version=1,
        error_correction=ERROR_CORRECT_L,
        box_size=1,
        border=border,
    )
    code.add_data(payload, optimize=0)
    code.make(fit=True)
    return np.asarray(code.get_matrix(), dtype=np.uint8)


def decode_qrs(reader, image: np.ndarray) -> list[dict]:
    """Decode every QReader detection and retain its bounding-box geometry."""
    decoded, detections = reader.detect_and_decode(
        image=image,
        return_detections=True,
        is_bgr=True,
    )
    results = []
    for raw, detection in zip(decoded, detections or ()):
        box = np.asarray(detection["bbox_xyxy"], dtype=float).reshape(4)
        x1, y1, x2, y2 = box.tolist()
        results.append({
            "raw": raw,
            "bbox": box,
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "confidence": float(detection.get("confidence", 0.0)),
        })
    return results


def grid_shape(grid_qrs: int) -> tuple[int, int]:
    """Return the configured row and column count."""
    try:
        return GRID_LAYOUTS[int(grid_qrs)]
    except (KeyError, TypeError, ValueError) as error:
        choices = ", ".join(str(value) for value in GRID_LAYOUTS)
        raise ValueError(f"QR grid must contain one of: {choices}") from error


def grid_positions(grid_qrs: int) -> tuple[tuple[int, int], ...]:
    """Return cell positions in a snake order that preserves legacy 2x2 order."""
    rows, columns = grid_shape(grid_qrs)
    positions = []
    for row in range(rows):
        column_order = range(columns) if row % 2 == 0 else range(columns - 1, -1, -1)
        positions.extend((row, column) for column in column_order)
    return tuple(positions)


def grid_cell_names(grid_qrs: int) -> tuple[str, ...]:
    if grid_qrs == 4:
        return ("Top-left", "Top-right", "Bottom-right", "Bottom-left")
    return tuple(
        f"Row {row + 1}, column {column + 1}"
        for row, column in grid_positions(grid_qrs)
    )


def grid_bounds(
    width: int,
    height: int,
    grid_qrs: int,
) -> tuple[tuple[int, int, int, int], ...]:
    rows, columns = grid_shape(grid_qrs)
    return tuple(
        (
            width * column // columns,
            height * row // rows,
            width * (column + 1) // columns,
            height * (row + 1) // rows,
        )
        for row, column in grid_positions(grid_qrs)
    )


def cell_index_for(
    center: tuple[float, float],
    size: tuple[int, int],
    grid_qrs: int,
) -> int:
    """Classify a detection by its center in the configured grid."""
    x, y = center
    width, height = size
    rows, columns = grid_shape(grid_qrs)
    row = min(rows - 1, max(0, int(y * rows / max(1, height))))
    column = min(columns - 1, max(0, int(x * columns / max(1, width))))
    return grid_positions(grid_qrs).index((row, column))


def decode_qrs_with_grid_retries(
    reader,
    image: np.ndarray,
    grid_qrs: int,
) -> list[dict]:
    """Decode the full image, then retry cells with no readable QR value."""
    height, width = image.shape[:2]
    results = decode_qrs(reader, image)
    found = {
        cell_index_for(result["center"], (width, height), grid_qrs)
        for result in results
        if result["raw"] is not None
    }
    for cell, (left, top, right, bottom) in enumerate(
        grid_bounds(width, height, grid_qrs)
    ):
        if cell in found:
            continue
        for detection in decode_qrs(reader, image[top:bottom, left:right]):
            detection["bbox"] += np.array((left, top, left, top), dtype=float)
            detection["center"] = (
                detection["center"][0] + left,
                detection["center"][1] + top,
            )
            if cell_index_for(detection["center"], (width, height), grid_qrs) == cell:
                results.append(detection)
    return results


def decode_qrs_with_quadrant_retries(reader, image: np.ndarray) -> list[dict]:
    """Compatibility wrapper for the original four-cell layout."""
    return decode_qrs_with_grid_retries(reader, image, 4)


def create_qreader():
    """Create QReader while adapting QRDet's legacy Ultralytics precision argument."""
    from qreader import QReader

    reader = QReader(min_confidence=0.3)
    predict = reader.detector.model.predict

    @wraps(predict)
    def predict_with_quantize(*args, **kwargs):
        kwargs.pop("half", None)
        kwargs.setdefault("quantize", None)
        return predict(*args, **kwargs)

    reader.detector.model.predict = predict_with_quantize
    return reader


@dataclass(frozen=True, slots=True)
class Quadrant:
    index: int
    name: str


QUADRANTS = (
    Quadrant(0, "Top-left"),
    Quadrant(1, "Top-right"),
    Quadrant(2, "Bottom-right"),
    Quadrant(3, "Bottom-left"),
)


def quadrant_for(center: tuple[float, float], size: tuple[int, int]) -> Quadrant:
    """Classify a detection by its bounding-box center."""
    x, y = center
    width, height = size
    if y < height / 2:
        return QUADRANTS[0 if x < width / 2 else 1]
    return QUADRANTS[3 if x < width / 2 else 2]


def order_by_quadrant(detections: list[dict], size: tuple[int, int]) -> list[dict]:
    """Attach sectors and order detections clockwise, then spatially within a sector."""
    ordered = []
    for detection in detections:
        quadrant = quadrant_for(detection["center"], size)
        ordered.append({**detection, "quadrant": quadrant.index, "quadrant_name": quadrant.name})
    return sorted(ordered, key=lambda item: (
        item["quadrant"], item["center"][1], item["center"][0]
    ))


def order_by_cell(
    detections: list[dict],
    size: tuple[int, int],
    grid_qrs: int,
) -> list[dict]:
    """Attach configured grid cells and order detections by traversal position."""
    names = grid_cell_names(grid_qrs)
    ordered = []
    for detection in detections:
        cell = cell_index_for(detection["center"], size, grid_qrs)
        ordered.append({
            **detection,
            "cell": cell,
            "cell_name": names[cell],
            # Compatibility aliases for consumers of old four-quadrant results.
            "quadrant": cell,
            "quadrant_name": names[cell],
        })
    return sorted(ordered, key=lambda item: (
        item["cell"], item["center"][1], item["center"][0]
    ))
